"""Generate a 640x360 synthetic YOLO-pose dataset from the STRL-001 mesh and train yolo11n-pose.

    cd train && uv run python yolo_pose.py --data ../runs/yolo_pose_data/v1 --out ../runs/pose_train/yolo11n-pose-v1

Generates 16:9 grayscale car-camera frames (640x360) composited over real lab car-camera photos,
with pixel-perfect airframe bounding boxes (from the rendered silhouette `ids >= 0`) and all 15
named keypoints (`ugv.pose.mesh.keypoints(..., near=True)`), then trains `yolo11n-pose.pt` on
the GPU and exports a single-stage `model.onnx` (480x288, static batch=1) ready for the Pi 4.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ugv.park.camera import apply_image_faults
from ugv.park.geometry import CAMERA_X_M, CAMERA_Z_M, bay
from ugv.park.perception import CameraError
from ugv.pose.mesh import PARTS, keypoints, load_airframe
from ugv.pose.render import (
    CAR_VIEWS,
    Camera,
    CarLevelSpec,
    ViewSpec,
    procedural_texture,
    random_part_albedo,
    render,
    room_background,
    sample_bay_pose,
    sample_car_level_pose,
    sample_pose,
)

REPO = Path(__file__).resolve().parents[1]
W, H = 640, 360
# Horizontal-flip keypoint symmetry for the 15-point skeleton:
# 0:nose 1:tail 2:wingL 3:wingR 4:hubL 5:hubR 6:hubRear 7:vtailL 8:vtailR
# 9:gearN 10:gearL 11:gearR 12:tyreL 13:tyreR 14:tyreT
FLIP_IDX = [0, 1, 3, 2, 5, 4, 6, 8, 7, 9, 11, 10, 13, 12, 14]
_ST: dict = {}


def _strl_real_albedo(frame, rng: np.random.Generator) -> np.ndarray:
    """High-contrast white airframe body/wings with dark props, nacelles, metallic strut, and tyres."""
    body_l = rng.uniform(0.72, 0.98)
    dark_l = rng.uniform(0.05, 0.25)
    albedo = np.full((len(frame.tris), 3), body_l, dtype=np.float32)
    for part_name in ("prop_left", "prop_right", "prop_rear", "nacelle_left", "nacelle_right"):
        if part_name in PARTS:
            idx = PARTS.index(part_name)
            albedo[frame.part == idx] = rng.uniform(dark_l, dark_l + 0.12)
    # Darken the landing-gear struts (metallic gray) and tyres (dark rubber)
    floor = float(frame.tris[:, :, 2].min())
    strut_tri = frame.centroids[:, 2] < floor + 0.14
    albedo[strut_tri] = rng.uniform(0.25, 0.55)
    low_tri = frame.centroids[:, 2] < floor + 0.065
    albedo[low_tri] = rng.uniform(0.04, 0.18)
    return albedo


def _prepare_hdr_bgs(hdr_dir: Path, cache_dir: Path, max_hdrs: int = 120) -> list[str]:
    """Tonemap up to `max_hdrs` 2K panoramas into fast uint8 grayscale JPEGs for background diversity."""
    if not hdr_dir.is_dir():
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    hdrs = sorted(hdr_dir.glob("*.hdr"))[:max_hdrs]
    out_paths: list[str] = []
    for p in hdrs:
        dst = cache_dir / f"{p.stem}.jpg"
        if not dst.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(np.maximum(img, 0.0), cv2.COLOR_BGR2GRAY)
            # Simple Reinhard-style tonemap + gamma into [0, 255]
            p90 = float(np.percentile(gray, 90)) + 1e-4
            tm = gray / (gray + p90 * 0.5)
            u8 = np.clip((tm ** (1.0 / 2.2)) * 255.0, 0, 255).astype(np.uint8)
            u8 = cv2.resize(u8, (1280, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), u8, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if dst.is_file():
            out_paths.append(str(dst))
    return out_paths


def _init_worker(frames: list[str], hdr_bgs: list[str], out_dir: str, bay_frac: float) -> None:
    frame = load_airframe()
    _ST.update(
        frame=frame,
        kp=keypoints(frame, near=True),
        frames=frames,
        hdr_bgs=hdr_bgs,
        bay_frac=bay_frac,
        cam=Camera().scaled(W, H),
        fixed=procedural_texture(frame.centroids, seed=0),
        level=CarLevelSpec(floor_z=bay().floor_z, away=0.18),
        out_dir=Path(out_dir),
    )


def _bg(rng: np.random.Generator) -> np.ndarray | None:
    u = rng.random()
    if u < 0.12:
        return None
    use_hdr = bool(_ST["hdr_bgs"]) and (not _ST["frames"] or u < 0.62)
    pool = _ST["hdr_bgs"] if use_hdr else _ST["frames"]
    if not pool:
        return None
    img = cv2.imread(rng.choice(pool), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    if use_hdr:
        # Random horizontal roll (360 deg yaw) + lower-hemisphere crop mimicking a floor-level camera
        shift = int(rng.integers(0, iw))
        img = np.roll(img, shift, axis=1)
        cw = int(iw * rng.uniform(0.35, 0.65))
        ch = max(32, int(cw * H / W))
        ch = min(ch, ih)
        x0 = int(rng.integers(0, max(1, iw - cw + 1)))
        y_lo = int(ih * 0.25)
        y_hi = max(y_lo + 1, ih - ch + 1)
        y0 = int(rng.integers(y_lo, y_hi))
        img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
    else:
        s = rng.uniform(0.65, 1.0)
        cw, ch = int(iw * s), int(ih * s)
        x0, y0 = rng.integers(0, iw - cw + 1), rng.integers(0, ih - ch + 1)
        img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
    return np.clip(
        img.astype(np.float32) * rng.uniform(0.55, 1.45) + rng.uniform(-25, 25), 0, 255
    ).astype(np.uint8)


def _gen_one(args: tuple[str, int, int]) -> bool:
    split, idx, seed = args
    rng = np.random.default_rng(seed)
    st = _ST
    u = rng.random()
    bay_f = float(st["bay_frac"])
    level_f = max(0.35, 0.85 - bay_f)
    if u < level_f:
        R, t, _ = sample_car_level_pose(rng, st["level"])
    elif u < level_f + bay_f:
        b = bay()
        R, t = sample_bay_pose(rng, b.floor_z, b.target, CAMERA_X_M, CAMERA_Z_M)
    else:
        spec = CAR_VIEWS if u < 0.94 else ViewSpec(distance_m=(0.4, 3.8))
        R, t = sample_pose(rng, st["cam"], spec)

    u_alb = rng.random()
    if u_alb < 0.45:
        albedo = _strl_real_albedo(st["frame"], rng)
    elif u_alb < 0.80:
        albedo = random_part_albedo(st["frame"], rng)
    elif u_alb < 0.90:
        albedo = np.full_like(st["fixed"], rng.uniform(0.45, 0.95))
    else:
        albedo = procedural_texture(st["frame"].centroids, int(rng.integers(1 << 30)))

    bg = room_background(rng, st["cam"], R) if u < 0.80 and rng.random() < 0.25 else _bg(rng)
    v = render(st["frame"], st["kp"], R, t, st["cam"], rng, albedo, bg)

    g = cv2.cvtColor(v.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if rng.random() < 0.20:
        klen = int(rng.choice([5, 7, 9]))
        k = np.zeros((klen, klen), np.float32)
        k[klen // 2] = 1.0 / klen
        rot = cv2.getRotationMatrix2D((klen / 2 - 0.5, klen / 2 - 0.5), rng.uniform(0, 180), 1)
        g = cv2.filter2D(g, -1, cv2.warpAffine(k, rot, (klen, klen)))
    g = 255 * (np.clip(g, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
    gray = np.clip(g, 0, 255).astype(np.uint8)

    fx: dict = {}
    if rng.random() < 0.35:
        if rng.random() < 0.5:
            fx["blur"] = float(rng.uniform(0.4, 3.5))
        if rng.random() < 0.5:
            fx["exposure"] = float(rng.uniform(-1.2, 2.0))
    noise_px = float(rng.uniform(0.2, 1.5))
    gray = apply_image_faults(gray, CameraError(), fx, noise_px, rng, {})

    img_path = st["out_dir"] / "images" / split / f"{idx:06d}.jpg"
    lbl_path = st["out_dir"] / "labels" / split / f"{idx:06d}.txt"
    cv2.imwrite(str(img_path), gray, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ys, xs = np.nonzero(v.ids >= 0)
    if len(xs) < 60:
        lbl_path.write_text("")
        return False

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    if bw < 0.02 or bh < 0.02:
        lbl_path.write_text("")
        return False
    cx, cy = 0.5 * (x0 + x1) / W, 0.5 * (y0 + y1) / H

    names = list(st["kp"])
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    n_vis = 0
    for n in names:
        ku, kv = v.kp2d[n]
        in_frame = bool(v.kp_visible[n]) and 0.0 <= ku < W and 0.0 <= kv < H
        if in_frame:
            parts.append(f"{ku / W:.6f} {kv / H:.6f} 2")
            n_vis += 1
        else:
            parts.append("0.000000 0.000000 0")
    if n_vis < 2:
        lbl_path.write_text("")
        return False
    lbl_path.write_text(" ".join(parts) + "\n")
    return True


def generate_dataset(
    data_dir: Path,
    n_train: int,
    n_val: int,
    workers: int,
    seed: int,
    bay_frac: float = 0.35,
    hdr_dir: Path | None = None,
) -> Path:
    for split in ("train", "val"):
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    frames = sorted(str(p) for p in (REPO / "calibration" / "runs").glob("**/*.jpg"))
    hdr_bgs = (
        _prepare_hdr_bgs(hdr_dir, data_dir.parent / "hdr_bgs_cache") if hdr_dir is not None else []
    )
    jobs = [("train", i, seed + i) for i in range(n_train)] + [
        ("val", i, seed + 10_000_000 + i) for i in range(n_val)
    ]
    print(
        f"Generating {n_train} train + {n_val} val frames ({W}x{H}, "
        f"{len(frames)} real bgs, {len(hdr_bgs)} HDR panoramas, bay={bay_frac:.0%})..."
    )
    pos = 0
    with ProcessPoolExecutor(
        workers, initializer=_init_worker, initargs=(frames, hdr_bgs, str(data_dir), bay_frac)
    ) as ex:
        for i, ok in enumerate(ex.map(_gen_one, jobs, chunksize=64), 1):
            pos += int(ok)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} (positive: {pos})", flush=True)

    yaml_path = data_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {data_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"kpt_shape: [15, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        f"names:\n"
        f"  0: strl001\n"
    )
    return yaml_path


def _export_int8(fp32_onnx: Path, int8_onnx: Path, val_dir: Path, img_w: int, img_h: int) -> None:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    prep = int8_onnx.parent / "prep.onnx"
    quant_pre_process(str(fp32_onnx), str(prep), skip_symbolic_shape=False)
    om = onnx.load(str(prep))
    head_nodes = [n.name for n in om.graph.node if "/model.23/" in n.name]
    val_imgs = sorted(val_dir.glob("*.jpg"))[:120]

    class _Reader(CalibrationDataReader):
        def __init__(self, paths: list[Path]) -> None:
            self.paths = paths
            self.i = 0

        def get_next(self):
            if self.i >= len(self.paths):
                return None
            img = cv2.imread(str(self.paths[self.i]), cv2.IMREAD_GRAYSCALE)
            self.i += 1
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
            scale = min(img_w / 1280, img_h / 720)
            rw, rh = round(1280 * scale), round(720 * scale)
            px, py = (img_w - rw) // 2, (img_h - rh) // 2
            canvas = np.full((img_h, img_w), 114, dtype=np.uint8)
            canvas[py : py + rh, px : px + rw] = cv2.resize(
                img, (rw, rh), interpolation=cv2.INTER_AREA
            )
            x = np.broadcast_to(
                (canvas.astype(np.float32) / 255.0)[None, None], (1, 3, img_h, img_w)
            )
            return {"images": np.ascontiguousarray(x)}

    quantize_static(
        str(prep),
        str(int8_onnx),
        _Reader(val_imgs),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=head_nodes,
    )
    prep.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    ap.add_argument("--hdr-dir", type=Path, default=REPO / "runs/cache/hdri_v2")
    ap.add_argument("--bay-frac", type=float, default=0.35)
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz-w", type=int, default=320)
    ap.add_argument("--imgsz-h", type=int, default=192)
    ap.add_argument("--export-int8", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    yaml_path = args.data / "dataset.yaml"
    if not args.skip_gen or not yaml_path.is_file():
        yaml_path = generate_dataset(
            args.data,
            args.train_n,
            args.val_n,
            args.workers,
            args.seed,
            bay_frac=args.bay_frac,
            hdr_dir=args.hdr_dir,
        )

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz_w,
        batch=args.batch,
        device=0,
        workers=8,
        project=str(args.out.resolve()),
        name="run",
        exist_ok=True,
        patience=10,
        mosaic=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.15,
        scale=0.35,
        fliplr=0.5,
        pose=18.0,
        verbose=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    best = YOLO(str(best_pt))
    exported = Path(
        best.export(
            format="onnx",
            imgsz=[args.imgsz_h, args.imgsz_w],
            simplify=True,
            opset=17,
            dynamic=False,
        )
    )
    fp32_dst = args.out / ("model_fp32.onnx" if args.export_int8 else "model.onnx")
    shutil.copy2(exported, fp32_dst)
    if args.export_int8:
        int8_dst = args.out / "model.onnx"
        _export_int8(fp32_dst, int8_dst, args.data / "images" / "val", args.imgsz_w, args.imgsz_h)
        dst = int8_dst
    else:
        dst = fp32_dst
    meta = {
        "type": "yolo-pose",
        "arch": "yolo11n-pose",
        "imgsz": [args.imgsz_h, args.imgsz_w],
        "quant": "int8-qdq" if args.export_int8 else "fp32",
        "keypoints": list(keypoints(load_airframe(), near=True)),
        "train_n": args.train_n,
        "val_n": args.val_n,
        "bay_frac": args.bay_frac,
        "epochs": args.epochs,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {dst} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
