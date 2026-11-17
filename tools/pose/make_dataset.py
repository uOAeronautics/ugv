"""Render a keypoint training set: the airframe over real car-camera frames, domain-randomised.

    uv run python tools/pose/make_dataset.py --out runs/pose_data/v1 --n 24000

Each sample: a car-camera view (grayscale, the real camera is mono) of the airframe at a random
pose, drawn over a random real frame from `calibration/runs/**/frames` (70%) or a noise field,
with a random or fixed texture, lighting, blur, sensor noise, gamma and occluders. Labels: the
12 keypoints in pixels, their visibility, and the pose. CPU only, S0.

Writes images.npy (N, H, W) uint8, kp.npy (N, 12, 2) float32, vis.npy (N, 12) bool,
pose.npy (N, 12) float32 (R row-major, t), meta.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.render import CAR_VIEWS, Camera, ViewSpec, procedural_texture, render, sample_pose

REPO = Path(__file__).resolve().parents[2]
W, H = 320, 180
_STATE: dict = {}


def _init(frames: list[str]) -> None:
    frame = load_airframe()
    _STATE.update(frame=frame, kp=keypoints(frame), frames=frames, cam=Camera().scaled(W, H))
    _STATE["fixed"] = procedural_texture(frame.centroids, seed=0)


def _background(rng: np.random.Generator) -> np.ndarray | None:
    if not _STATE["frames"] or rng.random() < 0.3:
        return None
    img = cv2.imread(rng.choice(_STATE["frames"]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = rng.uniform(0.6, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
    return np.clip(
        img.astype(np.float32) * rng.uniform(0.5, 1.5) + rng.uniform(-30, 30), 0, 255
    ).astype(np.uint8)


def _one(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    st = _STATE
    spec = CAR_VIEWS if rng.random() < 0.6 else ViewSpec(distance_m=(0.4, 4.0))
    R, t = sample_pose(rng, st["cam"], spec)
    albedo = (
        st["fixed"]
        if rng.random() < 0.3
        else procedural_texture(st["frame"].centroids, int(rng.integers(1 << 30)))
    )
    if rng.random() < 0.3:  # plain airframe paint, as the real one may be
        albedo = np.full_like(albedo, rng.uniform(0.4, 0.95))
    v = render(st["frame"], st["kp"], R, t, st["cam"], rng, albedo, _background(rng))
    g = cv2.cvtColor(v.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if rng.random() < 0.6:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.4))
    if rng.random() < 0.2:  # motion blur
        k = np.zeros((7, 7), np.float32)
        k[3] = 1 / 7
        g = cv2.filter2D(
            g,
            -1,
            cv2.warpAffine(k, cv2.getRotationMatrix2D((3, 3), rng.uniform(0, 180), 1), (7, 7)),
        )
    g = 255 * (np.clip(g, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    g += rng.normal(0, rng.uniform(1, 8), g.shape)
    for _ in range(rng.integers(0, 3) if rng.random() < 0.3 else 0):  # occluders
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.rectangle(
            g,
            (x, y),
            (x + rng.integers(10, 80), y + rng.integers(10, 60)),
            float(rng.uniform(0, 255)),
            -1,
        )
    names = list(st["kp"])
    kp = np.array([v.kp2d[n] for n in names], np.float32)
    vis = np.array([v.kp_visible[n] for n in names], bool)
    pose = np.concatenate([R.ravel(), t]).astype(np.float32)
    return np.clip(g, 0, 255).astype(np.uint8), kp, vis, pose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    frames = sorted(str(p) for p in (REPO / "calibration" / "runs").glob("**/*.jpg"))
    args.out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(args.out / "images.npy", "w+", np.uint8, (args.n, H, W))
    kps = np.zeros((args.n, 12, 2), np.float32)
    vis = np.zeros((args.n, 12), bool)
    poses = np.zeros((args.n, 12), np.float32)
    seeds = [args.seed * 10_000_019 + i for i in range(args.n)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(frames,)) as ex:
        for i, (img, kp, v, pose) in enumerate(ex.map(_one, seeds, chunksize=64)):
            images[i], kps[i], vis[i], poses[i] = img, kp, v, pose
            if i % 2000 == 0:
                print(f"{i}/{args.n}", flush=True)
    images.flush()
    np.save(args.out / "kp.npy", kps)
    np.save(args.out / "vis.npy", vis)
    np.save(args.out / "pose.npy", poses)
    cam = Camera().scaled(W, H)
    meta = {
        "n": args.n,
        "width": W,
        "height": H,
        "K": cam.K.tolist(),
        "keypoints": list(keypoints(load_airframe())),
        "real_backgrounds": len(frames),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    sheet = np.vstack([np.hstack(list(images[r * 6 : r * 6 + 6])) for r in range(4)])
    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for i in range(24):
        for (u, w_), ok in zip(kps[i], vis[i], strict=True):
            if ok:
                cv2.circle(
                    sheet, (int(u) + (i % 6) * W, int(w_) + (i // 6) * H), 2, (0, 0, 255), -1
                )
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet)
    print("done", meta)


if __name__ == "__main__":
    main()
