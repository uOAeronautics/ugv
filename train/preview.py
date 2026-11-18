"""Run a trained keypoint detector (ONNX, CPU) on held-out views and save what it saw, for the Lab.

    cd train && uv run python preview.py --run ../runs/pose_train/kp-crop-v1 --n 48

Uses the same validation split as `pose_kp.py` (permutation seed 123). For each sample writes the
image (JPEG) and, in `preview.json`, the true and predicted keypoints, visibility, the PnP pose
error and the 12 model points reprojected through the solved pose, so the dashboard can draw the
skeleton the detector found next to the pose PnP made of it. Synthetic views: debugging, not evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.solve import pose_error, solve_pnp

FULL_FX = 968.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--min-span", type=float, default=0.35, help="keypoint extent / image side")
    args = ap.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    data = (
        (args.run / summary["data"]).resolve()
        if summary["data"].startswith("..")
        else Path(summary["data"])
    )
    if not data.exists():
        data = (Path.cwd() / summary["data"]).resolve()
    meta = json.loads((data / "meta.json").read_text())
    images = np.load(data / "images.npy", mmap_mode="r")
    kp, vis, poses = (np.load(data / f, mmap_mode="r") for f in ("kp.npy", "vis.npy", "pose.npy"))
    Ks = np.load(data / "K.npy") if (data / "K.npy").exists() else None
    n = len(images)
    perm = np.random.default_rng(123).permutation(n)
    val = np.sort(perm[: max(256, n // 12)])
    # Views where the whole airframe is in frame (all 12 keypoints inside the image) and spans at
    # least `min_span` of it: the point is to see the airframe, not a crop of a wingtip.
    h, w = images.shape[1:]
    inside = (
        (kp[val, :, 0] >= 0) & (kp[val, :, 0] < w) & (kp[val, :, 1] >= 0) & (kp[val, :, 1] < h)
    ).all(1)
    span = np.ptp(kp[val], axis=1).max(1)
    big = val[inside & (span >= args.min_span * min(h, w))]
    pick = big[np.linspace(0, len(big) - 1, min(args.n, len(big))).astype(int)]

    kpd = keypoints(load_airframe())
    names = meta["keypoints"]
    model_pts = np.array([kpd[k] for k in names])
    sess = ort.InferenceSession(str(args.run / "model.onnx"), providers=["CPUExecutionProvider"])
    out_dir = args.run / "preview"
    out_dir.mkdir(exist_ok=True)
    samples = []
    for j, i in enumerate(pick):
        img = np.asarray(images[i])
        x = (img.astype(np.float32) / 255)[None, None]
        pk, pv, _ = sess.run(None, {"image": x})
        pk, prob = pk[0].astype(np.float64), 1 / (1 + np.exp(-pv[0]))
        K = (Ks[i] if Ks is not None else np.array(meta["K"])).astype(np.float64)
        use = prob > 0.5
        sol = solve_pnp(model_pts[use], pk[use], K, reproj_px=3.0) if use.sum() >= 6 else None
        R_gt, t_gt = poses[i, :9].reshape(3, 3).astype(np.float64), poses[i, 9:].astype(np.float64)
        rec = {
            "file": f"preview/{j:03d}.jpg",
            "index": int(i),
            "size": [int(img.shape[1]), int(img.shape[0])],
            "px_to_1280": float(FULL_FX / K[0, 0]),
            "true": np.round(kp[i], 2).tolist(),
            "true_vis": [bool(v) for v in vis[i]],
            "pred": np.round(pk, 2).tolist(),
            "pred_prob": np.round(prob, 3).tolist(),
            "range_m": round(float(np.linalg.norm(t_gt)), 3),
            "K": np.round(K, 3).tolist(),
            "R_true": np.round(R_gt, 5).tolist(),
            "t_true": np.round(t_gt, 4).tolist(),
        }
        if sol is not None:
            R, t = sol[0], sol[1]
            rot, pos = pose_error(R, t, R_gt, t_gt)
            proj, _ = cv2.projectPoints(model_pts, cv2.Rodrigues(R)[0], t, K, None)
            rec |= {
                "pnp": {"rot_deg": round(float(rot), 2), "pos_cm": round(100 * float(pos), 1)},
                "R_pnp": np.round(R, 5).tolist(),
                "t_pnp": np.round(np.ravel(t), 4).tolist(),
                "reproj": np.round(proj.reshape(-1, 2), 2).tolist(),
            }
        cv2.imwrite(str(out_dir / f"{j:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        samples.append(rec)
    out = {"model": args.run.name, "data": data.name, "keypoints": names, "samples": samples}
    (out_dir / "preview.json").write_text(json.dumps(out, separators=(",", ":")))
    ok = [s for s in samples if "pnp" in s]
    print(f"{out_dir}: {len(samples)} samples, PnP solved {len(ok)}")


if __name__ == "__main__":
    main()
