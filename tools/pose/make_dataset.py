"""Render a keypoint training set: the airframe over real car-camera frames, domain-randomised.

    uv run python tools/pose/make_dataset.py --out runs/pose_data/v1 --n 24000

Each sample: a car-camera view (grayscale, the real camera is mono). 55% use the real car's
level camera 6.4 cm off the floor, anywhere 0.25-3.5 m around the airframe and facing anywhere
(`render.sample_car_level_pose`): partial views, and 20% of them facing away (no airframe in
the picture, all keypoints invisible). `--bay` takes a share from the 25% low views aimed at the
airframe for views on the way into the bay (`render.sample_bay_pose`: nose gear and tyres in
view, the rest cut off); 20% are anywhere on the sphere. Backgrounds: real car frames from `calibration/runs/**/frames`, noise fields, or a
clean bright room (`render.room_background`). Random or fixed texture, lighting, blur, sensor
noise, gamma and occluders. Labels: the 15 keypoints (12 skeleton + 3 nose-tyre) in pixels, their visibility, and the
pose. CPU only, S0.

`--crop` makes the second-stage set instead: a square crop of the full 1280x720 camera, rendered
directly with the crop's own intrinsics at 256x256. For level views the crop is the one the
deployed first stage proposes (box of the keypoints in the picture, 1.1-1.6x), or a random box
when fewer than two are in view (stage 2 must call it empty); otherwise the airframe's extent.

Writes images.npy (N, H, W) uint8, kp.npy (N, 15, 2) float32, vis.npy (N, 15) bool,
pose.npy (N, 12) float32 (R row-major, t), K.npy (N, 3, 3) per-sample intrinsics, meta.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ugv.pose.dataset import CROP, REPO, H, W, init_worker, sample
from ugv.pose.mesh import keypoints, load_airframe

NKP = 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--bay", type=float, default=0.0, help="share of views on the way into the bay (0 = v2 mix)"
    )
    ap.add_argument("--crop", action="store_true", help="second-stage crops at 256x256")
    args = ap.parse_args()
    h, w = (CROP, CROP) if args.crop else (H, W)
    frames = sorted(str(p) for p in (REPO / "calibration" / "runs").glob("**/*.jpg"))
    args.out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(args.out / "images.npy", "w+", np.uint8, (args.n, h, w))
    Ks = np.zeros((args.n, 3, 3), np.float32)
    kps = np.zeros((args.n, NKP, 2), np.float32)
    vis = np.zeros((args.n, NKP), bool)
    poses = np.zeros((args.n, 12), np.float32)
    seeds = [args.seed * 10_000_019 + i for i in range(args.n)]
    with ProcessPoolExecutor(
        args.workers, initializer=init_worker, initargs=(frames, args.crop, args.bay)
    ) as ex:
        for i, (img, kp, v, pose, K) in enumerate(ex.map(sample, seeds, chunksize=64)):
            images[i], kps[i], vis[i], poses[i], Ks[i] = img, kp, v, pose, K
            if i % 2000 == 0:
                print(f"{i}/{args.n}", flush=True)
    images.flush()
    np.save(args.out / "kp.npy", kps)
    np.save(args.out / "vis.npy", vis)
    np.save(args.out / "pose.npy", poses)
    np.save(args.out / "K.npy", Ks)
    meta = {
        "n": args.n,
        "width": w,
        "height": h,
        "crop": args.crop,
        "K": Ks[0].tolist(),
        "keypoints": list(keypoints(load_airframe(), near=True)),
        "real_backgrounds": len(frames),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    sheet = np.vstack([np.hstack(list(images[r * 6 : r * 6 + 6])) for r in range(4)])
    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for i in range(24):
        for (u, v_), ok in zip(kps[i], vis[i], strict=True):
            if ok:
                cv2.circle(
                    sheet, (int(u) + (i % 6) * w, int(v_) + (i // 6) * h), 2, (0, 0, 255), -1
                )
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet)
    print("done", meta)


if __name__ == "__main__":
    main()
