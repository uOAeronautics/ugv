"""Smoke test of the synthetic pose pipeline: render views, then recover pose two ways.

1. Keypoint PnP: ground-truth 2D keypoints plus Gaussian pixel noise (an upper bound for a
   learned keypoint detector of that accuracy).
2. SIFT bank: reference renders with known pose, matched against held-out query renders.

    uv run python tools/pose/smoke.py --out runs/pose_smoke

Writes a contact sheet and `summary.json`; prints the medians. CPU only, S0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.render import CAR_VIEWS, Camera, ViewSpec, procedural_texture, render, sample_pose
from ugv.pose.solve import pose_error, sift_pose, sift_reference, solve_pnp


def _views(n, rng, frame, kp, cam, spec, albedo):
    return [render(frame, kp, *sample_pose(rng, cam, spec), cam, rng, albedo) for _ in range(n)]


def _summary(errs: list[tuple[float, float] | None]) -> dict[str, float]:
    ok = [e for e in errs if e is not None]
    rot = np.array([e[0] for e in ok]) if ok else np.array([np.nan])
    pos = np.array([e[1] for e in ok]) if ok else np.array([np.nan])
    hit = sum(e is not None and e[0] < 5 and e[1] < 0.10 for e in errs)
    return {
        "n": len(errs),
        "solved": len(ok),
        "within_5deg_10cm": hit,
        "median_rot_deg": float(np.median(rot)),
        "median_pos_m": float(np.median(pos)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("runs/pose_smoke"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--bank", type=int, default=48)
    ap.add_argument(
        "--views",
        choices=["generic", "car"],
        default="generic",
        help="car: camera 6 cm off the floor",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frame = load_airframe()
    kp = keypoints(frame)
    cam, spec = Camera(), (CAR_VIEWS if args.views == "car" else ViewSpec())
    albedo = procedural_texture(frame.centroids, seed=0)
    K = cam.K

    queries = _views(args.queries, rng, frame, kp, cam, spec, albedo)
    sheet = np.vstack([np.hstack([v.image for v in queries[r * 4 : r * 4 + 4]]) for r in range(2)])
    for i, v in enumerate(queries[:8]):
        for name, (u, w) in v.kp2d.items():
            if v.kp_visible[name]:
                c = (int(u) + (i % 4) * cam.width, int(w) + (i // 4) * cam.height)
                cv2.circle(sheet, c, 4, (0, 0, 255), -1)
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet)

    results = {"views": args.views, "keypoints": list(kp), "camera": vars(cam) | {"K": K.tolist()}}
    for sigma in (0.0, 1.0, 3.0):
        errs = []
        for v in queries:
            names = [n for n in kp if v.kp_visible[n]]
            p2 = np.array([v.kp2d[n] for n in names]) + rng.normal(0, sigma, (len(names), 2))
            sol = solve_pnp(np.array([kp[n] for n in names]), p2, K) if names else None
            errs.append(pose_error(sol[0], sol[1], v.R, v.t) if sol else None)
        results[f"keypoint_pnp_sigma{sigma:g}"] = _summary(errs)

    sift = cv2.SIFT_create(nfeatures=800)
    bank = [
        sift_reference(v, frame, sift) for v in _views(args.bank, rng, frame, kp, cam, spec, albedo)
    ]
    errs = []
    for v in queries:
        sol = sift_pose(v.image, bank, K, sift)
        errs.append(pose_error(sol[0], sol[1], v.R, v.t) if sol else None)
    results["sift_bank"] = _summary(errs) | {"bank_views": args.bank}

    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "n" in v:
            print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
