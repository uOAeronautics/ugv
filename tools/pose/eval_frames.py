"""Score the trained keypoint detector on recorded challenge frames (`ugv-lab --record <dir>`).

    uv run python tools/pose/eval_frames.py runs/challenge_frames/diag1 [--runs kp-v2,kp-crop-v2]
        [--stage1-only] [--threshold 0.5]

Prints recall, precision, left/right swaps, pixel error and false alarms on frames with no
airframe in view, overall and by range; writes <dir>/eval-<variant>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ugv.pose.detector import TRAIN_DIR, KeypointDetector, default_runs
from ugv.pose.frame_eval import load, score, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--runs", help="whole,crop run names under runs/pose_train")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--hit-px", type=float, default=20.0)
    args = ap.parse_args()
    whole, crop = (TRAIN_DIR / n for n in args.runs.split(",")) if args.runs else default_runs()
    det = KeypointDetector(whole, None if args.stage1_only else crop, threshold=args.threshold)
    res = summary(score(load(args.root), lambda f: det(f).points, args.root, args.hit_px))
    variant = f"{whole.name}-{'s1' if args.stage1_only else crop.name}-t{args.threshold}"
    (args.root / f"eval-{variant}.json").write_text(json.dumps(res, indent=1))
    print(variant, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
