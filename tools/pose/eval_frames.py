"""Score the trained YOLO-pose detector on recorded challenge frames (`ugv-lab --record <dir>`).

    uv run python tools/pose/eval_frames.py runs/challenge_frames/diag1 [--run yolo11n-pose-v3-416-int8]
        [--threshold 0.5]

Prints recall, precision, left/right swaps, pixel error and false alarms on frames with no
airframe in view, overall and by range; writes <dir>/eval-<variant>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ugv.pose.detector import TRAIN_DIR, YoloPoseDetector, default_runs
from ugv.pose.frame_eval import load, score, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--run", help="model run name under runs/pose_train")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-kp", type=int, help="score the first n keypoints")
    ap.add_argument("--hit-px", type=float, default=20.0)
    args = ap.parse_args()
    run_dir = (TRAIN_DIR / args.run) if args.run else default_runs()[0]
    det = YoloPoseDetector(run_dir, conf_threshold=args.threshold, kpt_threshold=args.threshold)
    res = summary(
        score(load(args.root), lambda f: det(f).points, args.root, args.hit_px, args.n_kp)
    )
    variant = f"{run_dir.name}-t{args.threshold}"
    (args.root / f"eval-{variant}.json").write_text(json.dumps(res, indent=1))
    print(variant, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
