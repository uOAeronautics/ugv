"""Pull and diagnose Auto Park runs recorded on adv034 (`~/ugv/runs/autopark/`).

uv run python tools/park/inspect_pi_run.py              # pull latest run from adv034 & inspect
uv run python tools/park/inspect_pi_run.py --frames     # also pull saved JPEG frames
uv run python tools/park/inspect_pi_run.py --local DIR  # inspect an already-pulled run dir
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO / "runs" / "autopark_pulled"


def pull_latest(run_id: str | None, with_frames: bool) -> Path:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    pick = f"'{run_id}'" if run_id else "$(ls -1 | sort | tail -n 1)"
    what = "$R" if with_frames else "$R/summary.json $R/telemetry.jsonl"
    remote = f"cd ~/ugv/runs/autopark && R={pick} && tar -czf - {what}"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "mac",
        f'ssh -o BatchMode=yes -o ConnectTimeout=8 advanced "{remote}"',
    ]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", str(LOCAL_ROOT)], stdin=p1.stdout)
    assert p1.stdout is not None
    p1.stdout.close()
    rc2 = p2.wait()
    rc1 = p1.wait()
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"pull failed (ssh={rc1}, tar={rc2})")
    if run_id:
        return LOCAL_ROOT / run_id
    return sorted(p for p in LOCAL_ROOT.iterdir() if p.is_dir())[-1]


def inspect_run(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "telemetry.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ticks = [r for r in rows if r.get("kind") == "tick"]
    vis = [r for r in rows if r.get("kind") == "vision"]
    if not ticks:
        print(f"{run_dir.name}: no ticks recorded")
        return

    ts = np.array([r["t"] for r in ticks], dtype=float)
    dts = np.diff(ts) * 1000.0 if len(ts) > 1 else np.array([0.0])
    dur = max(float(ts[-1] - ts[0]), 1e-3)
    print(f"=== {run_dir.name} ({summary.get('model')}) ===")
    print(
        f"Outcome: {summary.get('final_stage')} ({summary.get('final_status')}) | "
        f"dur={summary.get('duration_s')}s | ticks={len(ticks)} ({len(ticks) / dur:.1f} Hz, "
        f"dt p50/p95/max={np.median(dts):.0f}/{np.percentile(dts, 95):.0f}/{dts.max():.0f} ms)"
    )

    print("\nStage transitions:")
    prev = None
    for r in ticks:
        cur = (r["stage"], r["status"])
        if cur != prev:
            pose = r.get("pose")
            pstr = (
                f"({pose[0]:+.2f},{pose[1]:+.2f},{math.degrees(pose[2]):+.0f}°)" if pose else "None"
            )
            print(
                f"  t={r['t']:5.2f}s  {r['stage']:9s} | {r['status']:32s} | "
                f"pose={pstr:18s} | cmd=({r['forward']:+.2f},{r['strafe']:+.2f},{r['turn']:+.2f})"
            )
            prev = cur

    print("\nPer-stage summary:")
    for st in dict.fromkeys(r["stage"] for r in ticks):
        sub = [r for r in ticks if r["stage"] == st]
        fixes = sum(r.get("meas") is not None for r in sub)
        max_pwm = max(max(abs(v) for v in r["wheels"].values()) for r in sub)
        st_dur = sub[-1]["t"] - sub[0]["t"]
        print(
            f"  {st:9s}: {len(sub):4d} ticks ({st_dur:4.1f}s) | fixes={fixes:3d} | max_pwm={max_pwm:4d}"
        )

    w_cmd = np.array([r["cmd_vel"][2] for r in ticks], dtype=float)
    gyro = np.array([r["gyro_ccw"] for r in ticks], dtype=float)
    active = np.abs(w_cmd) > 0.08
    if active.any():
        err = w_cmd[active] - gyro[active]
        flips = int(np.sum(np.diff(np.sign(w_cmd[active])) != 0))
        over = int(
            np.sum(
                (np.sign(gyro[active]) == np.sign(w_cmd[active]))
                & (np.abs(gyro[active]) > np.abs(w_cmd[active]) * 1.4)
            )
        )
        print(
            f"\nYaw control ({active.sum()} active ticks): |w_cmd| med={np.median(np.abs(w_cmd[active])):.2f}, "
            f"|gyro| med/max={np.median(np.abs(gyro[active])):.2f}/{np.max(np.abs(gyro[active])):.2f}, "
            f"MAE={np.mean(np.abs(err)):.2f}, overshoot={over} ({100 * over / active.sum():.0f}%), flips={flips}"
        )

    ms_v = [v["ms"] for v in vis if v.get("ms") is not None]
    pts_v = [len(v.get("points") or []) for v in vis]
    if vis:
        print(
            f"Vision ({len(vis)} saved frames): pts p50={np.median(pts_v):.1f}, "
            f"zero_pts={sum(p == 0 for p in pts_v)}/{len(pts_v)}, "
            f"yolo_ms p50={np.median(ms_v):.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="specific run_id on adv034")
    ap.add_argument("--frames", action="store_true", help="also pull JPEG frames")
    ap.add_argument("--local", type=Path, default=None, help="inspect an already-pulled directory")
    args = ap.parse_args()
    run_dir = args.local if args.local is not None else pull_latest(args.run, args.frames)
    inspect_run(run_dir)


if __name__ == "__main__":
    main()
