"""Fan out Gazebo L6 eval shards across N parallel headless worlds.

Each shard is one subprocess running run_policy_episodes.py --bridge gazebo
with its own --out dir (fresh world_headless.sdf + gz log per world).
Topic isolation comes from lockstep_env.Gazebo, which mints a random
GZ_PARTITION per instance; we additionally force a distinct GZ_PARTITION
prefix per shard in the child env so topics can never collide even if the
uuid path changes. No shared dirs, no shared ports beyond loopback.

Sharding is by (policy, level, seed): episode RNG is seeded by
np.random.default_rng([seed, ep]), so shards of the same (policy, level)
MUST use different seeds to draw disjoint episodes. Aggregation pools
counts; time_median is a shard-size-weighted median of shard medians
(exact pooling only when per-episode lists exist, e.g. baseline script).

Usage:
  python simulation/gazebo/parallel_worlds.py --plan shards.json --workers 6
  python simulation/gazebo/parallel_worlds.py --smoke   # 6 x 3-ep baseline L1

Shard JSON: [{"policy": "baseline"|"student", "weights": "...", "level": "L1",
              "episodes": 10, "seed": 11, "name": "student_L2_a"}, ...]
Aggregation writes <out>/aggregate.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUNNER = HERE / "run_policy_episodes.py"

from drive_guard import require_drivable_car  # noqa: E402


def run_shard(sh: dict, out_root: Path, max_time: float) -> dict:
    require_drivable_car(car=sh.get("car"), world=sh.get("world"))
    name = sh["name"]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    js = out / f"{name}.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--bridge",
        "gazebo",
        "--level",
        sh["level"],
        "--episodes",
        str(sh["episodes"]),
        "--seed",
        str(sh["seed"]),
        "--out",
        str(out),
        "--json",
        str(js),
        "--max-time",
        str(max_time),
    ]
    if sh["policy"] == "baseline":
        cmd.append("--baseline")
    elif sh["policy"] == "student":
        cmd += ["--weights", sh["weights"]]
    else:
        raise ValueError(f"unknown policy {sh['policy']}")
    env = dict(__import__("os").environ, GZ_PARTITION=f"ugv-par-{uuid.uuid4().hex[:8]}-{name}")
    wall = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    wall_s = time.monotonic() - wall
    (out / f"{name}.stdout.log").write_text(p.stdout[-20000:])
    (out / f"{name}.stderr.log").write_text(p.stderr[-20000:])
    rec = {"name": name, "returncode": p.returncode, "wall_s": round(wall_s, 1)}
    if js.exists():
        summary = json.loads(js.read_text())
        rec["summary"] = summary
        n = summary.get("episodes", 0) or 0
        rec["eps_per_min"] = round(n / max(wall_s, 1e-6) * 60, 2)
        med = summary.get("time_median")
        if med is not None and n and wall_s:
            # approx sim D-steps per wall second: (median sim-s / DT) / (wall/n)
            rec["dsteps_per_s"] = round((med / 0.02) / (wall_s / n), 1)
    else:
        rec["error"] = "no json produced"
    print(
        json.dumps(
            {k: (v if k != "summary" else {kk: vv for kk, vv in v.items()}) for k, v in rec.items()}
        ),
        flush=True,
    )
    return rec


def aggregate(recs: list[dict]) -> dict:
    tot = {
        "episodes": 0,
        "success": 0,
        "collisions": 0,
        "timeouts": 0,
        "wall_s_max": 0.0,
        "wall_s_sum": 0.0,
    }
    meds = []
    for r in recs:
        s = r.get("summary")
        if not s:
            continue
        n = s.get("episodes", 0)
        tot["episodes"] += n
        tot["success"] += s.get("success", 0)
        tot["collisions"] += s.get("collisions", 0)
        tot["timeouts"] += s.get("timeouts", 0)
        tot["wall_s_max"] = max(tot["wall_s_max"], r.get("wall_s", 0))
        tot["wall_s_sum"] += r.get("wall_s", 0)
        if s.get("time_median") is not None:
            meds.append((s["time_median"], n))
    if meds:
        meds.sort()
        tot["time_median_weighted"] = round(
            sum(m * n for m, n in meds) / sum(n for _, n in meds), 2
        )
    tot["success_rate"] = round(tot["success"] / max(tot["episodes"], 1), 4)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--world", type=Path, default=HERE / "drone_docking_world.sdf")
    ap.add_argument("--car", choices=("freenove", "payload"), default=None)
    ap.add_argument("--out", type=Path, default=REPO / "rl/runs/gazebo_l6/parallel")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-time", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    require_drivable_car(world=a.world, car=a.car)
    if a.smoke:
        shards = [
            {
                "policy": "baseline",
                "level": "L1",
                "episodes": 3,
                "seed": 11 + i,
                "name": f"smoke_w{i}",
            }
            for i in range(6)
        ]
    else:
        shards = json.loads(Path(a.plan).read_text())
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(lambda sh: run_shard(sh, a.out, a.max_time), shards))
    agg = {"shards": recs, "aggregate": aggregate(recs), "wall_s": round(time.monotonic() - t0, 1)}
    (a.out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"aggregate": agg["aggregate"], "wall_s": agg["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
