"""Dream-RSI style experiment archive and Pareto frontier tracker (arXiv:2609.14858).

Backed by rl/runs/archive.jsonl. Records each run's lineage (run_id, parent_id),
track (teacher|student|baseline|oracle), level, config_delta, wall_s, eval_metrics,
and pareto_score, and computes non-dominated Pareto frontiers per level.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
ARCHIVE_PATH = RUNS / "archive.jsonl"


def compute_pareto_score(metrics: dict) -> float:
    succ_lb = float(metrics.get("success_lb", 0.0) or 0.0)
    coll = float(metrics.get("collision_rate", 0.0) or 0.0)
    fp = float(metrics.get("false_parked_rate", 0.0) or 0.0)
    t_med = float(metrics.get("time_median", 45.0) or 45.0)
    return round(succ_lb - 2.0 * coll - 2.0 * fp - 0.005 * t_med, 6)


def load_archive(path: Path = ARCHIVE_PATH) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def save_archive(entries: list[dict], path: Path = ARCHIVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def record_run(
    run_id: str,
    parent_id: str | None,
    track: str,
    level: str,
    config_delta: dict,
    wall_s: float | None,
    eval_metrics: dict,
    path: Path = ARCHIVE_PATH,
) -> dict:
    entry = {
        "run_id": run_id,
        "parent_id": parent_id,
        "track": track,
        "level": level,
        "config_delta": config_delta,
        "wall_s": wall_s,
        "eval_metrics": eval_metrics,
        "pareto_score": compute_pareto_score(eval_metrics),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    entries = [e for e in load_archive(path) if e["run_id"] != run_id]
    entries.append(entry)
    save_archive(entries, path)
    return entry


def _dominates(a: dict, b: dict) -> bool:
    ma, mb = a["eval_metrics"], b["eval_metrics"]
    sa, sb = float(ma.get("success_lb", 0.0)), float(mb.get("success_lb", 0.0))
    ca, cb = float(ma.get("collision_rate", 1.0)), float(mb.get("collision_rate", 1.0))
    ta, tb = float(ma.get("time_median", 1e9)), float(mb.get("time_median", 1e9))
    not_worse = (sa >= sb) and (ca <= cb) and (ta <= tb)
    strict = (sa > sb) or (ca < cb) or (ta < tb)
    return not_worse and strict


def pareto_frontier(
    entries: list[dict], level: str | None = None, track: str | None = None
) -> list[dict]:
    pool = [
        e
        for e in entries
        if (level is None or e["level"] == level) and (track is None or e["track"] == track)
    ]
    front = []
    for cand in pool:
        same_lvl = [
            o for o in pool if o["level"] == cand["level"] and o["run_id"] != cand["run_id"]
        ]
        if not any(_dominates(o, cand) for o in same_lvl):
            front.append(cand)
    return sorted(front, key=lambda e: (e["level"], -e["pareto_score"]))


def format_frontier(entries: list[dict], level: str | None = None, track: str | None = None) -> str:
    front = pareto_frontier(entries, level=level, track=track)
    lines = ["run_id | parent | track | lvl | succ/eps (lb) | coll | med_s | pareto"]
    lines.append("-" * 78)
    for e in front:
        m = e["eval_metrics"]
        succ = m.get("success", 0)
        eps = m.get("episodes", 0)
        lb = float(m.get("success_lb", 0.0))
        coll = m.get("collisions", 0)
        med = float(m.get("time_median", 0.0))
        lines.append(
            f"{e['run_id']} | {e['parent_id'] or '-'} | {e['track']} | {e['level']} | "
            f"{succ}/{eps} ({lb:.3f}) | {coll} | {med:.2f}s | {e['pareto_score']:.4f}"
        )
    return "\n".join(lines)


def seed_existing(path: Path = ARCHIVE_PATH) -> list[dict]:
    seeds = [
        (
            "teacher_L0_gate",
            None,
            "teacher",
            "L0",
            {"rate_w": 0.0, "yaw_w_train": 0.3, "envs": 512, "iters": 120},
            None,
            RUNS / "teacher_L0_gate.json",
        ),
        (
            "student_L0_lat_s1",
            None,
            "student",
            "L0",
            {"hidden": 64, "head_w": 64, "lat_w": 2.0, "iters": 300, "seed": 1},
            None,
            RUNS / "student_L0_lat_s1.json",
        ),
    ]
    for p in sorted(RUNS.glob("baseline_ekf_staged_L*.json")):
        lvl = p.stem.split("_")[-1]
        seeds.append((p.stem, None, "baseline", lvl, {"controller": "ekf_staged"}, None, p))
    for p in sorted(RUNS.glob("oracle_pd_L*.json")):
        lvl = p.stem.split("_")[-1]
        seeds.append((p.stem, None, "oracle", lvl, {"controller": "oracle_pd"}, None, p))
    recorded = []
    for run_id, parent_id, track, lvl, delta, wall_s, json_path in seeds:
        if json_path.exists():
            metrics = json.loads(json_path.read_text())
            recorded.append(
                record_run(run_id, parent_id, track, lvl, delta, wall_s, metrics, path=path)
            )
    return recorded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="seed archive from existing runs/*.json")
    ap.add_argument("--frontier", action="store_true", help="print Pareto frontier summary")
    ap.add_argument("--level", default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--config-delta", default="{}")
    ap.add_argument("--wall-s", type=float, default=None)
    ap.add_argument("--json", default=None, help="eval JSON file to record")
    args = ap.parse_args()
    if args.seed:
        n = len(seed_existing())
        print(f"seeded {n} runs into {ARCHIVE_PATH}")
    if args.record:
        if not args.run_id or not args.track or not args.level or not args.json:
            ap.error("--record requires --run-id, --track, --level, --json")
        metrics = json.loads(Path(args.json).read_text())
        delta = json.loads(args.config_delta)
        e = record_run(
            args.run_id, args.parent_id, args.track, args.level, delta, args.wall_s, metrics
        )
        print(f"recorded {e['run_id']} pareto={e['pareto_score']:.4f}")
    if args.frontier or (not args.seed and not args.record):
        print(format_frontier(load_archive(), level=args.level, track=args.track))


if __name__ == "__main__":
    main()
