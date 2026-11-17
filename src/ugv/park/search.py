"""Failure-guided search over the controller's parameters, kept as a discovery tree.

AutoRL-style outer loop (arXiv:2611.17790): propose a candidate, score it on a fixed batch of
episodes, keep it only if it beats its parent by more than noise (PASS), record it either way.
Dream-RSI-style memory (arXiv:2609.14858): every candidate is a node with a parent, its
parameter change, its score and its failure modes, appended to `tree.jsonl`, so later searches
(and people) can see which branches were tried and why they failed.

Proposals are guided by the parent's failures: each failure outcome names the parameters that
plausibly cause it (`LEVERS`), and mutation favours those. Every candidate in a run is scored on
the same episodes (common random numbers); the final pick is re-scored on held-out seeds.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from ugv.park.controller import ParkParams
from ugv.park.evaluate import Conditions, evaluate, wilson

# Searchable parameters: (low, high). Integers are rounded.
SPACE = {
    "orbit_r": (0.8, 1.4),
    "orbit_v": (0.12, 0.4),
    "k_radial": (0.5, 3.0),
    "k_face": (1.0, 5.0),
    "orbit_tol_deg": (3.0, 12.0),
    "approach_v": (0.08, 0.3),
    "k_lat": (0.8, 5.0),
    "k_head": (1.0, 5.0),
    "enter_d": (0.25, 0.6),
    "align_lat_m": (0.01, 0.05),
    "align_yaw_deg": (3.0, 12.0),
    "abort_lat_m": (0.015, 0.045),
    "abort_yaw_deg": (5.0, 15.0),
    "settle_s": (0.1, 0.6),
    "measure_s": (0.2, 1.0),
    "step_v": (0.04, 0.15),
    "step_w": (0.2, 0.9),
    "step_frac": (0.3, 0.9),
    "max_step_m": (0.04, 0.2),
    "lat_tol_m": (0.003, 0.015),
    "yaw_tol_deg": (0.5, 3.0),
    "stop_tol_m": (0.003, 0.015),
    "shield_inflate_m": (0.0, 0.02),
    "max_retries": (1, 6),
}

# Which parameters plausibly drive each failure outcome.
LEVERS = {
    "collision": [
        "shield_inflate_m",
        "abort_lat_m",
        "abort_yaw_deg",
        "lat_tol_m",
        "yaw_tol_deg",
        "step_v",
        "max_step_m",
        "k_lat",
    ],
    "parked-off-target": [
        "stop_tol_m",
        "lat_tol_m",
        "yaw_tol_deg",
        "step_frac",
        "max_step_m",
        "measure_s",
        "settle_s",
        "step_v",
    ],
    "abort:misaligned in the bay; out of retries": [
        "align_lat_m",
        "align_yaw_deg",
        "abort_lat_m",
        "abort_yaw_deg",
        "max_retries",
        "k_lat",
        "k_head",
        "enter_d",
    ],
    "abort:timeout": ["orbit_v", "approach_v", "step_v", "measure_s", "settle_s", "orbit_tol_deg"],
    "geofence": ["orbit_r", "k_radial", "orbit_v"],
}


def _clip(name: str, v: float):
    lo, hi = SPACE[name]
    v = float(np.clip(v, lo, hi))
    return int(round(v)) if isinstance(lo, int) else round(v, 4)


def mutate(
    params: ParkParams, failures: Counter, rng: np.random.Generator, k: int = 3
) -> tuple[ParkParams, dict]:
    """Change `k` parameters, chosen mostly among the levers of the parent's failures."""
    weights = Counter()
    for outcome, n in failures.items():
        key = (
            outcome
            if outcome in LEVERS
            else next((o for o in LEVERS if outcome.startswith(o)), None)
        )
        for name in LEVERS.get(key, []):
            weights[name] += n
    names = list(SPACE)
    w = np.array(
        [1.0 + 3.0 * weights[n] / max(1, sum(weights.values())) * len(names) for n in names]
    )
    chosen = rng.choice(names, size=k, replace=False, p=w / w.sum())
    change = {}
    for name in chosen:
        lo, hi = SPACE[name]
        cur = getattr(params, name)
        step = (hi - lo) * rng.normal(0, 0.15)
        change[str(name)] = _clip(name, cur + step)
    return replace(params, **change), change


def search(
    out: Path,
    generations: int = 6,
    children: int = 10,
    n: int = 128,
    seed: int = 7,
    start: ParkParams | None = None,
    cond: Conditions | None = None,
    log=print,
) -> ParkParams:
    out.mkdir(parents=True, exist_ok=True)
    tree = out / "tree.jsonl"
    rng = np.random.default_rng(seed)
    cond = cond or Conditions()
    best = start or ParkParams()
    t0 = time.time()

    def score(p: ParkParams) -> dict:
        return evaluate(p, n=n, seed=seed, cond=cond)

    def write(node: dict) -> None:
        with tree.open("a") as f:
            f.write(json.dumps(node) + "\n")

    root = score(best)
    best_node = {
        "id": "n0",
        "parent": None,
        "gen": 0,
        "change": {},
        "success": root["success"],
        "n": n,
        "wilson95": root["wilson95"],
        "outcomes": root["outcomes"],
        "verdict": "ROOT",
        "collisions": root["collisions"],
        "params": asdict(best),
        "t": round(time.time() - t0, 1),
    }
    write(best_node)
    log(f"root {root['success']}/{n} {root['outcomes']}")
    counter = 1
    for gen in range(1, generations + 1):
        failures = Counter({k: v for k, v in best_node["outcomes"].items() if k != "parked"})
        results = []
        for _ in range(children):
            cand, change = mutate(best, failures, rng)
            r = score(cand)
            node = {
                "id": f"n{counter}",
                "parent": best_node["id"],
                "gen": gen,
                "change": change,
                "success": r["success"],
                "n": n,
                "wilson95": r["wilson95"],
                "outcomes": r["outcomes"],
                "params": asdict(cand),
                "t": round(time.time() - t0, 1),
            }
            counter += 1
            # PASS: beats the parent by more than the paired noise we accept (2 episodes) and
            # does not collide more often. A collision on the car needs a person; it is never
            # traded for success.
            node["collisions"] = r["collisions"]
            better = r["success"] >= best_node["success"] + 2
            safe = r["collisions"] <= best_node.get("collisions", root["collisions"])
            node["verdict"] = "PASS" if better and safe else "FAIL"
            write(node)
            results.append((node, cand))
            log(f"gen {gen} {node['id']} {node['verdict']} {r['success']}/{n} {change}")
        winner, cand = max(results, key=lambda nc: nc[0]["success"])
        if winner["verdict"] == "PASS":
            best, best_node = cand, winner
            log(f"gen {gen}: new best {winner['id']} {winner['success']}/{n}")
        else:
            log(f"gen {gen}: no improvement, keep {best_node['id']}")
    (out / "best.json").write_text(
        json.dumps({"node": best_node["id"], "params": asdict(best)}, indent=1)
    )
    return best


def load_params(path: Path) -> ParkParams:
    data = json.loads(path.read_text())
    data = data.get("params", data)
    known = {f.name for f in fields(ParkParams)}
    return ParkParams(**{k: v for k, v in data.items() if k in known})


__all__ = ["LEVERS", "SPACE", "load_params", "mutate", "search", "wilson"]
