"""Score a parameter set on many episodes, in parallel, with common random numbers.

Episode `i` of a batch with base seed `s` always uses scenario seed `s * 100003 + i`, so two
parameter sets compared on the same `(seed, n)` see the same airframes, cars and cameras: their
difference is the parameters, not luck. Success is reported with a Wilson 95% interval.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ugv.park.controller import ParkParams
from ugv.park.episode import Result, Scenario, as_dict, run
from ugv.park.perception import DetectorModel


@dataclass(frozen=True)
class Conditions:
    """What the world throws at the controller. `plant_scale` 0 = the nominal car; `faults` is the
    severity of per-episode car and sensor faults (faults.py), `only` one fault family; `start`
    "ring" (1.3-3 m away) or "close" (0.4-1.3 m, any side); `floor` is the traction severity
    (plant.Floor), 0 = clean."""

    plant_scale: float = 1.0
    detector: DetectorModel = DetectorModel()
    faults: float = 0.0
    only: str | None = None
    start: str = "ring"
    floor: float = 0.0  # traction severity: patches, skid, lost strafe/yaw authority


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _one(args: tuple) -> dict:
    i, seed, params, cond = args
    s = seed * 100003 + i
    rng = np.random.default_rng(s)
    sc = Scenario.sample(
        rng, cond.plant_scale, cond.detector, cond.faults, cond.only, cond.start, cond.floor
    )
    return as_dict(run(sc, params, seed=s))


def evaluate(
    params: ParkParams,
    n: int = 200,
    seed: int = 11,
    cond: Conditions | None = None,
    workers: int | None = None,
) -> dict:
    cond = cond or Conditions()
    jobs = [(i, seed, params, cond) for i in range(n)]
    workers = workers or min(16, os.cpu_count() or 1)
    if workers == 1:
        rows = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_one, jobs, chunksize=max(1, n // (workers * 4))))
    return summarize(rows, params, n, seed, cond)


def summarize(rows: list[dict], params: ParkParams, n: int, seed: int, cond: Conditions) -> dict:
    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, n)
    ok = [r for r in rows if r["success"]]
    outcomes = Counter(r["outcome"] for r in rows)
    return {
        "n": n,
        "seed": seed,
        "success": k,
        "rate": k / n,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "collisions": outcomes.get("collision", 0),
        "outcomes": dict(outcomes.most_common()),
        "median_time_s": float(np.median([r["time_s"] for r in ok])) if ok else None,
        "mean_retries": float(np.mean([r["retries"] for r in rows])),
        "params": asdict(params),
        "conditions": {**asdict(cond), "detector": asdict(cond.detector)},
        "episodes": rows,
    }


def failure_examples(rows: list[dict], k: int = 3) -> dict[str, list[int]]:
    """Episode indices per failure outcome, to replay (`run(..., keep_trace=True)`)."""
    out: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if not r["success"]:
            out.setdefault(r["outcome"], [])
            if len(out[r["outcome"]]) < k:
                out[r["outcome"]].append(i)
    return out


__all__ = ["Conditions", "Result", "evaluate", "failure_examples", "wilson"]
