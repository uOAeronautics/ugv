"""Why episodes fail: state at the end, the estimate's error then, and which faults were present.

    uv run python tools/park/diagnose.py --params config/park_params.json --faults 1 --n 400

For every failure outcome: how many, in which controller state it happened, the median gap
between the car's estimate and the truth at that moment, and the failure rate with and without
each fault family (the lift says which fault drives which failure). S0, simulation only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ugv.park.controller import ParkParams
from ugv.park.episode import Episode, Scenario
from ugv.park.faults import FAMILIES, Faults
from ugv.park.search import load_params


def _families(f: Faults) -> list[str]:
    d = Faults()
    on = {
        "wheel": f.wheel_gain != d.wheel_gain,
        "battery": f.battery != 1 or f.sag_per_min > 0,
        "stall": f.stall_wheel >= 0,
        "slip": f.slip_std > 0,
        "delay": f.delay_ticks > 0,
        "drops": f.drop_rate > 0,
        "phantom": f.phantom_rate > 0,
        "gyro": f.gyro_scale != 1 or f.gyro_walk > 0,
        "exposure": f.exposure_rate > 0,
        "blur": f.blur_px > 0,
        "occlusion": f.occluder is not None,
        "shake": f.shake_deg > 0,
        "bump": f.bump_deg > 0,
        "lens": f.lens is not None,
    }
    return [k for k in FAMILIES if on.get(k)]


def one(args) -> dict:
    i, seed, params, faults, start = args
    s = seed * 100003 + i
    sc = Scenario.sample(np.random.default_rng(s), 1.0, None, faults, None, start)
    ep = Episode(sc, params, s)
    while not ep.tick():
        pass
    r = ep.result()
    est = ep.est.car_in_airframe()
    gap = None if est is None else float(np.hypot(*(np.asarray(est[:2]) - ep.true_rel()[:2])))
    return {
        "i": i,
        "outcome": r.outcome,
        "state": ep.state,
        "gap": gap,
        "families": _families(ep.faults.f) if ep.faults else [],
        "faults": r.faults,
        "time_s": r.time_s,
        "final_error": r.final_error,
    }


def diagnose(params: ParkParams, n: int, seed: int, faults: float, start: str) -> dict:
    jobs = [(i, seed, params, faults, start) for i in range(n)]
    with ProcessPoolExecutor(min(16, os.cpu_count() or 1)) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    out: dict = {"n": n, "outcomes": dict(Counter(r["outcome"] for r in rows).most_common())}
    by = defaultdict(list)
    for r in rows:
        by[r["outcome"]].append(r)
    for outcome, rs in by.items():
        if outcome == "parked":
            continue
        gaps = [r["gap"] for r in rs if r["gap"] is not None]
        lift = {}
        for fam in FAMILIES:
            with_f = [r for r in rows if fam in r["families"]]
            without = [r for r in rows if fam not in r["families"]]
            if with_f and without:
                a = sum(r["outcome"] == outcome for r in with_f) / len(with_f)
                b = sum(r["outcome"] == outcome for r in without) / len(without)
                lift[fam] = round(a - b, 3)
        out[outcome] = {
            "count": len(rs),
            "state": dict(Counter(r["state"] for r in rs).most_common()),
            "median_estimate_gap_cm": round(100 * float(np.median(gaps)), 1) if gaps else None,
            "rate_with_minus_without": dict(sorted(lift.items(), key=lambda kv: -kv[1])[:6]),
            "examples": [r["i"] for r in rs[:5]],
        }
    out["rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--faults", type=float, default=1.0)
    ap.add_argument("--start", choices=["ring", "close"], default="ring")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    params = load_params(args.params) if args.params else ParkParams()
    d = diagnose(params, args.n, args.seed, args.faults, args.start)
    if args.out:
        args.out.write_text(json.dumps(d))
    d.pop("rows")
    print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
