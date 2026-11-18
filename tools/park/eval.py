"""Score the parking controller in simulation (S0) and write a summary for the Lab dashboard.

    uv run python tools/park/eval.py --name default                 # 1000 episodes, seed 11
    uv run python tools/park/eval.py --params config/park_params.json --name tuned --sweep
    uv run python tools/park/eval.py --name default --traces 12      # also save replays

`--sweep` adds the robustness conditions: ideal world, detector-only, plant-only, worse
detector (4 px, 20% misses, 8% outliers), worse camera calibration, and a 1.5x wider spread
of cars, starts close to the airframe on any side, and per-episode faults (`faults.py`).
`--families` scores each fault family alone. `--faults S --start close` set the main run.
Output: `runs/park_eval/<name>.json` (and `<name>_traces.json`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ugv.park.controller import ParkParams
from ugv.park.episode import Scenario, run
from ugv.park.evaluate import Conditions, evaluate
from ugv.park.faults import FAMILIES as FAULT_FAMILIES
from ugv.park.perception import DetectorModel
from ugv.park.search import load_params

IDEAL = DetectorModel(
    pixel_std=0.3, miss_rate=0, outlier_rate=0, focal_rel_std=0, mount_yaw_std=0, mount_pitch_std=0
)
SWEEP = {
    "full": Conditions(),
    "ideal": Conditions(0.0, IDEAL),
    "detector-only": Conditions(
        0.0, replace(DetectorModel(), focal_rel_std=0, mount_yaw_std=0, mount_pitch_std=0)
    ),
    "plant-only": Conditions(1.0, IDEAL),
    "worse-detector": Conditions(
        1.0, DetectorModel(pixel_std=4.0, miss_rate=0.2, outlier_rate=0.08)
    ),
    "worse-calibration": Conditions(
        1.0,
        DetectorModel(
            focal_rel_std=0.04, mount_yaw_std=np.radians(2.5), mount_pitch_std=np.radians(3)
        ),
    ),
    "wider-cars": Conditions(1.5, DetectorModel()),
    "close-start": Conditions(start="close"),
    "faults": Conditions(faults=1.0),
    "close+faults": Conditions(faults=1.0, start="close"),
    "slippery": Conditions(floor=1.0),
    "slippery+faults": Conditions(faults=1.0, floor=1.0),
}
FAMILIES = {f"fault:{f}": Conditions(only=f) for f in FAULT_FAMILIES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-n", type=int, default=300)
    ap.add_argument("--families", action="store_true", help="sweep each fault family alone")
    ap.add_argument("--faults", type=float, default=0.0, help="fault severity of the main run")
    ap.add_argument(
        "--floor", type=float, default=0.0, help="floor traction severity of the main run"
    )
    ap.add_argument("--start", choices=["ring", "close"], default="ring")
    ap.add_argument("--traces", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/park_eval"))
    args = ap.parse_args()
    params = load_params(args.params) if args.params else ParkParams()
    args.out.mkdir(parents=True, exist_ok=True)
    main_cond = Conditions(faults=args.faults, start=args.start, floor=args.floor)
    main_r = evaluate(params, n=args.n, seed=args.seed, cond=main_cond)
    print(
        f"full {main_r['success']}/{args.n} wilson95 {main_r['wilson95']} {main_r['outcomes']}",
        flush=True,
    )
    summary = {
        "name": args.name,
        "params_file": str(args.params) if args.params else None,
        "main": main_r,
        "sweep": {},
    }
    sweep = (SWEEP if args.sweep else {}) | (FAMILIES if args.families else {})
    if sweep:
        for label, cond in sweep.items():
            if label == "full":
                continue
            r = evaluate(params, n=args.sweep_n, seed=args.seed + 1, cond=cond)
            r.pop("episodes")
            summary["sweep"][label] = r
            print(f"{label:18s} {r['success']}/{r['n']} {r['outcomes']}", flush=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(summary))
    if args.traces:
        traces = []
        for i in range(args.traces):
            s = args.seed * 100003 + i
            sc = Scenario.sample(
                np.random.default_rng(s), 1.0, None, args.faults, None, args.start, args.floor
            )
            r = run(sc, params, seed=s, keep_trace=True)
            traces.append(
                {
                    "episode": i,
                    "outcome": r.outcome,
                    "final_error": r.final_error,
                    "faults": r.faults,
                    "trace": r.trace,
                }
            )
        (args.out / f"{args.name}_traces.json").write_text(json.dumps(traces))
    print(f"wrote {args.out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
