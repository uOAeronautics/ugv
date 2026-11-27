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
from ugv.park.episode import run
from ugv.park.evaluate import Conditions, _scenario, evaluate
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
    ap.add_argument("--start", choices=["ring", "close", "presets", "mixed"], default="ring")
    ap.add_argument(
        "--perception",
        choices=["model", "network"],
        default="model",
        help="sampled DetectorModel or closed-loop rendered frames + ONNX KeypointDetector",
    )
    ap.add_argument(
        "--preset-repeats",
        type=int,
        default=0,
        help="repeats per named preset in --start mixed (0 = ~30%% of n)",
    )
    ap.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: cpu count, <=16)"
    )
    ap.add_argument("--dump-frames", type=Path, default=None, help="DAgger dataset output dir")
    ap.add_argument(
        "--dump-max", type=int, default=6000, help="max frames written to --dump-frames"
    )
    ap.add_argument(
        "--dump-err-px",
        type=float,
        default=10.0,
        help="keypoint error threshold (px) for DAgger priority",
    )
    ap.add_argument(
        "--failure-overlays",
        type=Path,
        default=None,
        help="dir to save worst-frame PNG overlays for failed episodes",
    )
    ap.add_argument("--traces", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/park_eval"))
    args = ap.parse_args()
    params = load_params(args.params) if args.params else ParkParams()
    args.out.mkdir(parents=True, exist_ok=True)
    main_cond = Conditions(
        faults=args.faults,
        start=args.start,
        floor=args.floor,
        perception=args.perception,
        preset_repeats=args.preset_repeats,
    )
    main_r = evaluate(
        params,
        n=args.n,
        seed=args.seed,
        cond=main_cond,
        workers=args.workers,
        dump_frames=args.dump_frames,
        dump_max=args.dump_max,
        dump_err_px=args.dump_err_px,
        failure_overlays=args.failure_overlays,
    )
    print(
        f"full {main_r['success']}/{args.n} wilson95 {main_r['wilson95']} {main_r['outcomes']}",
        flush=True,
    )
    if "timing" in main_r:
        print(f"timing {main_r['timing']}", flush=True)
    if "dagger_dump" in main_r:
        print(f"dagger_dump {main_r['dagger_dump']}", flush=True)
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
            c = replace(cond, perception=args.perception)
            r = evaluate(params, n=args.sweep_n, seed=args.seed + 1, cond=c, workers=args.workers)
            r.pop("episodes")
            summary["sweep"][label] = r
            print(f"{label:18s} {r['success']}/{r['n']} {r['outcomes']}", flush=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(summary))
    if args.traces:
        net = None
        if args.perception == "network":
            from ugv.park.camera import NetworkPerception

            net = NetworkPerception(threads=1)
        traces = []
        for i in range(args.traces):
            s, start_name, sc = _scenario(i, args.n, args.seed, main_cond)
            if net is not None:
                net.reset_episode()
            r = run(sc, params, seed=s, keep_trace=True, perception=net)
            traces.append(
                {
                    "episode": i,
                    "start": start_name,
                    "outcome": r.outcome,
                    "end_state": r.end_state,
                    "final_error": r.final_error,
                    "faults": r.faults,
                    "trace": r.trace,
                }
            )
        (args.out / f"{args.name}_traces.json").write_text(json.dumps(traces))
    print(f"wrote {args.out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
