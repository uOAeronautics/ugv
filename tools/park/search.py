"""Failure-guided parameter search for the parking controller (simulation only, S0).

    uv run python tools/park/search.py --out runs/park_search/<name> [--generations 6 --children 10 --n 128]

Writes `tree.jsonl` (one node per candidate: parent, change, score, failure modes, PASS/FAIL)
and `best.json`. Validate the result with `tools/park/eval.py --params <dir>/best.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ugv.park.search import load_params, search


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=Path, help="params JSON to start from")
    args = ap.parse_args()
    start = load_params(args.start) if args.start else None
    search(
        args.out,
        args.generations,
        args.children,
        args.n,
        args.seed,
        start,
        log=lambda s: print(s, flush=True),
    )


if __name__ == "__main__":
    main()
