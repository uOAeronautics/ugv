"""Read-only views of the repo for the Lab dashboard: ledger, parking runs, pose runs.

Everything is computed from files on disk at request time (`runs/` and `research/`), so the
dashboard shows what is true now and owns no state of its own.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def ledger() -> dict:
    from ugv.research import as_row, load
    from ugv.research.tasks import roadmap_lines, status_lines

    records = load()
    rows = [as_row(r) for r in records]
    for r, rec in zip(rows, records, strict=True):
        r["path"] = str(rec.path.relative_to(REPO))
    return {"records": rows, "status": status_lines(records), "roadmap": roadmap_lines(records)}


def park_evals() -> list[dict]:
    """Eval summaries, newest first, with compact per-episode columns for charts."""
    out = []
    for path in sorted((RUNS / "park_eval").glob("*.json"), key=lambda p: -p.stat().st_mtime):
        if path.name.endswith("_traces.json"):
            continue
        d = _read(path)
        if not d or "main" not in d:
            continue
        main = d["main"]
        eps = main.pop("episodes", [])
        main["columns"] = {
            "outcome": [e["outcome"] for e in eps],
            "time_s": [e["time_s"] for e in eps],
            "final_error": [e["final_error"] for e in eps],
            "retries": [e["retries"] for e in eps],
            "states": _mean_states(eps),
        }
        d["file"] = path.name
        d["mtime"] = path.stat().st_mtime
        d["has_traces"] = (path.parent / f"{path.stem}_traces.json").exists()
        out.append(d)
    return out


def _mean_states(eps: list[dict]) -> dict[str, float]:
    tot: Counter = Counter()
    for e in eps:
        tot.update(e.get("states", {}))
    return {k: round(v / max(1, len(eps)), 2) for k, v in tot.most_common()}


def park_traces(name: str) -> list | None:
    path = RUNS / "park_eval" / f"{Path(name).name}_traces.json"
    return _read(path)


def park_searches() -> list[dict]:
    out = []
    for tree in sorted((RUNS / "park_search").glob("*/tree.jsonl")):
        nodes = []
        for line in tree.read_text().splitlines():
            try:
                n = json.loads(line)
            except json.JSONDecodeError:
                continue
            n.pop("params", None)
            nodes.append(n)
        best = _read(tree.parent / "best.json")
        out.append({"name": tree.parent.name, "nodes": nodes, "best": best})
    return out


@lru_cache(maxsize=1)
def bay() -> dict:
    """Floor geometry for drawing and for the challenge: obstacles (subsampled), target, car."""
    from ugv.park import geometry as g

    b = g.bay()
    obstacles = np.unique(np.round(b.obstacles, 2), axis=0)
    tall = np.unique(np.round(b.tall, 2), axis=0)

    return {
        "obstacles": obstacles.tolist(),
        "tall": tall.tolist(),
        "target": [float(v) for v in b.target],
        "clearance_m": b.clearance_m,
        "car": {
            "length": g.CAR_LENGTH_M,
            "width": g.CAR_WIDTH_M,
            "camera_x": g.CAMERA_X_M,
            "camera_z": g.CAMERA_Z_M,
            "tower_x": list(g.TOWER_X_M),
            "tower_y": list(g.TOWER_Y_M),
        },
        "tolerance": {"x": 0.02, "y": 0.02, "yaw_deg": 5.0},
    }


@lru_cache(maxsize=1)
def plan() -> dict:
    """The airframe seen from above, for drawing: every triangle's (x, y) corners and its lowest
    and highest point above the floor, all in mm. Highest first, so the low parts the car can hit
    paint over the overhead structure it drives under. Plus the two heights that matter: the car
    body top (anything lower is hit by the whole car) and the camera stand top."""
    from ugv.park import geometry as g
    from ugv.pose.mesh import load_airframe

    tris = load_airframe().tris
    z = tris[:, :, 2] - tris[:, :, 2].min()
    low = z.min(axis=1)
    order = np.argsort(-low)
    xy = np.round(tris[order, :, :2] * 1000).astype(int).reshape(len(order), 6)
    zz = np.round(z.max(axis=1)[order] * 1000).astype(int)
    lo = np.round(low[order] * 1000).astype(int)
    return {
        "tris": np.column_stack([xy, lo, zz]).tolist(),
        "body_top_mm": round(g.CAR_BODY_TOP_M * 1000),
        "car_top_mm": round(g.CAR_TOP_M * 1000),
    }


@lru_cache(maxsize=1)
def mesh() -> list:
    """The airframe mesh in its model frame (the frame PnP solves in), triangles in mm."""
    from ugv.pose.mesh import load_airframe

    return np.round(load_airframe().tris.reshape(-1, 9) * 1000).astype(int).tolist()


def plant() -> dict:
    from ugv.park.plant import DUTY_CAP, NOMINAL

    return {
        "gain": NOMINAL.gain,
        "deadzone": NOMINAL.deadzone,
        "tau": NOMINAL.tau,
        "duty_cap": DUTY_CAP,
    }


def pose_runs() -> dict:
    smoke = []
    for path in sorted(RUNS.glob("pose_smoke*/summary.json")):
        d = _read(path) or {}
        d["name"] = path.parent.name
        d["sheet"] = (path.parent / "sheet.jpg").exists()
        smoke.append(d)
    training = []
    for path in sorted(RUNS.glob("pose_train/*/metrics.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        summary = _read(path.parent / "summary.json")
        preview = _read(path.parent / "preview" / "preview.json")
        training.append(
            {"name": path.parent.name, "metrics": rows, "summary": summary, "preview": preview}
        )
    return {"smoke": smoke, "training": training}


def run_file(rel: str) -> Path | None:
    """A file under runs/ (images for the pose page). Refuses anything outside runs/."""
    p = (RUNS / rel).resolve()
    return p if p.is_file() and RUNS.resolve() in p.parents else None
