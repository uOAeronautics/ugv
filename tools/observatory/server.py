"""FastAPI backend for the UGV Research Observatory.

Routes only. The logic lives in leaderboard.py, bundles.py, rollout.py, contract.py,
replay.py and data.py; the module-level path globals here are read at call time so
tests can point them at a temporary tree.
"""

import argparse
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import bundles, contract, leaderboard, rollout
from .data import (
    CALIB_RUNS,
    LEVELS,
    REPO,
    RL_RUNS,
    curve_runs,
    l7_runs,
    parse_log,
    parse_metrics_jsonl,
    read_json,
    safe_path,
    sanitize,
)
from .replay import eval_bundle, l7_bundle

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _count_files(directory: Path, pattern: str) -> int:
    """Count matching *files*.

    A run directory can be named like ``baseline_L1_20ep_s11.json``; globbing alone
    counts it, so require ``is_file``.
    """
    if not directory.exists():
        return 0
    return sum(1 for p in directory.glob(pattern) if p.is_file())


def _counts():
    traces = _count_files(RL_RUNS / "traces", "*.json")
    traces += _count_files(RL_RUNS, "*.traces.json")
    traces += _count_files(RL_RUNS / "gazebo_l6", "*.traces.json")
    archive = RL_RUNS / "archive.jsonl"
    archive_n = 0
    if archive.exists():
        archive_n = sum(1 for line in archive.open() if line.strip())
    l7 = [d for d in CALIB_RUNS.iterdir() if d.is_dir()] if CALIB_RUNS.exists() else []
    return {
        "archive": archive_n,
        "logs": _count_files(RL_RUNS, "*.log"),
        "metrics": _count_files(RL_RUNS, "*.metrics.jsonl"),
        # probe runs (dbg_/smoke_/obs_/live_), plan JSON and sub-gate summaries are
        # not graded eval runs; this matches the gazebo_l6_runs KPI
        "gazebo_l6": len(leaderboard.gazebo_entries(RL_RUNS)),
        "l7_runs": len(l7),
        "traces": traces,
    }


def _pi_step_ms(bench):
    if not isinstance(bench, dict):
        return None
    lat = bench.get("latency_ms_single_core")
    p50 = None
    if isinstance(lat, dict) and isinstance(lat.get("stateful_step_plus_shield"), dict):
        p50 = lat["stateful_step_plus_shield"].get("p50")
    if p50 is None:
        p50 = bench.get("step_ms_mean", bench.get("mean_ms"))
    try:
        return float(p50) if p50 is not None else None
    except (TypeError, ValueError):
        return None


def _leaderboard():
    return leaderboard.build(RL_RUNS)


def _eval_contract(run_id):
    return contract.eval_contract(run_id, CALIB_RUNS)


@app.get("/")
def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx), headers={"Cache-Control": "no-cache"})
    return HTMLResponse("<html><body><h1>UGV Observatory</h1><p>static not built</p></body></html>")


@app.get("/api/overview")
def overview():
    bench = None
    for p in (RL_RUNS / "pi_inference_bench.json", REPO / "pi_inference_bench.json"):
        if p.exists():
            bench = read_json(p)
            break
    entries = _leaderboard()
    kpi = {
        "eval_runs": sum(1 for e in entries if e.get("source") in ("archive", "eval_json")),
        "gazebo_l6_runs": sum(1 for e in entries if e.get("source") == "gazebo_l6"),
        "l7_runs": sum(1 for e in entries if e.get("source") == "l7"),
        "pi_step_ms_p50": _pi_step_ms(bench),
    }
    return sanitize({"counts": _counts(), "kpi": kpi, "levels": LEVELS, "pi_bench": bench})


@app.get("/api/leaderboard")
def leaderboard_route():
    return {"entries": sanitize(_leaderboard()), "l7_runs": sanitize(l7_runs())}


@app.get("/api/curves/runs")
def curves_runs():
    return {"runs": sanitize(curve_runs(RL_RUNS))}


@app.get("/api/curves")
def curves(runs: str = Query("")):
    series = {}
    for rid in (r.strip() for r in runs.split(",")):
        if not rid:
            continue
        if "/" in rid or ".." in rid or "\\" in rid:
            raise HTTPException(400, "bad run id")
        mj = RL_RUNS / f"{rid}.metrics.jsonl"
        lg = RL_RUNS / f"{rid}.log"
        if mj.is_file():
            series[rid] = {"format": "jsonl", "points": parse_metrics_jsonl(mj)}
        elif lg.is_file():
            series[rid] = {"format": "log", "points": parse_log(lg)}
    return {"series": sanitize(series)}


@app.get("/api/replay/list")
def replay_list():
    return {"bundles": sanitize(bundles.list_bundles(RL_RUNS, CALIB_RUNS))}


@app.get("/api/replay/bundle/{bundle_id:path}")
def replay_bundle(bundle_id: str):
    try:
        kind, path = bundles.resolve(bundle_id, RL_RUNS, CALIB_RUNS)
    except ValueError:
        raise HTTPException(400, "bad path") from None
    except LookupError:
        raise HTTPException(404, "not found") from None
    if kind == "l7":
        return sanitize(l7_bundle(path))
    try:
        return sanitize(eval_bundle(path))
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/api/aruco/{marker_id}")
def aruco_marker(marker_id: int):
    """6x6 bit grid (border included, 1 = white) of a DICT_4X4_50 marker for the POV renderer."""
    if not 0 <= marker_id < 50:
        raise HTTPException(404, "DICT_4X4_50 has ids 0-49")
    try:
        import cv2
    except ImportError:
        raise HTTPException(503, "opencv not installed") from None
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, marker_id, 6)
    return {"id": marker_id, "n": 6, "rows": [[int(v > 127) for v in row] for row in img]}


@app.get("/api/l7/{run_id}/end.jpg")
def l7_endjpg(run_id: str):
    if ".." in run_id or "/" in run_id:
        raise HTTPException(400, "bad path")
    p = safe_path(CALIB_RUNS, f"{run_id}/end.jpg")
    if p is None or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))


@app.get("/api/l7/{run_id}/frame/{fname}")
def l7_frame_jpg(run_id: str, fname: str):
    if ".." in run_id or "/" in run_id or ".." in fname or "/" in fname:
        raise HTTPException(400, "bad path")
    frames_dir = safe_path(CALIB_RUNS, f"{run_id}/frames")
    if frames_dir is None:
        raise HTTPException(400, "bad path")
    p = safe_path(frames_dir, fname)
    if p is None or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))


@app.get("/api/rollout/options")
def rollout_options():
    return rollout.options(RL_RUNS)


@app.post("/api/rollout")
async def rollout_start(req: dict):
    try:
        return await rollout.start(req, RL_RUNS)
    except rollout.RolloutError as e:
        raise HTTPException(400, str(e)) from None
    except rollout.RolloutBusy as e:
        raise HTTPException(409, str(e)) from None


@app.get("/api/rollout/{job_id}")
def rollout_status(job_id: str):
    job = rollout.status(job_id)
    if job is None:
        raise HTTPException(404, "not found")
    return job


@app.get("/api/contract/runs")
def contract_runs():
    return {"runs": contract.contract_runs(CALIB_RUNS)}


@app.get("/api/contract")
def contract_route(run_id: str | None = Query(default=None)):
    if run_id is not None:
        if ".." in run_id or "/" in run_id:
            raise HTTPException(400, "bad path")
        if not (CALIB_RUNS / run_id).is_dir():
            raise HTTPException(404, f"L7 run not found: {run_id}")
        return _eval_contract(run_id)
    latest = contract.latest_run(CALIB_RUNS)
    if latest is None:
        raise HTTPException(404, "no L7 runs with ticks.jsonl found")
    return _eval_contract(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if not args.no_open:
        try:
            webbrowser.open(f"http://{args.host}:{args.port}/")
        except Exception:
            pass
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
