"""FastAPI backend for the UGV Research Observatory."""

import argparse
import asyncio
import json
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .data import (
    CALIB_RUNS,
    LEVELS,
    REPO,
    RL_RUNS,
    l7_runs,
    parse_log,
    parse_metrics_jsonl,
    safe_path,
    sanitize,
)
from .replay import eval_bundle, l7_bundle
from .wilson import wilson_ci

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

JOBS = {}


def _counts():
    traces = list((RL_RUNS / "traces").glob("*.json")) if (RL_RUNS / "traces").exists() else []
    traces += list(RL_RUNS.glob("*.traces.json"))
    evals = list(RL_RUNS.glob("*.json"))
    archive = RL_RUNS / "archive.jsonl"
    archive_n = 0
    if archive.exists():
        archive_n = sum(1 for _ in archive.open() if _.strip())
    logs = list(RL_RUNS.glob("*.log"))
    metrics = list(RL_RUNS.glob("*.metrics.jsonl"))
    gazebo = (
        list((RL_RUNS / "gazebo_l6").glob("*.json")) if (RL_RUNS / "gazebo_l6").exists() else []
    )
    l7 = [d for d in CALIB_RUNS.iterdir() if d.is_dir()] if CALIB_RUNS.exists() else []
    return {
        "evals": len(evals),
        "archive": archive_n,
        "logs": len(logs),
        "metrics": len(metrics),
        "gazebo_l6": len(gazebo),
        "l7_runs": len(l7),
        "traces": len(traces),
    }


@app.get("/")
def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse("<html><body><h1>UGV Observatory</h1><p>static not built</p></body></html>")


@app.get("/api/overview")
def overview():
    pi_bench = None
    for p in (RL_RUNS / "pi_inference_bench.json", REPO / "pi_inference_bench.json"):
        if p.exists():
            try:
                pi_bench = json.loads(p.read_text())
            except Exception:
                pi_bench = None
            break
    return {"counts": _counts(), "levels": LEVELS, "pi_bench": pi_bench}


def _track_for(d, source, run_id):
    track = d.get("track")
    if track:
        return track
    rid = (run_id or "").lower()
    if "student" in rid:
        return "student"
    if "teacher" in rid:
        return "teacher"
    if source == "gazebo_l6":
        return "gazebo_l6"
    if source == "l7":
        return "l7"
    return "baseline"


def _flatten_metrics(d):
    out = dict(d)
    em = d.get("eval_metrics")
    if isinstance(em, dict):
        for k, v in em.items():
            out.setdefault(k, v)
    return out


def _leaderboard():
    by_id = {}

    def add(entry):
        rid = entry["run_id"]
        if rid in by_id:
            # merge: keep existing values, fill from the new entry
            cur = by_id[rid]
            for k, v in entry.items():
                if cur.get(k) is None:
                    cur[k] = v
        else:
            by_id[rid] = entry

    # archive.jsonl
    arch = RL_RUNS / "archive.jsonl"
    if arch.exists():
        for line in arch.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            add(_norm_entry(_flatten_metrics(d), "archive"))
    # rl/runs/*.json evals
    for p in RL_RUNS.glob("*.json"):
        if p.name.endswith(".traces.json"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if "success" in d and "episodes" in d:
            add(_norm_entry(_flatten_metrics(d), "eval_json", run_id=d.get("run_id") or p.stem))
    # gazebo_l6
    gz = RL_RUNS / "gazebo_l6"
    if gz.exists():
        for p in gz.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            summary = d.get("summary") if isinstance(d.get("summary"), dict) else {}
            flat = _flatten_metrics(d)
            merged = dict(summary)
            for k, v in flat.items():
                if v is None:
                    continue
                # keep a scalar summary value over a nested top-level value
                # (e.g. summary.episodes=100 vs a top-level per-episode list)
                if isinstance(v, (list, dict)) and isinstance(
                    summary.get(k), (int, float, str, bool)
                ):
                    continue
                merged[k] = v

            def _pick(*keys, _merged=merged):
                for k in keys:
                    v = _merged.get(k)
                    if v is not None:
                        return v
                return None

            add(
                _norm_entry(
                    {
                        "run_id": p.stem,
                        "success": _pick("success"),
                        "episodes": _pick("episodes"),
                        "collisions": _pick("collisions"),
                        "timeouts": _pick("timeouts"),
                        "false_parked": _pick("false_parked"),
                        "time_median": _pick("time_median", "median_time_s"),
                        "time_p95": _pick("time_p95"),
                        "pos_p95": _pick("pos_p95", "pos_err_p95_m"),
                        "yaw_p95_deg": _pick("yaw_p95_deg", "yaw_err_p95_deg"),
                        "level": _pick("level"),
                        "track": _pick("track"),
                        "tags": _pick("tags"),
                        "parent_id": _pick("parent_id"),
                        "config_delta": _pick("config_delta"),
                    },
                    "gazebo_l6",
                )
            )
    # l7 calibration runs
    for r in l7_runs():
        succ = 1 if r.get("outcome") in ("success", "parked") else 0
        lo, hi = wilson_ci(succ, 1)
        by_id[r["run_id"]] = {
            "run_id": r["run_id"],
            "track": "l7",
            "level": "L_real",
            "success": succ,
            "collisions": 0,
            "false_parked": 0,
            "timeouts": 0,
            "episodes": 1,
            "success_rate": float(succ),
            "wilson_lo": lo,
            "wilson_hi": hi,
            "time_median": r.get("seconds"),
            "time_p95": None,
            "pos_p95": None,
            "yaw_p95_deg": None,
            "jerk_mean": None,
            "parent_id": None,
            "tags": None,
            "pareto": False,
            "pareto_score": None,
            "source": "l7",
            "config_delta": None,
        }
    entries = list(by_id.values())
    # compute wilson for entries lacking, but never fabricate rates for
    # entries with no success/episode data (e.g. empty eval_metrics)
    for e in entries:
        if e.get("success") is None or e.get("episodes") is None:
            e["success_rate"] = None
            e["wilson_lo"] = None
            e["wilson_hi"] = None
            continue
        if e.get("wilson_lo") is None or e.get("wilson_hi") is None:
            lo, hi = wilson_ci(int(e.get("success") or 0), int(e.get("episodes") or 1))
            e["wilson_lo"] = lo
            e["wilson_hi"] = hi
        if "success_rate" not in e or e.get("success_rate") is None:
            ep = e.get("episodes") or 1
            e["success_rate"] = (e.get("success") or 0) / ep
    # pareto per level
    by_level = {}
    for e in entries:
        by_level.setdefault(e.get("level"), []).append(e)
    for _lvl, grp in by_level.items():
        # non-dominated on (success_rate, -collision_rate)
        for e in grp:
            sr = e.get("success_rate")
            sr = sr if sr is not None else 0.0
            coll_rate = (e.get("collisions") or 0) / max(1, e.get("episodes") or 1)
            dominated = False
            for o in grp:
                if o is e:
                    continue
                osr = o.get("success_rate")
                osr = osr if osr is not None else 0.0
                ocr = (o.get("collisions") or 0) / max(1, o.get("episodes") or 1)
                if osr >= sr and ocr <= coll_rate and (osr > sr or ocr < coll_rate):
                    dominated = True
                    break
            e["pareto"] = not dominated
            e["pareto_score"] = sr - coll_rate
    for e in entries:
        e["badges"] = (["Pareto"] if e.get("pareto") else []) + (
            [e["source"]] if e.get("source") else []
        )
    return entries


def _norm_entry(d, source, run_id=None):
    rid = run_id or d.get("run_id") or d.get("id") or "unknown"
    succ = d.get("success")
    eps = d.get("episodes")
    if isinstance(eps, (list, dict)):
        eps = None
    try:
        succ = int(succ) if succ is not None else None
    except (TypeError, ValueError):
        succ = None
    try:
        eps = int(eps) if eps is not None else None
    except (TypeError, ValueError):
        eps = None
    if succ is None or eps is None or eps <= 0:
        sr, lo, hi = None, None, None
    else:
        sr = succ / eps
        lo, hi = wilson_ci(succ, eps)
    return {
        "run_id": rid,
        "track": _track_for(d, source, rid),
        "level": d.get("level"),
        "success": succ,
        "collisions": d.get("collisions", 0),
        "false_parked": d.get("false_parked", 0),
        "timeouts": d.get("timeouts", 0),
        "episodes": eps,
        "success_rate": sr,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "time_median": d.get("time_median"),
        "time_p95": d.get("time_p95"),
        "pos_p95": d.get("pos_p95"),
        "yaw_p95_deg": d.get("yaw_p95_deg"),
        "jerk_mean": d.get("jerk_mean"),
        "parent_id": d.get("parent_id"),
        "tags": d.get("tags"),
        "pareto": False,
        "pareto_score": None,
        "source": source,
        "config_delta": d.get("config_delta"),
    }


@app.get("/api/leaderboard")
def leaderboard():
    entries = _leaderboard()
    return {"entries": sanitize(entries), "l7_runs": sanitize(l7_runs())}


@app.get("/api/curves/runs")
def curves_runs():
    runs = []
    for p in RL_RUNS.glob("*.metrics.jsonl"):
        runs.append(
            {
                "run_id": p.stem.replace(".metrics", ""),
                "format": "jsonl",
                "n_iters": sum(1 for _ in p.open() if _.strip()),
                "mtime": p.stat().st_mtime,
                "active": (time.time() - p.stat().st_mtime) < 60,
                "last": None,
            }
        )
    for p in RL_RUNS.glob("*.log"):
        runs.append(
            {
                "run_id": p.stem,
                "format": "log",
                "n_iters": sum(1 for _l in p.read_text().splitlines() if "iter" in _l),
                "mtime": p.stat().st_mtime,
                "active": (time.time() - p.stat().st_mtime) < 60,
                "last": None,
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": sanitize(runs)}


@app.get("/api/curves")
def curves(runs: str = Query("")):
    ids = [r.strip() for r in runs.split(",") if r.strip()]
    series = {}
    for rid in ids:
        # find files
        mj = RL_RUNS / f"{rid}.metrics.jsonl"
        lg = RL_RUNS / f"{rid}.log"
        if mj.exists():
            series[rid] = {"format": "jsonl", "points": parse_metrics_jsonl(mj)}
        elif lg.exists():
            series[rid] = {"format": "log", "points": parse_log(lg)}
        else:
            # try glob
            cand = list(RL_RUNS.glob(f"{rid}*.metrics.jsonl")) or list(RL_RUNS.glob(f"{rid}*.log"))
            if cand:
                p = cand[0]
                if p.suffix == ".jsonl":
                    series[rid] = {"format": "jsonl", "points": parse_metrics_jsonl(p)}
                else:
                    series[rid] = {"format": "log", "points": parse_log(p)}
    return {"series": sanitize(series)}


@app.get("/api/replay/list")
def replay_list():
    items = []
    tr = RL_RUNS / "traces"
    if tr.exists():
        for p in tr.glob("*.json"):
            items.append({"bundle_id": p.stem, "source": "eval"})
    for p in RL_RUNS.glob("*.traces.json"):
        items.append({"bundle_id": p.stem, "source": "eval"})
    if CALIB_RUNS.exists():
        for d in CALIB_RUNS.iterdir():
            if d.is_dir() and (d / "ticks.jsonl").exists():
                items.append({"bundle_id": f"l7:{d.name}", "source": "l7"})
    return {"bundles": items}


@app.get("/api/replay/bundle/{bundle_id:path}")
def replay_bundle(bundle_id: str):
    # guard traversal
    if ".." in bundle_id:
        raise HTTPException(400, "bad path")
    if bundle_id.startswith("l7:"):
        rid = bundle_id[3:]
        if ".." in rid or "/" in rid:
            raise HTTPException(400, "bad path")
        return sanitize(l7_bundle(rid))
    # eval trace
    candidates = [
        RL_RUNS / "traces" / f"{bundle_id}.json",
        RL_RUNS / f"{bundle_id}.traces.json",
        RL_RUNS / f"{bundle_id}.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            safe = safe_path(RL_RUNS, str(c.relative_to(RL_RUNS)))
            if safe is None:
                raise HTTPException(400, "bad path")
            return sanitize(eval_bundle(c))
    raise HTTPException(404, "not found")


@app.get("/api/l7/{run_id}/end.jpg")
def l7_endjpg(run_id: str):
    if ".." in run_id or "/" in run_id:
        raise HTTPException(400, "bad path")
    p = CALIB_RUNS / run_id / "end.jpg"
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))


@app.get("/api/rollout/options")
def rollout_options():
    pts = [p.name for p in (RL_RUNS).glob("*.pt")]
    return {
        "policies": pts + ["oracle_pd", "brake"],
        "baselines": ["oracle_pd", "brake"],
        "levels": LEVELS,
    }


@app.post("/api/rollout")
async def rollout(req: dict):
    policy = req.get("policy", "oracle_pd")
    level = req.get("level", "L0")
    episodes = max(1, min(20, int(req.get("episodes", 5))))
    seed = int(req.get("seed", 11))
    ts = int(time.time())
    bundle_id = f"obs_{ts}"
    json_path = RL_RUNS / f"obs_{ts}.json"
    trace_path = RL_RUNS / "traces" / f"obs_{ts}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    py = REPO / "rl" / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("python")
    cmd = [
        str(py),
        "evaluate.py",
        "--episodes",
        str(episodes),
        "--level",
        level,
        "--policy",
        policy,
        "--seed",
        str(seed),
        "--save-traces",
        str(trace_path),
        "--json",
        str(json_path),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(REPO / "rl"))
    job_id = f"job_{ts}"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "bundle_id": bundle_id,
        "error": None,
        "json_path": str(json_path),
        "proc": proc,
        "trace_path": str(trace_path),
    }
    asyncio.ensure_future(_watch(job_id, proc, trace_path))
    return {"job_id": job_id, "status": "running"}


async def _watch(job_id, proc, trace_path):
    await proc.wait()
    job = JOBS[job_id]
    if proc.returncode == 0:
        job["status"] = "done"
        job["bundle_id"] = Path(trace_path).stem
    else:
        job["status"] = "failed"
        job["error"] = f"exit {proc.returncode}"


@app.get("/api/rollout/{job_id}")
def rollout_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "bundle_id": job["bundle_id"],
        "error": job["error"],
        "json_path": job["json_path"],
    }


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _norm_contract_checks(raw, real_metrics):
    checks = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            v = v if isinstance(v, dict) else {}
            checks.append(
                {
                    "name": k,
                    "pass": bool(v.get("pass", True)),
                    "measured": v.get("value", v.get("real", "—")),
                    "limit": v.get("threshold", v.get("target", "—")),
                }
            )
        return checks
    if isinstance(raw, list):
        for v in raw:
            if not isinstance(v, dict):
                continue
            status = str(v.get("status", "")).upper()
            passed = v.get("pass")
            if passed is None:
                passed = status != "FAIL"
            checks.append(
                {
                    "name": v.get("name", "check"),
                    "pass": bool(passed),
                    "measured": v.get("value", v.get("real", "—")),
                    "limit": v.get("limit", v.get("threshold", v.get("target", "—"))),
                }
            )
        return checks
    # synthesize from real_metrics when no checks present
    rm = real_metrics or {}
    vr = _num(rm.get("valid_rate"))
    if vr is not None:
        checks.append(
            {
                "name": "valid_pose_rate",
                "pass": vr >= 0.5,
                "measured": round(vr, 3),
                "limit": ">= 0.50",
            }
        )
    hz = _num(rm.get("hz"))
    if hz is not None:
        checks.append(
            {
                "name": "loop_rate_hz",
                "pass": hz >= 40.0,
                "measured": round(hz, 2),
                "limit": ">= 40.0",
            }
        )
    loop_ms = rm.get("loop_ms") if isinstance(rm.get("loop_ms"), dict) else {}
    p95 = _num(loop_ms.get("p95"))
    if p95 is not None:
        checks.append(
            {
                "name": "loop_ms_p95",
                "pass": p95 <= 25.0,
                "measured": round(p95, 2),
                "limit": "<= 25.0",
            }
        )
    return checks


def _channel_row(name, real_mean, real_std, sim_mean, sim_std):
    return {
        "name": name,
        "real_mean": _num(real_mean),
        "real_std": _num(real_std),
        "sim_mean": _num(sim_mean),
        "sim_std": _num(sim_std),
    }


def _norm_contract_channels(real_metrics, sim):
    rm = real_metrics or {}
    sim = sim or {}
    sim_ch = sim.get("channels") if isinstance(sim.get("channels"), dict) else {}
    real_ch = rm.get("channels") if isinstance(rm.get("channels"), dict) else {}

    # vision_hz: sim = 1 / inter_valid_dt.mean; real likewise (fallback: hz * valid_rate)
    def hz_of(src):
        iv = src.get("inter_valid_dt") if isinstance(src.get("inter_valid_dt"), dict) else {}
        m = _num(iv.get("mean"))
        if m and m > 0:
            return 1.0 / m, (_num(iv.get("std")) or 0.0) / (m * m)
        h = _num(src.get("hz"))
        vr = _num(src.get("valid_rate"))
        if h is not None and vr is not None:
            return h * vr, None
        return h, None

    real_hz, real_hz_std = hz_of(rm)
    sim_hz, sim_hz_std = hz_of(sim)
    if sim_hz is None:
        sim_hz, sim_hz_std = 50.0, 0.0  # nominal 50 Hz loop, tag every frame

    # frame_age_p50 (tag_age channel, seconds -> ms)
    fa_r = real_ch.get("tag_age") if isinstance(real_ch.get("tag_age"), dict) else {}
    fa_s = sim_ch.get("tag_age") if isinstance(sim_ch.get("tag_age"), dict) else {}
    fa_real_p50 = _num(fa_r.get("p50"))
    fa_sim_p50 = _num(fa_s.get("p50"))

    # dropout_rate = 1 - valid_rate
    vr_r, vr_s = _num(rm.get("valid_rate")), _num(sim.get("valid_rate"))

    # imu_w std
    iw_r = (
        _num((real_ch.get("imu_w") or {}).get("std"))
        if isinstance(real_ch.get("imu_w"), dict)
        else None
    )
    iw_s = (
        _num((sim_ch.get("imu_w") or {}).get("std"))
        if isinstance(sim_ch.get("imu_w"), dict)
        else None
    )

    # loop_ms p50/p95 (sim has no control loop; nominal 20 ms for 50 Hz)
    loop_r = rm.get("loop_ms") if isinstance(rm.get("loop_ms"), dict) else {}
    lm_p50, lm_p95 = _num(loop_r.get("p50")), _num(loop_r.get("p95"))
    lm_std = _num(loop_r.get("std"))

    return [
        _channel_row("vision_hz", real_hz, real_hz_std, sim_hz, sim_hz_std),
        _channel_row(
            "frame_age_p50_ms",
            fa_real_p50 * 1000 if fa_real_p50 is not None else None,
            (_num(fa_r.get("std")) or 0.0) * 1000 if _num(fa_r.get("std")) is not None else None,
            fa_sim_p50 * 1000 if fa_sim_p50 is not None else 0.0,
            (_num(fa_s.get("std")) or 0.0) * 1000 if _num(fa_s.get("std")) is not None else None,
        ),
        _channel_row(
            "dropout_rate",
            1.0 - vr_r if vr_r is not None else None,
            None,
            1.0 - vr_s if vr_s is not None else 0.0,
            None,
        ),
        _channel_row("imu_w_std", iw_r, None, iw_s if iw_s is not None else 0.05, None),
        _channel_row("loop_ms_p50", lm_p50, lm_std, 20.0, 0.0),
        _channel_row("loop_ms_p95", lm_p95, lm_std, 20.0, 0.0),
    ]


def _eval_contract(run_id):
    from tools.l7.audit_contract import evaluate_contract, extract_run_metrics

    run_dir = CALIB_RUNS / run_id
    sim_ref = REPO / "tools" / "l7" / "sim_reference_L1.npz"
    note = None
    try:
        report = evaluate_contract(run_dir, sim_ref_path=sim_ref)
    except Exception as e:
        # graceful: compute real metrics + probe + sim profile
        try:
            real = extract_run_metrics(run_dir)
        except Exception as e2:
            real = {"run_id": run_id, "error": str(e2)}
        report = {"run_id": run_id, "real_metrics": real, "note": f"sim_ref unavailable: {e}"}
        note = str(e)
    if note:
        report.setdefault("note", note)
    real_metrics = report.get("real_metrics") or report.get("real") or {}
    sim = report.get("sim")
    sim_ok = isinstance(sim, dict) and bool(sim)
    report["sim_source"] = "sim_reference_L1.npz" if sim_ok else "nominal_contract"
    report["status"] = report.get("verdict") or ("PASS" if report.get("passed") else "INFO")
    report["checks"] = _norm_contract_checks(report.get("checks"), real_metrics)
    report["channels"] = _norm_contract_channels(real_metrics, sim if sim_ok else {})
    return sanitize(report)


@app.get("/api/contract/runs")
def contract_runs():
    runs = []
    if CALIB_RUNS.exists():
        for d in CALIB_RUNS.iterdir():
            if d.is_dir() and (d / "ticks.jsonl").exists():
                runs.append(d.name)
    return {"runs": runs}


@app.get("/api/contract")
def contract(run_id: str | None = Query(default=None)):
    if run_id is not None:
        if ".." in run_id or "/" in run_id:
            raise HTTPException(400, "bad path")
        run_dir = CALIB_RUNS / run_id
        if not run_dir.is_dir():
            raise HTTPException(404, f"L7 run not found: {run_id}")
        return _eval_contract(run_id)
    # default: latest real L7 directory that has ticks.jsonl
    latest = None
    if CALIB_RUNS.exists():
        candidates = [
            d for d in CALIB_RUNS.iterdir() if d.is_dir() and (d / "ticks.jsonl").exists()
        ]
        if candidates:
            latest = max(candidates, key=lambda d: (d / "ticks.jsonl").stat().st_mtime)
    if latest is None:
        raise HTTPException(404, "no L7 runs with ticks.jsonl found")
    return _eval_contract(latest.name)


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
