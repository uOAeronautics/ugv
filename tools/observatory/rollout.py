"""Live rollouts from the dashboard: one background job at a time.

Two engines:
  sim     rl/evaluate.py (VecSim, torch). Checkpoints are passed as ``runs/<name>.pt``
          because the subprocess runs with cwd = rl/.
  gazebo  simulation/gazebo/run_policy_episodes.py on the Gazebo Harmonic bridge.
          Either the EKF/staged baseline (``--baseline``) or an exported ``.npz``.
"""

import asyncio
import sys
import time
from pathlib import Path

from .data import LEVELS, REPO, RL_RUNS, safe_path

ENGINES = ("sim", "gazebo")
SIM_BASELINES = ("oracle_pd", "brake")
GZ_BASELINE = "baseline"
MAX_EPISODES = 20
TAIL_CHARS = 800

JOBS: dict[str, dict] = {}


class RolloutError(ValueError):
    """Bad request (HTTP 400)."""


class RolloutBusy(RuntimeError):
    """Another job holds the runner (HTTP 409)."""


def _rl_python() -> str:
    venv = REPO / "rl" / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _gz_weights(rl_runs: Path) -> list[str]:
    paths = list((rl_runs / "gazebo_l6").glob("*.npz")) + list(rl_runs.glob("*.npz"))
    return sorted(str(p.relative_to(rl_runs)) for p in paths)


def _sim_checkpoints(rl_runs: Path) -> list[str]:
    return sorted(p.name for p in rl_runs.glob("*.pt"))


def options(rl_runs: Path = RL_RUNS) -> dict:
    sim = list(SIM_BASELINES) + _sim_checkpoints(rl_runs)
    gz = [GZ_BASELINE] + _gz_weights(rl_runs)
    return {
        "engines": {
            "sim": {"label": "Fast VecSim", "policies": sim},
            "gazebo": {"label": "Gazebo Harmonic L6", "policies": gz},
        },
        # legacy flat list, kept for older pages
        "policies": sim,
        "baselines": list(SIM_BASELINES),
        "levels": LEVELS,
        "episodes_max": MAX_EPISODES,
    }


def _policy_allowed(engine: str, policy: str, rl_runs: Path) -> bool:
    if engine == "sim":
        if policy in SIM_BASELINES:
            return True
        return policy in _sim_checkpoints(rl_runs)
    if policy == GZ_BASELINE:
        return True
    if policy not in _gz_weights(rl_runs):
        return False
    p = safe_path(rl_runs, policy)
    return p is not None and p.suffix == ".npz" and p.is_file()


def _command(engine, policy, level, episodes, seed, stamp, rl_runs):
    """Return (argv, cwd, json_path, trace_path)."""
    if engine == "sim":
        pol_arg = policy if policy in SIM_BASELINES else f"runs/{policy}"
        trace = rl_runs / "traces" / f"obs_{stamp}.json"
        js = rl_runs / f"obs_{stamp}.json"
        argv = [
            _rl_python(),
            "evaluate.py",
            "--episodes",
            str(episodes),
            "--level",
            level,
            "--policy",
            pol_arg,
            "--seed",
            str(seed),
            "--save-traces",
            str(trace),
            "--json",
            str(js),
        ]
        return argv, REPO / "rl", js, trace
    trace = rl_runs / "traces" / f"gazebo_{stamp}.json"
    js = rl_runs / "gazebo_l6" / f"live_{stamp}.json"
    argv = [
        sys.executable,
        "simulation/gazebo/run_policy_episodes.py",
        "--bridge",
        "gazebo",
        "--level",
        level,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--save-traces",
        str(trace),
        "--json",
        str(js),
    ]
    if policy == GZ_BASELINE:
        argv.append("--baseline")
    else:
        argv += ["--weights", str((rl_runs / policy).resolve())]
    return argv, REPO, js, trace


def _running() -> str | None:
    for jid, job in JOBS.items():
        if job["status"] == "running":
            return jid
    return None


def parse_request(req: dict) -> dict:
    engine = str(req.get("engine") or "sim")
    if engine not in ENGINES:
        raise RolloutError(f"engine must be one of {ENGINES}")
    level = str(req.get("level") or "L0")
    if level not in LEVELS:
        raise RolloutError(f"level must be one of {LEVELS}")
    try:
        episodes = int(req.get("episodes", 5))
        seed = int(req.get("seed", 11))
    except (TypeError, ValueError) as e:
        raise RolloutError("episodes and seed must be integers") from e
    return {
        "engine": engine,
        "policy": str(req.get("policy") or ("oracle_pd" if engine == "sim" else GZ_BASELINE)),
        "level": level,
        "episodes": max(1, min(MAX_EPISODES, episodes)),
        "seed": seed,
    }


async def start(req: dict, rl_runs: Path = RL_RUNS) -> dict:
    cfg = parse_request(req)
    if not _policy_allowed(cfg["engine"], cfg["policy"], rl_runs):
        raise RolloutError(f"unknown {cfg['engine']} policy {cfg['policy']!r}")
    busy = _running()
    if busy:
        raise RolloutBusy(f"job {busy} is still running; wait for it to finish")
    stamp = int(time.time())
    argv, cwd, js, trace = _command(
        cfg["engine"], cfg["policy"], cfg["level"], cfg["episodes"], cfg["seed"], stamp, rl_runs
    )
    trace.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    job_id = f"job_{stamp}_{len(JOBS)}"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "engine": cfg["engine"],
        "policy": cfg["policy"],
        "level": cfg["level"],
        "episodes": cfg["episodes"],
        "bundle_id": trace.stem,
        "error": None,
        "json_path": str(js),
        "trace_path": str(trace),
    }
    asyncio.ensure_future(_watch(job_id, proc))
    return {"job_id": job_id, "status": "running", "bundle_id": trace.stem, "engine": cfg["engine"]}


async def _watch(job_id: str, proc) -> None:
    _out, err = await proc.communicate()
    job = JOBS[job_id]
    trace = Path(job["trace_path"])
    if proc.returncode == 0 and trace.is_file():
        job["status"] = "done"
        return
    job["status"] = "failed"
    tail = (err or b"").decode(errors="replace")[-TAIL_CHARS:].strip()
    if proc.returncode == 0:
        job["error"] = "run finished but wrote no trace bundle"
    else:
        job["error"] = f"exit {proc.returncode}" + (f": {tail}" if tail else "")


def status(job_id: str) -> dict | None:
    job = JOBS.get(job_id)
    if job is None:
        return None
    return {k: job[k] for k in ("job_id", "status", "engine", "bundle_id", "error", "json_path")}
