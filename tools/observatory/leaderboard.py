"""Leaderboard assembly: archive, sim eval JSONs, Gazebo L6 summaries and L7 field runs."""

import json
from pathlib import Path

from .data import l7_runs, read_json
from .wilson import wilson_ci

# Debug, smoke, observatory-rollout and live-stream runs are probes, not graded
# evals: they must not sit beside 1000-episode gates or win stage cards.
PROBE_PREFIXES = ("dbg_", "smoke_", "obs_", "live_")
# Smallest run that may appear on the primary leaderboard outside L7. L7 field
# runs are one episode by nature; sim and Gazebo probes of 1..10 episodes are not
# comparable to a 1000-episode gate.
GRADED_MIN_EPISODES = 20


def _is_probe(run_id) -> bool:
    return str(run_id or "").startswith(PROBE_PREFIXES)


def _is_graded(e) -> bool:
    """True when a leaderboard row belongs on the primary board and stage cards."""
    if _is_probe(e.get("run_id")):
        return False
    if e.get("source") == "l7":
        return True
    eps = e.get("episodes")
    # an unevaluated archive row has no episode count; a reported count below the
    # gate size is a probe, not a comparable run
    return eps is None or eps >= GRADED_MIN_EPISODES


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


def _count(v):
    """Coerce a stored count to int; lists/dicts/garbage become None."""
    if isinstance(v, bool) or isinstance(v, (list, dict)) or v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_entry(d, source, run_id=None):
    rid = run_id or d.get("run_id") or d.get("id") or "unknown"
    succ = _count(d.get("success"))
    eps = _count(d.get("episodes"))
    if eps is not None and eps <= 0:
        eps = None
    if succ is None or eps is None:
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
        "has_rate": sr is not None,
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


def _gazebo_entry(p):
    d = read_json(p)
    if not isinstance(d, dict):
        return None
    summary = d.get("summary") if isinstance(d.get("summary"), dict) else {}
    merged = dict(summary)
    for k, v in _flatten_metrics(d).items():
        if v is None:
            continue
        # keep a scalar summary value over a nested top-level value
        if isinstance(v, (list, dict)) and isinstance(summary.get(k), (int, float, str, bool)):
            continue
        merged[k] = v

    def pick(*keys):
        for k in keys:
            v = merged.get(k)
            if v is not None:
                return v
        return None

    success = pick("success")
    episodes = pick("episodes")
    if _count(success) is None or _count(episodes) is None:
        # a plan list, config dump or aborted run: not an eval summary
        return None

    return _norm_entry(
        {
            "success": success,
            "episodes": episodes,
            "collisions": pick("collisions"),
            "timeouts": pick("timeouts"),
            "false_parked": pick("false_parked"),
            "time_median": pick("time_median", "median_time_s"),
            "time_p95": pick("time_p95"),
            "pos_p95": pick("pos_p95", "pos_err_p95_m"),
            "yaw_p95_deg": pick("yaw_p95_deg", "yaw_err_p95_deg"),
            "level": pick("level"),
            "track": pick("track"),
            "tags": pick("tags"),
            "parent_id": pick("parent_id"),
            "config_delta": pick("config_delta"),
        },
        "gazebo_l6",
        run_id=p.stem,
    )


def gazebo_entries(rl_runs: Path) -> list[dict]:
    """Gazebo L6 runs that belong on the primary board: graded eval summaries only.

    ``gz.glob("*.json")`` also returns directories, plan/config JSON and probe
    runs; treating those as eval runs used to inflate the count and the board.
    """
    gz = rl_runs / "gazebo_l6"
    if not gz.is_dir():
        return []
    out = []
    for p in sorted(gz.glob("*.json")):
        if not p.is_file() or _is_probe(p.stem):
            continue
        entry = _gazebo_entry(p)
        if entry is not None and _is_graded(entry):
            out.append(entry)
    return out


def _l7_entry(r):
    succ = 1 if r.get("outcome") in ("success", "parked") else 0
    lo, hi = wilson_ci(succ, 1)
    return {
        "run_id": r["run_id"],
        "track": "l7",
        "level": "L_real",
        "success": succ,
        "collisions": 0,
        "false_parked": 0,
        "timeouts": 0,
        "episodes": 1,
        "success_rate": float(succ),
        "has_rate": True,
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
        # the raw field outcome: "arrived" is estimate-based and not a confirmed park
        "outcome": r.get("outcome"),
        "has_end_frame": bool(r.get("has_end_jpg")),
    }


def _mark_pareto(entries):
    """Non-dominated on (success_rate, -collision_rate), computed per level.

    A run with no measured success (``success_rate`` of 0 or None) never earns a
    badge: a 0/20 debug run is a failure, not a frontier trade-off.
    """
    by_level = {}
    for e in entries:
        by_level.setdefault(e.get("level"), []).append(e)
    for grp in by_level.values():
        for e in grp:
            sr = e.get("success_rate")
            if not sr:
                e["pareto"] = False
                e["pareto_score"] = None
                continue
            coll_rate = (e.get("collisions") or 0) / max(1, e.get("episodes") or 1)
            dominated = False
            for o in grp:
                if o is e:
                    continue
                osr = o.get("success_rate") or 0.0
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


def build(rl_runs: Path) -> list[dict]:
    by_id: dict[str, dict] = {}

    def add(entry):
        rid = entry["run_id"]
        if rid not in by_id:
            by_id[rid] = entry
            return
        cur = by_id[rid]
        for k, v in entry.items():
            if cur.get(k) is None:
                cur[k] = v

    arch = rl_runs / "archive.jsonl"
    if arch.exists():
        for line in arch.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                add(_norm_entry(_flatten_metrics(d), "archive"))

    for p in sorted(rl_runs.glob("*.json")):
        if p.name.endswith(".traces.json"):
            continue
        d = read_json(p)
        if isinstance(d, dict) and "success" in d and "episodes" in d:
            add(_norm_entry(_flatten_metrics(d), "eval_json", run_id=d.get("run_id") or p.stem))

    gz = rl_runs / "gazebo_l6"
    if gz.exists():
        for entry in gazebo_entries(rl_runs):
            add(entry)

    for r in l7_runs():
        # real-Pi runs replace any same-id sim row so the field outcome is shown
        by_id[r["run_id"]] = _l7_entry(r)

    # probe and tiny non-L7 rows stay out of the primary board and stage cards
    entries = [e for e in by_id.values() if _is_graded(e)]
    for e in entries:
        if e.get("success") is None or e.get("episodes") is None:
            e.update(success_rate=None, wilson_lo=None, wilson_hi=None, has_rate=False)
    _mark_pareto(entries)
    return entries
