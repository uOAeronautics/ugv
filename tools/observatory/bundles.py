"""Replay bundle index: discovery, source tagging, episode counts, safe resolution.

Bundle ids are namespaced: ``l7:<run>`` for real Pi field runs,
``gazebo_l6:<name>`` for Gazebo summaries, and a bare stem for sim traces.
"""

from pathlib import Path

from .data import read_json, safe_path

SUCCESS_OUTCOMES = {"success", "parked"}

# path -> ((mtime_ns, size), (episodes, successes)); big traces are parsed once
_STATS_CACHE: dict[Path, tuple[tuple[int, int], tuple[int | None, int | None]]] = {}


def bundle_source(bundle_id: str) -> str:
    if bundle_id.startswith("l7:"):
        return "l7"
    if "gazebo" in bundle_id.lower():
        return "gazebo_l6"
    return "sim"


def _trace_stats(path: Path) -> tuple[int | None, int | None]:
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _STATS_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    data = read_json(path, None)
    eps = data.get("episodes") if isinstance(data, dict) else None
    if isinstance(eps, list):
        stats = (
            len(eps),
            sum(1 for e in eps if isinstance(e, dict) and e.get("outcome") == "success"),
        )
    else:
        stats = (None, None)
    _STATS_CACHE[path] = (key, stats)
    return stats


def _label(bundle_id: str) -> str:
    return bundle_id.removeprefix("gazebo_l6:").removeprefix("l7:")


def _item(bundle_id: str, counts: tuple[int | None, int | None]) -> dict:
    return {
        "bundle_id": bundle_id,
        "label": _label(bundle_id),
        "source": bundle_source(bundle_id),
        "episodes_count": counts[0],
        "success_count": counts[1],
    }


def list_bundles(rl_runs: Path, calib_dir: Path) -> list[dict]:
    items: list[dict] = []
    tr = rl_runs / "traces"
    if tr.exists():
        for p in sorted(tr.glob("*.json")):
            items.append(_item(p.stem, _trace_stats(p)))
    for p in sorted(rl_runs.glob("*.traces.json")):
        items.append(_item(p.name.removesuffix(".traces.json"), _trace_stats(p)))
    gz_dir = rl_runs / "gazebo_l6"
    if gz_dir.exists():
        for p in sorted(gz_dir.glob("*.traces.json")):
            name = p.name.removesuffix(".traces.json")
            items.append(_item(f"gazebo_l6:{name}", _trace_stats(p)))
    if calib_dir.exists():
        for d in sorted(calib_dir.iterdir()):
            if d.is_dir() and (d / "ticks.jsonl").exists():
                summary = read_json(d / "summary.json", {})
                ok = isinstance(summary, dict) and summary.get("outcome") in SUCCESS_OUTCOMES
                items.append(_item(f"l7:{d.name}", (1, 1 if ok else 0)))
    return items


def _clean(bundle_id: str) -> None:
    if not bundle_id or ".." in bundle_id or "\\" in bundle_id or bundle_id.startswith("/"):
        raise ValueError("bad path")


def resolve(bundle_id: str, rl_runs: Path, calib_dir: Path):
    """Return ``("l7", run_dir)`` or ``("trace", file)``; raise LookupError if absent."""
    _clean(bundle_id)
    if bundle_id.startswith("l7:"):
        rid = bundle_id[3:]
        if not rid or "/" in rid:
            raise ValueError("bad path")
        d = calib_dir / rid
        if d.is_dir() and (d / "ticks.jsonl").exists():
            return "l7", d
        raise LookupError(bundle_id)
    if bundle_id.startswith("gazebo_l6:"):
        rid = bundle_id[len("gazebo_l6:") :]
        if not rid or "/" in rid:
            raise ValueError("bad path")
        candidates = [f"gazebo_l6/{rid}.traces.json", f"gazebo_l6/{rid}.json"]
    else:
        candidates = [
            f"traces/{bundle_id}.json",
            f"{bundle_id}.traces.json",
            f"{bundle_id}.json",
            f"gazebo_l6/{bundle_id}.traces.json",
        ]
    for rel in candidates:
        p = safe_path(rl_runs, rel)
        if p is not None and p.is_file():
            return "trace", p
    raise LookupError(bundle_id)
