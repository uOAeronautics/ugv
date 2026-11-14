"""Data helpers for the UGV Research Observatory."""

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RL_RUNS = REPO / "rl" / "runs"
CALIB_RUNS = REPO / "calibration" / "runs" / "l7"

LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L_real", "L_chaos"]


def safe_path(base: Path, rel: str) -> Path | None:
    try:
        p = (base / rel).resolve()
        base_r = base.resolve()
        if base_r == p or base_r in p.parents:
            return p
    except Exception:
        pass
    return None


def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def parse_log(path: Path):
    points = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"iter\s+(\d+)", line)
        if not m:
            continue
        it = int(m.group(1))

        def num(key, _line=line):
            mm = re.search(rf"\b{key}\b\s*[=:]?\s*([-+]?[\d.eE+]+)", _line)
            return float(mm.group(1)) if mm else None

        point = {"iter": it}
        # short key -> (metric value, long key)
        pairs = [
            ("dr_scale", "dr_scale", "dr"),
            ("anchor_frac", "anchor_frac", "anchor_frac"),
            ("std", "std", "std"),
            ("entropy", "entropy", "ent"),
            ("sps", "sps", "sps"),
            ("wall_s", "wall_s", "wall_s"),
            ("succ", "succ_rate", "succ"),
            ("coll", "coll_rate", "coll"),
            ("timeout", "timeout_rate", "timeout"),
            ("false_park", "false_park", "false_park"),
            ("pol", "pol_loss", "pol"),
            ("val", "val_loss", "val"),
            ("aux", "aux_loss", "aux"),
            ("dyn", "dyn_loss", "dyn"),
            ("return", "mean_return", "return"),
        ]
        for short, long, key in pairs:
            v = num(key)
            if v is not None:
                point[short] = v
                point[long] = v
        points.append(point)
    return points


def parse_metrics_jsonl(path: Path):
    points = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue

        def pick(*keys, _d=d):
            for k in keys:
                v = _d.get(k)
                if v is not None:
                    return v
            return None

        succ = pick("succ_rate", "succ")
        coll = pick("coll_rate", "coll")
        timeout = pick("timeout_rate", "timeout")
        false_park = pick("false_park", "false_park_rate")
        pol = pick("pol_loss", "pol")
        val = pick("val_loss", "val")
        aux = pick("aux_loss", "aux")
        dyn = pick("dyn_loss", "dyn")
        ret = pick("mean_return", "return")
        points.append(
            {
                "iter": d.get("iter"),
                "dr_scale": d.get("dr_scale"),
                "anchor_frac": d.get("anchor_frac"),
                "std": d.get("std"),
                "entropy": d.get("entropy"),
                "sps": d.get("sps"),
                "wall_s": d.get("wall_s"),
                "succ": succ,
                "succ_rate": succ,
                "coll": coll,
                "coll_rate": coll,
                "timeout": timeout,
                "timeout_rate": timeout,
                "false_park": false_park,
                "pol": pol,
                "pol_loss": pol,
                "val": val,
                "val_loss": val,
                "aux": aux,
                "aux_loss": aux,
                "dyn": dyn,
                "dyn_loss": dyn,
                "return": ret,
                "mean_return": ret,
            }
        )
    return points


def l7_runs():
    out = []
    if not CALIB_RUNS.exists():
        return out
    for d in sorted(CALIB_RUNS.iterdir()):
        if not d.is_dir():
            continue
        s = d / "summary.json"
        m = d / "meta.json"
        t = d / "ticks.jsonl"
        summary = json.loads(s.read_text()) if s.exists() else {}
        meta = json.loads(m.read_text()) if m.exists() else {}
        has_end = (d / "end.jpg").exists()
        ticks = 0
        if t.exists():
            ticks = sum(1 for _ in t.open())
        out.append(
            {
                "run_id": d.name,
                "mode": meta.get("mode") or summary.get("mode"),
                "outcome": summary.get("outcome"),
                "seconds": summary.get("seconds"),
                "hz": summary.get("hz"),
                "ticks": ticks,
                "start_pose": summary.get("start_pose") or meta.get("start_pose"),
                "acquired_pose": summary.get("acquired_pose"),
                "end_pose_tag": summary.get("end_pose_tag"),
                "end_est": summary.get("end_est"),
                "vision_ms_p50": summary.get("vision_ms_p50"),
                "loop_ms_p50": summary.get("loop_ms_p50"),
                "loop_ms_p95": summary.get("loop_ms_p95"),
                "has_end_jpg": has_end,
            }
        )
    return out
