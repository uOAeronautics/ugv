"""Replay bundle normalization."""

import json
import math
from pathlib import Path

from .data import CALIB_RUNS, sanitize

GEOMETRY = {
    "car_w": 0.16,
    "car_l": 0.22,
    "skid_x": 0.157,
    "gear_len": 0.3048,
    "gear_r": 0.03,
    "inflation": 0.015,
    "goal_x": 0.03,
    "goal_y": 0.05,
    "goal_yaw_deg": 5.0,
    "fov_half_deg": 33.5,
    "min_tag_range": 0.40,
    "area": 1.5,
    "tags": [
        {"id": 14, "x": -0.028, "y": 0.0},
        {"id": 6, "x": 0.06, "y": -0.05},
        {"id": 3, "x": -0.105, "y": -0.049},
    ],
}


def _tag_valid(r) -> bool:
    tag = r.get("tag")
    if isinstance(tag, bool):
        return tag
    src = r.get("src")
    if isinstance(src, str):
        return src.lower() == "tag"
    return False


def _tag_pose(r):
    # explicit tag_x/tag_y/tag_yaw keys take precedence
    if any(k in r for k in ("tag_x", "tag_y", "tag_yaw")):
        try:
            return [
                float(r.get("tag_x", 0.0)),
                float(r.get("tag_y", 0.0)),
                float(r.get("tag_yaw", 0.0)),
            ]
        except (TypeError, ValueError):
            return None
    est = r.get("est")
    if est is None:
        est = r.get("pose")
    return _as_triple(est)


def _first_num(r, keys):
    for k in keys:
        v = r.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def l7_bundle(run_id: str) -> dict:
    d = CALIB_RUNS / run_id
    ticks_path = d / "ticks.jsonl"
    summary = {}
    if (d / "summary.json").exists():
        summary = json.loads((d / "summary.json").read_text())
    meta = {}
    if (d / "meta.json").exists():
        meta = json.loads((d / "meta.json").read_text())
    rows = []
    if ticks_path.exists():
        for line in ticks_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    t0 = rows[0].get("t", 0.0) if rows else 0.0
    poses = []
    for r in rows:
        valid = _tag_valid(r)
        pose = _as_triple(r.get("est")) or _as_triple(r.get("pose"))
        tag = _tag_pose(r) if valid else None
        act = r.get("act")
        want = r.get("want")
        reason = r.get("reason") or ""
        imu = r.get("imu") or [0.0, 0.0]
        src = r.get("src") if isinstance(r.get("src"), str) else ("tag" if valid else "none")
        tag_age = _first_num(r, ("age_s", "tag_age", "frame_age", "steps_since_valid"))
        imu_w = _first_num(r, ("imu_wz", "wz", "imu_w_fed", "imu_w"))
        if imu_w is None and isinstance(imu, (list, tuple)) and len(imu) > 1:
            try:
                imu_w = float(imu[1])
            except (TypeError, ValueError):
                imu_w = None
        imu_yaw = None
        if isinstance(imu, (list, tuple)) and len(imu) > 0:
            try:
                imu_yaw = float(imu[0])
            except (TypeError, ValueError):
                imu_yaw = None
        act_v = act if act is not None else [0, 0, 0]
        poses.append(
            {
                "t": (r.get("t", 0.0) - t0),
                "pose": pose,
                "vel": None,
                "tag": tag,
                "valid": valid,
                "blackout": False,
                "outlier": False,
                "tag_age": tag_age,
                "imu_w": imu_w,
                "imu_yaw": imu_yaw,
                "aux_pose": pose,
                "act": act_v if isinstance(act_v, (list, tuple)) else [0, 0, 0],
                "shield_act": (
                    want if want is not None else (act if act is not None else [0, 0, 0])
                ),
                "shield_blocked": reason not in ("policy", None, ""),
                "reason": reason,
                "src": src or "none",
            }
        )
    bundle = {
        "format": "ugv_replay_v1",
        "run_id": run_id,
        "source": "l7",
        "summary": summary,
        "meta": meta,
        "poses": poses,
        "episodes": [
            {
                "episode_id": 0,
                "outcome": summary.get("outcome", "unknown"),
                "duration": round(poses[-1]["t"], 2) if poses else summary.get("seconds", 0.0),
                "steps": len(poses),
                "frames": poses,
                "end_frame_url": (
                    f"/api/l7/{run_id}/end.jpg" if (d / "end.jpg").exists() else None
                ),
            }
        ],
        "geometry": GEOMETRY,
    }
    if (d / "end.jpg").exists():
        bundle["end_frame_url"] = f"/api/l7/{run_id}/end.jpg"
    return sanitize(bundle)


def _as_triple(v):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return [float(v[0]), float(v[1]), float(v[2])]
        except (TypeError, ValueError):
            return None
    if isinstance(v, dict):
        try:
            return [float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("yaw", 0.0))]
        except (TypeError, ValueError):
            return None
    return None


def _norm_frame(frame):
    pose = _as_triple(frame.get("true_pose")) or _as_triple(frame.get("pose"))
    valid = bool(frame.get("valid", True))
    tag_src = frame.get("delayed_tag")
    if tag_src is None:
        tag_src = frame.get("tag")
    tag = _as_triple(tag_src) if valid else None
    aux = _as_triple(frame.get("aux_pose")) or pose
    if "shield_clear" in frame:
        shield_blocked = not bool(frame.get("shield_clear"))
    else:
        shield_blocked = bool(frame.get("shield_blocked", False))
    act = frame.get("action") or frame.get("act") or [0, 0, 0]
    try:
        t = float(frame.get("t", 0.0) or 0.0)
    except (TypeError, ValueError):
        t = 0.0
    try:
        imu_w = float(frame.get("imu_w", 0.0) or 0.0)
    except (TypeError, ValueError):
        imu_w = 0.0
    try:
        imu_yaw = float(frame.get("imu_yaw", 0.0) or 0.0)
    except (TypeError, ValueError):
        imu_yaw = 0.0
    try:
        tag_age = int(frame.get("steps_since_valid", frame.get("tag_age", 0)) or 0)
    except (TypeError, ValueError):
        tag_age = 0
    return {
        **frame,
        "t": t,
        "pose": pose,
        "tag": tag,
        "aux_pose": aux,
        "valid": valid,
        "blackout": bool(frame.get("blackout", False)),
        "outlier": bool(frame.get("outlier", False)),
        "tag_age": tag_age,
        "imu_w": imu_w,
        "imu_yaw": imu_yaw,
        "act": list(act) if isinstance(act, (list, tuple)) else [0, 0, 0],
        "shield_blocked": shield_blocked,
    }


def _norm_episodes(data):
    data["format"] = "ugv_replay_v1"
    data["geometry"] = GEOMETRY
    for ep in data.get("episodes", []) or []:
        if not isinstance(ep, dict):
            continue
        frames = [_norm_frame(f) for f in (ep.get("frames") or []) if isinstance(f, dict)]
        ep["frames"] = frames
        if ep.get("duration") is None:
            ep["duration"] = round(frames[-1]["t"], 2) if frames else 0.0
        if ep.get("final_error") is None:
            last_pose = frames[-1].get("pose") if frames else None
            ep["final_error"] = (
                round(math.hypot(last_pose[0], last_pose[1]) * 1000, 1) if last_pose else None
            )


def eval_bundle(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        _norm_episodes(data)
    return sanitize(data)
