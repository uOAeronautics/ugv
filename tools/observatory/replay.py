"""Replay bundle normalization."""

import json
import math
from bisect import bisect_right
from pathlib import Path

from .data import CALIB_RUNS, read_json, sanitize

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


def _load_l7_camera_frames(d: Path, run_id: str, t0: float):
    frames_dir = d / "frames"
    frames_jsonl = d / "frames.jsonl"
    if not (frames_dir.is_dir() and frames_jsonl.exists()):
        return [], []
    cam_rows = []
    det_rows = []
    for line in frames_jsonl.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        cap = row.get("cap")
        if cap is None:
            continue
        try:
            cap_f = float(cap)
        except (TypeError, ValueError):
            continue
        tags_raw = row.get("tags") or []
        tags_px = []
        for t in tags_raw:
            if not isinstance(t, dict):
                continue
            c = t.get("c")
            if c is None:
                continue
            tags_px.append(
                {
                    "id": t.get("id"),
                    "how": t.get("how"),
                    "px": t.get("px"),
                    "c": c,
                }
            )
        det_rows.append({"cap": cap_f, "t": round(cap_f - t0, 3), "tags_px": tags_px})
        jpg = row.get("jpg")
        if jpg and isinstance(jpg, str) and (frames_dir / jpg).is_file():
            cam_rows.append(
                {
                    "t": round(cap_f - t0, 3),
                    "cap": cap_f,
                    "seq": row.get("seq"),
                    "fname": jpg,
                    "frame_url": f"/api/l7/{run_id}/frame/{jpg}",
                    "tags_px": tags_px,
                }
            )
    cam_rows.sort(key=lambda x: x["cap"])
    det_rows.sort(key=lambda x: x["cap"])
    return cam_rows, det_rows


def l7_bundle(run_id: str | Path) -> dict:
    d = run_id if isinstance(run_id, Path) else CALIB_RUNS / run_id
    rid = d.name
    ticks_path = d / "ticks.jsonl"
    summary = read_json(d / "summary.json", {}) or {}
    meta = read_json(d / "meta.json", {}) or {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(meta, dict):
        meta = {}
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
    cam_frames, det_frames = _load_l7_camera_frames(d, rid, t0)
    cam_caps = [c["cap"] for c in cam_frames]
    det_caps = [c["cap"] for c in det_frames]

    def _match(items, caps, t_abs, max_lag=1.5):
        if not items:
            return None
        idx = bisect_right(caps, t_abs)
        if idx > 0 and (t_abs - caps[idx - 1]) <= max_lag:
            return items[idx - 1]
        if idx < len(items) and (caps[idx] - t_abs) <= 0.5:
            return items[idx]
        return None

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
        t_abs = r.get("t", t0)
        cam_m = _match(cam_frames, cam_caps, t_abs)
        det_m = _match(det_frames, det_caps, t_abs, max_lag=0.5)
        frame_url = cam_m["frame_url"] if cam_m else None
        tags_px = det_m["tags_px"] if det_m else (cam_m["tags_px"] if cam_m else [])
        poses.append(
            {
                "t": (t_abs - t0),
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
                "frame_url": frame_url,
                "camera_frame_url": frame_url,
                "tags_px": tags_px,
            }
        )
    end_url = f"/api/l7/{rid}/end.jpg" if (d / "end.jpg").exists() else None
    bundle = {
        "format": "ugv_replay_v1",
        "run_id": rid,
        "source": "l7",
        "summary": summary,
        "meta": meta,
        "poses": poses,
        "camera_frames": cam_frames,
        "episodes": [
            {
                "episode_id": 0,
                "outcome": summary.get("outcome", "unknown"),
                "duration": round(poses[-1]["t"], 2) if poses else summary.get("seconds", 0.0),
                "steps": len(poses),
                "frames": poses,
                "camera_frames": cam_frames,
                "end_frame_url": end_url,
            }
        ],
        "geometry": GEOMETRY,
    }
    if end_url:
        bundle["end_frame_url"] = end_url
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
    if frame.get("tag_age") is not None:
        try:
            tag_age: int | float = float(frame["tag_age"])
        except (TypeError, ValueError):
            tag_age = 0
    else:
        try:
            tag_age = int(frame.get("steps_since_valid", 0) or 0)
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
    raw_geo = data.get("geometry")
    merged_geo = {**GEOMETRY, **(raw_geo if isinstance(raw_geo, dict) else {})}
    if isinstance(raw_geo, dict) and "skid_half_len" in raw_geo and "gear_len" not in raw_geo:
        try:
            merged_geo["gear_len"] = 2.0 * float(raw_geo["skid_half_len"])
        except (TypeError, ValueError):
            pass
    data["geometry"] = merged_geo
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
    """Load a sim or Gazebo trace bundle. Raises ValueError when it is not JSON."""
    try:
        data = json.loads(path.read_text(errors="replace"))
    except ValueError as e:
        raise ValueError(f"unreadable trace bundle {path.name}: {e}") from e
    if isinstance(data, dict):
        _norm_episodes(data)
    return sanitize(data)
