"""Score a keypoint detector on recorded challenge frames (`ugv-lab --record`), against the true
keypoints written next to each frame. Pure measurement: no training, no tuning.

A detection counts as correct when it lands within `hit_px` of its own keypoint's true pixel;
anything else is a false detection (including a point placed on the mirrored keypoint, which is
counted separately as a left/right swap). Frames with no keypoint in view measure false alarms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ugv.pose.mesh import keypoints, load_airframe

MIRROR = {
    "wingtip_left": "wingtip_right",
    "hub_left": "hub_right",
    "vtail_tip_left": "vtail_tip_right",
    "gear_main_left": "gear_main_right",
}


def _mirror_index() -> np.ndarray:
    names = list(keypoints(load_airframe()))
    m = {**MIRROR, **{v: k for k, v in MIRROR.items()}}
    return np.array([names.index(m.get(n, n)) for n in names])


@dataclass
class FrameScore:
    range_m: float
    n_true: int  # keypoints in view
    n_det: int
    n_hit: int
    n_swap: int
    errors: list[float]  # px, correct detections only


def load(root: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(root) / "labels.jsonl").open()]


def score(rows: list[dict], detect, root: Path, hit_px: float = 20.0) -> list[FrameScore]:
    """`detect(frame) -> [[i, u, v], ...]`; frames are read from `root`."""
    mirror = _mirror_index()
    out = []
    for r in rows:
        frame = cv2.imread(str(Path(root) / r["file"]), cv2.IMREAD_GRAYSCALE)
        dets = detect(frame)
        uv, inside = np.array(r["true_uv"]), np.array(r["true_in"])
        hits, swaps, errs = 0, 0, []
        for i, u, v in dets:
            i = int(i)
            e = float(np.hypot(u - uv[i, 0], v - uv[i, 1]))
            if inside[i] and e <= hit_px:
                hits += 1
                errs.append(e)
            elif inside[mirror[i]] and mirror[i] != i:
                if np.hypot(u - uv[mirror[i], 0], v - uv[mirror[i], 1]) <= hit_px:
                    swaps += 1
        out.append(FrameScore(r["range_m"], int(inside.sum()), len(dets), hits, swaps, errs))
    return out


def summary(scores: list[FrameScore]) -> dict:
    empty = [s for s in scores if s.n_true == 0]
    seen = [s for s in scores if s.n_true > 0]
    errs = [e for s in seen for e in s.errors]
    n_det = sum(s.n_det for s in seen)
    out = {
        "frames": len(scores),
        "frames_empty": len(empty),
        "false_alarm_frames": round(np.mean([s.n_det > 0 for s in empty]), 3) if empty else None,
        "false_points_per_empty_frame": round(np.mean([s.n_det for s in empty]), 2)
        if empty
        else None,
        "frames_with_airframe": len(seen),
        "recall": round(sum(s.n_hit for s in seen) / max(1, sum(s.n_true for s in seen)), 3),
        "precision": round(sum(s.n_hit for s in seen) / max(1, n_det), 3),
        "swap_rate": round(sum(s.n_swap for s in seen) / max(1, n_det), 3),
        "median_px": round(float(np.median(errs)), 1) if errs else None,
        "frames_6_correct": round(np.mean([s.n_hit >= 6 for s in seen]), 3) if seen else None,
    }
    bins = [(0, 1), (1, 2), (2, 3), (3, 9)]
    out["by_range"] = {}
    for lo, hi in bins:
        b = [s for s in seen if lo <= s.range_m < hi]
        if b:
            out["by_range"][f"{lo}-{hi} m"] = {
                "frames": len(b),
                "recall": round(sum(s.n_hit for s in b) / max(1, sum(s.n_true for s in b)), 3),
                "precision": round(sum(s.n_hit for s in b) / max(1, sum(s.n_det for s in b)), 3),
                "frames_6_correct": round(np.mean([s.n_hit >= 6 for s in b]), 3),
            }
    return out
