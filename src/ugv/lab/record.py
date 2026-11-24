"""Record challenge camera frames with their true keypoints, to measure the trained detector on
exactly the pictures it is asked to read (`tools/pose/eval_frames.py`).

    uv run ugv-lab --record runs/challenge_frames/<name>

Each frame is saved as a lossless PNG next to one JSON line: the true pixel of every skeleton
keypoint (projected with the pose and camera mount the browser rendered, the episode's focal,
principal point and lens), whether it is in the image, and what the network reported. The
truth is written to disk only; it never reaches the detector.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ugv.park.geometry import inverse
from ugv.park.perception import HEIGHT, WIDTH, CameraError, _true_projection, camera_points


class Recorder:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._n = len(list(self.root.glob("*.png")))

    def save(
        self,
        sid: str,
        t: float,
        frame: np.ndarray,
        cam: CameraError,
        car: np.ndarray,
        mount: tuple[float, float],
        det,
    ) -> None:
        """`car`: the car pose the browser rendered, in the airframe frame (as the challenge
        draws it); `mount`: the rendered camera yaw and pitch (base mount, shake and knock)."""
        rel = inverse(np.asarray(car, float))  # the airframe in the car frame
        uv, front = _true_projection(camera_points(rel), replace(cam, yaw=mount[0], pitch=mount[1]))
        inside = (
            front & (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
        )
        with self._lock:
            name = f"{self._n:06d}.png"
            self._n += 1
        cv2.imwrite(str(self.root / name), frame)
        row = {
            "file": name,
            "session": sid,
            "t": round(t, 2),
            "car": np.round(car, 4).tolist(),
            "mount": [round(float(m), 5) for m in mount],
            "range_m": round(float(np.hypot(*car[:2])), 3),
            "true_uv": np.round(uv, 1).tolist(),
            "true_in": inside.tolist(),
            "det": det.points,
            "prob": det.prob,
            "box": det.box,
            "ms": det.ms,
        }
        with self._lock, (self.root / "labels.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")


RECORDER: Recorder | None = None
