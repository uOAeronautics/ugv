"""Single-stage YOLO-pose keypoint detector on one car-camera frame: ONNX Runtime on CPU, no torch.

Predicts the full airframe bounding box (x0, y0, x1, y1) and all 15 keypoints
(`ugv.pose.mesh.keypoints(..., near=True)` order: 12 skeleton plus 3 nose-tyre points) in one pass
using aspect-preserving letterbox resize. Nothing here knows the true pose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
REPO = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO / "runs/pose_train"
DEFAULT = "yolo11n-pose-v3-416-int8"


@dataclass(frozen=True)
class Detection:
    points: list[list[float]]  # [[keypoint index, u, v], ...] in 1280x720 pixels
    prob: list[float]  # keypoint confidence per keypoint
    box: tuple[float, float, float, float] | None  # (x0, y0, x1, y1) bbox in 1280x720 pixels
    ms: float  # inference time
    score: float | None = None  # object detection confidence


class YoloPoseDetector:
    """Single-stage YOLO-pose ONNX detector (CPU via onnxruntime, no torch/ultralytics)."""

    def __init__(
        self,
        model_dir: Path,
        conf_threshold: float = 0.50,
        kpt_threshold: float = 0.50,
        threads: int = 2,
    ) -> None:
        import onnxruntime as ort  # deferred: the Pi runtime imports this package without it

        path = model_dir / "model.onnx" if model_dir.is_dir() else model_dir
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing (train/yolo_pose.py)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        if threads == 1:
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # (1, 3, h, w)
        self.h, self.w = int(shape[2]), int(shape[3])
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.names = (model_dir.name if model_dir.is_dir() else model_dir.stem, None)

    def __call__(
        self,
        frame: np.ndarray,
        *,
        close_range: bool = False,
    ) -> Detection:
        """frame: (720, 1280) uint8 mono."""
        if frame.shape != (H, W):
            raise ValueError(f"frame must be {H}x{W} mono, got {frame.shape}")
        t0 = cv2.getTickCount()
        scale = min(self.w / W, self.h / H)
        rw, rh = round(W * scale), round(H * scale)
        pad_x = (self.w - rw) // 2
        pad_y = (self.h - rh) // 2
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.h, self.w), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized
        x = np.broadcast_to((canvas.astype(np.float32) / 255.0)[None, None], (1, 3, self.h, self.w))
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]
        ms = 1000.0 * (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        conf_thr = 0.30 if close_range else self.conf_threshold

        # Support both YOLO11/v8 `(50, N)` [cx,cy,w,h,score,kp...] and YOLO26 end-to-end
        # `(300, 51)` [x1,y1,x2,y2,score,cls,kp...] ONNX layouts.
        if out.ndim == 2 and out.shape[0] > out.shape[1] and out.shape[1] >= 6:
            scores = out[:, 4]
            best = int(np.argmax(scores))
            best_score = float(scores[best])
            n_kp = (out.shape[1] - 6) // 3
            if best_score < conf_thr:
                return Detection([], [0.0] * n_kp, None, round(ms, 1), round(best_score, 3))
            bx0, by0, bx1, by1 = out[best, 0:4]
            x0 = float(np.clip((bx0 - pad_x) / scale, 0.0, W))
            y0 = float(np.clip((by0 - pad_y) / scale, 0.0, H))
            x1 = float(np.clip((bx1 - pad_x) / scale, 0.0, W))
            y1 = float(np.clip((by1 - pad_y) / scale, 0.0, H))
            kpts = out[best, 6:].reshape(n_kp, 3)
        else:
            scores = out[4]
            best = int(np.argmax(scores))
            best_score = float(scores[best])
            n_kp = (out.shape[0] - 5) // 3
            if best_score < conf_thr:
                return Detection([], [0.0] * n_kp, None, round(ms, 1), round(best_score, 3))
            cx, cy, bw, bh = out[0:4, best]
            x0 = float(np.clip((cx - bw / 2 - pad_x) / scale, 0.0, W))
            y0 = float(np.clip((cy - bh / 2 - pad_y) / scale, 0.0, H))
            x1 = float(np.clip((cx + bw / 2 - pad_x) / scale, 0.0, W))
            y1 = float(np.clip((cy + bh / 2 - pad_y) / scale, 0.0, H))
            kpts = out[5:, best].reshape(n_kp, 3)
        ku = (kpts[:, 0] - pad_x) / scale
        kv = (kpts[:, 1] - pad_y) / scale
        kc = kpts[:, 2]
        kpt_thr = 0.40 if close_range else self.kpt_threshold
        pts = [
            [int(i), round(float(ku[i]), 1), round(float(kv[i]), 1)]
            for i in range(n_kp)
            if kc[i] >= kpt_thr and 0.0 <= ku[i] <= W and 0.0 <= kv[i] <= H
        ]
        # Reject weak clutter false positives (e.g. rolling chair/table casters with 1-3 stray pts)
        # while keeping valid 3-4 pt side/bay views (e.g. hubRear + gearL + gearR or nose tyre).
        if not close_range and (len(pts) < 3 or (len(pts) < 4 and best_score < 0.58)):
            return Detection([], [0.0] * n_kp, None, round(ms, 1), round(best_score, 3))
        if close_range and len(pts) < 3:
            return Detection([], [0.0] * n_kp, None, round(ms, 1), round(best_score, 3))
        return Detection(
            pts,
            np.round(kc.astype(np.float64), 3).tolist(),
            (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
            round(ms, 1),
            round(best_score, 3),
        )


def default_runs() -> tuple[Path, ...]:
    """The trained YOLO-pose run: `UGV_DETECTOR=<run>` (under runs/pose_train) or `DEFAULT`."""
    name = os.environ.get("UGV_DETECTOR", "").strip() or DEFAULT
    return (TRAIN_DIR / name,)


def load_default(threads: int = 2) -> YoloPoseDetector:
    return YoloPoseDetector(default_runs()[0], threads=threads)
