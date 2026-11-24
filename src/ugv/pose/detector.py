"""The trained keypoint detector on one car-camera frame: ONNX Runtime on the CPU, no torch.

Two stages, as trained (`train/pose_kp.py`, `tools/pose/make_dataset.py`):

1. the whole 1280x720 frame shrunk to the first model's input (320x180) finds the airframe;
2. a square crop around what stage 1 found (1.35x its keypoint extent, 64-1400 px, reflected at
   the image edges like the training crops) goes to the crop model (256x256) for precise points.

Both models output the 12 skeleton keypoints (`ugv.pose.mesh.keypoints` order) in input pixels
and a visibility logit each. A point is reported when its visibility probability exceeds
`threshold`; its pixel is mapped back to the full frame. Nothing here knows the true pose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
CROP_MARGIN = 1.35  # middle of the 1.1-1.6x the crop model was trained on
CROP_MIN, CROP_MAX = 64.0, 1400.0
REPO = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO / "runs/pose_train"
DEFAULT = ("kp-v2", "kp-crop-v2")  # (whole frame, crop) run names


@dataclass(frozen=True)
class Detection:
    points: list[list[float]]  # [[keypoint index, u, v], ...] in 1280x720 pixels
    prob: list[float]  # visibility probability per keypoint (final stage)
    box: tuple[float, float, float] | None  # (x0, y0, size) of the stage-2 crop
    ms: float  # inference time


class _Model:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort  # deferred: the Pi runtime imports this package without it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        shape = self.sess.get_inputs()[0].shape  # (1, 1, h, w)
        self.h, self.w = int(shape[2]), int(shape[3])

    def __call__(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (gray.astype(np.float32) / 255.0)[None, None]
        kp, vis, _ = self.sess.run(None, {"image": x})
        return kp[0].astype(np.float64), 1.0 / (1.0 + np.exp(-vis[0].astype(np.float64)))


class KeypointDetector:
    def __init__(self, whole: Path, crop: Path | None, threshold: float = 0.5) -> None:
        for p in (whole, crop):
            if p is not None and not (p / "model.onnx").exists():
                raise FileNotFoundError(f"{p / 'model.onnx'} is missing (train/pose_kp.py)")
        self.whole = _Model(whole / "model.onnx")
        self.crop = _Model(crop / "model.onnx") if crop is not None else None
        self.threshold = threshold
        self.names = (whole.name, crop.name if crop is not None else None)

    def __call__(self, frame: np.ndarray) -> Detection:
        """frame: (720, 1280) uint8, the car camera's mono picture."""
        if frame.shape != (H, W):
            raise ValueError(f"frame must be {H}x{W} mono, got {frame.shape}")
        t0 = cv2.getTickCount()
        small = cv2.resize(frame, (self.whole.w, self.whole.h), interpolation=cv2.INTER_AREA)
        kp, prob = self.whole(small)
        kp = kp * [W / self.whole.w, H / self.whole.h]
        box = None
        if self.crop is not None:
            seen = prob > self.threshold
            if seen.sum() >= 2:
                lo, hi = kp[seen].min(0), kp[seen].max(0)
                size = float(np.clip(max(hi - lo) * CROP_MARGIN, CROP_MIN, CROP_MAX))
                x0, y0 = (lo + hi) / 2 - size / 2
                box = (float(x0), float(y0), size)
                s = self.crop.w / size
                M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
                patch = cv2.warpAffine(
                    frame,
                    M,
                    (self.crop.w, self.crop.h),
                    flags=cv2.INTER_AREA,
                    borderMode=cv2.BORDER_REFLECT,
                )
                kp, prob = self.crop(patch)
                kp = kp / s + [x0, y0]
        keep = (
            (prob > self.threshold)
            & (kp[:, 0] >= 0)
            & (kp[:, 0] <= W)
            & (kp[:, 1] >= 0)
            & (kp[:, 1] <= H)
        )
        pts = [
            [int(i), round(float(kp[i, 0]), 1), round(float(kp[i, 1]), 1)]
            for i in np.flatnonzero(keep)
        ]
        ms = 1000.0 * (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return Detection(pts, np.round(prob, 3).tolist(), box, round(ms, 1))


def default_runs() -> tuple[Path, Path]:
    """The two trained runs the Lab uses: `UGV_DETECTOR=whole,crop` (run names under
    runs/pose_train) or `DEFAULT`. Not in Git: copy them with the repo."""
    names = os.environ.get("UGV_DETECTOR", "").split(",")
    whole, crop = names if len(names) == 2 and all(names) else DEFAULT
    return TRAIN_DIR / whole, TRAIN_DIR / crop


def load_default() -> KeypointDetector:
    return KeypointDetector(*default_runs())
