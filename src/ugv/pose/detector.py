"""The trained keypoint detector on one car-camera frame: ONNX Runtime on the CPU, no torch.

Two stages, as trained (`train/pose_kp.py`, `tools/pose/make_dataset.py`):

1. the whole 1280x720 frame shrunk to the first model's input (320x180) finds the airframe;
2. a square crop around what stage 1 found (1.35x its keypoint extent, 64-1400 px, reflected at
   the image edges like the training crops) goes to the crop model (256x256) for precise points.

Both models output the 15 keypoints (`ugv.pose.mesh.keypoints(..., near=True)` order: 12 skeleton
plus 3 nose-tyre points) in input pixels and a visibility logit each. A point is reported when its
visibility probability exceeds `threshold`; its pixel is mapped back to the full frame. Nothing
here knows the true pose.
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
DEFAULT = ("kp-v3c", "kp-crop-v3c")  # (whole frame, crop) run names


@dataclass(frozen=True)
class Detection:
    points: list[list[float]]  # [[keypoint index, u, v], ...] in 1280x720 pixels
    prob: list[float]  # visibility probability per keypoint (final stage)
    box: tuple[float, ...] | None  # (x0, y0, size) square crop OR (x0, y0, x1, y1) bbox
    ms: float  # inference time
    score: float | None = None  # object detection confidence (YOLO-pose)


class _Model:
    def __init__(self, path: Path, threads: int = 2) -> None:
        import onnxruntime as ort  # deferred: the Pi runtime imports this package without it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        if threads == 1:
            opts.inter_op_num_threads = 1
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
    def __init__(
        self, whole: Path, crop: Path | None, threshold: float = 0.5, threads: int = 2
    ) -> None:
        for p in (whole, crop):
            if p is not None and not (p / "model.onnx").exists():
                raise FileNotFoundError(f"{p / 'model.onnx'} is missing (train/pose_kp.py)")
        self.whole = _Model(whole / "model.onnx", threads=threads)
        self.crop = _Model(crop / "model.onnx", threads=threads) if crop is not None else None
        self.threshold = threshold
        self.names = (whole.name, crop.name if crop is not None else None)

    def _run_crop(
        self, frame: np.ndarray, box: tuple[float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, size = box
        assert self.crop is not None
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
        return kp / s + [x0, y0], prob

    @staticmethod
    def _box_from_kp(
        kp: np.ndarray, seen: np.ndarray, margin: float = CROP_MARGIN
    ) -> tuple[float, float, float] | None:
        if seen.sum() < 2:
            return None
        lo, hi = kp[seen].min(0), kp[seen].max(0)
        size = float(np.clip(max(hi - lo) * margin, CROP_MIN, CROP_MAX))
        x0, y0 = (lo + hi) / 2 - size / 2
        return (float(x0), float(y0), size)

    def __call__(
        self,
        frame: np.ndarray,
        prev_box: tuple[float, float, float] | None = None,
        nose_box_min: float = 400.0,
    ) -> Detection:
        """frame: (720, 1280) uint8, the car camera's mono picture.

        When `prev_box` is supplied and a crop model is loaded, stage 1 is skipped and the crop
        model runs directly on `prev_box`; if fewer than 3 keypoints survive, it falls back to
        stage 1 + stage 2 on the same frame.
        """
        if frame.shape != (H, W):
            raise ValueError(f"frame must be {H}x{W} mono, got {frame.shape}")
        t0 = cv2.getTickCount()
        box: tuple[float, float, float] | None = None
        kp = prob = None
        if prev_box is not None and self.crop is not None:
            kp, prob = self._run_crop(frame, prev_box)
            seen = (
                (prob > self.threshold)
                & (kp[:, 0] >= 0)
                & (kp[:, 0] <= W)
                & (kp[:, 1] >= 0)
                & (kp[:, 1] <= H)
            )
            if seen.sum() >= 3:
                box = self._box_from_kp(kp, seen, margin=1.45)
            else:
                kp = prob = None

        if kp is None or prob is None:
            small = cv2.resize(frame, (self.whole.w, self.whole.h), interpolation=cv2.INTER_AREA)
            kp, prob = self.whole(small)
            kp = kp * [W / self.whole.w, H / self.whole.h]
            if self.crop is not None:
                box = self._box_from_kp(kp, prob > self.threshold)
                if box is not None:
                    kp, prob = self._run_crop(frame, box)

        keep = (
            (prob > self.threshold)
            & (kp[:, 0] >= 0)
            & (kp[:, 0] <= W)
            & (kp[:, 1] >= 0)
            & (kp[:, 1] <= H)
        )
        # When the main crop is wide (> nose_box_min px), the 5 cm nose wheel is tiny inside the
        # 256x256 patch; a tight crop around its detected points (indices 9, 12-14) sharpens
        # bay-range localization from ~7 px to ~3 px.
        if self.crop is not None and box is not None and box[2] > nose_box_min and len(kp) >= 15:
            nose_ids = [i for i in (9, 12, 13, 14) if keep[i]]
            if len(nose_ids) >= 2:
                npts = kp[nose_ids]
                nlo, nhi = npts.min(0), npts.max(0)
                nsize = float(np.clip(max(nhi - nlo) * 2.5, 96.0, 320.0))
                nx0, ny0 = (nlo + nhi) / 2 - nsize / 2
                nkp, nprob = self._run_crop(frame, (float(nx0), float(ny0), nsize))
                for i in (9, 12, 13, 14):
                    if nprob[i] > self.threshold and 0 <= nkp[i, 0] <= W and 0 <= nkp[i, 1] <= H:
                        kp[i], prob[i], keep[i] = nkp[i], nprob[i], True
        pts = [
            [int(i), round(float(kp[i, 0]), 1), round(float(kp[i, 1]), 1)]
            for i in np.flatnonzero(keep)
        ]
        ms = 1000.0 * (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return Detection(pts, np.round(prob, 3).tolist(), box, round(ms, 1))


class YoloPoseDetector:
    """Single-stage YOLO-pose ONNX detector (CPU via onnxruntime, no torch/ultralytics).

    Predicts the full airframe bounding box (x0, y0, x1, y1) and all 15 keypoints in one pass
    using aspect-preserving letterbox resize.
    """

    def __init__(
        self,
        model_dir: Path,
        conf_threshold: float = 0.25,
        kpt_threshold: float = 0.45,
        threads: int = 2,
    ) -> None:
        import onnxruntime as ort

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
        prev_box: tuple[float, ...] | None = None,
        nose_box_min: float = 400.0,
    ) -> Detection:
        """frame: (720, 1280) uint8 mono. Extra kwargs accepted for interface parity."""
        del prev_box, nose_box_min
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
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]  # (C, N)

        scores = out[4]
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        n_kp = (out.shape[0] - 5) // 3
        ms = 1000.0 * (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        if best_score < self.conf_threshold:
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
        pts = [
            [int(i), round(float(ku[i]), 1), round(float(kv[i]), 1)]
            for i in range(n_kp)
            if kc[i] >= self.kpt_threshold and 0.0 <= ku[i] <= W and 0.0 <= kv[i] <= H
        ]
        return Detection(
            pts,
            np.round(kc.astype(np.float64), 3).tolist(),
            (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
            round(ms, 1),
            round(best_score, 3),
        )


def default_runs() -> tuple[Path, Path]:
    """The two trained runs the Lab uses: `UGV_DETECTOR=whole,crop` (run names under
    runs/pose_train) or `DEFAULT`. Not in Git: copy them with the repo."""
    names = os.environ.get("UGV_DETECTOR", "").split(",")
    whole, crop = names if len(names) == 2 and all(names) else DEFAULT
    return TRAIN_DIR / whole, TRAIN_DIR / crop


def load_default(threads: int = 2) -> KeypointDetector:
    return KeypointDetector(*default_runs(), threads=threads)
