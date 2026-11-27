"""Live read-only airframe keypoint detection and planar pose solver for the Pi dashboard.

Runs the two-stage ONNX `KeypointDetector` (`ugv.pose.detector`) on frames from `CameraStream`
when toggled on from the web UI. Strictly read-only (no actuation, no fabricated stand-ins):
if `onnxruntime` or the trained `model.onnx` files are missing, it reports the error visibly.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .pose.detector import TRAIN_DIR, KeypointDetector, default_runs

if TYPE_CHECKING:
    from .web import CameraStream, RobotController

KEYPOINT_NAMES = (
    "nose",
    "tail",
    "wingtip_left",
    "wingtip_right",
    "hub_left",
    "hub_right",
    "hub_rear",
    "vtail_tip_left",
    "vtail_tip_right",
    "gear_nose",
    "gear_main_left",
    "gear_main_right",
    "tyre_left",
    "tyre_right",
    "tyre_top",
)

PRIOR_TTL_S = 1.0


def discover_models(train_dir: Path = TRAIN_DIR) -> dict[str, tuple[Path, Path]]:
    """Return `{tag: (whole_dir, crop_dir)}` for trained two-stage pairs present on disk."""
    found: dict[str, tuple[Path, Path]] = {}
    if not train_dir.is_dir():
        return found
    for whole in sorted(train_dir.iterdir()):
        if not whole.is_dir() or not whole.name.startswith("kp-"):
            continue
        if whole.name.startswith("kp-crop-"):
            continue
        tag = whole.name[len("kp-") :]
        crop = train_dir / f"kp-crop-{tag}"
        if (whole / "model.onnx").is_file() and (crop / "model.onnx").is_file():
            found[tag] = (whole, crop)
    return found


def _default_tag(available: dict[str, tuple[Path, Path]]) -> str:
    whole, _ = default_runs()
    tag = whole.name[len("kp-") :] if whole.name.startswith("kp-") else whole.name
    if tag in available:
        return tag
    if available:
        return next(iter(available))
    return tag


def _decode_gray_1280x720(jpeg: bytes) -> np.ndarray | None:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    if gray.shape != (720, 1280):
        gray = cv2.resize(gray, (1280, 720), interpolation=cv2.INTER_AREA)
    return gray


class PlaneDetectorRunner:
    """Background keypoint detector for live manual inspection on the dashboard."""

    def __init__(
        self,
        camera: CameraStream,
        controller: RobotController,
        train_dir: Path = TRAIN_DIR,
    ) -> None:
        self.camera = camera
        self.controller = controller
        self.train_dir = train_dir
        self.threads = max(1, int(os.environ.get("UGV_PLANE_THREADS", "2")))
        self.enabled = False
        self.available = discover_models(train_dir)
        self.model_tag = _default_tag(self.available)
        self._detector: KeypointDetector | None = None
        self._loaded_tag: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._closed = False

        self.status = "disabled"
        self.points: list[list[float]] = []
        self.box: list[float] | None = None
        self.pose: dict[str, Any] | None = None
        self.last_ms: float | None = None
        self.last_at = 0.0
        self._times: deque[float] = deque(maxlen=30)
        self._prior: np.ndarray | None = None
        self._prior_at = 0.0
        self._last_seq = -1

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="plane-detector")

    async def close(self) -> None:
        self._closed = True
        self.enabled = False
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def configure(self, enabled: bool | None = None, model: str | None = None) -> None:
        self.available = discover_models(self.train_dir)
        if model is not None:
            tag = str(model).strip()
            if tag.startswith("kp-"):
                tag = tag[len("kp-") :]
            if tag not in self.available:
                raise ValueError(f"unknown plane model: {model}")
            if tag != self.model_tag:
                self.model_tag = tag
                self._detector = None
                self._loaded_tag = None
                self._prior = None
        if enabled is not None:
            self.enabled = bool(enabled)
            if not self.enabled:
                self._clear_detection("disabled")
            else:
                self.status = "starting"
                self._last_seq = -1
        self._wake.set()
        await self.controller.broadcast()

    def _clear_detection(self, status: str) -> None:
        self.status = status
        self.points = []
        self.box = None
        self.pose = None
        self.last_ms = None
        self._times.clear()
        self._prior = None

    def _ensure_detector(self) -> KeypointDetector:
        if self._detector is not None and self._loaded_tag == self.model_tag:
            return self._detector
        self.available = discover_models(self.train_dir)
        if self.model_tag in self.available:
            whole, crop = self.available[self.model_tag]
        else:
            whole, crop = default_runs()
        det = KeypointDetector(whole, crop, threads=self.threads)
        self._detector = det
        self._loaded_tag = self.model_tag
        return det

    def _infer_frame(self, jpeg: bytes) -> dict[str, Any] | None:
        gray = _decode_gray_1280x720(jpeg)
        if gray is None:
            return None
        det = self._ensure_detector()
        res = det(gray)
        pts = [[idx, u, v, res.prob[idx]] for idx, u, v in res.points]
        box = [round(float(v), 1) for v in res.box] if res.box is not None else None

        pose_out: dict[str, Any] | None = None
        if len(res.points) >= 2:
            try:
                from .park.perception import DetectorModel, model_points, solve

                det_cfg = DetectorModel()
                all_model = model_points(det_cfg)
                idxs = [int(p[0]) for p in res.points if int(p[0]) < len(all_model)]
                if len(idxs) >= 2:
                    m = all_model[idxs]
                    o = np.array([[p[1], p[2]] for p in res.points if int(p[0]) < len(all_model)])
                    now = time.monotonic()
                    prior = self._prior if (now - self._prior_at) <= PRIOR_TTL_S else None
                    sol = solve(m, o, det_cfg, prior)
                    if sol is not None:
                        self._prior = sol[:4].copy()
                        self._prior_at = now
                        pose_out = {
                            "x_m": round(float(sol[0]), 3),
                            "y_m": round(float(sol[1]), 3),
                            "yaw_deg": round(math.degrees(float(sol[2])), 1),
                            "pitch_deg": round(math.degrees(float(sol[3])), 1),
                            "used": int(round(float(sol[4]))),
                        }
            except Exception:
                pose_out = None

        return {"points": pts, "box": box, "pose": pose_out, "ms": float(res.ms)}

    async def _loop(self) -> None:
        while not self._closed:
            if not self.enabled:
                self._wake.clear()
                await self._wake.wait()
                continue
            if not self.camera.enabled:
                self.enabled = False
                self._clear_detection("camera-paused")
                await self.controller.broadcast()
                continue

            frame, seq, _ = self.camera.latest_sample()
            if frame is None or seq == self._last_seq:
                await asyncio.sleep(0.03)
                continue
            self._last_seq = seq

            try:
                if self._detector is None or self._loaded_tag != self.model_tag:
                    self.status = f"loading {self.model_tag}..."
                    await self.controller.broadcast()
                out = await asyncio.to_thread(self._infer_frame, frame)
            except Exception as exc:
                self.enabled = False
                self._clear_detection(f"error: {exc}")
                await self.controller.broadcast()
                continue

            if not self.enabled:
                continue
            if out is None:
                await asyncio.sleep(0.03)
                continue

            now = time.monotonic()
            self.points = out["points"]
            self.box = out["box"]
            self.pose = out["pose"]
            self.last_ms = round(out["ms"], 1)
            self.last_at = now
            self._times.append(now)
            self.controller.perf.record("plane_ms", out["ms"])

            n = len(self.points)
            if self.pose is not None:
                p = self.pose
                self.status = (
                    f"{n} pts · x={p['x_m']:.2f}m y={p['y_m']:+.2f}m yaw={p['yaw_deg']:+.0f}°"
                )
            elif n > 0:
                self.status = f"{n} pts · {self.last_ms:.0f} ms"
            else:
                self.status = f"no airframe · {self.last_ms:.0f} ms"

            await self.controller.broadcast()
            await asyncio.sleep(0.01)

    def metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        recent = [t for t in self._times if now - t < 3.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
        age_ms = round((now - self.last_at) * 1000) if (self.enabled and self.last_at) else None
        return {
            "plane_enabled": self.enabled,
            "plane_model": self.model_tag,
            "plane_available_models": list(self.available.keys()),
            "plane_status": self.status,
            "plane_points": self.points if self.enabled else [],
            "plane_box": self.box if self.enabled else None,
            "plane_pose": self.pose if self.enabled else None,
            "plane_ms": self.last_ms if self.enabled else None,
            "plane_fps": round(fps, 1) if self.enabled else 0.0,
            "plane_age_ms": age_ms,
        }
