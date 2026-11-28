"""Live read-only airframe keypoint detection and planar pose solver for the Pi dashboard.

Runs the two-stage ONNX `KeypointDetector` (`ugv.pose.detector`) on frames from `CameraStream`
when toggled on from the web UI. Strictly read-only (no actuation, no fabricated stand-ins):
if `onnxruntime` or the trained `model.onnx` files are missing, it reports the error visibly.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .pose.detector import TRAIN_DIR, KeypointDetector, YoloPoseDetector, default_runs

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
TRACK_REANCHOR_EVERY = 12
NOSE_BOX_MIN_PX = 750.0


def discover_models(train_dir: Path = TRAIN_DIR) -> dict[str, tuple[Path, Path | None]]:
    """Return `{tag: (primary_dir, crop_dir_or_None)}` for trained models present on disk.

    Discovers single-stage YOLO-pose models (`yolo*/model.onnx`, listed first) and two-stage
    keypoint models (`kp-<tag>` + `kp-crop-<tag>`).
    """
    found: dict[str, tuple[Path, Path | None]] = {}
    if not train_dir.is_dir():
        return found
    for d in sorted(train_dir.iterdir()):
        if d.is_dir() and d.name.startswith("yolo") and (d / "model.onnx").is_file():
            found[d.name] = (d, None)
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


def _default_tag(available: dict[str, tuple[Path, Path | None]]) -> str:
    for tag in available:
        if tag.startswith("yolo") and "int8" in tag:
            return tag
    for tag in available:
        if tag.startswith("yolo"):
            return tag
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
        self._track_box: tuple[float, float, float] | None = None
        self._track_at = 0.0
        self._track_steps = 0
        self._last_seq = -1
        self._external_active = False

    def set_external(self, active: bool) -> None:
        """Hand detector execution to an external loop (e.g. Auto Park) without double-running."""
        self._external_active = bool(active)
        if not active:
            self._last_seq = -1
            self._wake.set()

    def detect_once(
        self, jpeg: bytes, prior: np.ndarray | None = None, *, close_range: bool = False
    ) -> tuple[dict[str, Any] | None, np.ndarray | None]:
        """Run one frame synchronously (for Auto Park), updating the dashboard overlay."""
        gray = _decode_gray_1280x720(jpeg)
        if gray is None:
            return None, None
        det = self._ensure_detector()
        if isinstance(det, YoloPoseDetector):
            res = det(gray, close_range=close_range)
        else:
            res = det(gray, nose_box_min=NOSE_BOX_MIN_PX)
        pts = [[idx, u, v, res.prob[idx]] for idx, u, v in res.points]
        box = [round(float(v), 1) for v in res.box] if res.box is not None else None
        pose_out, meas = self._solve_pose(res.points, prior=prior)

        now = time.monotonic()
        self.points = pts
        self.box = box
        self.pose = pose_out
        self.last_ms = round(float(res.ms), 1)
        self.last_at = now
        self._times.append(now)
        self.controller.perf.record("plane_ms", float(res.ms))
        n = len(self.points)
        if self.pose is not None:
            p = self.pose
            self.status = f"{n} pts · x={p['x_m']:.2f}m y={p['y_m']:+.2f}m yaw={p['yaw_deg']:+.0f}°"
        elif n > 0:
            self.status = f"{n} pts · {self.last_ms:.0f} ms"
        else:
            self.status = f"no airframe · {self.last_ms:.0f} ms"
        return {"points": pts, "box": box, "pose": pose_out, "ms": float(res.ms)}, meas

    def _solve_pose(
        self, res_points: list[list[float]], prior: np.ndarray | None = None
    ) -> tuple[dict[str, Any] | None, np.ndarray | None]:
        if len(res_points) < 2:
            return None, None
        try:
            from .park.perception import DetectorModel, model_points, solve

            det_cfg = DetectorModel()
            all_model = model_points(det_cfg)
            idxs = [int(p[0]) for p in res_points if int(p[0]) < len(all_model)]
            if len(idxs) < 2:
                return None, None
            m = all_model[idxs]
            o = np.array([[p[1], p[2]] for p in res_points if int(p[0]) < len(all_model)])
            now = time.monotonic()
            use_prior = (
                prior
                if prior is not None
                else (self._prior if (now - self._prior_at) <= PRIOR_TTL_S else None)
            )
            sol = solve(m, o, det_cfg, use_prior)
            if sol is None:
                return None, None
            self._prior = sol[:4].copy()
            self._prior_at = now
            pose_out = {
                "x_m": round(float(sol[0]), 3),
                "y_m": round(float(sol[1]), 3),
                "yaw_deg": round(math.degrees(float(sol[2])), 1),
                "pitch_deg": round(math.degrees(float(sol[3])), 1),
                "used": int(round(float(sol[4]))),
            }
            return pose_out, sol
        except Exception:
            return None, None

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
                self._track_box = None
                self._track_steps = 0
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
        self._track_box = None
        self._track_steps = 0

    def _ensure_detector(self) -> KeypointDetector | YoloPoseDetector:
        if self._detector is not None and self._loaded_tag == self.model_tag:
            return self._detector
        self.available = discover_models(self.train_dir)
        if self.model_tag in self.available:
            whole, crop = self.available[self.model_tag]
        else:
            whole, crop = default_runs()
        det: KeypointDetector | YoloPoseDetector = (
            YoloPoseDetector(whole, threads=self.threads)
            if crop is None
            else KeypointDetector(whole, crop, threads=self.threads)
        )
        self._detector = det
        self._loaded_tag = self.model_tag
        return det

    def _infer_frame(self, jpeg: bytes) -> dict[str, Any] | None:
        gray = _decode_gray_1280x720(jpeg)
        if gray is None:
            return None
        det = self._ensure_detector()
        now = time.monotonic()
        use_prev = (
            self._track_box is not None
            and (now - self._track_at) <= PRIOR_TTL_S
            and self._track_steps < TRACK_REANCHOR_EVERY
        )
        res = det(
            gray,
            prev_box=self._track_box if use_prev else None,
            nose_box_min=NOSE_BOX_MIN_PX,
        )
        if res.box is not None and len(res.box) == 3 and len(res.points) >= 3:
            self._track_box = (res.box[0], res.box[1], res.box[2])
            self._track_at = now
            self._track_steps = self._track_steps + 1 if use_prev else 1
        else:
            self._track_box = None
            self._track_steps = 0
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
            if not self.enabled or self._external_active:
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
        active = self.enabled or self._external_active
        recent = [t for t in self._times if now - t < 3.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
        age_ms = round((now - self.last_at) * 1000) if (active and self.last_at) else None
        return {
            "plane_enabled": active,
            "plane_model": self.model_tag,
            "plane_available_models": list(self.available.keys()),
            "plane_status": self.status,
            "plane_points": self.points if active else [],
            "plane_box": self.box if active else None,
            "plane_pose": self.pose if active else None,
            "plane_ms": self.last_ms if active else None,
            "plane_fps": round(fps, 1) if active else 0.0,
            "plane_age_ms": age_ms,
        }


class PlaneAutoPark:
    """Closed-loop Auto Park using `ugv.park` (Estimator + ParkController) on YOLO-pose fixes.

    Sign mapping between `ugv.park` body frame `(vx fwd, vy left, w CCW)` and `web.py`
    `_apply_vector(forward, turn, limit=1800, strafe=...)`:
      - `forward = duty[0] / DUTY_CAP`   (+x forward)
      - `strafe  = -duty[1] / DUTY_CAP`  (+strafe is RIGHT in `_apply_vector`, +y is LEFT in park)
      - `turn    = duty[2] / DUTY_CAP`   (+turn is LEFT / CCW in both)
    """

    def __init__(self, start_t: float | None = None) -> None:
        from .park.controller import ParkController, ParkParams
        from .park.estimator import Estimator
        from .park.geometry import bay

        self.t0 = time.monotonic() if start_t is None else float(start_t)
        self.last_t = self.t0
        self.est = Estimator(calib_min_points=4)
        self.ctrl = ParkController(
            bay=bay(),
            params=ParkParams(
                search_w=0.42,
                orbit_r=1.15,
                orbit_v=0.14,
                k_radial=0.8,
                k_face=1.4,
                approach_v=0.12,
                k_lat=1.2,
                k_head=1.5,
                align_lat_m=0.05,
                align_yaw_deg=10.0,
                abort_lat_m=0.09,
                abort_yaw_deg=16.0,
                settle_s=0.15,
                measure_s=0.25,
                step_v=0.12,
                step_w=0.40,
                lat_tol_m=0.025,
                yaw_tol_deg=4.0,
                stop_tol_m=0.025,
                k_hold=1.5,
                shield_backoff=True,
                backoff_extra_m=0.40,
                backoff_straight_in_bay=True,
                approach_align_first=True,
                misalign_confirm=2,
                camera_silent_s=1.0,
            ),
        )
        self.last_duty = np.zeros(3, dtype=float)
        self._yaw_int = 0.0
        self._yaw_sign = 0.0

    def prior_for_solver(self) -> np.ndarray | None:
        from .park.geometry import inverse

        q = self.est.car_in_airframe()
        if q is None:
            return None
        rel = inverse(q)
        return np.array([rel[0], rel[1], rel[2], self.est.pitch], dtype=float)

    def step(
        self,
        now: float,
        *,
        imu_age_s: float | None,
        frame_age_s: float | None,
        gyro_ccw_rad_s: float,
        meas: np.ndarray | None = None,
        cap_t: float | None = None,
    ) -> dict[str, Any]:
        from .park.plant import DUTY_CAP

        if imu_age_s is None or imu_age_s > 0.25:
            self.last_duty = np.zeros(3)
            return {
                "stage": "error",
                "status": "imu-stale",
                "reason": "imu-stale",
                "stop": True,
                "done": True,
                "forward": 0.0,
                "strafe": 0.0,
                "turn": 0.0,
                "pose": None,
            }
        if frame_age_s is None or frame_age_s > 0.45:
            self.last_duty = np.zeros(3)
            return {
                "stage": "error",
                "status": "frame-stale",
                "reason": "frame-stale",
                "stop": True,
                "done": True,
                "forward": 0.0,
                "strafe": 0.0,
                "turn": 0.0,
                "pose": None,
            }

        t = max(0.0, float(now) - self.t0)
        dt = float(np.clip(float(now) - self.last_t, 1e-3, 0.25))
        self.last_t = float(now)

        self.est.predict(t, self.last_duty, float(gyro_ccw_rad_s), dt)
        jumped = False
        if cap_t is not None:
            t_cap = max(0.0, float(cap_t) - self.t0)
            self.est.last_frame_t = t_cap
            if meas is not None:
                upd = self.est.update(t_cap, meas, t)
                jumped = upd == "reinit"

        cmd_vel = self.ctrl.step(t, self.est, jumped)
        cmd_vel = np.array(
            [
                float(np.clip(cmd_vel[0], -0.22, 0.22)),
                float(np.clip(cmd_vel[1], -0.18, 0.18)),
                float(np.clip(cmd_vel[2], -0.65, 0.65)),
            ],
            dtype=float,
        )
        duty = self.est.actuator.to_duty(cmd_vel)
        # Closed-loop IMU gyro PI yaw-rate controller with deadband & smooth deadzone taper:
        # eliminates bang-bang yaw chatter around 0, ramps through static friction when stalled,
        # and damps overshoot at 50 Hz.
        w_cmd = float(cmd_vel[2])
        if abs(w_cmd) < 0.08:
            self._yaw_int = 0.0
            self._yaw_sign = 0.0
            duty[2] = 0.0
        else:
            sgn = 1.0 if w_cmd > 0 else -1.0
            if self._yaw_sign != 0.0 and sgn != self._yaw_sign:
                self._yaw_int = 0.0
            self._yaw_sign = sgn
            ax_w = self.est.actuator.axes[2]
            dz_taper = min(1.0, abs(w_cmd) / 0.25)
            d_ff = sgn * (ax_w.deadzone * dz_taper + abs(w_cmd) / ax_w.gain)
            err_w = w_cmd - float(gyro_ccw_rad_s)
            self._yaw_int = float(np.clip(self._yaw_int + 0.35 * err_w * dt, -0.25, 0.25))
            d_w = d_ff + 0.08 * err_w + self._yaw_int
            duty[2] = float(np.clip(d_w, -DUTY_CAP, DUTY_CAP))
        state = self.ctrl.state
        if state == "PARKED":
            stage, done = "parked", True
        elif state == "ABORT":
            stage, done = "aborted", True
        else:
            stage, done = state.lower(), False

        stop = bool(done or np.allclose(duty, 0.0, atol=1e-4))
        self.last_duty = np.zeros(3) if stop else duty.copy()
        q = self.est.car_in_airframe()
        pose = [round(float(v), 4) for v in q] if q is not None else None
        status = self.ctrl.reason or stage
        return {
            "stage": stage,
            "status": status,
            "reason": status,
            "stop": stop,
            "done": done,
            "forward": 0.0 if stop else float(np.clip(duty[0] / DUTY_CAP, -1.0, 1.0)),
            "strafe": 0.0 if stop else float(np.clip(-duty[1] / DUTY_CAP, -1.0, 1.0)),
            "turn": 0.0 if stop else float(np.clip(duty[2] / DUTY_CAP, -1.0, 1.0)),
            "pose": pose,
            "cmd_vel": [round(float(v), 4) for v in cmd_vel],
            "duty": [round(float(v), 4) for v in self.last_duty],
            "yaw_int": round(float(self._yaw_int), 4),
        }


AUTOPARK_RUNS_DIR = TRAIN_DIR.parent / "autopark"


class AutoParkRecorder:
    """Records 50 Hz Auto Park telemetry + sampled real JPEG frames for post-mortem & training.

    Writes `telemetry.jsonl`, `summary.json`, and `frames/<seq>.jpg` under
    `runs/autopark/<YYYYMMDD-HHMMSS>/`. Auto-prunes oldest runs beyond `max_runs` and caps
    frames per run so the Pi's SD card never fills up.
    """

    def __init__(
        self,
        root: Path = AUTOPARK_RUNS_DIR,
        *,
        model: str = "",
        max_runs: int = 12,
        max_frames: int = 180,
        frame_interval_s: float = 0.35,
    ) -> None:
        self.root = root
        self.max_frames = max_frames
        self.frame_interval_s = frame_interval_s
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = root / self.run_id
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._prune_old_runs(max_runs)
        self._fp = (self.run_dir / "telemetry.jsonl").open("w", buffering=1)
        self._t0 = time.monotonic()
        self._last_frame_saved = 0.0
        self.frames_saved = 0
        self.ticks = 0
        self.model = model
        self._closed = False

    def _prune_old_runs(self, max_runs: int) -> None:
        if not self.root.is_dir():
            return
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        for old in dirs[:-max_runs]:
            if old != self.run_dir:
                shutil.rmtree(old, ignore_errors=True)

    def save_frame(self, seq: int, jpeg: bytes, det: dict[str, Any] | None) -> str | None:
        if self._closed or self.frames_saved >= self.max_frames:
            return None
        now = time.monotonic()
        if self._last_frame_saved and (now - self._last_frame_saved) < self.frame_interval_s:
            return None
        self._last_frame_saved = now
        self.frames_saved += 1
        rel = f"frames/{seq:06d}.jpg"
        (self.run_dir / rel).write_bytes(jpeg)
        if det is not None:
            meta = {
                "t": round(now - self._t0, 3),
                "seq": seq,
                "frame": rel,
                "points": det.get("points"),
                "box": det.get("box"),
                "pose": det.get("pose"),
                "ms": det.get("ms"),
            }
            self._fp.write(json.dumps({"kind": "vision", **meta}) + "\n")
        return rel

    def record_tick(self, row: dict[str, Any]) -> None:
        if self._closed:
            return
        self.ticks += 1
        self._fp.write(
            json.dumps({"kind": "tick", "t": round(time.monotonic() - self._t0, 3), **row}) + "\n"
        )

    def close(self, final_stage: str, final_status: str) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fp.close()
        finally:
            summary = {
                "run_id": self.run_id,
                "model": self.model,
                "duration_s": round(time.monotonic() - self._t0, 2),
                "ticks": self.ticks,
                "frames_saved": self.frames_saved,
                "final_stage": final_stage,
                "final_status": final_status,
            }
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
