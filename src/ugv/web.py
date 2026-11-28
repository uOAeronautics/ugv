from __future__ import annotations

import asyncio
import json
import math
import os
import select
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from smbus2 import SMBus

from . import SERVO_CHANNELS, WHEEL_CHANNELS, _configure_pca9685, _set_pulse, _set_pwm, _stop_wheels
from .aruco import ArucoFollower
from .autopark import TERMINAL_STAGES
from .dynamics_telemetry import DynamicsTelemetry
from .heading import RelativeHeading
from .imu import ImuStream
from .motion import GyroMotion, MotionFault, ramp_wheels
from .parking import DockingFeedback, ParkingEstimator, Pose2, gear_obstacles
from .perf import PerfMonitor
from .plane_vision import AutoParkRecorder, PlaneAutoPark, PlaneDetectorRunner

STATIC = Path(__file__).parent / "static"
WATCHDOG_SECONDS = 0.60


@dataclass
class RobotState:
    connected: bool = False
    owner: str | None = None
    armed: bool = False
    forward: float = 0.0
    strafe: float = 0.0
    turn: float = 0.0
    turn_target_dps: float = 0.0
    motion_mode: str = "idle"
    speed_limit: int = 1800
    pan_us: int = 1500
    tilt_us: int = 1500
    aruco_enabled: bool = False
    aruco_follow: bool = False
    aruco_visible: bool = False
    aruco_id: int | None = None
    aruco_distance_m: float | None = None
    aruco_error_x: float | None = None
    aruco_corners: list[list[float]] = field(default_factory=list)
    aruco_markers: list[dict] = field(default_factory=list)
    aruco_status: str = "aruco-disabled"
    imu_connected: bool = False
    imu_status: str = "starting"
    imu_age_ms: int | None = None
    imu_rate_hz: float = 0.0
    imu_accel_g: dict[str, float] = field(default_factory=lambda: {axis: 0.0 for axis in "xyz"})
    imu_gyro_dps: dict[str, float] = field(default_factory=lambda: {axis: 0.0 for axis in "xyz"})
    heading_deg: float = 0.0
    heading_rate_dps: float = 0.0
    heading_calibrated: bool = False
    heading_stationary: bool = False
    heading_status: str = "hold still · zeroing"
    heading_progress: float = 0.0
    wheels: dict[str, int] = field(default_factory=lambda: {n: 0 for n in WHEEL_CHANNELS})
    stopped: bool = True
    reason: str = "startup"
    revision: int = 0
    controller: str = "manual"
    controller_stage: str = "idle"
    swept_clear_margin_m: float | None = None
    inference_ms: float | None = None
    tick_ms: float | None = None
    aruco_age_ms: int | None = None
    imu_bias_dps: dict[str, float] = field(default_factory=lambda: {axis: 0.0 for axis in "xyz"})
    follow_debug: dict = field(
        default_factory=lambda: {
            "mode": "IDLE",
            "err_lead": 0.0,
            "z_lead": 0.0,
            "vx": 0.0,
            "vz": 0.0,
            "sigma_m": 0.05,
            "coast_age_ms": 0,
            "cmd_vec": [0.0, 0.0, 0.0],
        }
    )
    rls_scale: float = 1.0
    joystick_deadzone: float = 0.08
    sysid_ready: bool = False
    auto_park: bool = False
    auto_park_stage: str = "idle"
    auto_park_status: str = ""
    auto_park_pose: list[float] | None = None
    perf: dict[str, Any] = field(default_factory=dict)


class CameraStream:
    def __init__(self):
        self.frame: bytes | None = None
        self.seq = 0
        self.received_at = 0.0
        self.frame_times = deque(maxlen=90)
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.process = None
        self.thread = None
        self.enabled = True

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-stream")
        self.thread.start()

    def _run(self):
        cmd = [
            "rpicam-vid",
            "--timeout",
            "0",
            "--nopreview",
            "--codec",
            "mjpeg",
            "--width",
            "1280",
            "--height",
            "720",
            "--framerate",
            "30",
            "--quality",
            os.environ.get("UGV_CAMERA_JPEG_QUALITY", "95"),
            "--shutter",
            os.environ.get("UGV_CAMERA_SHUTTER_US", "8000"),
            "--gain",
            os.environ.get("UGV_CAMERA_GAIN", "1.5"),
            "--denoise",
            "cdn_off",
            "--flush",
            "--output",
            "-",
        ]
        while not self.stop.is_set():
            if not self.enabled:
                self.stop.wait(0.25)
                continue
            try:
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                buf = bytearray()
                last_frame_at = time.monotonic()
                received_frame = False
                while not self.stop.is_set() and self.enabled:
                    # A live rpicam process can stall indefinitely without EOF.
                    # Bound pipe waits and recover capture independently of the
                    # controller. Stale-frame checks still clear old overlays.
                    frame_timeout = 2.0 if received_frame else 5.0
                    if time.monotonic() - last_frame_at > frame_timeout:
                        break
                    ready, _, _ = select.select([self.process.stdout], [], [], 0.25)
                    if not ready:
                        continue
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while True:
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                        if a < 0 or b < 0:
                            if len(buf) > 2_000_000:
                                del buf[:-2]
                            break
                        frame = bytes(buf[a : b + 2])
                        del buf[: b + 2]
                        last_frame_at = time.monotonic()
                        received_frame = True
                        with self.cv:
                            if not self.enabled:
                                break
                            self.received_at = time.monotonic()
                            self.frame_times.append(self.received_at)
                            self.frame = frame
                            self.seq += 1
                            self.cv.notify_all()
            except Exception:
                pass
            finally:
                if self.process:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                    self.process = None
            self.stop.wait(1)

    def set_enabled(self, enabled: bool):
        with self.cv:
            self.enabled = enabled
            self.frame = None
            self.received_at = 0.0
            self.frame_times.clear()
            self.cv.notify_all()

    def frames(self) -> Iterator[bytes]:
        import cv2
        import numpy as np

        seen = -1
        last_sent = 0.0
        min_interval = 1.0 / max(1.0, float(os.environ.get("UGV_PREVIEW_FPS", "20")))
        preview_w = int(os.environ.get("UGV_PREVIEW_WIDTH", "640"))
        preview_q = int(os.environ.get("UGV_PREVIEW_QUALITY", "65"))
        while not self.stop.is_set():
            with self.cv:
                self.cv.wait_for(
                    lambda seen=seen: self.seq != seen or self.stop.is_set(), timeout=2
                )
                if self.stop.is_set():
                    return
                if self.seq == seen:
                    continue
                frame, seen = self.frame, self.seq
            if not frame:
                continue
            now = time.monotonic()
            if now - last_sent < min_interval:
                continue
            last_sent = now
            out = frame
            if preview_w > 0 and preview_w < 1280:
                img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape[1] > preview_w:
                    ph = max(2, round(img.shape[0] * preview_w / img.shape[1]))
                    small = cv2.resize(img, (preview_w, ph), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(
                        ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), preview_q]
                    )
                    if ok:
                        out = buf.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n"
                + out
                + b"\r\n"
            )

    def latest(self) -> tuple[bytes | None, int]:
        with self.cv:
            return self.frame, self.seq

    def latest_sample(self):
        with self.cv:
            return self.frame, self.seq, self.received_at

    def metrics(self):
        with self.cv:
            now = time.monotonic()
            recent = [t for t in self.frame_times if now - t < 3]
            age = now - self.received_at if self.received_at else None
            fps = (
                (len(recent) - 1) / (recent[-1] - recent[0])
                if len(recent) > 1 and age < 0.5
                else 0.0
            )
            return {
                "camera_enabled": self.enabled,
                "camera_fps": round(fps, 1),
                "camera_frame_age_ms": round(age * 1000) if age is not None else None,
                "camera_sequence": self.seq,
            }

    def close(self):
        self.stop.set()
        with self.cv:
            self.cv.notify_all()
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=3)


class RobotController:
    def __init__(self):
        self.state = RobotState()
        self.bus = None
        self.lock = asyncio.Lock()
        self.hw_lock = threading.RLock()
        self.clients = {}
        self.last_drive = 0.0
        self.guard_stop = threading.Event()
        self.guard_thread = None
        self.motion = GyroMotion()
        self.motion_task: asyncio.Task | None = None
        self.last_operator_command = 0.0
        self.last_pwm_at = time.monotonic()
        self.drive_deadline: float | None = None
        self.calibration_up = (0.0, 0.0, 1.0)
        self.imu_sample_at: float | None = None
        self.heading = RelativeHeading()
        self.dynamics = DynamicsTelemetry()
        self.parking = ParkingEstimator()
        self.parking_active = False
        self.parking_owner = None
        self.parking_task = None
        self.parking_status = "observing"
        self.parking_epoch = 0
        self.controller_stage = "idle"
        self.inference_ms: float | None = None
        self.tick_ms: float | None = None
        self.rls_lateral_scale: float = 1.0
        self.auto_park_task: asyncio.Task | None = None
        self.auto_park_owner: str | None = None
        self.auto_park_epoch: int = 0
        self.perf = PerfMonitor()
        self._perf_snapshot_at: float = 0.0

    CONTROLLERS = ("manual", "staged_ekf")

    def imu_sample(self, acceleration: list[float], gyroscope: list[float], rate_hz: float) -> None:
        self.state.imu_accel_g = dict(zip("xyz", acceleration, strict=False))
        self.state.imu_gyro_dps = dict(zip("xyz", gyroscope, strict=False))
        self.state.imu_rate_hz = rate_hz
        self.state.imu_connected = True
        self.state.imu_status = "streaming"
        self.imu_sample_at = time.monotonic()
        self.dynamics.append(self.imu_sample_at, acceleration, gyroscope, self.state)
        estimate = self.heading.update(
            acceleration, gyroscope, self.imu_sample_at, moving=not self.state.stopped
        )
        for key, value in estimate.items():
            setattr(self.state, key, value)

    def imu_status(self, connected: bool, status: str) -> None:
        self.state.imu_connected = connected
        self.state.imu_status = status
        if not connected:
            self.state.imu_rate_hz = 0.0
            self.heading.disconnected()
            self.state.heading_status = "imu-disconnected"
            self.state.heading_rate_dps = 0.0

    async def start(self):
        self.bus = SMBus(1)
        self.bus.read_byte_data(0x40, 0)
        _configure_pca9685(self.bus)
        _stop_wheels(self.bus)
        self.state.connected = True
        self.guard_thread = threading.Thread(target=self._guard, daemon=True, name="motor-deadman")
        self.guard_thread.start()
        self.motion_task = asyncio.create_task(self._motion_loop(), name="gyro-drive")

    def _imu_ready(self, now):
        return (
            self.state.imu_connected
            and self.state.heading_calibrated
            and self.imu_sample_at is not None
            and now - self.imu_sample_at <= 0.15
        )

    def _brake_locked(self, reason, release=False):
        self.drive_deadline = None
        self.motion.cancel()
        if self.bus:
            _stop_wheels(self.bus)
        self.state.forward = self.state.strafe = self.state.turn = self.state.turn_target_dps = 0
        self.state.wheels = {n: 0 for n in WHEEL_CHANNELS}
        self.state.stopped = True
        self.state.motion_mode = "idle"
        self.state.reason = reason
        self.state.revision += 1
        if release:
            self.state.armed = False
            self.state.owner = None

    def _guard_check(self, now):
        with self.hw_lock:
            if self.state.controller != "manual":
                breach = None
                if not self.state.armed:
                    breach = "controller-disarmed"
                elif not self._imu_ready(now):
                    breach = "controller-imu-stale"
                else:
                    try:
                        blockers = self.parking.snapshot()["blockers"]
                    except Exception:
                        blockers = ["parking snapshot unavailable"]
                    if blockers:
                        breach = "controller-blocked"
                if breach:
                    self._fallback_manual(breach)
                    return
            if not self.state.armed:
                if self.state.auto_park:
                    self.state.auto_park = False
                    self.state.auto_park_stage = "idle"
                    self.state.auto_park_status = "disarmed"
                    self._brake_locked("autopark-disarmed", release=False)
                return
            reason = None
            if self.state.auto_park:
                imu_age = (
                    (now - self.imu_sample_at)
                    if (
                        self.state.imu_connected
                        and self.state.heading_calibrated
                        and self.imu_sample_at is not None
                    )
                    else None
                )
                try:
                    cam_age_ms = camera.metrics().get("camera_frame_age_ms")
                except Exception:
                    cam_age_ms = None
                if imu_age is None or imu_age > 0.25:
                    self.state.auto_park = False
                    self.state.auto_park_stage = "error"
                    self.state.auto_park_status = "autopark-imu-stale"
                    reason = "autopark-imu-stale"
                elif cam_age_ms is None or cam_age_ms > 450:
                    self.state.auto_park = False
                    self.state.auto_park_stage = "error"
                    self.state.auto_park_status = "autopark-frame-stale"
                    reason = "autopark-frame-stale"
            if reason:
                self._brake_locked(reason, release=False)
                return
            if self.parking_active and (
                not self._imu_ready(now) or now - self.parking.last_localized > 0.30
            ):
                self.parking_active = False
                reason = "parking-sensors-stale"
            elif self.drive_deadline is not None and now >= self.drive_deadline:
                reason = (
                    "calibration-pulse-complete"
                    if self.state.motion_mode == "calibration"
                    else "aruco-pivot-complete"
                )
            elif self.state.motion_mode == "calibration" and not self._imu_ready(now):
                reason = "calibration-imu-stale"
            elif self.state.motion_mode == "calibration":
                gyro = tuple(self.state.imu_gyro_dps[a] for a in "xyz")
                yaw = sum(a * b for a, b in zip(gyro, self.calibration_up, strict=False))
                tilt = math.sqrt(max(0, sum(v * v for v in gyro) - yaw * yaw))
                if not all(math.isfinite(v) for v in gyro) or abs(yaw) > 850 or tilt > 180:
                    reason = "calibration-sensor-motion-limit"
            elif self.motion.mode and now - self.last_operator_command > WATCHDOG_SECONDS:
                reason = "input-watchdog"
            elif self.motion.requires_gyro and not self._imu_ready(now):
                reason = "gyro-unavailable"
            elif not self.state.stopped and now - self.last_drive > WATCHDOG_SECONDS:
                reason = "watchdog"
            if reason:
                self._brake_locked(
                    reason,
                    release=reason in ("calibration-imu-stale", "calibration-sensor-motion-limit"),
                )

    def _guard(self):
        while not self.guard_stop.wait(0.01):
            self._guard_check(time.monotonic())

    async def _motion_loop(self):
        broadcast_at = 0.0
        while True:
            await asyncio.sleep(0.02)
            async with self.lock:
                with self.hw_lock:
                    if not self.motion.mode or not self.state.armed:
                        continue
                    now = time.monotonic()
                    try:
                        tick0 = time.perf_counter()
                        output = self.motion.step(
                            now,
                            self.state.heading_deg,
                            self.state.heading_rate_dps,
                            self._imu_ready(now),
                        )
                        self.tick_ms = round((time.perf_counter() - tick0) * 1000, 3)
                        self.state.tick_ms = self.tick_ms
                        if output is None:
                            continue
                        self.state.motion_mode = self.motion.mode
                        self.state.turn_target_dps = output.target_rate
                        self._apply_vector(
                            output.forward,
                            output.turn,
                            output.limit,
                            output.reason,
                            smooth=bool(output.forward or output.turn),
                        )
                    except MotionFault as error:
                        self._brake_locked(str(error), True)
                    except Exception:
                        self._brake_locked("motor-control-error", True)
            if now - broadcast_at >= 0.10:
                broadcast_at = now
                await self.broadcast()

    async def close(self):
        await self.cancel_auto_park("shutdown")
        await self.cancel_parking("shutdown")
        if self.motion_task:
            self.motion_task.cancel()
            try:
                await self.motion_task
            except asyncio.CancelledError:
                pass
        self.guard_stop.set()
        if self.guard_thread:
            self.guard_thread.join(timeout=1)
        await self.stop("shutdown", True)
        if self.bus:
            self.bus.close()
            self.bus = None

    def _swept_margin(self) -> float | None:
        try:
            snap = self.parking.snapshot()
        except Exception:
            return None
        try:
            margin = float(self.parking.config.get("clearance_margin_m", 0.015))
            sigma = snap.get("position_sigma_m")
            if sigma is not None:
                margin += float(sigma)
            return round(margin, 4)
        except Exception:
            return None

    def _fallback_manual(self, reason: str) -> None:
        self._brake_locked(reason)
        self.state.controller = "manual"
        self.controller_stage = "idle"
        self.state.controller_stage = "idle"

    async def select_controller(self, cid, name: str):
        if name not in self.CONTROLLERS:
            raise ValueError("unknown controller")
        async with self.lock:
            if name == "manual":
                self.state.controller = "manual"
                self.controller_stage = "idle"
                self.state.controller_stage = "idle"
                self.state.reason = "controller-manual"
                self.state.revision += 1
            else:
                if self.state.owner != cid or not self.state.armed:
                    raise PermissionError("arm controls before selecting controller")
                now = time.monotonic()
                if not self._imu_ready(now):
                    raise PermissionError("fresh calibrated IMU required")
                blockers = self.parking.snapshot()["blockers"]
                if blockers:
                    raise PermissionError("controller unavailable: " + "; ".join(blockers))
                self.state.controller = name
                self.controller_stage = "armed-hold"
                self.state.controller_stage = self.controller_stage
                self.state.reason = f"controller-{name}"
                self.state.revision += 1
            self.state.swept_clear_margin_m = self._swept_margin()
            self.state.tick_ms = self.tick_ms
        await self.broadcast()

    async def arm(self, cid):
        async with self.lock:
            if self.state.owner not in (None, cid):
                raise PermissionError("controller busy")
            self.state.owner = cid
            self.state.armed = True
            self.state.reason = "armed"
            self.state.revision += 1
        await self.broadcast()

    async def camera_move(self, cid, axis, delta):
        async with self.lock:
            if self.parking_active:
                raise PermissionError("stop parking before moving camera")
            if self.state.owner != cid or not self.state.armed:
                raise PermissionError("arm controls first")
            if axis not in SERVO_CHANNELS:
                raise ValueError
            delta = max(-50, min(50, int(delta)))
            attr = "pan_us" if axis == "pan" else "tilt_us"
            pulse = max(1000, min(2000, getattr(self.state, attr) + delta))
            with self.hw_lock:
                if not self.bus:
                    raise RuntimeError("I2C unavailable")
                _set_pulse(self.bus, SERVO_CHANNELS[axis], pulse)
            setattr(self.state, attr, pulse)
            self.state.reason = f"camera-{axis}"
            self.state.revision += 1
        await self.broadcast()

    def _write(self, wheels):
        if not self.bus:
            raise RuntimeError("I2C unavailable")
        with self.hw_lock:
            for name, duty in wheels.items():
                rev, fwd = WHEEL_CHANNELS[name]
                if duty > 0:
                    _set_pwm(self.bus, rev, 0)
                    _set_pwm(self.bus, fwd, duty)
                elif duty < 0:
                    _set_pwm(self.bus, fwd, 0)
                    _set_pwm(self.bus, rev, abs(duty))
                else:
                    _set_pwm(self.bus, rev, 4095)
                    _set_pwm(self.bus, fwd, 4095)

    def _apply_vector(
        self,
        forward: float,
        turn: float,
        limit: int,
        reason: str = "drive",
        smooth=False,
        strafe: float = 0.0,
    ) -> None:
        motor_forward = -forward
        # Mecanum inverse kinematics: forward, rightward strafe, and yaw.
        # Q/positive yaw retains the established left-turn convention.
        values = {
            "front-left": motor_forward - strafe + turn,
            "front-right": motor_forward + strafe - turn,
            "rear-left": motor_forward + strafe + turn,
            "rear-right": motor_forward - strafe - turn,
        }
        scale = max(1.0, *(abs(value) for value in values.values()))
        wheels = {name: round(value / scale * limit) for name, value in values.items()}
        with self.hw_lock:
            if not self.state.armed:
                return
            now = time.monotonic()
            if smooth:
                wheels = ramp_wheels(self.state.wheels, wheels, now - self.last_pwm_at)
            self._write(wheels)
            self.last_drive = self.last_pwm_at = now
            self.state.forward = forward
            self.state.strafe = strafe
            self.state.turn = turn
            self.state.speed_limit = limit
            self.state.wheels = wheels
            self.state.stopped = not any(wheels.values())
            self.state.reason = "command-zero" if self.state.stopped else reason
            self.state.revision += 1

    async def calibration_step(self, cid, forward, strafe, turn, limit, duration):
        """Explicit identification pulse: full PCA range, at most 250ms lease."""
        async with self.lock:
            if self.state.owner != cid or not self.state.armed:
                raise PermissionError("arm controls first")
            if self.parking_active or aruco.follow:
                raise PermissionError("disable autonomous motion before calibration")
            if not all(math.isfinite(v) for v in (forward, strafe, turn, limit, duration)):
                raise ValueError("nonfinite calibration input")
            if not 0 <= limit <= 4095 or not 0.01 <= duration <= 0.25:
                raise ValueError("calibration PWM/duration out of range")
            if any(abs(v) > 1 for v in (forward, strafe, turn)):
                raise ValueError("calibration vector out of range")
            with self.hw_lock:
                now = time.monotonic()
                if not self._imu_ready(now):
                    raise PermissionError("fresh calibrated IMU required")
                if self.state.motion_mode != "calibration":
                    gravity = tuple(self.state.imu_accel_g[a] for a in "xyz")
                    norm = math.sqrt(sum(v * v for v in gravity))
                    if norm > 0.5:
                        self.calibration_up = tuple(v / norm for v in gravity)
                self.motion.cancel()
                self.state.motion_mode = "calibration"
                self.last_operator_command = now
                self._apply_vector(forward, turn, int(limit), reason="calibration", strafe=strafe)
                self.drive_deadline = now + duration
        await self.broadcast()

    async def sidestep(self, cid: str, direction: int, limit: int) -> None:
        await self.drive(cid, 0, float(direction), 0, limit)

    async def survey_step(self, cid, forward, turn, limit, duration):
        if self.parking_active:
            raise PermissionError("stop parking before surveying")
        if aruco.follow:
            raise PermissionError("disable follow before surveying")
        if not self._imu_ready(time.monotonic()):
            raise PermissionError("fresh calibrated IMU required for survey")
        frame_age = camera.metrics()["camera_frame_age_ms"]
        if frame_age is None or frame_age > 250:
            raise PermissionError("fresh camera required for survey")
        if not all(math.isfinite(v) for v in (forward, turn, duration)):
            raise ValueError("invalid survey step")
        if abs(forward) + abs(turn) > 1 or not 0.01 <= duration <= 0.20:
            raise ValueError("survey steps must be a single axis and <=200ms")
        await self.drive(
            cid, forward, 0, turn, max(500, min(1800, limit)), autonomous=True, duration=duration
        )
        self.state.motion_mode = "survey"
        self.state.reason = "survey-step"
        await self.broadcast()

    async def cancel_parking(self, reason="parking-stopped"):
        self.parking_active = False
        self.parking_epoch += 1
        task = self.parking_task
        if task and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.parking_task = None
        self.parking_owner = None
        self.parking_status = reason
        await self.stop(reason, release=False)

    async def start_parking(self, cid):
        if not self._imu_ready(time.monotonic()):
            raise PermissionError("fresh calibrated IMU required")
        if self.state.owner != cid or not self.state.armed:
            raise PermissionError("arm controls before parking")
        blockers = self.parking.snapshot()["blockers"]
        if blockers:
            raise PermissionError("parking unavailable: " + "; ".join(blockers))
        await self.cancel_parking("parking-planning")
        epoch = self.parking_epoch
        await aruco.disable_follow("parking-start")
        plan = await asyncio.to_thread(self.parking.plan_snapshot)
        if not plan["ok"]:
            raise PermissionError("; ".join(plan["blockers"]))
        if self.state.owner != cid or not self.state.armed or epoch != self.parking_epoch:
            raise PermissionError("parking request cancelled or ownership changed")
        if not self._imu_ready(time.monotonic()) or self.parking.snapshot()["blockers"]:
            raise PermissionError("parking localization changed while planning")
        self.parking_owner = cid
        self.parking_active = True
        self.parking_status = "following verified route"
        route = [Pose2(**pose) for pose in plan["path"]]
        config = self.parking.config
        feedback = DockingFeedback(
            route,
            config["motion_model"],
            config["vehicle"],
            gear_obstacles(config["gear"], config["vehicle"]["height_m"]),
            config["clearance_margin_m"] + self.parking.sigma,
        )
        self.parking_task = asyncio.create_task(
            self._parking_loop(cid, feedback), name="drone-parking"
        )

    async def _parking_loop(self, cid, feedback):
        measured_at = -1
        try:
            while self.parking_active:
                if not self._imu_ready(time.monotonic()):
                    self.parking_status = "parking-imu-stale"
                    break
                if self.state.owner != cid or not self.state.armed:
                    self.parking_status = "parking ownership lost"
                    break
                snapshot = self.parking.snapshot()
                if snapshot["blockers"]:
                    self.parking_status = "; ".join(snapshot["blockers"])
                    break
                if self.parking.last_localized == measured_at:
                    await asyncio.sleep(0.02)
                    continue
                measured_at = self.parking.last_localized
                pose = snapshot["car_pose"]
                command = feedback.step(
                    Pose2(pose["x_m"], pose["z_m"], math.radians(pose["yaw_deg"]))
                )
                self.parking_status = command.get("reason", command["action"])
                if command["action"] == "stop":
                    await self.stop(self.parking_status, release=False)
                    if self.parking_status != "settling":
                        break
                    await asyncio.sleep(0.06)
                    continue
                duration = command["duration"]
                await self.drive(
                    cid,
                    command["forward"],
                    0,
                    command["turn"],
                    1800,
                    autonomous=True,
                    duration=duration,
                )
                self.state.motion_mode = "parking"
                pulse_end = time.monotonic() + duration
                await asyncio.sleep(duration + 0.02)
                await self.stop("parking-observe", release=False)
                await asyncio.sleep(0.12)
                deadline = time.monotonic() + 0.5
                while self.parking_active and self.parking.last_localized < pulse_end:
                    if time.monotonic() > deadline:
                        self.parking_status = "post-motion localization unavailable"
                        break
                    await asyncio.sleep(0.02)
                if self.parking.last_localized < pulse_end:
                    break
        finally:
            self.parking_active = False
            self.parking_owner = None
            await self.stop("parking-stopped", release=False)
            await self.broadcast()

    async def cancel_auto_park(self, reason="auto-park-stopped", cid=None):
        if (
            cid is not None
            and self.state.owner not in (None, cid)
            and self.auto_park_owner not in (None, cid)
        ):
            raise PermissionError("only the controller can stop auto-park")
        self.auto_park_epoch += 1
        was_active = self.state.auto_park or self.auto_park_task is not None
        self.state.auto_park = False
        task = self.auto_park_task
        self.auto_park_task = None
        if task and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.auto_park_owner = None
        if was_active:
            if self.state.auto_park_stage not in TERMINAL_STAGES:
                self.state.auto_park_stage = "idle"
            self.state.auto_park_status = reason
            await self.stop(reason, release=False)

    async def start_auto_park(self, cid):
        if self.state.owner != cid or not self.state.armed:
            raise PermissionError("arm controls before auto-park")
        if not camera.enabled:
            raise PermissionError("enable camera capture before auto-park")
        now = time.monotonic()
        if not (
            self.state.imu_connected
            and self.state.heading_calibrated
            and self.imu_sample_at is not None
            and (now - self.imu_sample_at) <= 0.25
        ):
            raise PermissionError("fresh calibrated IMU required for auto-park")
        await self.cancel_parking("auto-park-started")
        self.motion.cancel()
        await aruco.disable_follow("auto-park-started")
        try:
            await aruco.configure(aruco.owner or cid, False, False)
        except Exception:
            aruco.enabled = False
        await self.cancel_auto_park("auto-park-restarted")
        self.auto_park_epoch += 1
        epoch = self.auto_park_epoch
        self.auto_park_owner = cid
        self.state.auto_park = True
        self.state.auto_park_stage = "acquiring"
        self.state.auto_park_status = "starting"
        self.state.auto_park_pose = None
        self.auto_park_task = asyncio.create_task(
            self._auto_park_loop(cid, epoch), name="auto-park"
        )
        await self.broadcast()

    async def _auto_park_loop(self, cid, epoch):
        plane.set_external(True)
        vision_task: asyncio.Task[None] | None = None
        recorder: AutoParkRecorder | None = None
        try:
            recorder = AutoParkRecorder(model=plane.model_tag)
            runner = PlaneAutoPark(start_t=time.monotonic())
            latest_fix: dict[str, Any] = {"seq": -1, "meas": None, "cap": None, "det": None}

            async def _vision_worker() -> None:
                last_seq = -1
                while self.state.auto_park and self.auto_park_epoch == epoch:
                    frame, seq, cap = camera.latest_sample()
                    now_v = time.monotonic()
                    if frame is None or seq == last_seq or not cap or (now_v - cap) > 0.45:
                        await asyncio.sleep(0.01)
                        continue
                    last_seq = seq
                    prior = runner.prior_for_solver()
                    close_range = runner.ctrl.state == "ENTER"
                    det, meas = await asyncio.to_thread(
                        plane.detect_once, frame, prior, close_range=close_range
                    )
                    if recorder is not None:
                        await asyncio.to_thread(recorder.save_frame, seq, frame, det)
                    latest_fix["seq"] = seq
                    latest_fix["meas"] = meas
                    latest_fix["cap"] = cap
                    latest_fix["det"] = det

            vision_task = asyncio.create_task(_vision_worker(), name="auto-park-vision")
            consumed_seq = -1
            while self.state.auto_park and self.auto_park_epoch == epoch:
                loop_t0 = time.perf_counter()
                if self.state.owner != cid or not self.state.armed:
                    self.state.auto_park_stage = "error"
                    self.state.auto_park_status = "auto-park ownership lost"
                    break
                now = time.monotonic()
                frame, _, cap = camera.latest_sample()
                frame_age = (now - cap) if (frame is not None and cap) else None
                imu_age = (
                    (now - self.imu_sample_at)
                    if (
                        self.state.imu_connected
                        and self.state.heading_calibrated
                        and self.imu_sample_at is not None
                    )
                    else None
                )
                meas = None
                cap_used = None
                gear_nose_u: float | None = None
                if latest_fix["seq"] != consumed_seq:
                    consumed_seq = latest_fix["seq"]
                    meas = latest_fix["meas"]
                    cap_used = latest_fix["cap"]
                    det = latest_fix["det"]
                    if det and det.get("points"):
                        for pt in det["points"]:
                            if int(pt[0]) == 9:
                                gear_nose_u = float(pt[1])
                                break

                plan_t0 = time.perf_counter()
                # RelativeHeading emits CW+ heading_rate_dps; PlaneAutoPark expects CCW+ rad/s.
                gyro_ccw = -math.radians(float(self.state.heading_rate_dps))
                cmd = runner.step(
                    now,
                    imu_age_s=imu_age,
                    frame_age_s=frame_age,
                    gyro_ccw_rad_s=gyro_ccw,
                    meas=meas,
                    cap_t=cap_used,
                    gear_nose_u=gear_nose_u,
                )
                self.perf.record("plan_ms", (time.perf_counter() - plan_t0) * 1000.0)
                self.state.auto_park_stage = cmd["stage"]
                self.state.auto_park_status = cmd["status"]
                self.state.auto_park_pose = cmd["pose"]
                if recorder is not None:
                    recorder.record_tick(
                        {
                            "stage": cmd["stage"],
                            "status": cmd["status"],
                            "stop": cmd["stop"],
                            "forward": round(cmd["forward"], 3),
                            "strafe": round(cmd["strafe"], 3),
                            "turn": round(cmd["turn"], 3),
                            "cmd_vel": cmd.get("cmd_vel"),
                            "duty": cmd.get("duty"),
                            "yaw_int": cmd.get("yaw_int"),
                            "upd_status": cmd.get("upd_status"),
                            "est_innov": cmd.get("est_innov"),
                            "shield_hits": cmd.get("shield_hits"),
                            "gyro_ccw": round(gyro_ccw, 4),
                            "accel_g": dict(self.state.imu_accel_g),
                            "heading_deg": round(float(self.state.heading_deg), 2),
                            "imu_age_ms": round(imu_age * 1000) if imu_age is not None else None,
                            "frame_age_ms": (
                                round(frame_age * 1000) if frame_age is not None else None
                            ),
                            "pose": cmd["pose"],
                            "meas": (
                                [round(float(v), 4) for v in meas] if meas is not None else None
                            ),
                            "wheels": dict(self.state.wheels),
                        }
                    )
                drive_t0 = time.perf_counter()
                if cmd["stop"]:
                    await self.stop(cmd["reason"] or "auto-park-stop", release=False)
                    self.perf.record("drive_ms", (time.perf_counter() - drive_t0) * 1000.0)
                    self.perf.record("autopark_loop_ms", (time.perf_counter() - loop_t0) * 1000.0)
                    if cmd["done"] or cmd["stage"] in TERMINAL_STAGES:
                        break
                else:
                    await self.drive(
                        cid,
                        cmd["forward"],
                        cmd["strafe"],
                        cmd["turn"],
                        1800,
                        autonomous=True,
                        mode="auto-park",
                    )
                    self.perf.record("drive_ms", (time.perf_counter() - drive_t0) * 1000.0)
                    self.perf.record("autopark_loop_ms", (time.perf_counter() - loop_t0) * 1000.0)
                rem = 0.02 - (time.perf_counter() - loop_t0)
                await asyncio.sleep(max(0.002, rem))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.auto_park_stage = "error"
            self.state.auto_park_status = f"auto-park-error: {exc}"
        finally:
            if vision_task is not None:
                vision_task.cancel()
                try:
                    await vision_task
                except asyncio.CancelledError:
                    pass
            if recorder is not None:
                recorder.close(self.state.auto_park_stage, self.state.auto_park_status)
            plane.set_external(False)
            if self.auto_park_epoch == epoch:
                self.state.auto_park = False
                self.auto_park_owner = None
                self.auto_park_task = None
                if self.state.auto_park_stage not in TERMINAL_STAGES:
                    self.state.auto_park_stage = "idle"
                await self.stop(self.state.auto_park_status or "auto-park-stopped", release=False)
                await self.broadcast()

    async def camera_capture(self, cid, enabled):
        if self.state.owner not in (None, cid):
            raise PermissionError("only the controller can change camera capture")
        if not isinstance(enabled, bool):
            raise ValueError("camera enabled must be a boolean")
        if not enabled:
            await self.cancel_auto_park("camera-paused")
            await self.cancel_parking("camera-paused")
            await aruco.configure(cid, False, False)
        camera.set_enabled(enabled)
        await self.broadcast()

    async def drive(
        self,
        cid,
        forward,
        strafe,
        turn,
        limit,
        autonomous=False,
        duration=None,
        mode: str = "aruco-follow",
    ):
        if not autonomous and self.state.auto_park and self.state.owner == cid:
            await self.cancel_auto_park("manual-drive-override", cid=cid)
        async with self.lock:
            if self.state.owner != cid or not self.state.armed:
                raise PermissionError("arm controls first")
            if self.parking_active and not autonomous:
                raise PermissionError("stop parking before manual driving")
            if aruco.follow and not autonomous:
                raise PermissionError("disable ArUco follow before manual drive")
            if not all(math.isfinite(value) for value in (forward, strafe, turn)):
                raise ValueError
            forward = max(-1.0, min(1.0, forward))
            strafe = max(-1.0, min(1.0, strafe))
            turn = max(-1.0, min(1.0, turn))
            limit = max(500, min(1800, limit))
            if duration is not None:
                if not autonomous or not math.isfinite(duration):
                    raise ValueError("invalid pulse duration")
                duration = max(0.01, min(0.25, float(duration)))
            with self.hw_lock:
                now = time.monotonic()
                self.drive_deadline = None
                self.state.strafe = strafe
                if autonomous:
                    self.motion.cancel()
                    self.state.motion_mode = mode
                    self.state.turn_target_dps = 0
                    self._apply_vector(forward, turn, limit, strafe=strafe)
                    if duration is not None:
                        self.drive_deadline = now + duration
                elif not forward and not strafe and not turn:
                    self._brake_locked("key-release")
                else:
                    self.motion.cancel()
                    self.last_operator_command = now
                    self.state.motion_mode = "manual"
                    self.state.turn_target_dps = 0
                    self._apply_vector(forward, turn, limit, strafe=strafe)
        now_b = time.monotonic()
        if not autonomous or (now_b - getattr(self, "_last_auto_broadcast", 0.0)) >= 0.065:
            self._last_auto_broadcast = now_b
            await self.broadcast()

    async def stop(self, reason, release=False):
        if release and self.state.auto_park and self.auto_park_task is not asyncio.current_task():
            await self.cancel_auto_park(reason)
        async with self.lock:
            with self.hw_lock:
                self._brake_locked(reason, release)
        await self.broadcast()

    async def disconnect(self, cid):
        if self.auto_park_owner == cid or (self.state.owner == cid and self.state.auto_park):
            await self.cancel_auto_park("auto-park-owner-disconnected")
        if self.parking_owner == cid:
            await self.cancel_parking("parking-owner-disconnected")
        self.clients.pop(cid, None)
        if self.state.owner == cid:
            await self.stop("controller-disconnected", True)
        else:
            await self.broadcast()

    def _aruco_age_ms(self) -> int | None:
        try:
            last = aruco.last_detection_at
        except Exception:
            return None
        if not last:
            return None
        return round((time.monotonic() - last) * 1000)

    def _imu_bias_dps(self) -> dict[str, float]:
        try:
            bias = getattr(self.heading, "_bias", (0.0, 0.0, 0.0))
            return {axis: round(float(value), 3) for axis, value in zip("xyz", bias, strict=False)}
        except Exception:
            return {axis: 0.0 for axis in "xyz"}

    def _rls_scale(self) -> float:
        # Bounded lateral dynamics gain estimate for SysID readiness display.
        # Defaults to 1.0 until an online estimator publishes a refined scale.
        try:
            scale = float(getattr(self, "rls_lateral_scale", 1.0))
        except Exception:
            return 1.0
        if not math.isfinite(scale):
            return 1.0
        return round(max(0.25, min(2.0, scale)), 3)

    def _sysid_ready(self) -> bool:
        now = time.monotonic()
        try:
            cam_age = camera.metrics()["camera_frame_age_ms"]
        except Exception:
            cam_age = None
        return bool(self._imu_ready(now) and cam_age is not None and cam_age <= 250)

    def payload(self, cid=None):
        self.state.imu_age_ms = (
            round((time.monotonic() - self.imu_sample_at) * 1000) if self.imu_sample_at else None
        )
        self.state.swept_clear_margin_m = self._swept_margin()
        self.state.tick_ms = self.tick_ms
        self.state.aruco_age_ms = self._aruco_age_ms()
        self.state.imu_bias_dps = self._imu_bias_dps()
        try:
            self.state.follow_debug = dict(aruco.follow_debug)
        except Exception:
            pass
        self.state.rls_scale = self._rls_scale()
        self.state.sysid_ready = self._sysid_ready()
        now_perf = time.monotonic()
        if not self.state.perf or (now_perf - self._perf_snapshot_at) >= 0.2:
            self.state.perf = self.perf.snapshot()
            self._perf_snapshot_at = now_perf
        p = asdict(self.state)
        p.update(
            clients=len(self.clients),
            you_are_owner=bool(cid and self.state.owner == cid),
            watchdog_ms=600,
        )
        p.update(camera.metrics())
        p.update(aruco.metrics())
        p.update(plane.metrics())
        p["parking"] = self.parking.snapshot()
        p["parking"].update(active=self.parking_active, execution_status=self.parking_status)
        p["motion_input_age_ms"] = (
            round((time.monotonic() - self.last_operator_command) * 1000)
            if self.motion.mode
            else None
        )
        return {"type": "state", "state": p}

    async def broadcast(self):
        t0 = time.perf_counter()
        dead = []
        for cid, ws in list(self.clients.items()):
            try:
                await ws.send_json(self.payload(cid))
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.clients.pop(cid, None)
        self.perf.record("broadcast_ms", (time.perf_counter() - t0) * 1000.0)


controller = RobotController()
camera = CameraStream()
aruco = ArucoFollower(camera, controller)
plane = PlaneDetectorRunner(camera, controller)
imu = ImuStream(controller.imu_sample, controller.imu_status)
app = FastAPI(title="UGV Control", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
async def startup():
    await controller.start()
    camera.start()
    aruco.start()
    plane.start()
    imu.start()


@app.on_event("shutdown")
async def shutdown():
    await plane.close()
    await aruco.close()
    imu.close()
    camera.close()
    await controller.close()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/stream.mjpg")
def stream():
    return StreamingResponse(
        camera.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/state")
async def state():
    return controller.payload()


@app.get("/api/calibration/frame")
async def calibration_frame():
    frame, sequence, captured_at = camera.latest_sample()
    if frame is None:
        return Response(status_code=503)
    return Response(
        frame,
        media_type="image/jpeg",
        headers={
            "X-Pi-Monotonic": str(captured_at),
            "X-Frame-Sequence": str(sequence),
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/dynamics")
async def dynamics(after: int = 0):
    return controller.dynamics.snapshot(max(0, after))


@app.get("/api/parking")
async def parking_state():
    return controller.parking.snapshot()


@app.get("/api/parking/plan")
async def parking_plan():
    return await asyncio.to_thread(controller.parking.plan_snapshot)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cid = uuid.uuid4().hex[:8]
    controller.clients[cid] = ws
    try:
        await ws.send_json(controller.payload(cid))
        await controller.broadcast()
        while True:
            try:
                m = json.loads(await ws.receive_text())
                kind = m.get("type")
                if kind == "drive":
                    await controller.drive(
                        cid,
                        float(m.get("forward", 0)),
                        float(m.get("strafe", 0)),
                        float(m.get("turn", 0)),
                        int(m.get("speed_limit", 1800)),
                    )
                elif kind == "calibration-step":
                    await controller.calibration_step(
                        cid,
                        float(m.get("forward", 0)),
                        float(m.get("strafe", 0)),
                        float(m.get("turn", 0)),
                        float(m.get("speed_limit", 0)),
                        float(m.get("duration", 0)),
                    )
                elif kind == "sidestep":
                    await controller.sidestep(
                        cid, int(m.get("direction", 0)), int(m.get("speed_limit", 1200))
                    )
                elif kind == "survey-step":
                    await controller.survey_step(
                        cid,
                        float(m.get("forward", 0)),
                        float(m.get("turn", 0)),
                        int(m.get("speed_limit", 1800)),
                        float(m.get("duration", 0.1)),
                    )
                elif kind == "arm":
                    await controller.arm(cid)
                elif kind == "camera":
                    await controller.camera_move(cid, str(m.get("axis")), int(m.get("delta", 0)))
                elif kind == "camera-capture":
                    await controller.camera_capture(cid, m.get("enabled"))
                elif kind == "aruco":
                    if m.get("enabled") and controller.state.auto_park:
                        continue
                    if m.get("enabled") and not camera.enabled:
                        raise PermissionError("enable camera capture before ArUco")
                    await aruco.configure(
                        cid,
                        bool(m.get("enabled")),
                        m.get("follow"),
                        m.get("speed_limit"),
                        m.get("follow_mode"),
                    )
                elif kind == "plane":
                    if m.get("enabled") and not camera.enabled:
                        raise PermissionError("enable camera capture before Plane detector")
                    await plane.configure(
                        m.get("enabled") if "enabled" in m else None,
                        m.get("model"),
                    )
                elif kind == "controller":
                    await controller.select_controller(cid, str(m.get("controller", "manual")))
                elif kind == "auto-park":
                    if not isinstance(m.get("enabled"), bool):
                        raise ValueError("auto-park enabled must be a boolean")
                    if m["enabled"]:
                        await controller.start_auto_park(cid)
                    else:
                        await controller.cancel_auto_park("client-auto-park-stop", cid=cid)
                elif kind == "parking":
                    if m.get("action") == "start":
                        await controller.start_parking(cid)
                    elif m.get("action") == "stop" and controller.state.owner == cid:
                        await controller.cancel_parking()
                    else:
                        raise ValueError("invalid parking request")
                elif kind == "heading-calibrate":
                    controller.heading.request_calibration()
                elif kind == "heartbeat":
                    await ws.send_json(controller.payload(cid))
                elif kind == "stop" and controller.state.owner in (None, cid):
                    stop_reason = str(m.get("reason") or "client-stop").strip() or "client-stop"
                    await controller.cancel_auto_park(stop_reason)
                    await controller.cancel_parking(stop_reason)
                    await aruco.disable_follow(stop_reason)
                    await controller.stop(stop_reason, stop_reason == "disarm")
                else:
                    await ws.send_json({"type": "error", "error": "unknown message"})
            except PermissionError as e:
                await ws.send_json({"type": "error", "error": str(e)})
            except (TypeError, ValueError, json.JSONDecodeError):
                await ws.send_json({"type": "error", "error": "invalid message"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if aruco.owner == cid:
            await aruco.disable_follow("aruco-owner-disconnected")
        await controller.disconnect(cid)


def main():
    uvicorn.run("ugv.web:app", host="0.0.0.0", port=8765, reload=False)
