"""Plane keypoint detector runner for the Pi dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ugv.plane_vision import AutoParkRecorder, PlaneAutoPark, PlaneDetectorRunner, discover_models
from ugv.web import CameraStream, RobotController


def test_autopark_recorder_writes_telemetry_frames_and_summary(tmp_path: Path):
    import json

    rec = AutoParkRecorder(root=tmp_path, model="yolo11n-pose-v1-320-int8", frame_interval_s=0.0)
    rec.record_tick({"stage": "search", "turn": 0.4, "gyro_ccw": 0.5})
    rel = rec.save_frame(1, b"\xff\xd8fake\xff\xd9", {"points": [], "box": None, "ms": 80.0})
    assert rel == "frames/000001.jpg"
    assert (rec.run_dir / rel).is_file()
    rec.close("parked", "arrived")
    summary = json.loads((rec.run_dir / "summary.json").read_text())
    assert summary["ticks"] == 1
    assert summary["frames_saved"] == 1
    assert summary["final_stage"] == "parked"


def test_plane_autopark_search_acquire_orbit_and_safety():
    import numpy as np

    ap = PlaneAutoPark(start_t=10.0)
    # 1. No measurement yet -> SEARCH spinning CCW (+turn)
    c0 = ap.step(10.05, imu_age_s=0.01, frame_age_s=0.02, gyro_ccw_rad_s=0.0)
    assert c0["stage"] == "search"
    assert c0["stop"] is False
    assert c0["turn"] > 0.0

    # 2. Airframe seen at 1.4 m ahead -> transitions to ACQUIRE and stops
    meas = np.array([1.4, 0.0, 2.8, 0.0, 12.0])
    c1 = ap.step(
        10.15, imu_age_s=0.01, frame_age_s=0.02, gyro_ccw_rad_s=0.0, meas=meas, cap_t=10.10
    )
    assert c1["stage"] == "acquire"
    assert c1["stop"] is True

    # 3. After acquire_s + acquire_samples -> transitions to ORBIT
    for i in range(1, 10):
        t = 10.15 + i * 0.1
        c = ap.step(
            t, imu_age_s=0.01, frame_age_s=0.02, gyro_ccw_rad_s=0.0, meas=meas, cap_t=t - 0.02
        )
    assert c["stage"] == "orbit"
    assert c["pose"] is not None

    # 4. Stale IMU immediately stops with error
    c_err = ap.step(11.5, imu_age_s=0.5, frame_age_s=0.02, gyro_ccw_rad_s=0.0)
    assert c_err["stage"] == "error"
    assert c_err["stop"] is True and c_err["done"] is True


def test_plane_autopark_gyro_pi_ramps_on_stall_and_damps_overshoot():
    ap = PlaneAutoPark(start_t=0.0)
    # Stalled in SEARCH (gyro=0): turn command should ramp up automatically to break friction
    turns = []
    for i in range(1, 20):
        c = ap.step(i * 0.02, imu_age_s=0.01, frame_age_s=0.02, gyro_ccw_rad_s=0.0)
        turns.append(c["turn"])
    assert turns[-1] > turns[0] + 0.1

    # Overshooting target rate (gyro=1.5 rad/s >> 0.55 rad/s): turn command should drop
    c_over = ap.step(0.42, imu_age_s=0.01, frame_age_s=0.02, gyro_ccw_rad_s=1.5)
    assert c_over["turn"] < turns[-1]


def test_discover_models_finds_pairs_and_yolo(tmp_path: Path):
    for name in ("yolo11n-pose-v1", "kp-v3c", "kp-crop-v3c", "kp-v4", "kp-crop-v4", "kp-orphan"):
        d = tmp_path / name
        d.mkdir()
        (d / "model.onnx").write_bytes(b"fake")
    found = discover_models(tmp_path)
    assert list(found.keys()) == ["yolo11n-pose-v1", "v3c", "v4"]
    assert found["yolo11n-pose-v1"] == (tmp_path / "yolo11n-pose-v1", None)
    assert found["v4"] == (tmp_path / "kp-v4", tmp_path / "kp-crop-v4")


def test_payload_exposes_plane_fields():
    controller = RobotController()
    state = controller.payload()["state"]
    for key in (
        "plane_enabled",
        "plane_model",
        "plane_available_models",
        "plane_status",
        "plane_points",
        "plane_box",
        "plane_pose",
        "plane_ms",
        "plane_fps",
    ):
        assert key in state, key
    assert state["plane_enabled"] is False


def test_plane_runner_toggle_and_unknown_model(tmp_path: Path):
    for name in ("kp-v3c", "kp-crop-v3c", "kp-v4", "kp-crop-v4"):
        d = tmp_path / name
        d.mkdir()
        (d / "model.onnx").write_bytes(b"fake")

    async def scenario():
        cam = CameraStream()
        ctrl = RobotController()
        ctrl.broadcast = AsyncMock()
        runner = PlaneDetectorRunner(cam, ctrl, train_dir=tmp_path)
        assert runner.model_tag == "v3c"
        await runner.configure(enabled=True, model="v4")
        assert runner.enabled is True
        assert runner.model_tag == "v4"
        with pytest.raises(ValueError):
            await runner.configure(model="does-not-exist")
        await runner.configure(enabled=False)
        assert runner.enabled is False
        assert runner.metrics()["plane_points"] == []

    asyncio.run(scenario())
