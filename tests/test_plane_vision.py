"""Plane keypoint detector runner for the Pi dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ugv.plane_vision import PlaneAutoPark, PlaneDetectorRunner, discover_models
from ugv.web import CameraStream, RobotController


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
