"""Plane keypoint detector runner for the Pi dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ugv.plane_vision import PlaneDetectorRunner, discover_models
from ugv.web import CameraStream, RobotController


def test_discover_models_finds_pairs(tmp_path: Path):
    for name in ("kp-v3c", "kp-crop-v3c", "kp-v4", "kp-crop-v4", "kp-orphan"):
        d = tmp_path / name
        d.mkdir()
        (d / "model.onnx").write_bytes(b"fake")
    found = discover_models(tmp_path)
    assert list(found.keys()) == ["v3c", "v4"]
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
