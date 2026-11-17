"""Dashboard controller selector: explicit transitions, gating, fallback, snapshot."""

import asyncio
from unittest.mock import patch

import pytest

from ugv.web import RobotController


def _armed(c, cid="t"):
    c.state.owner = cid
    c.state.armed = True
    return cid


def test_manual_is_default_and_snapshot_exposes_fields():
    c = RobotController()
    p = c.payload()["state"]
    assert p["controller"] == "manual"
    for k in ("controller_stage", "swept_clear_margin_m", "inference_ms", "tick_ms"):
        assert k in p


def test_staged_requires_armed_imu_and_zero_blockers():
    async def scenario():
        c = RobotController()
        with pytest.raises(PermissionError):
            await c.select_controller("t", "staged_ekf")
        _armed(c)
        with patch.object(c, "_imu_ready", return_value=False):
            with pytest.raises(PermissionError):
                await c.select_controller("t", "staged_ekf")
        with (
            patch.object(c, "_imu_ready", return_value=True),
            patch.object(
                c.parking, "snapshot", return_value={"blockers": ["x"], "position_sigma_m": None}
            ),
        ):
            with pytest.raises(PermissionError):
                await c.select_controller("t", "staged_ekf")
        assert c.state.controller == "manual"

    asyncio.run(scenario())


def test_staged_select_and_guard_fallback_to_manual():
    async def scenario():
        c = RobotController()
        _armed(c)
        c._write = lambda wheels: None
        snap = {"blockers": [], "position_sigma_m": 0.005}
        with (
            patch.object(c, "_imu_ready", return_value=True),
            patch.object(c.parking, "snapshot", return_value=snap),
        ):
            await c.select_controller("t", "staged_ekf")
        assert c.state.controller == "staged_ekf"
        p = c.payload()["state"]
        assert p["controller"] == "staged_ekf"
        assert p["swept_clear_margin_m"] is not None
        # breach: IMU stale -> brake + manual
        with patch.object(c, "_imu_ready", return_value=False):
            c._guard_check(99.0)
        assert c.state.controller == "manual" and c.state.stopped

    asyncio.run(scenario())


def test_rl_student_is_gone():
    async def scenario():
        c = RobotController()
        with pytest.raises(ValueError):
            await c.select_controller("t", "rl_student")

    asyncio.run(scenario())


def test_unknown_controller_rejected():
    async def scenario():
        c = RobotController()
        with pytest.raises(ValueError):
            await c.select_controller("t", "autopilot")

    asyncio.run(scenario())
