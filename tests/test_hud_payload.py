"""HUD payload: additive telemetry fields, controller round-trip, static IDs."""
import asyncio
from pathlib import Path
from unittest.mock import patch

from ugv.web import RobotController

STATIC = Path(__file__).resolve().parent.parent / "src" / "ugv" / "static"

REQUIRED_STATE_KEYS = (
    "tick_ms", "inference_ms", "swept_clear_margin_m",
    "aruco_age_ms", "imu_bias_dps", "follow_debug", "rls_scale",
)

LEGACY_IDS = (
    "camera-feed", "aruco-overlay", "feed-stats",
    "connection", "parking-panel", "heading-needle", "heading-degrees",
    "heading-status", "heading-calibrate", "motion", "reason",
    "turn-feedback", "ownership", "clients", "forward", "strafe", "turn",
    "pan", "tilt", "aruco-readout", "vision-rate", "vision-time",
    "imu-status", "imu-accel", "imu-gyro", "joystick", "joystick-knob",
    "arm", "camera-toggle",
    "aruco-toggle", "follow-toggle", "map-toggle", "speed", "speed-value",
    "speed-meter", "stop", "watchdog",
)


def test_payload_has_additive_telemetry_fields():
    controller = RobotController()
    state = controller.payload()["state"]
    for key in REQUIRED_STATE_KEYS:
        assert key in state, key
    assert isinstance(state["imu_bias_dps"], dict)
    assert set(state["imu_bias_dps"]) == {"x", "y", "z"}
    fd = state["follow_debug"]
    for key in ("mode", "err_lead", "z_lead", "vx", "vz", "sigma_m",
                "coast_age_ms", "cmd_vec"):
        assert key in fd, key
    assert len(fd["cmd_vec"]) == 3
    assert isinstance(state["rls_scale"], float)
    # legacy keys intact
    for key in ("controller", "controller_stage", "forward", "strafe", "turn",
                "wheels", "reason", "watchdog_ms", "parking"):
        assert key in state, key


def test_controller_selector_round_trip():
    async def scenario():
        controller = RobotController()
        controller.state.owner = "t"
        controller.state.armed = True
        snap = {"blockers": [], "position_sigma_m": 0.005}
        with patch.object(controller, "_imu_ready", return_value=True), \
             patch.object(controller.parking, "snapshot", return_value=snap):
            await controller.select_controller("t", "staged_ekf")
        payload = controller.payload("t")["state"]
        assert payload["controller"] == "staged_ekf"
        assert payload["swept_clear_margin_m"] is not None
        with patch.object(controller, "_imu_ready", return_value=False):
            controller._guard_check(99.0)
        assert controller.payload()["state"]["controller"] == "manual"
    asyncio.run(scenario())


def test_static_hud_ids_present_and_legacy_preserved():
    html = (STATIC / "index.html").read_text()
    for dom_id in ("controller-select", "tactical-bev", "telemetry-rail",
                   "follow-state", "l7-banner", "follow-mode-toggle",
                   "view-fit-toggle", "spark-loop", "spark-cam", "spark-imu",
                   "wheel-fl", "wheel-fr", "wheel-rl", "wheel-rr"):
        assert f'id="{dom_id}"' in html, dom_id
    for dom_id in LEGACY_IDS:
        assert f'id="{dom_id}"' in html, dom_id
    for attr in ('data-key="KeyW"', 'data-wheel="front-left"'):
        assert attr in html, attr
    assert '<select id="controller-select"' in html
    for value in ("manual", "staged_ekf", "rl_student"):
        assert f'value="{value}"' in html, value
    assert 'send({type: "controller", controller:' in (STATIC / "app.js").read_text() \
        or 'send({type:"controller"' in (STATIC / "app.js").read_text().replace(" ", "")
    tactical = (STATIC / "tactical.js").read_text()
    assert "tactical-bev" in tactical
    assert "parking.gear" in tactical or "parking" in tactical
    assert "sigma" in tactical  # covariance ellipse
    assert "wheel" in tactical.lower()
