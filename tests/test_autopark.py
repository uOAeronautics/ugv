import asyncio
import json
import math
import time

import cv2
import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from ugv import autopark, web
from ugv.parking import Capsule


def _make_pose_sols(tag_yaw: float, tx: float = 0.0, tz: float = 0.7, err: float = 0.1):
    alpha = math.pi - tag_yaw
    nx = -math.sin(alpha)
    nz = math.cos(alpha)
    return [tx, 0.0, tz, nx, 0.0, nz, err]


class _FakeWS:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_text(self):
        await asyncio.sleep(0.02)
        if not self.incoming:
            raise WebSocketDisconnect()
        item = self.incoming.pop(0)
        if isinstance(item, float):
            await asyncio.sleep(item)
            return await self.receive_text()
        return json.dumps(item)


def test_load_tagmap_skips_zero_obs(tmp_path):
    p = tmp_path / "tagmap.json"
    p.write_text(json.dumps({
        "14": [-0.028, 0.0, math.pi / 2, -1],
        "6": [0.06, -0.05, 0.0, 0],
        "3": [-0.10, -0.05, 2.55, 129],
    }))
    m = autopark.load_tagmap(p, tag_y=-0.08)
    assert 14 in m and 3 in m
    assert 6 not in m
    assert math.isclose(m[14][1], -0.08)


def test_tag_solutions_verifies_positive_camera_z():
    obj = autopark._OBJ.astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [0.8]], dtype=np.float64)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, autopark.K_DEFAULT, None)
    sols = autopark.tag_solutions(img_pts.reshape(4, 2))
    assert len(sols) >= 1
    assert all(s[2] > 0.0 for s in sols)


def test_tag_bits_and_detector_jpeg():
    bits = autopark.TagBits()
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 14, 120, borderBits=1)
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    canvas[120:240, 580:700] = marker
    pts = np.array([[580, 120], [699, 120], [699, 239], [580, 239]], dtype=np.float32)
    assert bits.errors(canvas, 14, pts) <= 1

    ok, buf = cv2.imencode(".jpg", canvas)
    assert ok
    det = autopark.TagDetector()
    rec = det.detect_jpeg(buf.tobytes(), cap=12.34)
    assert rec["cap"] == 12.34
    assert any(t["id"] == 14 and t["sol"] for t in rec["tags"])

    # Corrupted marker rejected by TagBits > 1
    corrupt = canvas.copy()
    corrupt[140:200, 600:660] = 255 - corrupt[140:200, 600:660]
    assert bits.errors(corrupt, 14, pts) > 1


def test_localizer_ippe_branch_disambiguation_and_outlier_rejection():
    tagmap = {"14": [-0.028, 0.0, math.pi / 2, -1], "6": [0.06, -0.05, 0.0, 0]}
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    assert 6 not in loc.map

    rec0 = {"cap": 10.0, "tags": [{"id": 14, "px": 900.0, "sol": [_make_pose_sols(0.0, tz=0.80, err=0.2)]}]}
    assert loc.update(rec0, gyro_at_cap=0.0, now=10.05, gyro_now=0.0, duty_fwd=0.2, duty_left=0.0)
    assert loc.dr_t == pytest.approx(10.05)

    # Secondary IPPE branch has error >= 1.2 * 0.10 + 0.15 = 0.27 -> rejected, keeps sols[0]
    s0 = _make_pose_sols(0.10, tz=0.82, err=0.10)
    s1 = _make_pose_sols(0.00, tz=0.82, err=0.40)
    rec_ippe = {"cap": 10.10, "tags": [{"id": 14, "px": 900.0, "sol": [s0, s1]}]}
    assert loc.update(rec_ippe, gyro_at_cap=0.0, now=10.10)
    assert loc.last["tag_yaw"] == pytest.approx(0.10, abs=1e-4)

    # Outlier |e| >= 20 deg -> rejected, pose/dr_pose/off untouched
    pose_before = loc.pose
    dr_before = list(loc.dr_pose)
    off_before = loc.off
    s_bad = _make_pose_sols(math.radians(45.0), tx=0.99, tz=2.50, err=0.1)
    rec_bad = {"cap": 10.20, "tags": [{"id": 14, "px": 900.0, "sol": [s_bad]}]}
    assert loc.update(rec_bad, gyro_at_cap=0.0, now=10.20) is False
    assert loc.pose == pose_before
    assert loc.dr_pose == dr_before
    assert loc.off == off_before
    assert loc.outliers == 1


def test_autopark_runner_swept_clear_blocks_with_cw_negation():
    tagmap = {"14": [-0.028, 0.0, math.pi / 2, -1]}
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    obs = [Capsule(ax=-0.10, az=0.72, bx=0.10, bz=0.72, radius=0.04)]
    runner = autopark.AutoParkRunner(localizer=loc, obstacles=obs)
    tz = 0.90 - (-0.08) - autopark.CAM_FWD_M
    rec = {"cap": 1.0, "tags": [{"id": 14, "px": 900.0, "sol": [_make_pose_sols(0.0, tx=-0.028, tz=tz, err=0.1)]}]}
    cmd = runner.step(1.0, imu_age_s=0.02, frame_age_s=0.02, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=rec)
    assert cmd.stage == "blocked"
    assert cmd.stop is True
    assert cmd.reason == "swept-clear-blocked"


def test_autopark_runner_corridor_and_blind_dr_timeout_and_parked():
    tagmap = {"14": [-0.028, 0.0, math.pi / 2, -1]}
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    runner = autopark.AutoParkRunner(localizer=loc, obstacles=[])

    cmd0 = runner.step(0.0, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd0.stage == "acquiring" and not cmd0.stop and abs(cmd0.turn) > 0.0

    tz_stage = 0.55 - (-0.08) - autopark.CAM_FWD_M
    rec_stage = {"cap": 1.0, "tags": [{"id": 14, "px": 900.0, "sol": [_make_pose_sols(0.0, tx=-0.028, tz=tz_stage, err=0.1)]}]}
    cmd1 = runner.step(1.0, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=rec_stage)
    assert cmd1.stage == "entering" and not cmd1.stop and cmd1.forward > 0.0

    loc.dr_pose = [0.0, 0.30, 0.0]
    loc.pose_t = 1.0
    cmd_to = runner.step(2.6, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd_to.stage == "aborted" and cmd_to.stop is True and cmd_to.reason == "blind-dr-timeout"

    loc2 = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    runner2 = autopark.AutoParkRunner(localizer=loc2, obstacles=[])
    runner2.stage = "entering"
    loc2.off = 0.0
    loc2.pose_t = 5.0
    loc2.dr_t = 5.0
    loc2.dr_pose = [0.16, 0.35, 0.0]
    cmd_x = runner2.step(5.02, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd_x.stage == "aborted" and cmd_x.reason == "corridor-x-limit"

    loc3 = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    runner3 = autopark.AutoParkRunner(localizer=loc3, obstacles=[])
    runner3.stage = "entering"
    loc3.off = 0.0
    loc3.pose_t = 5.0
    loc3.dr_t = 5.0
    loc3.dr_pose = [0.0, 0.03, 0.0]
    cmd_brk = runner3.step(5.02, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=5.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd_brk.stage == "braking" and cmd_brk.stop is True
    cmd_done = runner3.step(5.05, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.5, gyro_yaw_rad=0.0, rec=None)
    assert cmd_done.stage == "parked" and cmd_done.done is True and cmd_done.stop is True


def test_ws_autopark_start_stop_disarm_manual_override_and_stale(monkeypatch):
    ctrl = web.RobotController()
    cam = web.CameraStream()
    arc = web.ArucoFollower(cam, ctrl)

    img = np.full((720, 1280, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()

    writes = []
    monkeypatch.setattr(ctrl, "_write", lambda wheels: writes.append(dict(wheels)))
    monkeypatch.setattr(web, "controller", ctrl)
    monkeypatch.setattr(web, "camera", cam)
    monkeypatch.setattr(web, "aruco", arc)
    monkeypatch.setattr(cam, "latest_sample", lambda: (jpeg, 1, time.monotonic()))
    monkeypatch.setattr(cam, "metrics", lambda: {"camera_enabled": True, "camera_frame_age_ms": 10})

    async def _run():
        cam.set_enabled(True)
        ctrl.state.imu_connected = True
        ctrl.state.heading_calibrated = True
        ctrl.imu_sample_at = time.monotonic()
        arc.enabled = True

        # 1. Start auto-park then stop/disarm via WS
        ws1 = _FakeWS([
            {"type": "arm"},
            {"type": "auto-park", "enabled": True},
            0.08,
            {"type": "stop", "reason": "disarm"},
            0.05,
        ])
        await web.websocket_endpoint(ws1)
        states = [m["state"] for m in ws1.sent if m.get("type") == "state"]
        assert any(s["auto_park"] for s in states)
        assert arc.enabled is False
        assert states[-1]["auto_park"] is False
        assert states[-1]["armed"] is False
        assert states[-1]["stopped"] is True

        # 2. Manual drive cancels auto-park
        ctrl.imu_sample_at = time.monotonic()
        ws2 = _FakeWS([
            {"type": "arm"},
            {"type": "auto-park", "enabled": True},
            0.08,
            {"type": "drive", "forward": 0.1, "strafe": 0.0, "turn": 0.0, "speed_limit": 1200},
            0.05,
        ])
        await web.websocket_endpoint(ws2)
        assert ctrl.state.auto_park is False

        # 3. Stale IMU (> 0.25s) in _guard_check immediately stops auto-park & brakes
        ctrl.imu_sample_at = time.monotonic()
        ws3 = _FakeWS([
            {"type": "arm"},
            {"type": "auto-park", "enabled": True},
            0.12,
        ])
        task3 = asyncio.create_task(web.websocket_endpoint(ws3))
        await asyncio.sleep(0.08)
        assert ctrl.state.auto_park is True
        ctrl.imu_sample_at = time.monotonic() - 0.30
        ctrl._guard_check(time.monotonic())
        assert ctrl.state.auto_park is False
        assert ctrl.state.stopped is True
        await task3

        # 4. Stale frame (> 0.45s) immediately stops auto-park & brakes
        ctrl.imu_sample_at = time.monotonic()
        monkeypatch.setattr(cam, "latest_sample", lambda: (jpeg, 1, time.monotonic() - 0.50))
        monkeypatch.setattr(cam, "metrics", lambda: {"camera_enabled": True, "camera_frame_age_ms": 500})
        ws4 = _FakeWS([
            {"type": "arm"},
            {"type": "auto-park", "enabled": True},
            0.10,
        ])
        await web.websocket_endpoint(ws4)
        assert ctrl.state.auto_park is False
        assert ctrl.state.stopped is True

    asyncio.run(_run())


def test_autopark_runner_acquiring_timeout_aborts():
    tagmap = {"14": [-0.028, 0.0, math.pi / 2, -1]}
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    runner = autopark.AutoParkRunner(localizer=loc, obstacles=[], acquire_timeout_s=16.0)

    cmd0 = runner.step(10.0, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd0.stage == "acquiring" and cmd0.stop is False

    cmd1 = runner.step(25.9, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd1.stage == "acquiring" and cmd1.stop is False

    cmd_to = runner.step(26.1, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    assert cmd_to.stage == "aborted"
    assert cmd_to.stop is True
    assert cmd_to.status == "No tag found after 360 deg scan"
    assert cmd_to.forward == 0.0 and cmd_to.strafe == 0.0 and cmd_to.turn == 0.0
    assert runner.last_duty == (0.0, 0.0)


def test_uncalibrated_tag6_bootstraps_staging_and_blocks_entering_until_calibrated():
    tagmap = {
        "14": [-0.028, 0.0, math.pi / 2, -1],
        "6": [0.06, -0.05, 0.0, 0],
    }
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    assert 6 not in loc.map
    assert 6 in loc.uncal_map
    runner = autopark.AutoParkRunner(localizer=loc, obstacles=[])

    # Tag 6 (obs==0) drives acquiring -> staging toward (0.0, 0.55)
    # Place car right at (0.0, 0.55, 0.0) seen via uncalibrated Tag 6:
    # tag6_map = [0.06, -0.13, 0.0, 0] (with tag_y=-0.08), car at (0, 0.55, 0) => cam at (0, 0.55 + CAM_FWD_M)
    # f = (0, 1), r = (1, 0), cam = tag6 - (tz*f + tx*r) => tx = 0.06, tz = -0.13 - (0.55 + CAM_FWD_M)
    # Use a pose in front of Tag 6 (e.g., tag_yaw=0 => phi=0, nx=0, nz=1? Wait: phi=0 means tag faces +X)
    # Better: put car at (0.55 + CAM_FWD_M + 0.06?): let's compute exact tag6 sol for pose (0.0, 0.55, 0.0):
    # At yaw=0: th=-pi/2 => f=(0, 1), r=(-1, 0)? Wait: f=(cos(-pi/2), sin(-pi/2))=(0, -1), r=(sin(-pi/2), -cos(-pi/2))=(-1, 0).
    # cam = (0, 0.55) + CAM_FWD_M*(0, -1) = (0, 0.55 - CAM_FWD_M).
    # m - cam = (0.06, -0.13 - (0.55 - CAM_FWD_M)) = tz*(0, -1) + tx*(-1, 0) => tx = -0.06, tz = 0.68 - CAM_FWD_M.
    # For tag_yaw=0 with phi=0: _tag_yaw(0, sol) = 0 - atan2(-nx, nz) + pi/2 = 0 => atan2(-nx, nz) = pi/2 => nx=-1, nz=0.
    sol6 = [-0.06, 0.0, 0.68 - autopark.CAM_FWD_M, -1.0, 0.0, 0.0, 0.1]
    rec6 = {"cap": 1.0, "tags": [{"id": 6, "px": 700.0, "sol": [sol6]}]}

    cmd1 = runner.step(1.0, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=rec6)
    assert loc.last_used_calibrated is False
    # Even when already at (0.0, 0.55, 0.0), uncalibrated Tag 6 must NOT enter "entering"
    assert cmd1.stage == "staging"
    assert cmd1.stop is False

    # Now calibrated Tag 14 confirms alignment at (0.0, 0.55, 0.0) -> transitions to "entering"
    tz14 = 0.55 - (-0.08) - autopark.CAM_FWD_M
    rec14 = {"cap": 1.1, "tags": [{"id": 14, "px": 900.0, "sol": [_make_pose_sols(0.0, tx=-0.028, tz=tz14, err=0.1)]}]}
    cmd2 = runner.step(1.1, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=rec14)
    assert loc.last_used_calibrated is True
    assert cmd2.stage == "entering"
    assert cmd2.forward > 0.0


def test_first_calibrated_fix_snaps_dr_pose_within_25cm():
    tagmap = {
        "14": [-0.028, 0.0, math.pi / 2, -1],
        "6": [0.06, -0.05, 0.0, 0],
    }
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    sol6 = [-0.16, 0.0, 0.68 - autopark.CAM_FWD_M, -1.0, 0.0, 0.0, 0.1]
    assert loc.update({"cap": 1.0, "tags": [{"id": 6, "px": 700.0, "sol": [sol6]}]}, gyro_at_cap=0.0, allow_uncalibrated=True)
    assert loc.last_used_calibrated is False
    assert loc.dr_pose[:2] == pytest.approx([-0.10, 0.55], abs=1e-6)

    tz14 = 0.55 - (-0.08) - autopark.CAM_FWD_M
    rec14 = {"cap": 1.1, "tags": [{"id": 14, "px": 900.0, "sol": [_make_pose_sols(0.0, tx=0.028, tz=tz14, err=0.1)]}]}
    assert loc.update(rec14, gyro_at_cap=0.0)
    assert loc.last_used_calibrated is True
    assert loc.dr_pose[:2] == pytest.approx([0.0, 0.55], abs=1e-6)


def test_autopark_adaptive_imu_breakaway_ramp_and_damping():
    tagmap = {"14": [-0.028, 0.0, math.pi / 2, -1]}
    loc = autopark.Localizer(tagmap=tagmap, tag_y=-0.08)
    runner = autopark.AutoParkRunner(localizer=loc, obstacles=[])
    assert runner.stage_goto.dz_yaw == pytest.approx(0.16)

    cmd0 = runner.step(0.0, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    cmd1 = runner.step(0.25, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    cmd2 = runner.step(0.50, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)
    cmd3 = runner.step(0.80, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=0.0, gyro_yaw_rad=0.0, rec=None)

    assert cmd0.stage == "acquiring"
    assert abs(cmd1.turn) > abs(cmd0.turn)
    assert abs(cmd2.turn) > abs(cmd1.turn)
    assert abs(cmd3.turn) >= abs(cmd2.turn)
    assert abs(cmd3.turn) * autopark.KDUTY <= 0.26 + 1e-9

    # Once robot breaks away (imu_w_dps >= 6.0), stall timer resets and damping moderates turn
    cmd_run = runner.step(0.90, imu_age_s=0.01, frame_age_s=0.01, imu_w_dps=30.0, gyro_yaw_rad=0.0, rec=None)
    assert runner.stall_s == 0.0
    assert abs(cmd_run.turn) < abs(cmd3.turn)
