"""Unit tests for tools/l7/l7_run.py Localizer, TagBits, GoTo safety, and arrival validation."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_SPEC = importlib.util.spec_from_file_location("l7_run", REPO / "tools" / "l7" / "l7_run.py")
l7 = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(l7)


class _StubImu:
    def __init__(self, yaw: float = 0.0, rate: float = 0.0):
        self.yaw = yaw
        self.rate = rate
        self.t = 0.0

    def yaw_at(self, t: float) -> float:
        return self.yaw

    def snapshot(self):
        return self.yaw, self.rate, self.t, (0, 0, 0, 0)


def _make_args(**kw):
    defaults = dict(
        gain_fwd=2.3,
        gain_left=1.6,
        gain_yaw=15.7,
        dz_fwd=0.05,
        dz_left=0.13,
        dz_yaw=0.10,
        blend=0.35,
        kp_pos=1.2,
        kp_yaw=1.5,
        vcap_fwd=0.30,
        vcap_left=0.25,
        vcap_yaw=0.5,
        tol_pos=0.04,
        tol_yaw=5.0,
        calibrate_tagmap=False,
        allow_uncalibrated_tags=False,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _tag14_sol(tz: float = 0.8, tx: float = 0.0, yaw: float = 0.0, err: float = 0.15, nz: float = -1.0):
    # At yaw=0 (car facing -y), tag 14 normal (+y) in camera frame is (-sin(yaw), 0, -cos(yaw))?
    # Check Localizer._tag_yaw: phi - atan2(-nx, nz) + pi/2 = yaw -> with phi=pi/2, atan2(-nx, nz) = pi - yaw.
    # For yaw=0: atan2(-nx, nz) = pi -> nx=0, nz=-1. For small yaw: nx = -sin(yaw), nz = -cos(yaw).
    nx = -math.sin(yaw) * abs(nz)
    return [tx, 0.0, tz, nx, 0.0, nz, err]


def test_ippe_ambiguity_and_outlier_rejection():
    imu = _StubImu(yaw=0.0)
    loc = l7.Localizer(imu, l7.DEFAULT_MAP, tag_y=-0.08, primary=14, args=_make_args())

    # 1. Ambiguous IPPE pair on first frame (self.off is None) -> must wait (return False).
    ambig = {
        "cap": 100.0,
        "tags": [{
            "id": 14,
            "px": 40.0,
            "sol": [_tag14_sol(yaw=0.0, err=0.30), _tag14_sol(yaw=0.7, err=0.35)],
        }],
    }
    assert loc.update(ambig) is False
    assert loc.off is None
    assert loc.pose is None

    # 2. Back-facing normal (nz > -0.2) must be rejected.
    back_facing = {
        "cap": 100.0,
        "tags": [{"id": 14, "px": 40.0, "sol": [_tag14_sol(yaw=0.0, err=0.1, nz=0.5)]}],
    }
    assert loc.update(back_facing) is False
    assert loc.off is None

    # 3. Uncalibrated tag 6 (obs == 0) must be dropped when primary is absent.
    tag6_only = {
        "cap": 100.0,
        "tags": [{"id": 6, "px": 40.0, "sol": [[0.0, 0.0, 0.6, 0.0, 0.0, -1.0, 0.1]]}],
    }
    assert loc.update(tag6_only) is False
    assert loc.pose is None

    # 4. Clear IPPE frame initializes self.off, pose, dr_pose.
    clear = {
        "cap": 100.0,
        "tags": [{"id": 14, "px": 40.0, "sol": [_tag14_sol(tz=0.8, yaw=0.0, err=0.15),
                                                _tag14_sol(tz=0.8, yaw=0.7, err=0.60)]}],
    }
    assert loc.update(clear) is True
    assert loc.off is not None
    pose_before = loc.pose
    dr_before = list(loc.dr_pose)
    map6_before = list(loc.map[6])

    # 5. Repeated >= 20 deg outliers (even 10 in a row!) must return False and NOT corrupt pose, dr_pose, off, or map.
    off_before = loc.off
    for k in range(10):
        outlier = {
            "cap": 100.1 + 0.02 * k,
            "tags": [
                {"id": 14, "px": 40.0, "sol": [_tag14_sol(tz=1.4, yaw=math.radians(35), err=0.1)]},
                {"id": 6, "px": 30.0, "sol": [[0.1, 0.0, 0.5, 0.0, 0.0, -1.0, 0.1]]},
            ],
        }
        assert loc.update(outlier) is False
    assert loc.off == off_before
    assert loc.pose == pose_before
    assert loc.dr_pose == dr_before
    assert loc.map[6] == map6_before


def test_latency_cap_to_now_propagation(monkeypatch):
    imu = _StubImu(yaw=0.0)
    loc = l7.Localizer(imu, l7.DEFAULT_MAP, tag_y=-0.08, primary=14, args=_make_args())
    # Pretend car is moving forward (+v_fwd) at 0.5 m/s, captured 0.10 s ago.
    loc.v = np.array([0.5, 0.0])
    t_now = 200.10
    monkeypatch.setattr(l7.time, "monotonic", lambda: t_now)
    # Gyro yaw advanced +0.04 rad between cap and now.
    imu_hist = {200.00: 0.0, t_now: 0.04}
    imu.yaw_at = lambda t: imu_hist.get(t, 0.04)
    imu.snapshot = lambda: (0.04, 0.4, t_now, (0, 0, 0, 0))

    rec = {
        "cap": 200.00,
        "tags": [{"id": 14, "px": 45.0, "sol": [_tag14_sol(tz=0.8, yaw=0.0, err=0.1)]}],
    }
    assert loc.update(rec) is True
    assert loc.pose_t == 200.00
    assert loc.dr_t == t_now
    # At yaw=0, forward is -y. Over dt=0.10 s at 0.5 m/s, car moves ~0.05 m toward -y.
    assert math.isclose(loc.dr_pose[1] - loc.pose[1], -0.05 * math.cos(0.02), abs_tol=1e-3)
    assert math.isclose(loc.dr_pose[2], 0.04, abs_tol=1e-6)


def test_goto_swept_clear_yaw_negation_and_atan2_guard():
    ctrl = l7.GoTo((0.0, 0.9, 0.0), _make_args())
    # Near the right landing-gear bar (x=+0.17, y in [-0.24, +0.24]) and rotated CCW (+0.5 rad):
    # yaw -> -yaw in Pose2 means the car's front-right/rear-right corners swing into the bar.
    # Verify _is_clear sees the collision and GoTo refuses to drive into the bar.
    assert ctrl._is_clear(0.08, 0.10, 0.45, 0.25, -0.20, 0.20) is False
    cmd, done = ctrl.step((0.08, 0.10, 0.45), now=1.0)
    assert done is False
    assert ctrl.last_reason in ("goto:steer_clear", "brake: blocked")
    if ctrl.last_reason == "goto:steer_clear":
        assert ctrl._is_clear(0.08, 0.10, 0.45, *cmd) is True
    else:
        assert cmd == (0.0, 0.0, 0.0)

    # Guard y -> 0 atan2 flip: at (x=0.08, y=-0.02) targeting (0, 0.9, 0),
    # yaw_goal must stay target[2]=0 (not flip ~180 deg via -atan2(0.08, -0.02)).
    cmd_zero, done_zero = ctrl.step((0.05, -0.02, 0.0), now=2.0)
    assert done_zero is False
    assert abs(cmd_zero[2]) < 1e-6  # no spurious yaw spin toward pi


def test_tagbits_tightened_threshold():
    bits = l7.TagBits()
    marker = cv2.aruco.generateImageMarker(bits.d, 14, 120, borderBits=1)
    gray = np.full((200, 200), 220, dtype=np.uint8)
    gray[40:160, 40:160] = marker
    pts = np.array([[40, 40], [160, 40], [160, 160], [40, 160]], dtype=np.float32)
    assert bits.errors(gray, 14, pts) == 0
    assert bits.errors(gray, 6, pts) > 1


def test_validate_stage_arrival_false_arrive():
    target = (0.0, 0.9, 0.0)
    assert l7.validate_stage_arrival("arrived", target, None) == "false_arrive"
    assert l7.validate_stage_arrival("arrived", target, (0.20, 0.9, 0.0)) == "false_arrive"
    assert l7.validate_stage_arrival("arrived", target, (0.0, 0.9, math.radians(15))) == "false_arrive"
    assert l7.validate_stage_arrival("arrived", target, (0.03, 0.88, math.radians(4))) == "arrived"
    # Close-in target (y < 0.42) where tag 14 is out of FOV is not invalidated by missing rest_pose.
    assert l7.validate_stage_arrival("arrived", (0.0, 0.1, 0.0), None) == "arrived"
def test_should_abort_corridor_fresh_tag_vs_stale_dr():
    est_corridor_edge = (0.18, 0.40, 0.0)
    now = 100.10

    # Fresh tag fix (< 0.15 s old) must NOT abort even on k % 5 in (1, 4) where src == "none".
    fresh_pose_t = 100.00
    for k in range(5):
        src = "tag" if (k % 5 in (0, 2, 3)) else "none"
        assert (src == "none") == (k % 5 in (1, 4))
        assert l7.should_abort_corridor("stage", est_corridor_edge, fresh_pose_t, now) is False

    # Stale blind DR (>= 0.15 s or missing pose_t) inside y < 0.55 and |x| > 0.14 MUST abort.
    assert l7.should_abort_corridor("stage", est_corridor_edge, 99.94, now) is True
    assert l7.should_abort_corridor("stage", est_corridor_edge, 99.80, now) is True
    assert l7.should_abort_corridor("stage", est_corridor_edge, None, now) is True

    # Outside corridor trigger box or non-stage mode or missing est must NOT abort.
    assert l7.should_abort_corridor("stage", (0.14, 0.40, 0.0), 99.80, now) is False
    assert l7.should_abort_corridor("stage", (0.18, 0.55, 0.0), 99.80, now) is False
    assert l7.should_abort_corridor("policy", est_corridor_edge, 99.80, now) is False
    assert l7.should_abort_corridor("stage", None, 99.80, now) is False
