"""Fast shield + stale-latency compensation (scout fast-shield-and-latency).

- Vectorized swept_clear must match the scalar collides() loop exactly.
- Shield must brake under a 200 ms stale pose at max speed (yaw 0/30/60).
"""
import math
import time

import numpy as np

from ugv.parking import Pose2, angle_error, collides, gear_obstacles, swept_clear
from ugv.policy import V_X_MAX, StudentPolicy


def _reference_sweep(start, end, width, length, obstacles, inflation=0.):
    """Pre-vectorization scalar loop (old swept_clear body)."""
    delta = angle_error(end.yaw, start.yaw)
    distance = math.hypot(end.x - start.x, end.z - start.z) \
        + abs(delta) * math.hypot(width, length) / 2
    steps = max(1, math.ceil(distance / .005))
    for i in range(steps + 1):
        t = i / steps
        pose = Pose2(start.x + t * (end.x - start.x),
                     start.z + t * (end.z - start.z), start.yaw + t * delta)
        if collides(pose, width, length, obstacles, inflation + .0025):
            return False
    return True


def _default_obstacles():
    return gear_obstacles({"clear_width_m": 0.254, "bar_length_m": 0.3048,
                           "bar_radius_m": 0.03, "inward_slope": 0.0})


def test_swept_clear_parity_with_scalar_loop():
    rng = np.random.default_rng(7)
    obstacles = _default_obstacles()
    for _ in range(500):
        start = Pose2(float(rng.uniform(-.5, .5)), float(rng.uniform(-.5, .5)),
                      float(rng.uniform(-math.pi, math.pi)))
        end = Pose2(start.x + float(rng.uniform(-.5, .5)),
                    start.z + float(rng.uniform(-.5, .5)),
                    start.yaw + float(rng.uniform(-1, 1)))
        expected = _reference_sweep(start, end, .16, .22, obstacles, .015)
        assert swept_clear(start, end, .16, .22, obstacles, .015) == expected


def _contact_approach(yaw_deg):
    """March toward the right skid; return (true, stale) pose pair.

    True pose sits 1 cm before first contact along the drive direction;
    stale lags it by V_X_MAX * 0.2 s (200 ms camera latency at max speed).
    """
    yaw = math.radians(yaw_deg)
    direction = (math.cos(yaw), math.sin(yaw))
    obstacles = _default_obstacles()
    lam = 0.6
    while lam > 0 and collides(Pose2(-direction[0] * lam, -direction[1] * lam, yaw),
                               .16, .22, obstacles, .015):
        lam += 0.01  # start clear if 0.6 already clips
    while lam > 0 and not collides(Pose2(-direction[0] * (lam - 0.002),
                                         -direction[1] * (lam - 0.002), yaw),
                                   .16, .22, obstacles, .015):
        lam -= 0.002
    lag = V_X_MAX * 0.2
    true = Pose2(-direction[0] * (lam + 0.01), -direction[1] * (lam + 0.01), yaw)
    stale = Pose2(true.x - direction[0] * lag, true.z - direction[1] * lag, yaw)
    assert not collides(true, .16, .22, obstacles, .015), yaw_deg
    assert not collides(stale, .16, .22, obstacles, .015), yaw_deg
    return true, stale


def _max_speed_policy():
    p = StudentPolicy({})
    p.forward = lambda h: np.array([1.0, 0.0, 0.0])  # +x_body at V_X_MAX
    return p


def test_shield_brakes_under_200ms_stale_pose():
    for yaw_deg in (0, 30, 60):
        true, stale = _contact_approach(yaw_deg)
        assert true.x != stale.x or true.z != stale.z
        for dt_stale in (0.2, 0.0):
            p = _max_speed_policy()
            act, reason = p.step((stale.x, stale.z, stale.yaw), 0.05, True,
                                 0.0, 0.0, np.zeros(3), stale, dt_stale=dt_stale)
            assert (act == 0).all() and "blocked" in reason, (yaw_deg, dt_stale)


def test_shield_no_false_brake_when_truly_clear():
    p = StudentPolicy({})
    p.forward = lambda h: np.array([0.1, 0.0, 0.0])  # crawl, short sweep
    est = Pose2(-0.45, 0.0, 0.0)
    act, reason = p.step((-0.45, 0.0, 0.0), 0.05, True, 0.0, 0.0,
                         np.zeros(3), est, dt_stale=0.2)
    assert reason == "policy" and np.allclose(act, [0.1, 0.0, 0.0])


def test_swept_clear_under_point_15ms():
    obstacles = _default_obstacles()
    start, end = Pose2(0, 0, 0), Pose2(.3, .1, .2)
    for _ in range(200):
        swept_clear(start, end, .16, .22, obstacles, .015)
    t0 = time.perf_counter()
    for _ in range(2000):
        swept_clear(start, end, .16, .22, obstacles, .015)
    assert (time.perf_counter() - t0) / 2000 * 1000 < 0.15
