"""Swept-footprint shield: the vectorized swept_clear matches the scalar loop and stays fast."""

import math
import time

import numpy as np

from ugv.parking import Pose2, angle_error, collides, gear_obstacles, swept_clear


def _reference_sweep(start, end, width, length, obstacles, inflation=0.0):
    """Pre-vectorization scalar loop (old swept_clear body)."""
    delta = angle_error(end.yaw, start.yaw)
    distance = (
        math.hypot(end.x - start.x, end.z - start.z) + abs(delta) * math.hypot(width, length) / 2
    )
    steps = max(1, math.ceil(distance / 0.005))
    for i in range(steps + 1):
        t = i / steps
        pose = Pose2(
            start.x + t * (end.x - start.x), start.z + t * (end.z - start.z), start.yaw + t * delta
        )
        if collides(pose, width, length, obstacles, inflation + 0.0025):
            return False
    return True


def _default_obstacles():
    return gear_obstacles(
        {"clear_width_m": 0.254, "bar_length_m": 0.3048, "bar_radius_m": 0.03, "inward_slope": 0.0}
    )


def test_swept_clear_parity_with_scalar_loop():
    rng = np.random.default_rng(7)
    obstacles = _default_obstacles()
    for _ in range(500):
        start = Pose2(
            float(rng.uniform(-0.5, 0.5)),
            float(rng.uniform(-0.5, 0.5)),
            float(rng.uniform(-math.pi, math.pi)),
        )
        end = Pose2(
            start.x + float(rng.uniform(-0.5, 0.5)),
            start.z + float(rng.uniform(-0.5, 0.5)),
            start.yaw + float(rng.uniform(-1, 1)),
        )
        expected = _reference_sweep(start, end, 0.16, 0.22, obstacles, 0.015)
        assert swept_clear(start, end, 0.16, 0.22, obstacles, 0.015) == expected


def test_swept_clear_under_point_15ms():
    obstacles = _default_obstacles()
    start, end = Pose2(0, 0, 0), Pose2(0.3, 0.1, 0.2)
    for _ in range(200):
        swept_clear(start, end, 0.16, 0.22, obstacles, 0.015)
    t0 = time.perf_counter()
    for _ in range(2000):
        swept_clear(start, end, 0.16, 0.22, obstacles, 0.015)
    assert (time.perf_counter() - t0) / 2000 * 1000 < 0.15
