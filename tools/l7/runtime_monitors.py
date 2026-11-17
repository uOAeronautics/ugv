"""UNWIRED draft of the five online runtime monitors (audit_contract.py header).

Pure rolling-window brake predicates over the L1 sim reference thresholds. This
module is a DRAFT: it is not imported by l7_run.py or the dashboard and commands
no motors. Wiring it into the runtime is a hardware-gated change (SAFETY S3->S4,
docs/STATE.md); the dashboard remains the sole motor authority. Kept here so the
spec is executable and testable before that gate.

Windows are 1.0 s (50 ticks) and 2.0 s (100 ticks) at the 50 Hz loop. Each
monitor returns True when its brake condition trips; ``MonitorBank.brake`` is
the OR of the five (monitor 5 only after visual acquisition).
"""

from __future__ import annotations

from collections import deque

import numpy as np

DT = 0.02
WIN_1S = 50
WIN_2S = 100

GYRO_STD_MAX = 0.25  # rad/s, rolling 1 s std
SAT_FRAC_MAX = 0.25  # fraction |act_j| > 0.8 over 1 s, any axis
SAT_LEVEL = 0.8
GYRO_MU_SIM = 0.002  # rad/s, L1 sim reference mean
GYRO_SIGMA_SIM = 0.091  # rad/s, L1 sim reference std
GYRO_Z_MAX = 3.0  # RMS z-score over 1 s
FLIP_DEADBAND = 0.1
FLIP_RATE_A0_MAX = 2.5  # reversals/s over 2 s
FLIP_RATE_A1_MAX = 1.5
TAG_RATE_MIN = 0.25  # fraction src == "tag" over 2 s, after acquisition


def gyro_std_trip(imu_w: np.ndarray) -> bool:
    w = np.asarray(imu_w[-WIN_1S:], dtype=np.float64)
    return len(w) >= WIN_1S and float(w.std()) > GYRO_STD_MAX


def saturation_trip(act: np.ndarray) -> bool:
    a = np.asarray(act[-WIN_1S:], dtype=np.float64)
    if len(a) < WIN_1S:
        return False
    frac = (np.abs(a) > SAT_LEVEL).mean(axis=0)
    return bool((frac > SAT_FRAC_MAX).any())


def gyro_zscore_trip(imu_w: np.ndarray) -> bool:
    w = np.asarray(imu_w[-WIN_1S:], dtype=np.float64)
    if len(w) < WIN_1S:
        return False
    z = (w - GYRO_MU_SIM) / GYRO_SIGMA_SIM
    return bool(np.sqrt(np.mean(z * z)) > GYRO_Z_MAX)


def _flip_rate(axis: np.ndarray, seconds: float) -> float:
    a = np.asarray(axis, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    live = np.abs(a) > FLIP_DEADBAND
    s = np.sign(a)
    reversals = (s[1:] != s[:-1]) & live[1:] & live[:-1] & (s[1:] != 0) & (s[:-1] != 0)
    return float(reversals.sum()) / seconds


def flip_rate_trip(act: np.ndarray) -> bool:
    a = np.asarray(act[-WIN_2S:], dtype=np.float64)
    if len(a) < WIN_2S:
        return False
    seconds = len(a) * DT
    return (
        _flip_rate(a[:, 0], seconds) > FLIP_RATE_A0_MAX
        or _flip_rate(a[:, 1], seconds) > FLIP_RATE_A1_MAX
    )


def tag_rate_trip(is_tag: np.ndarray, acquired: bool) -> bool:
    t = np.asarray(is_tag[-WIN_2S:], dtype=bool)
    if not acquired or len(t) < WIN_2S:
        return False
    return bool(t.mean() < TAG_RATE_MIN)


class MonitorBank:
    """Rolling buffer of (imu_w, act[3], is_tag) ticks; evaluates the five monitors.

    UNWIRED: ``brake`` is a diagnostic recommendation only. No caller in the
    runtime consumes it yet; the dashboard owns the motors.
    """

    def __init__(self) -> None:
        self.imu_w: deque[float] = deque(maxlen=WIN_2S)
        self.act: deque[tuple[float, float, float]] = deque(maxlen=WIN_2S)
        self.is_tag: deque[bool] = deque(maxlen=WIN_2S)
        self.acquired = False

    def update(self, imu_w: float, act, is_tag: bool) -> None:
        self.imu_w.append(float(imu_w))
        self.act.append(tuple(float(x) for x in act))
        self.is_tag.append(bool(is_tag))
        if is_tag:
            self.acquired = True

    def trips(self) -> dict[str, bool]:
        w = np.array(self.imu_w, dtype=np.float64)
        a = np.array(self.act, dtype=np.float64).reshape(-1, 3)
        t = np.array(self.is_tag, dtype=bool)
        return {
            "gyro_std": gyro_std_trip(w),
            "saturation": saturation_trip(a),
            "gyro_zscore": gyro_zscore_trip(w),
            "flip_rate": flip_rate_trip(a),
            "tag_rate": tag_rate_trip(t, self.acquired),
        }

    @property
    def brake(self) -> bool:
        return any(self.trips().values())
