"""Known-answer tests for the UNWIRED runtime-monitor draft (tools/l7)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.l7 import runtime_monitors as rm


def _calm_gyro(n=rm.WIN_1S, seed=0):
    return np.random.default_rng(seed).normal(rm.GYRO_MU_SIM, 0.05, n)


def test_gyro_std_quiet_on_sim_trips_on_ep3():
    assert rm.gyro_std_trip(_calm_gyro()) is False
    assert rm.gyro_std_trip(np.full(rm.WIN_1S, 0.5) * np.tile([1, -1], rm.WIN_1S // 2)) is True


def test_saturation_quiet_on_small_trips_on_saturated_axis():
    assert rm.saturation_trip(np.zeros((rm.WIN_1S, 3))) is False
    a = np.zeros((rm.WIN_1S, 3))
    a[:, 2] = 0.9
    assert rm.saturation_trip(a) is True


def test_gyro_zscore_quiet_on_sim_trips_on_offset():
    assert rm.gyro_zscore_trip(_calm_gyro(seed=1)) is False
    assert rm.gyro_zscore_trip(np.full(rm.WIN_1S, 0.5)) is True


def test_flip_rate_quiet_on_smooth_trips_on_oscillation():
    smooth = np.zeros((rm.WIN_2S, 3))
    smooth[:, 0] = 0.05  # below deadband: no reversals
    assert rm.flip_rate_trip(smooth) is False
    osc = np.zeros((rm.WIN_2S, 3))
    osc[:, 0] = 0.3 * np.tile([1, -1], rm.WIN_2S // 2)  # ~49 reversals/s
    assert rm.flip_rate_trip(osc) is True


def test_tag_rate_needs_acquisition_and_trips_on_masking():
    full = np.ones(rm.WIN_2S, dtype=bool)
    sparse = np.zeros(rm.WIN_2S, dtype=bool)
    sparse[:10] = True  # 10% tag rate
    assert rm.tag_rate_trip(full, acquired=True) is False
    assert rm.tag_rate_trip(sparse, acquired=True) is True
    assert rm.tag_rate_trip(sparse, acquired=False) is False  # not yet acquired


def test_short_windows_do_not_trip():
    assert rm.gyro_std_trip(np.zeros(10)) is False
    assert rm.saturation_trip(np.zeros((10, 3))) is False
    assert rm.flip_rate_trip(np.zeros((10, 3))) is False
    assert rm.tag_rate_trip(np.zeros(10, dtype=bool), acquired=True) is False


def test_monitor_bank_brake_is_or_and_unwired():
    bank = rm.MonitorBank()
    for _ in range(rm.WIN_2S):
        bank.update(0.0, (0.0, 0.0, 0.0), True)
    assert bank.brake is False
    assert all(v is False for v in bank.trips().values())
    # a saturated yaw axis trips the bank
    for _ in range(rm.WIN_1S):
        bank.update(0.0, (0.0, 0.0, 0.9), True)
    assert bank.brake is True
    assert bank.trips()["saturation"] is True


def test_module_is_not_wired_into_runtime():
    # the draft must not be imported by the motor authority or the L7 runner
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/ugv/web.py", "tools/l7/l7_run.py"):
        text = (root / rel).read_text()
        assert "runtime_monitors" not in text
