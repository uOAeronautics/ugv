import importlib.util
import sys
from pathlib import Path

import numpy as np

from ugv.staged_baseline import MeanModel, StagedBaseline

HARNESS = Path(__file__).resolve().parents[1] / "simulation" / "gazebo" / "run_policy_episodes.py"


def _harness():
    spec = importlib.util.spec_from_file_location("run_policy_episodes", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_translation_inverse_round_trip():
    m = MeanModel()
    v = np.array([[0.1, 0.0], [0.0, -0.2], [0.05, -0.1], [-0.08, 0.15]])
    got = m.f(m.inv(v))[:, :2]
    assert np.abs(got - v).max() < 0.03


def test_searches_without_tag_and_holds_when_never_seen():
    c = StagedBaseline(1)
    for _ in range(10):
        act = c.step(np.zeros(3), False, 0.0)
    assert c.phase[0] == StagedBaseline.SEARCH and act[0, 2] != 0 and abs(act[0, :2]).max() == 0


def test_parks_on_fake_bridge_without_contact():
    h = _harness()
    step, reset = h.baseline_controller(0.0)
    res = h.run_episodes(step, reset, h.FakeBridge(), 5)
    assert all(r.success for r in res) and not any(r.collision for r in res)


def test_brakes_after_arrival():
    c = StagedBaseline(1)
    c.phase[:] = StagedBaseline.ENTER
    c.ekf.init[:] = True
    c.yaw.t[:] = 1.0  # identified
    act = c.step(np.array([0.0, 0.01, 0.0]), True, 0.0)
    assert c.phase[0] == StagedBaseline.HOLD and (act == 0).all()


def test_l4_search_acquires_and_locks_ekf():
    c = StagedBaseline(1)
    for _ in range(5):
        act = c.step(np.zeros(3), False, 0.0)
    assert c.phase[0] == StagedBaseline.SEARCH
    assert abs(act[0, :2]).max() == 0 and act[0, 2] != 0  # yaw-only sweep
    act = c.step(np.array([0.0, 1.0, 0.0]), True, 0.0)
    assert c.phase[0] == StagedBaseline.APPROACH
    assert c.ekf.init[0] and c.ekf.age[0] == 0.0


def test_3s_blackout_recovers():
    c = StagedBaseline(1)
    c.yaw.t[:] = 1.0  # identified
    c.step(np.array([0.0, 1.0, 0.0]), True, 0.0)
    p0 = np.diag(c.ekf.P[0]).copy()
    for _ in range(150):  # 3 s at 50 Hz
        c.step(np.zeros(3), False, 0.0)
    assert c.ekf.init[0]  # not lost
    assert np.diag(c.ekf.P[0])[:2].max() > p0[:2].max()  # uncertainty inflated
    c.step(np.array([0.0, 1.0, 0.0]), True, 0.0)
    assert c.ekf.age[0] == 0.0
    assert c.phase[0] == StagedBaseline.APPROACH


def test_yaw_coupling_adaptation():
    c = StagedBaseline(1)
    prior = c.sysid.A_lin[2, 0]
    c.sysid.u_lag[:] = np.array([0.5, 0.0, 0.0])
    for _ in range(100):
        c.sysid.observe(np.zeros((1, 3)), np.zeros(1, bool), np.array([0.5]),
                        0.02, np.zeros(1, int), np.zeros(1, bool), np.zeros(1, bool))
        c.sysid.u_lag[:] = np.array([0.5, 0.0, 0.0])  # hold excitation
    assert abs(c.sysid.A_hat[0, 2, 0] - prior) > 1e-4


def test_swept_clear_brakes_into_skids():
    c = StagedBaseline(1)
    c.yaw.t[:] = 1.0  # identified
    v_fwd = np.array([[0.0, -0.20]])
    x = np.array([0.12])
    y = np.array([0.10])
    blocked = c._swept_clear(x, y, np.array([0.0]), v_fwd, np.array([0.0]))
    assert not blocked[0]
    far = c._swept_clear(np.array([0.0]), np.array([1.2]), np.array([0.0]), v_fwd,
                         np.array([0.0]))
    assert far[0]
    # closed loop: FINAL near skid must not drive forward
    c.phase[:] = StagedBaseline.ENTER
    c.ekf.init[:] = True
    c.ekf.x[:] = np.array([0.12, 0.10, 0.0, 0.0, 0.0])
    act = c.step(np.array([0.12, 0.10, 0.0]), True, 0.0)
    assert act[0, 1] > -0.01
