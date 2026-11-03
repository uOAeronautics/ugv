import numpy as np

from ugv.parking import Pose2
from ugv.policy import DIM, StudentPolicy


def _policy():
    # Weights are irrelevant to the shield paths; forward is stubbed where used.
    return StudentPolicy({})


def test_shield_brakes_on_stale_pose():
    act, reason = _policy().step(None, 9.0, False, 0.0, 0.0, np.zeros(3), Pose2(0, 0, 0))
    assert (act == 0).all() and reason.startswith("brake")


def test_invalid_nan_brakes():
    act, _ = _policy().step((0.1, 0.2, 0.0), 0.1, True, float("nan"), 0.0, np.zeros(3), Pose2(0, 0, 0))
    assert (act == 0).all()


def test_shield_brakes_on_blocked_sweep():
    p = _policy()
    p.forward = lambda h: np.array([1.0, 0.0, 0.0])  # drive right into skid
    est = Pose2(0.10, 0.0, 0.0)  # close to right skid at x=0.157
    act, reason = p.step((0.1, 0.0, 0.0), 0.05, True, 0.0, 0.0, np.zeros(3), est)
    assert (act == 0).all() and "blocked" in reason


def test_reset_clears_episode_state():
    p = _policy()
    p.forward = lambda h: np.zeros(3)
    p.step((0.1, 0.5, 0.0), 0.0, True, 0.0, 0.0, np.zeros(3), Pose2(0.0, 0.5, 0.0))
    assert p.hist.any()
    p.reset()
    assert not p.hist.any() and not p._prev_valid and p.hist.shape == (50, DIM)


def test_golden_trace_matches_sim():
    # 0.8: one frame spec. A recorded sim sequence replayed through the deploy
    # policy must reproduce env.hist exactly. Fixture from rl/make_policy_fixture.py.
    from pathlib import Path
    z = np.load(Path(__file__).parent / "fixtures" / "policy_trace.npz")
    weights = {k[2:]: z[k] for k in z.files if k.startswith("w_")}
    p = StudentPolicy(weights)
    n = len(z["in_valid"])
    for i in range(n):
        tag = tuple(float(v) for v in z["in_tag"][i]) if z["in_valid"][i] else None
        est = Pose2(float(z["in_est"][i][0]), float(z["in_est"][i][1]), float(z["in_est"][i][2]))
        _, reason = p.step(tag, float(z["in_age"][i]), bool(z["in_valid"][i]),
                           float(z["in_imu_w"][i]), float(z["in_imu_yaw"][i]),
                           np.array(z["in_prev_act"][i]), est)
        assert reason == "policy", f"step {i} braked: {reason}"
    # Exact match except the recursive EMA channel (cols 10-12), which differs
    # at float32-vs-float64 rounding level between torch and NumPy.
    np.testing.assert_array_equal(p.hist[:, :10], z["hist"][:, :10])
    np.testing.assert_allclose(p.hist[:, 10:], z["hist"][:, 10:], atol=1e-6)
