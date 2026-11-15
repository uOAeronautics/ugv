"""Known-answer tests for the Observatory joint whitened-KDE overlap gate."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.observatory import contract


def test_identical_distributions_overlap_high():
    rng = np.random.default_rng(3)
    sim = rng.normal(size=(3000, 4))
    real = rng.normal(size=(3000, 4))  # independent draws, same distribution
    r = contract.joint_overlap(real, sim)
    assert r["overlap"] > 0.8
    assert r["pass"] is True
    assert r["channels"] == contract.JOINT_CHANNELS
    assert r["dropped"] == []
    assert 0.0 <= r["overlap"] <= 1.0
    assert "kde" in r["method"].lower()


def test_mean_shift_3sigma_overlap_below_threshold():
    rng = np.random.default_rng(4)
    sim = rng.normal(size=(3000, 4))
    real = sim + np.array([3.0, 0.0, 0.0, 0.0])  # 3-sigma shift on one channel
    r = contract.joint_overlap(real, sim)
    assert r["overlap"] < contract.JOINT_OVERLAP_MIN == 0.35
    assert r["pass"] is False


def test_correlation_flip_caught_while_marginals_identical():
    # Flip the sign of a strong correlation: every marginal (mean, std) is
    # unchanged in distribution, but the joint law differs, so a marginal-only
    # contract passes while the joint overlap must drop well below 1.
    rng = np.random.default_rng(5)
    n, rho = 6000, 0.9
    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    sim = np.column_stack([x, rho * x + np.sqrt(1 - rho**2) * eps])
    real = np.column_stack([x, -rho * x + np.sqrt(1 - rho**2) * eps])
    for j in range(2):
        assert abs(real[:, j].mean() - sim[:, j].mean()) < 0.05
        assert abs(real[:, j].std() - sim[:, j].std()) < 0.05
    r = contract.joint_overlap(real, sim, channels=["a", "b"])
    assert r["overlap"] < 0.8


def test_channels_below_min_samples_are_dropped():
    rng = np.random.default_rng(6)
    real = rng.normal(size=(100, 4))
    sim = rng.normal(size=(100, 4))
    real[:, 3] = np.nan
    real[:10, 3] = 0.5  # only 10 finite real samples on loop_ms_s
    sim[:, 2] = np.nan
    sim[:15, 2] = 0.1  # only 15 finite sim samples on imu_w_rad_s
    r = contract.joint_overlap(real, sim)
    assert r["dropped"] == ["imu_w_rad_s", "loop_ms_s"]
    assert r["channels"] == ["inter_tag_dt_s", "tag_age_s"]
    assert r["overlap"] is not None and r["pass"] is not None


def test_fewer_than_two_channels_left_is_unknown_not_fail():
    rng = np.random.default_rng(7)
    real = rng.normal(size=(100, 4))
    sim = rng.normal(size=(100, 4))
    real[:, 1:] = np.nan
    sim[:, 1:] = np.nan
    r = contract.joint_overlap(real, sim)
    assert r["overlap"] is None
    assert r["pass"] is None
    assert r["channels"] == ["inter_tag_dt_s"]
    assert len(r["dropped"]) == 3


def test_scattered_nans_keep_rows_aligned():
    # NaNs on different rows per channel: per-channel filtering would pair
    # channel-0 event i with channel-1 event i from a different row. The
    # result must equal the overlap of the complete-case submatrix.
    rng = np.random.default_rng(10)
    sim = rng.normal(size=(300, 3))
    real = sim + 0.3
    real[5, 0] = np.nan
    real[9, 1] = np.nan
    sim[7, 0] = np.nan
    sim[11, 2] = np.nan
    r = contract.joint_overlap(real, sim, channels=["a", "b", "c"])
    rm = np.isfinite(real).all(axis=1)
    sm = np.isfinite(sim).all(axis=1)
    ref = contract.joint_overlap(real[rm], sim[sm], channels=["a", "b", "c"])
    assert r["dropped"] == []
    assert r["overlap"] == ref["overlap"]


def test_overlap_is_deterministic():
    rng = np.random.default_rng(8)
    sim = rng.normal(size=(500, 4))
    real = rng.normal(size=(500, 4)) + 0.5
    a = contract.joint_overlap(real, sim)
    b = contract.joint_overlap(real, sim)
    assert a["overlap"] == b["overlap"]


def test_whiten_singular_sim_falls_back_to_diagonal():
    rng = np.random.default_rng(9)
    # Huge scale on one channel plus a constant channel: the regularized
    # covariance is numerically singular, so diagonal whitening must kick in.
    sim = np.column_stack([rng.normal(size=200) * 1e5, np.zeros(200)])
    real = np.column_stack([rng.normal(size=200) * 1e5, rng.normal(size=200) * 0.1])
    rw, sw = contract.whiten_by_sim(real, sim)
    assert np.all(np.isfinite(rw)) and np.all(np.isfinite(sw))
    assert np.allclose(sw.mean(axis=0), 0.0, atol=1e-9)


def test_verdict_fail_when_joint_fails():
    report = {
        "verdict": "PASS",
        "checks": [
            {"name": "valid_pose_rate", "pass": True},
            {"name": contract.JOINT_CHECK_NAME, "pass": False},
        ],
    }
    assert contract.verdict_of(report) == "FAIL"


def test_verdict_pass_when_joint_passes_and_unknown_is_not_fail():
    checks = [{"name": "valid_pose_rate", "pass": True}]
    joint_pass = {"name": contract.JOINT_CHECK_NAME, "pass": True}
    assert contract.verdict_of({"checks": checks + [joint_pass]}) == "PASS"
    joint_unknown = {"name": contract.JOINT_CHECK_NAME, "pass": None}
    assert contract.verdict_of({"checks": checks + [joint_unknown]}) == "INFO"


def test_real_joint_samples_builder(tmp_path):
    rows = []
    for i in range(40):
        src = "tag" if i % 4 == 0 else ("dr" if i % 4 == 2 else "none")
        row = {
            "t": 100.0 + 0.02 * i,
            "src": src,
            "imu": [0.1, 0.01 * i],
            "loop_ms": 2.0 + 0.1 * i,
        }
        if i == 8:
            row["imu_w_fed"] = 0.5
        rows.append(row)
    (tmp_path / "ticks.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = contract.real_joint_samples(tmp_path)
    assert out.shape == (9, 4)  # 10 tags -> 9 inter-tag intervals
    assert np.allclose(out[:, 0], 0.08)
    assert np.allclose(out[:, 1], 0.02)  # peak age: tick before the tag, after the dr fix
    assert out[1, 2] == 0.5  # imu_w_fed wins over raw imu at tick 8
    assert out[0, 2] == 0.04  # raw imu[1] at tick 4
    assert np.allclose(out[:, 3], (2.0 + 0.1 * np.arange(4, 40, 4)) / 1000.0)


def test_sim_joint_samples_builder(tmp_path):
    frames = np.zeros((12, 13))
    frames[:, 4] = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    frames[:, 3] = np.arange(12) * 0.01  # tag_age
    frames[:, 5] = np.arange(12) * 0.001  # imu_w
    ep_id = np.array([0] * 6 + [1] * 6)
    ref = tmp_path / "ref.npz"
    np.savez(ref, frames=frames, ep_id=ep_id, dt=0.02)
    out = contract.sim_joint_samples(ref)
    assert out.shape == (4, 4)  # 2 valid-index pairs per episode
    assert np.allclose(out[:, 0], [0.06, 0.04, 0.04, 0.06])
    assert np.allclose(out[:, 3], 0.02)  # nominal loop budget
    assert out[0, 1] == 0.02 and out[0, 2] == 0.003  # age before arrival, imu at arrival
