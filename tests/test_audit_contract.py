"""Tests for tools/l7/audit_contract.py using synthetic fixtures (no real-log deps)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "tools" / "l7" / "audit_contract.py"


def _make_sim_ref(path: Path, n_ticks: int = 200, dt: float = 0.02) -> None:
    rng = np.random.default_rng(42)
    frames = np.zeros((n_ticks, 13), dtype=np.float32)
    valid = np.array([(i % 5 in (0, 2, 3)) for i in range(n_ticks)], dtype=np.float32)
    frames[:, 0] = rng.normal(0.0, 0.03, n_ticks) * valid
    frames[:, 1] = np.linspace(0.8, 0.05, n_ticks, dtype=np.float32) * valid
    frames[:, 2] = rng.normal(0.0, 0.08, n_ticks) * valid
    age = np.zeros(n_ticks, dtype=np.float32)
    cur_age = 0.0
    for i in range(n_ticks):
        cur_age = 0.0 if valid[i] > 0.5 else cur_age + dt
        age[i] = cur_age
    frames[:, 3] = age
    frames[:, 4] = valid
    frames[:, 5] = rng.normal(0.0, 0.08, n_ticks)
    frames[:, 6] = rng.normal(0.0, 0.12, n_ticks)
    acts = rng.normal(0.0, 0.05, (n_ticks, 3)).astype(np.float32)
    acts[:, 1] = -0.08
    frames[1:, 7:10] = acts[:-1]
    ep_id = np.zeros(n_ticks, dtype=np.int32)
    pose = np.zeros((n_ticks, 3), dtype=np.float32)
    pose[:, 1] = np.linspace(0.8, 0.05, n_ticks, dtype=np.float32)
    np.savez_compressed(
        path,
        frames=frames,
        acts=acts,
        ep_id=ep_id,
        pose=pose,
        dt=np.float64(dt),
        meta_json=np.array(json.dumps({"level": "L1", "episodes": 1, "sha": "deadbeef12345678"})),
    )


def _make_profile_and_probe(tmp_path: Path, mismatch: bool = False) -> tuple[Path, Path]:
    # Build a clean profile whose steady-state matches the probe when mismatch=False.
    coef = np.zeros((16, 3), dtype=float)
    # _mix_np: fwd ([0,-1,0]) -> wheels=[-1,1,1,-1]; left ([1,0,0]) -> [-1,-1,-1,-1]
    coef[0:4, 1] = [0.25, -0.25, -0.25, 0.25]
    coef[4:8, 1] = [0.25, -0.25, -0.25, 0.25]
    coef[0:4, 0] = -0.25
    coef[4:8, 0] = -0.25
    half_base = (0.21 + 0.149) / 2.0
    w_cmd = half_base * 4.0
    coef[0:4, 2] = [-0.25 / w_cmd, -0.25 / w_cmd, 0.25 / w_cmd, 0.25 / w_cmd]
    coef[4:8, 2] = [-0.25 / w_cmd, -0.25 / w_cmd, 0.25 / w_cmd, 0.25 / w_cmd]
    profile = {
        "schema_version": 2,
        "deadzone": 0.0,
        "coefficients": coef.tolist(),
        "drive_tau_s": [0.25, 0.25, 0.25],
        "brake_tau_s": [0.12, 0.12, 0.12],
        "innovation_covariance": np.eye(3).tolist(),
        "posterior_particles": [
            {
                "deadzone": 0.0,
                "coefficients": coef.tolist(),
                "drive_tau_s": [0.25, 0.25, 0.25],
                "brake_tau_s": [0.12, 0.12, 0.12],
                "innovation_covariance": np.eye(3).tolist(),
            }
        ],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    probe_dir = tmp_path / "20261113-000000-probe-axes"
    probe_dir.mkdir()
    if mismatch:
        fit = {
            "pulses": [{"axis": "left", "duty": 0.15, "v_ss": 0.01}],
            "fit": {
                "fwd": {"gain": 2.3, "deadzone": 0.05, "unit": "m/s per duty"},
                "left": {"gain": 1.6, "deadzone": 0.13, "unit": "m/s per duty"},
                "ccw": {"gain": 900.0, "deadzone": 0.10, "unit": "deg/s per duty"},
            },
        }
    else:
        fit = {
            "pulses": [],
            "fit": {
                "fwd": {"gain": 1.0, "deadzone": 0.0, "unit": "m/s per duty"},
                "left": {"gain": 1.0, "deadzone": 0.0, "unit": "m/s per duty"},
                "ccw": {"gain": math.degrees(1.0 / w_cmd), "deadzone": 0.0, "unit": "deg/s per duty"},
            },
        }
    (probe_dir / "fit.json").write_text(json.dumps(fit))
    return profile_path, probe_dir


def _make_run(
    run_dir: Path,
    mode: str = "policy",
    noisy_gyro: bool = False,
    saturate_turn: bool = False,
    sha: str = "deadbeef12345678",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 150
    dt = 0.02
    rows = []
    for k in range(n):
        valid = k % 5 in (0, 2, 3)
        y = 0.9 - 0.002 * k
        imu_w = float(rng.normal(0.0, 0.65 if noisy_gyro else 0.08))
        rec = {
            "k": k,
            "t": 100.0 + k * dt,
            "src": "tag" if valid else "none",
            "pose": [0.0, y, 0.0] if valid else None,
            "est": [0.0, y, 0.0],
            "imu": [0.0, imu_w],
            "loop_ms": 5.0,
        }
        if mode == "policy":
            a2 = (0.95 if (k % 2 == 0) else -0.95) if saturate_turn else float(rng.normal(0.0, 0.05))
            rec["act"] = [0.0, -0.08, a2]
        rows.append(json.dumps(rec))
    (run_dir / "ticks.jsonl").write_text("\n".join(rows) + "\n")
    meta = {"run_id": run_dir.name, "args": {"mode": mode}}
    if mode == "policy":
        meta["policy"] = {"sha": sha, "stateful": False}
    (run_dir / "meta.json").write_text(json.dumps(meta))


def test_audit_passes_on_matching_synthetic_contract(tmp_path: Path) -> None:
    sim_ref = tmp_path / "sim_ref.npz"
    _make_sim_ref(sim_ref)
    profile_path, probe_dir = _make_profile_and_probe(tmp_path, mismatch=False)
    l6_dir = tmp_path / "l6"
    l6_dir.mkdir()
    (l6_dir / "ok.json").write_text(json.dumps({"sha": "deadbeef12345678", "bridge": "gazebo", "success": 15}))

    run_dir = tmp_path / "run_ok"
    _make_run(run_dir, mode="policy", noisy_gyro=False, saturate_turn=False)
    out_json = tmp_path / "report.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(run_dir),
            "--sim-ref",
            str(sim_ref),
            "--probe",
            str(probe_dir),
            "--profile",
            str(profile_path),
            "--l6-dir",
            str(l6_dir),
            "--json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out_json.read_text())
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert "channel" in proc.stdout and "OVERALL: PASS" in proc.stdout


def test_audit_fails_on_gyro_saturation_plant_and_l6_gate(tmp_path: Path) -> None:
    sim_ref = tmp_path / "sim_ref.npz"
    _make_sim_ref(sim_ref)
    profile_path, probe_dir = _make_profile_and_probe(tmp_path, mismatch=True)
    l6_dir = tmp_path / "l6"
    l6_dir.mkdir()
    (l6_dir / "fail.json").write_text(json.dumps({"sha": "deadbeef12345678", "bridge": "gazebo", "success": 0}))

    run_dir = tmp_path / "run_bad"
    _make_run(run_dir, mode="policy", noisy_gyro=True, saturate_turn=True)
    out_json = tmp_path / "bad_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(run_dir),
            "--sim-ref",
            str(sim_ref),
            "--probe",
            str(probe_dir),
            "--profile",
            str(profile_path),
            "--l6-dir",
            str(l6_dir),
            "--json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    report = json.loads(out_json.read_text())
    assert report["passed"] is False
    assert "imu_w" in report["failed_channels"]
    assert "act_2" in report["failed_channels"]
    assert "gyro_imu_w_std" in report["failed_checks"]
    assert "action_sat_act_2" in report["failed_checks"]
    assert "plant_deadzone_mismatch" in report["failed_checks"]
    assert "gazebo_l6_gate" in report["failed_checks"]


def test_audit_handles_stage_run_without_actions(tmp_path: Path) -> None:
    sim_ref = tmp_path / "sim_ref.npz"
    _make_sim_ref(sim_ref)
    run_dir = tmp_path / "run_stage"
    _make_run(run_dir, mode="stage", noisy_gyro=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(run_dir),
            "--sim-ref",
            str(sim_ref),
            "--no-probe",
            "--no-l6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "mode=stage (no act)" in proc.stdout
    assert "OVERALL: PASS" in proc.stdout
