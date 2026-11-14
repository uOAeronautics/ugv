"""30 s SysID + turn-column refit on a synthetic perturbed particle.

True plant: deadzone 0.14, yaw_scale 0.7, yaw-translation cross-coupling
(w += 0.25 * vy, ~25% drift ratio during strafe — within full-turn yaw
authority so the decoupling is physically testable). Prior: Bayesian profile mean (dz 0.0, unscaled turn
column). Pass bar: refit recovers dz / taus / yaw_scale within 8%, and
JoystickMapper eliminates yaw drift during pure strafe.
"""

import json
import math
from pathlib import Path

import numpy as np

from ugv.calibration import (
    CalibratedModel,
    ExcitationSchedule,
    JoystickMapper,
    ProfileRefitter,
    simulate_particle,
    smooth_dz_comp,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "simulation" / "gazebo" / "mecanum_bayesian_profile.json"

TRUE_DZ = 0.14
TRUE_DRIVE = 0.22
TRUE_BRAKE = 0.10
TRUE_YAW_SCALE = 0.7
COUPLE = 0.25  # w += 0.25 * vy: ~25% drift ratio on strafe, cancellable by
# full turn authority (~0.15 rad/s) so the decoupling is testable
DT = 0.02


def _prior_coef():
    prof = json.loads(PROFILE.read_text())
    parts = prof["posterior_particles"]
    return np.mean([p["coefficients"] for p in parts], axis=0)


def _run_sysid():
    C0 = _prior_coef()
    C_true = C0.copy()
    C_true[:, 2] *= TRUE_YAW_SCALE
    sched = ExcitationSchedule()
    cmds, _ = sched.to_array(DT)
    v, yaw = simulate_particle(
        cmds,
        DT,
        C_true,
        TRUE_DZ,
        TRUE_DRIVE,
        TRUE_BRAKE,
        cross_couple=COUPLE,
        noise_std=0.004,
        seed=0,
    )
    ref = ProfileRefitter(C0)
    res = ref.fit(cmds, DT, v, yaw)
    return sched, cmds, C0, C_true, res


def test_schedule_is_30s_bounded_safe():
    sched = ExcitationSchedule()
    assert abs(sched.total_s - 30.0) < 1e-6
    assert sched.max_abs_cmd() <= 0.75 + 1e-9
    assert sched.max_pulse_s() <= 0.80 + 1e-9
    cmds, labels = sched.to_array(DT)
    assert len(cmds) == int(30.0 / DT)
    text = " ".join(labels)
    assert "dz-ramp-fwd" in text and "dz-ramp-strafe" in text
    assert "dz-ramp-turn" in text and "yaw-probe" in text


def test_refit_recovers_true_parameters():
    _, _, C0, _, res = _run_sysid()
    assert abs(res.deadzone - TRUE_DZ) / TRUE_DZ < 0.08
    assert abs(res.drive_tau - TRUE_DRIVE) / TRUE_DRIVE < 0.08
    assert abs(res.brake_tau - TRUE_BRAKE) / TRUE_BRAKE < 0.08
    assert abs(res.yaw_scale - TRUE_YAW_SCALE) / TRUE_YAW_SCALE < 0.08
    # Turn-column steady-state prediction matches the coupled plant.
    assert res.n_steady > 500
    assert res.steady_rmse < 0.03
    assert C0.shape == (16, 3)


def test_turn_column_prediction_within_8pct():
    _, cmds, _, C_true, res = _run_sysid()
    from ugv.calibration import mix as _mix, wheel_features as _wf

    model = CalibratedModel(res.coef, res.deadzone)
    yaw_only = (np.abs(cmds[:, 2]) > 0.2) & (np.abs(cmds[:, :2]).max(axis=1) < 1e-9)
    pred = model.f(cmds[yaw_only])
    # true steady state (noise-free, coupling included)
    v_ss = _wf(_mix(cmds[yaw_only]), TRUE_DZ) @ C_true
    v_ss[:, 2] += COUPLE * v_ss[:, 1]
    err = np.abs(pred[:, 2] - v_ss[:, 2]) / np.maximum(np.abs(v_ss[:, 2]), 0.05)
    assert float(np.median(err)) < 0.08


def _settle(cmd, C_true):
    n = int(2.0 / DT)
    v, _ = simulate_particle(
        np.tile(cmd, (n, 1)),
        DT,
        C_true,
        TRUE_DZ,
        TRUE_DRIVE,
        TRUE_BRAKE,
        cross_couple=COUPLE,
        noise_std=0.0,
    )
    return v[-1]


def test_joystick_mapper_kills_strafe_yaw_drift():
    _, _, C0, C_true, res = _run_sysid()
    model = CalibratedModel(res.coef, res.deadzone, res.drive_tau, res.brake_tau)
    mapper = JoystickMapper(model)
    naive = np.array([0.0, 0.5, 0.0])
    v_naive = _settle(naive, C_true)
    assert abs(v_naive[2]) > 0.05  # coupling really drifts naive strafe
    assert abs(v_naive[2] / v_naive[1]) > 0.10
    cmd = mapper.map_stick(0.0, 0.5, 0.0, heading_rate_dps=0.0)
    v_map = _settle(cmd, C_true)
    assert abs(v_map[2]) < 0.2 * abs(v_naive[2])
    assert abs(v_map[2]) < 0.03  # rad/s residual
    # decoupling must preserve lateral authority around the desired
    # stick->velocity operating point (not the raw-cmd point: the mapper
    # intentionally remaps stick fraction through dz-comp + VMAX).
    des_vy = smooth_dz_comp(0.5, res.deadzone) * JoystickMapper.VMAX
    assert abs(v_map[1] - des_vy) / des_vy < 0.50
    # pure forward decouples too
    cmd_f = mapper.map_stick(0.5, 0.0, 0.0, heading_rate_dps=0.0)
    v_f = _settle(cmd_f, C_true)
    assert abs(v_f[2]) < 0.03
    des_vx = smooth_dz_comp(0.5, res.deadzone) * JoystickMapper.VMAX
    assert abs(v_f[0] - des_vx) / des_vx < 0.50
    # centred stick with gyro drift triggers yaw-hold counter-command
    hold = mapper.map_stick(0.0, 0.0, 0.0, heading_rate_dps=20.0)
    assert hold[2] < -0.01
