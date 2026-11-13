"""Non-privileged classical parking baseline: EKF + RLS system-ID + staged mecanum controller. NumPy only.

The single implementation used by the batched sim benchmark (rl/baselines.py) and the
Gazebo L6 harness (simulation/gazebo/run_policy_episodes.py). Arrays are batched over N
robots; deploy uses N=1.

Inputs per 50 Hz step: tag pose of the car in the drone frame (x, y, yaw; yaw 0 faces the
drone along -y) with a valid flag, IMU yaw rate, and the commands this controller issued.
The tag capture latency is known per robot (camera timestamps). Output: (vx, vy, w)
normalised commands in [-1, 1], the same action space as the RL student.

estimator : EKF on (x, y, yaw, bx, by) with online 3x3 RLS velocity/yaw adaptation,
            innovation-covariance Mahalanobis gating + blackout uncertainty inflation,
            and exact dt_since_valid handling for asynchronous camera rates (e.g. 30 Hz).
controller: out-of-FOV in-place yaw sweep -> gear-aware corridor funneling -> smooth
            staging alignment -> swept-clear shielded entry with lateral/yaw integral trim
            and retreat on lateral drift -> brake at the dead-reckoned goal.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROFILE = Path(__file__).resolve().parents[2] / "simulation" / "gazebo" / "mecanum_bayesian_profile.json"
HALF_BASE = (0.21 + 0.149) / 2  # mixer arm, mirrors rl/sim.py
RING = 32                       # delayed-state history, 0.64 s at 50 Hz
ENTER_V = 0.20                  # entry speed, m/s
STOP_Y = 0.025                  # dead-reckoned brake point
RETREAT_Y = 0.75                # safe re-alignment staging distance in front of drone
SWEPT_HORIZON = 0.25            # stopping-horizon safety lookahead (s)


@dataclass(frozen=True)
class Geometry:
    """Gear + footprint in the drone frame (skids along y at x = +-skid_x). Defaults are the
    rl/sim.py provisional values (config/parking.json, unverified)."""
    skid_x: float = 0.157      # skid centre line |x|
    skid_r: float = 0.03
    skid_half_len: float = 0.1524
    car_w: float = 0.16
    car_l: float = 0.22

    @property
    def margin(self):          # lateral clearance per side when centred
        return self.skid_x - self.skid_r - self.car_w / 2

    @property
    def zone_y(self):          # |y| below which the footprint overlaps the skid span
        return self.skid_half_len + self.car_l / 2 + 0.04

    @property
    def band_y(self):
        return self.zone_y + 0.02

    @property
    def stage_y(self):
        return max(0.55, self.band_y + 0.23)

    @property
    def keep_x(self):
        return max(0.45, self.skid_x + self.skid_r + 0.27)


# Gazebo drone_docking world (simulation/gazebo/drone_dimensions.json + generated car).
GAZEBO_GEOMETRY = Geometry(skid_x=0.17, skid_r=0.015, skid_half_len=0.235, car_w=0.172, car_l=0.203)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def wheel_features(wheels, deadzone):
    dz = np.clip(np.asarray(deadzone, float), 0, 0.3)[..., None]
    q = np.sign(wheels) * np.clip(np.abs(wheels) - dz, 0, None) / (1 - dz + 1e-6)
    mean_abs = np.abs(q).mean(axis=-1, keepdims=True)
    return np.concatenate([np.clip(q, 0, None), np.clip(q, None, 0), q * np.abs(q), q * mean_abs], axis=-1)


def mix(cmd):
    """(vx, vy, w) command -> normalised wheel PWM; mirrors rl/sim.py (hardware polarity)."""
    vx, vy, w = cmd[..., 0], cmd[..., 1], cmd[..., 2] * HALF_BASE * 4.0
    trans = np.stack([vx - vy, vx + vy, vx + vy, vx - vy], axis=-1)
    turn = np.stack([-w, -w, w, w], axis=-1)
    return np.clip(-trans + turn, -1, 1)


def tag_std(d):
    """Default tag noise model (sensor characterisation used by rl/sim.py)."""
    return 0.008 + 0.020 * d, math.radians(2) + math.radians(2) * d


class MeanModel:
    """Posterior-mean steady-state body velocity. Translation is well identified across
    draws; the turn column is not, so only translation is inverted."""
    LEVELS = (0, .01, .02, .035, .05, .07, .09, .12, .15, .19, .24, .3, .38, .48, .6)

    def __init__(self, profile: dict | None = None, axes: str = "rl_sim"):
        """axes: how the fitted (forward, left, ccw) columns map to the drone-frame body
        axes. "rl_sim" reproduces rl/sim.py (forward -> body +x, i.e. along the footprint
        width, 90 deg from the camera). "physical" puts forward along the camera axis
        (body -y at yaw 0, facing the drone) and left on body +x, as on the real car."""
        prof = profile or json.loads(PROFILE.read_text())
        if axes not in ("rl_sim", "physical"):
            raise ValueError("axes must be rl_sim or physical")
        self.axes = axes
        self.coef = np.mean([p["coefficients"] for p in prof["posterior_particles"]], axis=0)
        self.dz = float(prof["deadzone"])
        self.tau = float(np.mean(prof["drive_tau_s"]))
        self.brake_tau = float(np.mean(prof["brake_tau_s"]))
        lv = np.array(self.LEVELS)
        lv = np.concatenate([-lv[::-1][:-1], lv])
        gx, gy = np.meshgrid(lv, lv, indexing="ij")
        self.grid = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
        self.grid_v = self.f(self.grid)[:, :2]

    def f(self, cmd):
        out = wheel_features(mix(cmd), self.dz) @ self.coef
        if self.axes == "physical":
            out = np.stack([out[..., 1], -out[..., 0], out[..., 2]], axis=-1)
        return out

    def inv(self, vb):
        """Translation command (turn 0) whose steady state is nearest the body (vx, vy)."""
        d = ((vb[:, None, :] - self.grid_v[None]) ** 2).sum(-1)
        return self.grid[d.argmin(1)].copy()

    def linearize(self, eps: float = 0.2) -> np.ndarray:
        """3x3 Jacobian of steady-state (vx_b, vy_b, w_b) wrt (ux, uy, uw)."""
        A = np.zeros((3, 3), dtype=float)
        for j in range(3):
            e = np.zeros((2, 3), dtype=float)
            e[0, j] = eps
            e[1, j] = -eps
            y = self.f(e)
            A[:, j] = (y[0] - y[1]) / (2.0 * eps)
        return A


class OnlineSysID:
    """Online 3x3 forgetting-factor RLS mapping lagged command u_lag -> (vx_b, vy_b, w_b).

    Rows 0..1 (vx_b, vy_b) update from valid tag pairs over exact dt_since_valid using
    a matched low-pass filter on both finite-differenced velocity and delayed lagged
    commands (suppressing tag differentiation noise by >20x). Row 2 (w_b) updates at
    every active step from the instantaneous IMU gyro. Ridge regularization toward A_lin
    keeps A_hat bounded; determinant/condition-number guards live at inversion time.
    """
    GAMMA = 0.985
    LAM_XY = 0.35
    LAM_W = 0.15
    TAU_FILT = 0.15
    DET_MIN = 0.03
    COND_MAX = 45.0

    def __init__(self, n: int, model: MeanModel):
        self.n = n
        self.tau = float(model.tau)
        self.brake_tau = float(model.brake_tau)
        self.A_lin = model.linearize()
        self.A_hat = np.broadcast_to(self.A_lin, (n, 3, 3)).copy()
        self.u_lag = np.zeros((n, 3))
        self.u_ring = np.zeros((n, RING, 3))
        self.head = 0
        self.S_uu_xy = np.zeros((n, 3, 3))
        self.S_eu_xy = np.zeros((n, 2, 3))
        self.S_uu_w = np.zeros((n, 3, 3))
        self.S_eu_w = np.zeros((n, 1, 3))
        self.v_xy_filt = np.zeros((n, 2))
        self.u_del_filt = np.zeros((n, 3))
        self.prev_tag = np.zeros((n, 3))
        self.has_prev = np.zeros(n, bool)
        self.dt_since_valid = np.zeros(n)

    def step_lag(self, cmd: np.ndarray, dt: float):
        braking = np.abs(cmd).max(1) < 1e-8
        tau = np.where(braking, self.brake_tau, self.tau)
        a = np.exp(-dt / np.maximum(tau, 1e-3))[:, None]
        self.u_lag = a * self.u_lag + (1.0 - a) * cmd
        self.u_ring[:, self.head % RING] = self.u_lag
        self.head += 1
        self.dt_since_valid += dt

    def _well_conditioned(self, A: np.ndarray) -> np.ndarray:
        det = np.linalg.det(A)
        sv = np.linalg.svd(A, compute_uv=False)
        cond = sv[:, 0] / np.maximum(sv[:, 2], 1e-9)
        finite = np.all(np.isfinite(A), axis=(1, 2))
        return finite & (np.abs(det) > self.DET_MIN) & (cond < self.COND_MAX)

    def observe(self, tag: np.ndarray, valid: np.ndarray, gyro: np.ndarray, dt: float,
                lag: np.ndarray, ekf_ok: np.ndarray, ekf_reinit: np.ndarray):
        # Row 2 (yaw rate + translation-to-yaw coupling) from instantaneous gyro
        u_cur = self.u_lag
        act_w = np.linalg.norm(u_cur, axis=1) > 0.03
        if act_w.any():
            e_w = gyro - (u_cur @ self.A_lin[2])
            uu = u_cur[:, :, None] * u_cur[:, None, :]
            eu = e_w[:, None, None] * u_cur[:, None, :]
            mw = act_w[:, None, None]
            self.S_uu_w = np.where(mw, self.GAMMA * self.S_uu_w + uu, self.S_uu_w)
            self.S_eu_w = np.where(mw, self.GAMMA * self.S_eu_w + eu, self.S_eu_w)
            Gw = self.S_uu_w + self.LAM_W * np.eye(3)[None]
            dAw = np.linalg.solve(Gw, self.S_eu_w.transpose(0, 2, 1)).transpose(0, 2, 1)
            A_w = np.clip(self.A_lin[2:3][None] + dAw, -6.0, 6.0)
            self.A_hat[:, 2:3, :] = np.where(mw, A_w, self.A_hat[:, 2:3, :])

        # Reset matched velocity filter on discontinuities / blackout / reinit
        reset_filt = ekf_reinit | (self.dt_since_valid > 0.16)
        if reset_filt.any():
            self.v_xy_filt = np.where(reset_filt[:, None], 0.0, self.v_xy_filt)
            self.u_del_filt = np.where(reset_filt[:, None], 0.0, self.u_del_filt)

        # Rows 0..1 (body velocity + yaw-to-translation coupling) from valid tag pairs
        pair = valid & self.has_prev & ekf_ok & (self.dt_since_valid > 1e-4) & (self.dt_since_valid <= 0.16)
        if pair.any():
            ds = np.maximum(self.dt_since_valid, 1e-4)
            v_w = (tag[:, :2] - self.prev_tag[:, :2]) / ds[:, None]
            th_mid = wrap(self.prev_tag[:, 2] + 0.5 * wrap(tag[:, 2] - self.prev_tag[:, 2]))
            c, s = np.cos(th_mid), np.sin(th_mid)
            v_xy = np.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1]], axis=1)
            mid_lag = np.clip(lag + np.round(0.5 * ds / max(dt, 1e-4)).astype(int) - 1, 0, RING - 1)
            idx_u = (self.head - 1 - mid_lag) % RING
            u_del = self.u_ring[np.arange(self.n), idx_u]

            # Matched first-order low-pass on both v_xy and u_del using exact dt_since_valid
            af = np.exp(-ds / self.TAU_FILT)[:, None]
            v_f = af * self.v_xy_filt + (1.0 - af) * v_xy
            u_f = af * self.u_del_filt + (1.0 - af) * u_del
            self.v_xy_filt = np.where(pair[:, None], v_f, self.v_xy_filt)
            self.u_del_filt = np.where(pair[:, None], u_f, self.u_del_filt)

            e_xy = self.v_xy_filt - (self.u_del_filt @ self.A_lin[:2].T)
            good = (pair
                    & (np.linalg.norm(self.v_xy_filt, axis=1) < 1.5)
                    & (np.linalg.norm(e_xy, axis=1) < 1.0)
                    & (np.linalg.norm(self.u_del_filt, axis=1) > 0.03))
            if good.any():
                uu = self.u_del_filt[:, :, None] * self.u_del_filt[:, None, :]
                eu = e_xy[:, :, None] * self.u_del_filt[:, None, :]
                mg = good[:, None, None]
                self.S_uu_xy = np.where(mg, self.GAMMA * self.S_uu_xy + uu, self.S_uu_xy)
                self.S_eu_xy = np.where(mg, self.GAMMA * self.S_eu_xy + eu, self.S_eu_xy)
                Gxy = self.S_uu_xy + self.LAM_XY * np.eye(3)[None]
                dAxy = np.linalg.solve(Gxy, self.S_eu_xy.transpose(0, 2, 1)).transpose(0, 2, 1)
                A_xy = np.clip(self.A_lin[:2][None] + dAxy, -6.0, 6.0)
                self.A_hat[:, :2, :] = np.where(mg, A_xy, self.A_hat[:, :2, :])

        # Advance prev_tag only on accepted or newly initialized valid frames (skip outliers)
        upd_prev = valid & (ekf_ok | ekf_reinit | ~self.has_prev)
        if upd_prev.any():
            self.prev_tag = np.where(upd_prev[:, None], tag, self.prev_tag)
            self.has_prev |= upd_prev
            self.dt_since_valid = np.where(upd_prev, 0.0, self.dt_since_valid)


class EKF:
    CHI2_GATE = 16.3  # 3-DOF 99.9% Mahalanobis threshold

    def __init__(self, n, model, latency, dt=0.02):
        self.m, self.n = model, n
        self.x = np.zeros((n, 5))
        self.v = np.zeros((n, 3))
        self.P = np.tile(np.diag([1e3, 1e3, 1e3, 0.05 ** 2, 0.05 ** 2]), (n, 1, 1))
        self.init = np.zeros(n, bool)
        self.age = np.full(n, 1e9)
        self.ring = np.zeros((n, RING, 5))
        self.head = 0
        self.lag = np.clip(np.round(np.asarray(latency, float) / dt), 0, RING - 1).astype(int) * np.ones(n, int)
        self.rejected = np.zeros(n)
        self.H = np.eye(3, 5)
        self.last_ok = np.zeros(n, bool)
        self.last_reinit = np.zeros(n, bool)

    @property
    def pose(self):
        return self.x[:, :3]

    @property
    def pos_std(self):
        s = np.sqrt(np.clip(np.maximum(self.P[:, 0, 0], self.P[:, 1, 1]), 0, None))
        return np.where(self.init, s, 9.0)

    def predict(self, cmd, gyro, dt, A_hat: np.ndarray | None = None, A_lin: np.ndarray | None = None):
        target = self.m.f(cmd)
        if A_hat is not None and A_lin is not None:
            target = target + ((A_hat - A_lin[None]) @ cmd[:, :, None])[..., 0]
        braking = np.abs(cmd).max(1) < 1e-8
        tau = np.where(braking, self.m.brake_tau, self.m.tau)
        a = np.exp(-dt / tau)[:, None]
        self.v = a * self.v + (1 - a) * target
        moving = (np.linalg.norm(self.v[:, :2], axis=1) > 0.005) | (np.abs(cmd).max(1) > 0.02)
        mv = moving.astype(float)
        vb = self.v[:, :2] + self.x[:, 3:5] * mv[:, None]
        th = self.x[:, 2] + gyro * dt / 2
        c, s = np.cos(th), np.sin(th)
        x = self.x.copy()
        x[:, 0] += dt * (c * vb[:, 0] - s * vb[:, 1])
        x[:, 1] += dt * (s * vb[:, 0] + c * vb[:, 1])
        x[:, 2] = wrap(self.x[:, 2] + gyro * dt)
        F = np.tile(np.eye(5), (self.n, 1, 1))
        F[:, 0, 2] = dt * (-s * vb[:, 0] - c * vb[:, 1])
        F[:, 1, 2] = dt * (c * vb[:, 0] - s * vb[:, 1])
        F[:, 0, 3] = dt * c * mv; F[:, 0, 4] = -dt * s * mv
        F[:, 1, 3] = dt * s * mv; F[:, 1, 4] = dt * c * mv
        spd = np.linalg.norm(self.v[:, :2], axis=1)
        qb = (0.02 + 0.3 * np.abs(cmd[:, 2])) ** 2 * dt  # bias walks faster while turning
        prolonged = (self.init & (self.age > 0.5)).astype(float)
        q_pos_infl = prolonged * ((0.25 + 0.45 * spd) ** 2) * dt
        q_yaw_infl = prolonged * ((0.20 + 0.15 * np.abs(gyro)) ** 2) * dt
        q = np.stack([(0.15 * spd * dt) ** 2 + (0.01 * dt) ** 2 + q_pos_infl,
                      (0.15 * spd * dt) ** 2 + (0.01 * dt) ** 2 + q_pos_infl,
                      0.002 ** 2 * dt + (0.05 * np.abs(gyro) * dt) ** 2 + q_yaw_infl, qb, qb], 1)
        self.P = F @ self.P @ F.transpose(0, 2, 1) + q[:, :, None] * np.eye(5)
        self.x = x
        self.age += dt
        self.ring[:, self.head % RING] = self.x
        self.head += 1

    def update(self, tag, valid, std=None):
        if std is None:
            ps, ys = tag_std(np.linalg.norm(tag[:, :2], axis=1))
        else:
            ps, ys = std
        R = np.stack([ps ** 2, ps ** 2, ys ** 2], 1)[:, :, None] * np.eye(3)
        idx = (self.head - 1 - self.lag) % RING
        past = self.ring[np.arange(self.n), idx, :3]
        innov = tag - past
        innov[:, 2] = wrap(innov[:, 2])
        H = self.H
        S = H @ self.P @ H.T + R
        Sinv_i = np.linalg.solve(S, innov[:, :, None])[..., 0]
        m2 = (innov * Sinv_i).sum(1)
        ok = valid & self.init & (m2 < self.CHI2_GATE)
        self.rejected += (valid & self.init & ~ok)
        K = np.linalg.solve(S, H @ self.P).transpose(0, 2, 1)  # [n,5,3]
        # yaw from gyro + tag yaw only: position innovations after dead reckoning would
        # otherwise rotate the heading through the linearised cross-covariance
        K[:, 2, :2] = 0
        K[:, :2, 2] = 0
        corr = np.where(ok[:, None], (K @ innov[:, :, None])[..., 0], 0.0)
        self.x = self.x + corr
        self.x[:, 2] = wrap(self.x[:, 2])
        self.ring = self.ring + corr[:, None, :]
        IKH = np.eye(5) - K @ H
        Pn = IKH @ self.P @ IKH.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
        self.P = np.where(ok[:, None, None], Pn, self.P)
        reinit = (valid & ~self.init) | (valid & (self.rejected >= 6))
        if reinit.any():
            xi = np.concatenate([tag, np.zeros((self.n, 2))], 1)
            self.x = np.where(reinit[:, None], xi, self.x)
            self.ring = np.where(reinit[:, None, None], xi[:, None, :], self.ring)
            P0 = np.zeros((self.n, 5, 5))
            P0[:, :3, :3] = R
            P0[:, 3, 3] = P0[:, 4, 4] = 0.05 ** 2
            self.P = np.where(reinit[:, None, None], P0, self.P)
        self.rejected = np.where(reinit | ok, 0.0, self.rejected)
        self.init |= valid
        self.age = np.where(ok | reinit, 0.0, self.age)
        self.last_ok = ok
        self.last_reinit = reinit
        return ok


class AdaptiveYaw:
    """gyro ~ b+ max(u,0) + b- min(u,0) + kx ux + ky uy at steady command.

    Probe (0.8 s): +0.3 turn then -0.3; b+/b- = mean gyro over the last 0.15 s of each half
    divided by the command, so the estimate does not depend on the plant's response lag.
    Afterwards NLMS refines on steady-command samples only (|u - lowpass(u)| < STEADY)."""
    PROBE = 0.8
    WINDOW = 0.15
    STEADY = 0.08
    LIM = np.array([4.0, 4.0, 2.0, 2.0])

    def __init__(self, n, tau=0.25):
        self.th = np.zeros((n, 4))
        self.uf = np.zeros((n, 3))
        self.tau = tau
        self.t = np.full(n, -1.0)  # probe clock; < 0 = not started
        self.acc = np.zeros((n, 2))
        self.cnt = np.zeros((n, 2))

    def start(self, mask):
        self.t = np.where(mask & (self.t < 0), 0.0, self.t)

    def probing(self):
        return (self.t >= 0) & (self.t < self.PROBE)

    def identified(self):
        return self.t >= self.PROBE

    def probe_cmd(self):
        c = np.zeros((len(self.t), 3))
        c[:, 2] = np.where(self.t < self.PROBE / 2, 0.3, -0.3)
        return c

    def observe(self, cmd, gyro, dt):
        h = self.PROBE / 2
        # probe windows: last WINDOW seconds of each half (t is the time the command was issued)
        for k, (lo, hi) in enumerate(((h - self.WINDOW, h), (self.PROBE - self.WINDOW, self.PROBE))):
            inw = (self.t >= lo) & (self.t < hi)
            self.acc[:, k] += np.where(inw, gyro, 0.0)
            self.cnt[:, k] += inw
        a = math.exp(-dt / self.tau)
        self.uf = a * self.uf + (1 - a) * cmd
        nt = np.where(self.t >= 0, self.t + dt, self.t)
        done_probe = (self.t < self.PROBE) & (nt >= self.PROBE)
        if done_probe.any():
            mean = self.acc / np.maximum(self.cnt, 1)
            self.th[:, 0] = np.where(done_probe, mean[:, 0] / 0.3, self.th[:, 0])
            self.th[:, 1] = np.where(done_probe, mean[:, 1] / -0.3, self.th[:, 1])
        self.t = nt
        steady = self.identified() & (np.abs(cmd - self.uf).max(1) < self.STEADY)
        uw = cmd[:, 2]
        phi = np.stack([np.clip(uw, 0, None), np.clip(uw, None, 0), cmd[:, 0], cmd[:, 1]], 1)
        e = gyro - (phi * self.th).sum(1)
        # turning: refine the active b only; no turn: refine the cross terms only
        mask = np.where((np.abs(uw) > 0.08)[:, None], [1., 1., 0., 0.],
                        np.where((np.abs(uw) < 0.03)[:, None], [0., 0., 1., 1.], 0.0))
        ph = phi * mask * steady[:, None]
        self.th = np.clip(self.th + 0.03 * e[:, None] * ph / ((ph ** 2).sum(1, keepdims=True) + 1e-3),
                          -self.LIM, self.LIM)

    def turn(self, w_des, ux, uy, lim=0.6):
        r = w_des - self.th[:, 2] * ux - self.th[:, 3] * uy
        bp, bn = self.th[:, 0], self.th[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(np.abs(bp) > 0.15, r / bp, 0.0)
            un = np.where(np.abs(bn) > 0.15, r / bn, 0.0)
        up = np.where(up > 0, up, 0.0)
        un = np.where(un < 0, un, 0.0)
        u = np.where(((np.abs(bp) >= np.abs(bn)) & (up > 0)) | (un == 0), up, un)
        # no identified channel turns the needed way: excite an unidentified one
        need = (u == 0) & (np.abs(r) > 0.05)
        explore = np.where(np.abs(bp) < 0.15, 0.3, np.where(np.abs(bn) < 0.15, -0.3, 0.0))
        return np.clip(np.where(need, explore, u), -lim, lim)


class StagedBaseline:
    """Phases: 0 search, 1 approach, 2 enter, 3 hold (braked), 4 back out."""
    SEARCH, APPROACH, ENTER, HOLD, BACK = range(5)
    ALIGN, FINAL = APPROACH, ENTER  # phase aliases

    def __init__(self, n=1, latency=0.08, dt=0.02, model: MeanModel | None = None, v_max=0.25,
                 geometry: Geometry = Geometry()):
        self.n, self.dt, self.v_max = n, dt, v_max
        self.m = model or MeanModel()
        self.g = geometry
        self.latency = latency
        self.reset()

    def reset(self):
        n = self.n
        self.ekf = EKF(n, self.m, self.latency, self.dt)
        self.sysid = OnlineSysID(n, self.m)
        self.yaw = AdaptiveYaw(n, self.m.tau)
        self.phase = np.zeros(n, int)
        self.settle = np.zeros(n)
        self.ix = np.zeros(n)
        self.iyaw = np.zeros(n)
        self._seen_tag = np.zeros(n, bool)
        self._retreat = np.zeros(n, bool)
        self._search_t = np.zeros(n)
        self._last_gyro = np.zeros(n)
        self.cmd = np.zeros((n, 3))

    def step(self, tag, valid, gyro, std=None):
        """One 50 Hz step. tag [n,3] (ignored where not valid), valid [n], gyro [n]."""
        tag = np.asarray(tag, float).reshape(self.n, 3)
        valid = np.asarray(valid, bool).reshape(self.n)
        gyro = np.asarray(gyro, float).reshape(self.n)
        prev = np.clip(self.cmd, -1, 1)
        self.sysid.step_lag(prev, self.dt)
        self.ekf.predict(prev, gyro, self.dt, A_hat=self.sysid.A_hat, A_lin=self.sysid.A_lin)
        clean_tag = np.where(valid[:, None], tag, 0.0)
        self.ekf.update(clean_tag, valid, std)
        self.sysid.observe(clean_tag, valid, gyro, self.dt, self.ekf.lag,
                           self.ekf.last_ok, self.ekf.last_reinit)
        self.yaw.observe(prev, gyro, self.dt)
        self._seen_tag |= valid
        self._last_gyro = gyro
        self.cmd = self._control()
        return self.cmd.copy()

    def _inv_kinematics(self, vb: np.ndarray, w_des: np.ndarray) -> np.ndarray:
        """Invert adapted 3x3 A_hat with determinant/condition-number guards and fallback."""
        A = self.sysid.A_hat
        det3 = np.linalg.det(A)
        sv3 = np.linalg.svd(A, compute_uv=False)
        cond3 = sv3[:, 0] / np.maximum(sv3[:, 2], 1e-9)
        ok3 = np.all(np.isfinite(A), axis=(1, 2)) & (np.abs(det3) > 0.03) & (cond3 < 45.0)

        A2 = A[:, :2, :2]
        det2 = A2[:, 0, 0] * A2[:, 1, 1] - A2[:, 0, 1] * A2[:, 1, 0]
        sv2 = np.linalg.svd(A2, compute_uv=False)
        cond2 = sv2[:, 0] / np.maximum(sv2[:, 1], 1e-9)
        ok2 = np.all(np.isfinite(A2), axis=(1, 2)) & (np.abs(det2) > 0.05) & (cond2 < 30.0)

        A_safe3 = np.where(ok3[:, None, None], A, np.eye(3)[None])
        rhs3 = np.concatenate([vb, w_des[:, None]], axis=1)
        cmd3 = np.linalg.solve(A_safe3, rhs3[:, :, None])[..., 0]

        cmd_fb = self.m.inv(vb)
        w_fb = self.yaw.turn(w_des, cmd_fb[:, 0], cmd_fb[:, 1])
        use_rls_w = ok3 & (np.abs(A[:, 2, 2]) > 0.10) & self.yaw.identified()
        w_cmd = np.where(use_rls_w, np.clip(cmd3[:, 2], -0.6, 0.6), w_fb)
        w_cmd = np.where(self.yaw.identified(), w_cmd, 0.0)

        vb_net = vb - np.where(ok2[:, None], A[:, :2, 2] * w_cmd[:, None], 0.0)
        A_safe2 = np.where(ok2[:, None, None], A2, np.eye(2)[None])
        xy_rls = np.linalg.solve(A_safe2, vb_net[:, :, None])[..., 0]
        xy_cmd = np.where(ok2[:, None], xy_rls, cmd_fb[:, :2])

        # Preserve translation direction on saturation
        max_xy = np.maximum(np.abs(xy_cmd).max(axis=1, keepdims=True), 1.0)
        xy_cmd = xy_cmd / max_xy
        return np.concatenate([xy_cmd, w_cmd[:, None]], axis=1)

    def _swept_clear(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                     v_w: np.ndarray, gyro: np.ndarray, pad: np.ndarray | float = 0.004) -> np.ndarray:
        """Stopping-horizon (~0.25 s) swept-footprint collision check against skids."""
        g = self.g
        hw, hl = g.car_w / 2.0, g.car_l / 2.0
        pad_arr = np.asarray(pad, float)[:, None] if np.ndim(pad) > 0 else float(pad)
        # Blend current estimated world velocity and proposed reference over horizon
        c0, s0 = np.cos(yaw), np.sin(yaw)
        vb_est = self.ekf.v[:, :2]
        vw_est = np.stack([c0 * vb_est[:, 0] - s0 * vb_est[:, 1],
                           s0 * vb_est[:, 0] + c0 * vb_est[:, 1]], axis=1)
        v_pred = 0.5 * vw_est + 0.5 * v_w
        ts = np.linspace(0.05, SWEPT_HORIZON, 5)[None, :]  # [1, 5]
        xk = x[:, None] + v_pred[:, 0:1] * ts
        yk = y[:, None] + v_pred[:, 1:2] * ts
        yawk = yaw[:, None] + gyro[:, None] * ts
        ca, sa = np.abs(np.cos(yawk)), np.abs(np.sin(yawk))
        eff_hw = hw * ca + hl * sa
        eff_hl = hw * sa + hl * ca
        y_overlap = np.abs(yk) < (g.skid_half_len + eff_hl + g.skid_r + pad_arr)
        x_hit = (np.abs(xk) + eff_hw + pad_arr) > (g.skid_x - g.skid_r)
        return ~np.any(y_overlap & x_hit, axis=1)

    def _control(self):
        est, dt, g = self.ekf, self.dt, self.g
        STAGE_Y, KEEP_X, M = g.stage_y, g.keep_x, g.margin
        p = est.pose
        x, y, yaw = p[:, 0], p[:, 1], p[:, 2]
        dist_drone = np.hypot(x, y)

        # Prolonged total loss far from target -> search sweep; short blackout -> hold still in APPROACH
        prolonged_loss = (~self._seen_tag) | (~est.init) | ((est.age > 3.5) & (dist_drone > 0.45))
        blackout_wait = self._seen_tag & est.init & (est.age > 0.5) & (est.age <= 3.5)

        ph = self.phase.copy()
        ph = np.where(~self._seen_tag | ~est.init, self.SEARCH, ph)
        ph = np.where((ph == self.SEARCH) & self._seen_tag & est.init, self.APPROACH, ph)
        ph = np.where(prolonged_loss & (ph != self.HOLD), self.SEARCH, ph)

        # Retreat staging height: 0.75 m after lateral drift retreat, else normal STAGE_Y
        self._retreat = np.where((ph == self.SEARCH) | (ph == self.ENTER) | (ph == self.HOLD),
                                 False, self._retreat)
        stage_y = np.where(self._retreat, max(STAGE_Y, RETREAT_Y), STAGE_Y)
        self._retreat = np.where((ph == self.APPROACH) & (y >= max(STAGE_Y, RETREAT_Y) - 0.04),
                                 False, self._retreat)

        # Gear-aware corridor funneling waypoints
        side = np.where(x >= 0, 1.0, -1.0)
        in_front = y > stage_y - 0.08
        zero = np.zeros_like(x)
        stage = np.stack([zero, stage_y], 1)
        lane_up = np.stack([side * KEEP_X, np.maximum(stage_y + 0.05, STAGE_Y + 0.05)], 1)
        out_side = np.stack([side * KEEP_X, y], 1)
        to_centre = np.stack([zero, y], 1)
        band = np.abs(y) < g.band_y
        tgt = np.where((in_front | (y > g.band_y + 0.02))[:, None], stage,
              np.where((np.abs(x) >= KEEP_X - 0.05)[:, None], lane_up,
              np.where((band & (np.abs(x) < 0.75 * M))[:, None], stage,
              np.where((band & (np.abs(x) < g.skid_x))[:, None], to_centre, out_side))))
        err = tgt - p[:, :2]
        dist = np.linalg.norm(err, axis=1)
        face = -np.arctan2(x, np.where(np.abs(y) < 1e-3, 1e-3, y))
        yaw_ref = np.where(in_front & (dist < 0.12), 0.0, face)

        # Smooth ALIGN/APPROACH exit tolerances + leaky settle hysteresis
        lat_tol = min(0.025, 0.45 * M)
        at_stage = ((ph == self.APPROACH) & in_front
                    & (np.abs(x) < lat_tol)
                    & (np.abs(y - stage_y) < 0.04)
                    & (np.abs(yaw) < math.radians(4.5))
                    & (est.pos_std < 0.045)
                    & (est.age < 0.35))
        near_stage = ((ph == self.APPROACH) & in_front
                      & (np.abs(x) < 0.65 * M)
                      & (np.abs(y - stage_y) < 0.07)
                      & (np.abs(yaw) < math.radians(7.0)))
        self.settle = np.where(at_stage, self.settle + dt,
                      np.where(near_stage, np.maximum(0.0, self.settle - 0.5 * dt), 0.0))
        ph = np.where(self.settle >= 0.24, self.ENTER, ph)

        # Retreat FINAL -> APPROACH (y = 0.75 m) if |x| drifts dangerously while tags visible,
        # or BACK -> APPROACH if already inside the skid zone
        enter = ph == self.ENTER
        tags_fresh = est.age < 0.30
        drift_outside = enter & tags_fresh & (y >= g.zone_y) & (np.abs(x) > 0.64 * M)
        drift_inside = enter & (y < g.zone_y) & (np.abs(x) > 0.64 * M)
        ph = np.where(drift_outside, self.APPROACH, ph)
        self._retreat |= drift_outside
        ph = np.where(drift_inside, self.BACK, ph)
        back_done = (ph == self.BACK) & (y > max(STAGE_Y, RETREAT_Y) - 0.02)
        ph = np.where(back_done, self.APPROACH, ph)
        self._retreat |= drift_inside
        # Slot-margin gating: never enter/hold FINAL at/above the lateral slot
        # margin (|x_est| >= M). Closes the stale-tag hole where drift_outside
        # cannot fire (age >= 0.30 s) and EKF trim hides the true offset.
        over_margin = (ph == self.ENTER) & (np.abs(x) >= M)
        ph = np.where(over_margin, self.APPROACH, ph)
        self._retreat |= over_margin

        enter = ph == self.ENTER
        # Lateral and yaw integral trim during FINAL (frozen during under-drone blind phase)
        self.ix = np.where(enter, np.where(est.age < 0.25, np.clip(self.ix + x * dt, -0.025, 0.025), self.ix), 0.0)
        self.iyaw = np.where(enter, np.where(est.age < 0.25, np.clip(self.iyaw + wrap(-yaw) * dt, -0.08, 0.08), self.iyaw), 0.0)
        ph = np.where(enter & (y < STOP_Y), self.HOLD, ph)
        enter = ph == self.ENTER
        back = ph == self.BACK
        self.phase = ph

        # World velocity reference
        vmax = np.where(dist > 0.15, self.v_max, 0.12)
        v_w = err * (np.minimum(dist * 1.5, vmax) / np.maximum(dist, 1e-6))[:, None]
        # Corridor funneling: prioritize lateral centering when approaching the staging axis
        funnel_x = np.where((ph == self.APPROACH) & in_front & (np.abs(x) > 0.35 * M),
                            np.clip(-2.2 * x, -0.15, 0.15), v_w[:, 0])
        v_w[:, 0] = funnel_x

        # Entry velocity with lateral integral trim & funnel speed scaling
        lat_scale = np.clip(1.0 - (np.abs(x) / max(0.65 * M, 1e-3)) ** 2, 0.25, 1.0)
        ent_v = np.stack([np.clip(-4.0 * x - 4.0 * self.ix, -0.07, 0.07),
                          -ENTER_V * lat_scale], 1)
        ent_v[:, 1] = np.where((y > g.zone_y) & (np.abs(x) > 0.35 * M), 0.0, ent_v[:, 1])
        v_w = np.where(enter[:, None], ent_v, v_w)

        back_v = np.stack([np.clip(-2.0 * x, -0.04, 0.04), np.full_like(y, 0.08)], 1)
        v_w = np.where(back[:, None], back_v, v_w)

        # Swept-clear shield in APPROACH / FINAL (ENTER)
        pad = np.where(enter, 0.002, 0.006)
        clear = self._swept_clear(x, y, yaw, v_w, self._last_gyro, pad=pad)
        shield = (~clear) & (enter | (ph == self.APPROACH))
        if shield.any():
            # Brake forward motion into skids and steer laterally toward safe center
            safe_vx = np.where((y < g.zone_y) & (np.abs(x) > 0.6 * M),
                               np.clip(-3.0 * x, -0.08, 0.08),
                               np.where(np.abs(x) < g.skid_x, np.clip(-2.5 * x, -0.10, 0.10), side * 0.10))
            safe_vy = np.where(y < g.zone_y + 0.08, 0.04, 0.0)
            v_w = np.where(shield[:, None], np.stack([safe_vx, safe_vy], 1), v_w)

        # During short blackout in APPROACH, hold translation still while keeping yaw pointed at drone
        hold_blackout = blackout_wait & (ph == self.APPROACH)
        v_w = np.where(hold_blackout[:, None], 0.0, v_w)

        yaw_ref = np.where(enter | back | (ph == self.HOLD), 0.0, yaw_ref)
        c, s = np.cos(yaw), np.sin(yaw)
        vb = np.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1]], 1)
        ye = wrap(yaw_ref - yaw)
        big = np.abs(ye) > math.radians(25)
        # Smooth yaw proportional + integral trim in FINAL (ENTER)
        w_enter = np.clip(2.4 * ye + 2.0 * self.iyaw, -1.0, 1.0)
        w = np.where(enter, w_enter, np.clip(2.5 * ye, -1.2, 1.2))
        # Large yaw error: mostly rotate first (keeps the tag in view)
        vb = np.where((big & ~enter & ~back)[:, None], vb * 0.3, vb)

        cmd = self._inv_kinematics(vb, w)

        # L4 Out-of-FOV / prolonged total-loss search: safe in-place yaw sweep (w only, zero translation)
        search = (ph == self.SEARCH)
        self._search_t = np.where(search, self._search_t + dt, 0.0)
        # Sweep +0.75 rad/s first 1.8 s (~77 deg), then -0.75 rad/s continuously to cover full 360 deg quickly
        sweep_dir = np.where(self._search_t < 1.8, 1.0, -1.0)
        spin_w = self.yaw.turn(0.75 * sweep_dir, zero, zero)
        spin = np.zeros_like(cmd)
        spin[:, 2] = np.where(np.abs(spin_w) < 0.15, 0.3 * sweep_dir, spin_w)
        cmd = np.where(search[:, None], spin, cmd)

        # Probe the yaw channel once the drone is within 1.2 m or immediately when searching
        self.yaw.start((est.init & (dist_drone < 1.2)) | ~self._seen_tag | ~est.init)
        cmd[:, 2] = np.where(self.yaw.identified(), cmd[:, 2], 0.0)
        cmd = np.where(self.yaw.probing()[:, None], self.yaw.probe_cmd(), cmd)
        return np.where((ph == self.HOLD)[:, None], 0.0, cmd)
