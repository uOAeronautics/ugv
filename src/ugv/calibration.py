"""30-second real-car + sim excitation and Bayesian/RLS profile refit.

Standalone pure-NumPy module (stdlib + numpy only). Mirrors the feature/mixer
math of :mod:`ugv.staged_baseline` (``wheel_features`` / ``mix`` /
``MeanModel``) without importing it, so the Pi can run SysID from a log file
with no torch, no web server, and no simulator.

Workflow (see ``docs/L7_PREFLIGHT.md`` section 8)::

    python3 -m ugv.calibration --emit-csv /tmp/sysid_cmds.csv
    # drive the car (or sim) through the CSV at 50 Hz, logging
    # (cmd_vx,vy,w ; dt ; v_body ; imu_yaw_rate_dps) per step
    python3 -m ugv.calibration --fit-csv /tmp/sysid_log.csv \\
        --prior simulation/gazebo/mecanum_bayesian_profile.json \\
        --out simulation/gazebo/mecanum_bayesian_profile.json

The excitation schedule is bounded and safe: every pulse is <= 0.80 s,
|cmd| <= 0.75, total wall time exactly 30.0 s. It covers

(a) breakaway deadzone ramps (0.05..0.30 in 0.25 s steps) on forward /
    strafe / turn independently,
(b) forward + strafe step pulses at 3 magnitudes (0.35/0.55/0.75, 0.8 s
    on + 0.4 s brake) to fit ``drive_tau`` / ``brake_tau`` and the
    translational columns of the 16x3 ``coef`` matrix, and
(c) CW/CCW yaw probes (+/-0.35, +/-0.55, 0.6 s on + 0.4 s off) to excite
    the uncalibrated turn column (``yaw_scale`` + ``coef[:, 2]``) using
    IMU ``heading_rate`` plus ArUco pose deltas.

``ProfileRefitter`` ridge-regresses the 16x3 coefficients toward the Bayesian
prior mean, so unexcited terms stay bounded instead of blowing up.
``JoystickMapper`` adds smooth deadzone pre-compensation, full 3x3
body-velocity decoupling (cancels yaw-translation cross-coupling during pure
strafe/forward), and closed-loop gyro yaw-hold when ``turn == 0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass

import numpy as np

HALF_BASE = (0.21 + 0.149) / 2  # mixer arm, mirrors rl/sim.py + staged_baseline
LEVELS = (0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15, 0.19, 0.24, 0.3, 0.38, 0.48, 0.6)
SCHEDULE_TOTAL_S = 30.0
PULSE_DT = 0.02  # 50 Hz execution


# --------------------------------------------------------------------------
# Feature / mixer math (bit-identical to staged_baseline, duplicated so this
# module stays dependency-free).
# --------------------------------------------------------------------------


def wheel_features(wheels, deadzone):
    """(..., 4) wheel PWM -> (..., 16) nonlinear features."""
    dz = np.clip(np.asarray(deadzone, float), 0, 0.3)[..., None]
    q = np.sign(wheels) * np.clip(np.abs(wheels) - dz, 0, None) / (1 - dz + 1e-6)
    mean_abs = np.abs(q).mean(axis=-1, keepdims=True)
    return np.concatenate(
        [np.clip(q, 0, None), np.clip(q, None, 0), q * np.abs(q), q * mean_abs], axis=-1
    )


def mix(cmd):
    """(..., 3) (vx, vy, w) command -> (..., 4) normalised wheel PWM."""
    cmd = np.asarray(cmd, float)
    vx, vy, w = cmd[..., 0], cmd[..., 1], cmd[..., 2] * HALF_BASE * 4.0
    trans = np.stack([vx - vy, vx + vy, vx + vy, vx - vy], axis=-1)
    turn = np.stack([-w, -w, w, w], axis=-1)
    return np.clip(-trans + turn, -1, 1)


# --------------------------------------------------------------------------
# Excitation schedule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcitationSegment:
    cmd: tuple  # (vx, vy, w) in [-1, 1]
    duration: float  # seconds
    label: str


class ExcitationSchedule:
    """Bounded safe 30 s SysID pulse sequence (50 Hz, 250 ms quantum)."""

    FWD_MAGS = (0.35, 0.55, 0.75)
    STRAFE_MAGS = (0.35, 0.55, 0.75)
    YAW_MAGS = (0.35, -0.35, 0.55, -0.55)
    RAMP_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    STEP_ON_S = 0.80
    STEP_OFF_S = 0.40
    YAW_ON_S = 0.60
    YAW_OFF_S = 0.40
    RAMP_STEP_S = 0.25

    def __init__(self, segments: list | None = None):
        self.segments = list(segments) if segments is not None else self.build_default()

    @classmethod
    def _steps(cls) -> list:
        segs: list = []
        # (a) breakaway deadzone ramps, one axis at a time.
        axes = [((1.0, 0.0, 0.0), "fwd"), ((0.0, 1.0, 0.0), "strafe"), ((0.0, 0.0, 1.0), "turn")]
        for unit, name in axes:
            for lv in cls.RAMP_LEVELS:
                segs.append(
                    ExcitationSegment(
                        (unit[0] * lv, unit[1] * lv, unit[2] * lv),
                        cls.RAMP_STEP_S,
                        f"dz-ramp-{name}-{lv:.2f}",
                    )
                )
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), 0.50, f"dz-settle-after-{name}"))
        # (b) forward + strafe step pulses at 3 magnitudes (tau + columns 0..1).
        for mag in cls.FWD_MAGS:
            segs.append(ExcitationSegment((mag, 0.0, 0.0), cls.STEP_ON_S, f"step-fwd-{mag:.2f}"))
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), cls.STEP_OFF_S, "brake-settle"))
        for mag in cls.STRAFE_MAGS:
            segs.append(ExcitationSegment((0.0, mag, 0.0), cls.STEP_ON_S, f"step-strafe-{mag:.2f}"))
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), cls.STEP_OFF_S, "brake-settle"))
        # (c) CW/CCW yaw probes (turn column + yaw_scale).
        for mag in cls.YAW_MAGS:
            segs.append(ExcitationSegment((0.0, 0.0, mag), cls.YAW_ON_S, f"yaw-probe-{mag:+.2f}"))
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), cls.YAW_OFF_S, "brake-settle"))
        # (d) repeat of (b) for cross-coupling excitation (pure strafe/fwd
        # long holds let yaw-translation coupling accumulate measurably).
        for mag in cls.FWD_MAGS:
            segs.append(ExcitationSegment((mag, 0.0, 0.0), cls.STEP_ON_S, f"step-fwd2-{mag:.2f}"))
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), cls.STEP_OFF_S, "brake-settle"))
        for mag in cls.STRAFE_MAGS:
            segs.append(
                ExcitationSegment((0.0, mag, 0.0), cls.STEP_ON_S, f"step-strafe2-{mag:.2f}")
            )
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), cls.STEP_OFF_S, "brake-settle"))
        return segs

    @classmethod
    def build_default(cls) -> list:
        segs = cls._steps()
        total = sum(s.duration for s in segs)
        pad = SCHEDULE_TOTAL_S - total
        if pad < -1e-9:
            raise ValueError(f"schedule overshoots 30 s: {total:.3f} s")
        if pad > 1e-9:
            segs.append(ExcitationSegment((0.0, 0.0, 0.0), pad, "final-settle-pad"))
        return segs

    @property
    def total_s(self) -> float:
        return sum(s.duration for s in self.segments)

    def max_abs_cmd(self) -> float:
        return max(max(abs(c) for c in s.cmd) for s in self.segments)

    def max_pulse_s(self) -> float:
        return max(s.duration for s in self.segments if any(s.cmd))

    def to_array(self, dt: float = PULSE_DT):
        """Expand to (cmds[N,3], labels[N]) at fixed dt.

        Segment boundaries are rounded cumulatively so the output is
        exactly ``round(total_s / dt)`` steps with no drift.
        """
        n_total = int(round(self.total_s / dt))
        bounds = [0]
        cum = 0.0
        for s in self.segments:
            cum += s.duration
            bounds.append(int(round(cum / dt)))
        bounds[-1] = n_total
        cmds = np.zeros((n_total, 3))
        labels: list = []
        for s, (a, b) in zip(self.segments, zip(bounds[:-1], bounds[1:], strict=True), strict=True):
            n = max(b - a, 1 if s.duration > 0 else 0)
            b = a + n
            cmds[a:b] = np.array(s.cmd, float)
            labels += [s.label] * n
        return cmds[: bounds[-1]], labels

    def save_csv(self, path: str, dt: float = PULSE_DT) -> int:
        cmds, labels = self.to_array(dt)
        t = np.arange(len(cmds)) * dt
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "cmd_vx", "cmd_vy", "cmd_w", "label"])
            for i in range(len(cmds)):
                w.writerow(
                    [
                        f"{t[i]:.3f}",
                        f"{cmds[i, 0]:.4f}",
                        f"{cmds[i, 1]:.4f}",
                        f"{cmds[i, 2]:.4f}",
                        labels[i],
                    ]
                )
        return len(cmds)

    @staticmethod
    def load_csv(path: str):
        cmds = []
        with open(path) as f:
            for row in csv.DictReader(f):
                cmds.append([float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_w"])])
        return np.array(cmds, float)


# --------------------------------------------------------------------------
# Calibrated forward model (posterior-mean steady state + first-order lag)
# --------------------------------------------------------------------------


class CalibratedModel:
    """Steady-state body velocity + grid inverse, like MeanModel."""

    def __init__(
        self, coef, deadzone: float = 0.0, drive_tau: float = 0.25, brake_tau: float = 0.12
    ):
        self.coef = np.asarray(coef, float).reshape(16, 3)
        self.dz = float(deadzone)
        self.drive_tau = float(drive_tau)
        self.brake_tau = float(brake_tau)
        lv = np.array(LEVELS)
        lv = np.concatenate([-lv[::-1][:-1], lv])
        gx, gy = np.meshgrid(lv, lv, indexing="ij")
        self.grid = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
        self.grid_v = self.f(self.grid)[:, :2]

    def f(self, cmd):
        return wheel_features(mix(cmd), self.dz) @ self.coef

    def inv(self, vb):
        """Nearest translation grid point (turn 0) for body (vx, vy)."""
        vb = np.asarray(vb, float).reshape(-1, 2)
        d = ((vb[:, None, :] - self.grid_v[None]) ** 2).sum(-1)
        return self.grid[d.argmin(1)].copy()

    def linearize(self, eps: float = 0.2) -> np.ndarray:
        A = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros((2, 3))
            e[0, j] = eps
            e[1, j] = -eps
            y = self.f(e)
            A[:, j] = (y[0] - y[1]) / (2.0 * eps)
        return A


def simulate_particle(
    cmds,
    dt: float,
    coef,
    deadzone: float,
    drive_tau: float,
    brake_tau: float,
    cross_couple: float = 0.0,
    noise_std: float = 0.0,
    yaw_bias_dps: float = 0.0,
    seed: int = 0,
):
    """First-order-lag synthetic car. Returns (v_body[N,3], yaw_dps[N]).

    ``cross_couple`` adds ``w += cross_couple * vy_ss`` (yaw-translation
    cross-coupling, the pure-strafe yaw-drift failure mode).
    """
    rng = np.random.default_rng(seed)
    coef = np.asarray(coef, float).reshape(16, 3)
    cmds = np.asarray(cmds, float)
    v = np.zeros(3)
    out = np.zeros_like(cmds)
    for i, u in enumerate(cmds):
        v_ss = wheel_features(mix(u)[None], deadzone)[0] @ coef
        v_ss = v_ss.copy()
        v_ss[2] += cross_couple * v_ss[1]
        tau = brake_tau if np.abs(u).max() < 1e-8 else drive_tau
        a = math.exp(-dt / max(tau, 1e-3))
        v = a * v + (1 - a) * v_ss
        out[i] = v
    if noise_std > 0:
        out = out + rng.normal(0, noise_std, out.shape)
    yaw = out[:, 2] * 180.0 / math.pi + yaw_bias_dps
    return out, yaw


# --------------------------------------------------------------------------
# Profile refit
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    deadzone: float
    drive_tau: float
    brake_tau: float
    coef: np.ndarray  # (16, 3)
    yaw_scale: float
    cross_couple: float  # yaw-translation coupling gain (w += c * vy)
    steady_rmse: float
    n_steady: int


def _find_runs(cmds):
    """Contiguous constant-command runs: list of (start, end, cmd)."""
    runs = []
    start = 0
    for i in range(1, len(cmds) + 1):
        if i == len(cmds) or np.abs(cmds[i] - cmds[start]).max() > 1e-9:
            runs.append((start, i, cmds[start].copy()))
            start = i
    return runs


def _weighted_ridge(Phi, V, W, C0, lam: float):
    """Weighted ridge toward prior: (Phi'W Phi + lam I) C = Phi'W V + lam C0."""
    G = (Phi * W[:, None]).T @ Phi + lam * np.eye(Phi.shape[1])
    B = (Phi * W[:, None]).T @ V + lam * C0
    return np.linalg.solve(G, B)


def _ridge_fit(Phi, V, C0, lam: float):
    """Ridge toward prior: (Phi'Phi + lam I) C = Phi'V + lam C0."""
    return _weighted_ridge(Phi, V, np.ones(len(Phi)), C0, lam)


def _fit_tau_from_steps(t, v, mode: str) -> float:
    """Grid-search first-order tau with a free asymptote.

    Rise: ``v = A (1 - exp(-t/tau))``; fall: ``v = A exp(-t/tau)``. The
    amplitude ``A`` is solved per tau so a not-fully-settled reference
    cannot bias the time constant.
    """
    taus = np.arange(0.05, 0.61, 0.01)
    best, best_err = 0.25, np.inf
    for tau in taus:
        p = (1 - np.exp(-t / tau)) if mode == "rise" else np.exp(-t / tau)
        denom = float(p @ p)
        if denom < 1e-9:
            continue
        A = float(p @ v) / denom
        err = float(((A * p - v) ** 2).mean())
        if err < best_err:
            best, best_err = tau, err
    return float(best)


class ProfileRefitter:
    """Fit dz / taus / 16x3 coef from logged excitation data.

    Inputs: ``cmds`` [N,3], ``dt`` (float or [N]), ``v_meas`` [N,3] body
    velocity (from ArUco pose deltas), ``yaw_dps`` [N] IMU gyro (deg/s;
    averaged with ``v_meas[:, 2]`` when both present).
    """

    def __init__(
        self,
        prior_coef,
        prior_dz: float = 0.0,
        prior_drive_tau: float = 0.25,
        prior_brake_tau: float = 0.12,
        ridge_lam: float = 20.0,
    ):
        self.C0 = np.asarray(prior_coef, float).reshape(16, 3)
        self.prior_dz = float(prior_dz)
        self.prior_drive_tau = float(prior_drive_tau)
        self.prior_brake_tau = float(prior_brake_tau)
        self.lam = float(ridge_lam)

    @classmethod
    def from_profile(cls, profile: dict, ridge_lam: float = 20.0):
        parts = profile.get("posterior_particles") or []
        if parts:
            C0 = np.mean([p["coefficients"] for p in parts], axis=0)
            dz = float(np.mean([p.get("deadzone", 0.0) for p in parts]))
            dtau = profile.get("drive_tau_s", [0.25])
            btau = profile.get("brake_tau_s", [0.12])
        else:
            C0 = np.asarray(profile["coefficients"], float)
            dz = float(profile.get("deadzone", 0.0))
            dtau = profile.get("drive_tau_s", [0.25])
            btau = profile.get("brake_tau_s", [0.12])
        dtau = float(np.mean(dtau))
        btau = float(np.mean(btau))
        return cls(C0, dz, dtau, btau, ridge_lam)

    def fit(self, cmds, dt, v_meas, yaw_dps=None) -> FitResult:
        cmds = np.asarray(cmds, float).reshape(-1, 3)
        v = np.asarray(v_meas, float).reshape(-1, 3).copy()
        if yaw_dps is not None:
            yaw = np.asarray(yaw_dps, float).ravel() * math.pi / 180.0
            has_gyro = np.isfinite(yaw)
            v[has_gyro, 2] = np.where(
                np.abs(v[has_gyro, 2]) > 1e-9, 0.5 * (v[has_gyro, 2] + yaw[has_gyro]), yaw[has_gyro]
            )
        dt_arr = (
            np.full(len(cmds), float(dt)) if np.ndim(dt) == 0 else np.asarray(dt, float).ravel()
        )
        dt_med = float(np.median(dt_arr))
        runs = _find_runs(cmds)

        # 1) tau fits from step onsets/offsets, run-bounded and v0-aware
        # (dz-free: every curve is self-normalised).
        rises, brakes = [], []
        for idx, (a, b, u) in enumerate(runs):
            prev = runs[idx - 1][2] if idx > 0 else np.zeros(3)
            v0 = v[a - 1] if a > 0 else np.zeros(3)
            n = b - a
            if np.abs(u).max() > 0.20 and np.abs(prev).max() < 0.05 and n * dt_med >= 0.5:
                ax = int(np.argmax(np.abs(u)))
                seg = v[a:b, ax]
                t = (np.arange(n) + 1) * dt_med
                ref = np.mean(seg[-max(3, int(0.15 / dt_med)) :])
                if abs(ref - v0[ax]) > 0.03:
                    rises.append(_fit_tau_from_steps(t, (seg - v0[ax]) / (ref - v0[ax]), "rise"))
            if np.abs(u).max() < 0.05 and np.abs(prev).max() > 0.20 and n * dt_med >= 0.3:
                ax = int(np.argmax(np.abs(prev)))
                seg = v[a:b, ax]
                t = (np.arange(n) + 1) * dt_med
                if abs(v0[ax]) > 0.03:
                    brakes.append(_fit_tau_from_steps(t, seg / v0[ax], "fall"))
        drive_tau = float(np.median(rises)) if rises else self.prior_drive_tau
        brake_tau = float(np.median(brakes)) if brakes else self.prior_brake_tau

        # 2) inverse-lag steady-state estimates with trust weights.
        # v(t) = v_ss + (v0 - v_ss) e^{-t/tau} ->
        # v_ss = (v - v0 e^{-t/tau}) / (1 - e^{-t/tau}), weight (1-e^-)^2.
        Pest, Vest, W = [], [], []
        for a, b, u in runs:
            tau_eff = brake_tau if np.abs(u).max() < 1e-8 else drive_tau
            v0 = v[a - 1] if a > 0 else np.zeros(3)
            for k in range(2, b - a):  # skip switch transients
                t = (k + 1) * dt_med
                e = math.exp(-t / max(tau_eff, 1e-3))
                alpha = 1.0 - e
                Pest.append(u)
                Vest.append((v[a + k] - v0 * e) / alpha)
                W.append(alpha**2)
        Pest = np.array(Pest)
        Vest = np.array(Vest)
        W = np.array(W)

        # 3) deadzone grid search + ridge toward prior on lag-corrected data.
        # Translational columns: plain weighted ridge. Turn column: the
        # task model is C[:,2] = s*C0[:,2] + c*C1[:,1] + D, i.e. a scalar
        # yaw_scale s on the prior turn column, an explicit yaw-translation
        # cross-coupling gain c on the fitted lateral column (vy_hat), and
        # a small residual D (ridge-penalised). (+/-) yaw probes are nearly
        # collinear in feature space and strafe pulses carry large coupling
        # but small prior-turn response, so a free 16-dim turn-column fit
        # (or an (s, D) split without c) trades shared gain against D and
        # mis-predicts one probe polarity. vy_hat and P both lie in
        # span(Phi); the D-ridge selects the minimum-norm residual, and
        # the true plant (D = 0) is exactly recovered.
        best_dz, best_rmse = self.prior_dz, np.inf
        best_C, best_s, best_c = self.C0.copy(), 1.0, 0.0
        P_eye = np.eye(16)
        for dz in np.arange(0.0, 0.251, 0.005):
            Phi = wheel_features(mix(Pest), dz)
            C01 = _weighted_ridge(Phi, Vest[:, :2], W, self.C0[:, :2], self.lam)
            P = Phi @ self.C0[:, 2]
            Y = Phi @ C01[:, 1]
            WP = Phi * W[:, None]
            pP, pY = W * P, W * Y
            G18 = np.zeros((18, 18))
            G18[0, 0] = float(P @ pP)
            G18[1, 1] = float(Y @ pY)
            G18[0, 1] = G18[1, 0] = float(P @ pY)
            G18[0, 2:] = P @ WP
            G18[1, 2:] = Y @ WP
            G18[2:, 0] = G18[0, 2:]
            G18[2:, 1] = G18[1, 2:]
            # Weak ridge on the coupling residual D only: the true D = 0
            # needs no shrinkage, but unexcited dims must stay bounded.
            G18[2:, 2:] = WP.T @ Phi + 2.0 * P_eye
            b18 = np.zeros(18)
            wV2 = W * Vest[:, 2]
            b18[0] = float(P @ wV2)
            b18[1] = float(Y @ wV2)
            b18[2:] = WP.T @ Vest[:, 2]
            try:
                x18 = np.linalg.solve(G18, b18)
            except np.linalg.LinAlgError:
                continue
            s, c, D = float(x18[0]), float(x18[1]), x18[2:]
            C = np.column_stack([C01, s * self.C0[:, 2] + c * C01[:, 1] + D])
            r = Phi @ C - Vest
            rmse = float(np.sqrt((W[:, None] * r**2).sum() / (3 * W.sum())))
            if rmse < best_rmse:
                best_dz, best_rmse, best_C = float(dz), rmse, C
                best_s, best_c = s, c
        yaw_scale = float(np.clip(best_s, 0.2, 3.0))

        return FitResult(
            deadzone=best_dz,
            drive_tau=drive_tau,
            brake_tau=brake_tau,
            coef=best_C,
            yaw_scale=yaw_scale,
            cross_couple=best_c,
            steady_rmse=best_rmse,
            n_steady=int(len(W)),
        )

    def apply_to_profile(self, profile: dict, res: FitResult) -> dict:
        """Return an updated profile dict (mean + particles nudged)."""
        out = dict(profile)
        out["deadzone"] = res.deadzone
        out["coefficients"] = res.coef.tolist()
        out["drive_tau_s"] = [res.drive_tau] * 3
        out["brake_tau_s"] = [res.brake_tau] * 3
        out.setdefault("calibration", {})["sysid_30s"] = {
            "deadzone": res.deadzone,
            "drive_tau_s": res.drive_tau,
            "brake_tau_s": res.brake_tau,
            "yaw_scale": res.yaw_scale,
            "cross_couple": res.cross_couple,
            "steady_rmse": res.steady_rmse,
            "n_steady": res.n_steady,
        }
        return out


# --------------------------------------------------------------------------
# Joystick mapping: smooth dz pre-comp + 3x3 decoupling + gyro yaw-hold
# --------------------------------------------------------------------------


def smooth_dz_comp(u: float, dz: float, eps: float = 0.04) -> float:
    """Deadzone pre-compensation with a smoothstep blend over [dz, dz+eps]."""
    a = abs(float(u))
    if a <= dz:
        return 0.0
    full = (a - dz) / (1 - dz + 1e-9)
    if a >= dz + eps:
        return math.copysign(full, u)
    s = (a - dz) / eps
    return math.copysign(full * s * s * (3 - 2 * s), u)


class JoystickMapper:
    """Stick (fwd, strafe, turn in [-1,1]) -> (vx, vy, w) cmd.

    * smooth deadzone pre-compensation per channel,
    * translation via the model grid inverse (``CalibratedModel.inv``,
      the ``MeanModel.inv`` analogue): the desired body velocity is met
    * turn via exact steady-state inversion along the turn axis, so
      yaw-translation cross-coupling is cancelled during pure
      strafe/forward instead of merely ignored,
    * closed-loop gyro yaw-hold: when the raw turn stick is centred,
      ``w_des = -kp * yaw_rate`` damps residual drift.
    """

    VMAX = 0.25  # m/s at full stick (ENTER_V)
    TURN_EPS = 0.03
    YAW_HOLD_KP = 0.5

    def __init__(self, model: CalibratedModel):
        self.m = model
        wmax = max(abs(model.f([[0, 0, 1]])[0, 2]), abs(model.f([[0, 0, -1]])[0, 2]))
        self.WMAX = float(max(wmax, 0.05))
        self.YAW_HOLD_MAX = 0.5 * self.WMAX

    def turn_cmd_for(self, cmd_xy, w_des: float) -> float:
        """Turn command whose steady-state yaw is nearest ``w_des``.

        Bracketed bisection on the (continuous) model yaw curve: robust
        to deadzone flats and feature wiggles that make dense-grid argmin
        jump to spurious minima. Falls back to the best-effort endpoint
        when ``w_des`` is outside turn authority.
        """
        cmd_xy = np.asarray(cmd_xy, float).reshape(3)

        def err(w: float) -> float:
            c = cmd_xy.copy()
            c[2] = w
            return float(self.m.f(c[None])[0, 2]) - w_des

        lo, hi, flo, fhi = -1.0, 1.0, err(-1.0), err(1.0)
        if flo * fhi > 0:  # no blanket bracket: scan for a crossing
            ws = np.linspace(-1.0, 1.0, 21)
            ys = [err(w) for w in ws]
            for i in range(len(ws) - 1):
                if ys[i] == 0.0:
                    return float(ws[i])
                if ys[i] * ys[i + 1] < 0.0:
                    lo, hi, flo = ws[i], ws[i + 1], ys[i]
                    break
            else:
                return float(ws[int(np.argmin(np.abs(ys)))])
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            fm = err(mid)
            if flo * fm <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    def map_stick(
        self, fwd: float, strafe: float, turn: float, heading_rate_dps: float = 0.0
    ) -> np.ndarray:
        f = smooth_dz_comp(fwd, self.m.dz)
        s = smooth_dz_comp(strafe, self.m.dz)
        cmd_xy = self.m.inv([[f * self.VMAX, s * self.VMAX]])[0]
        if abs(turn) < self.TURN_EPS:
            w_des = -self.YAW_HOLD_KP * math.radians(heading_rate_dps)
            w_des = float(np.clip(w_des, -self.YAW_HOLD_MAX, self.YAW_HOLD_MAX))
        else:
            w_des = smooth_dz_comp(turn, self.m.dz) * self.WMAX
        w_cmd = self.turn_cmd_for(cmd_xy, w_des)
        return np.clip([cmd_xy[0], cmd_xy[1], w_cmd], -1, 1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="30 s SysID excitation + refit")
    ap.add_argument("--emit-csv", default=None, help="write excitation commands to CSV")
    ap.add_argument(
        "--fit-csv",
        default=None,
        help="log CSV with cmd_vx,vy,w + v_vx,vy,w (+ yaw_dps, dt_s optional)",
    )
    ap.add_argument("--prior", default=None, help="prior profile JSON")
    ap.add_argument("--out", default=None, help="updated profile JSON")
    ap.add_argument(
        "--demo-sim", action="store_true", help="run synthetic-plant demo (dz=0.14, yaw_scale=0.7)"
    )
    ap.add_argument("--dt", type=float, default=PULSE_DT)
    a = ap.parse_args(argv)

    sched = ExcitationSchedule()
    if a.emit_csv:
        n = sched.save_csv(a.emit_csv, a.dt)
        print(f"wrote {n} steps ({sched.total_s:.2f} s) to {a.emit_csv}")
    if a.demo_sim:
        prior = (
            json.load(open(a.prior))
            if a.prior
            else {"coefficients": np.zeros((16, 3)).tolist(), "deadzone": 0.0}
        )
        C0 = np.asarray(prior["coefficients"] if "coefficients" in prior else prior, float)
        cmds, _ = sched.to_array(a.dt)
        v, yaw = simulate_particle(
            cmds,
            a.dt,
            C0 * np.array([1, 1, 0.7]),
            0.14,
            0.22,
            0.10,
            cross_couple=0.6,
            noise_std=0.004,
            seed=0,
        )
        res = ProfileRefitter(C0).fit(cmds, a.dt, v, yaw)
        print(
            f"dz={res.deadzone:.3f} drive={res.drive_tau:.3f} "
            f"brake={res.brake_tau:.3f} yaw_scale={res.yaw_scale:.3f} "
            f"rmse={res.steady_rmse:.4f} n={res.n_steady}"
        )
    if a.fit_csv:
        if not a.prior or not a.out:
            raise SystemExit("--fit-csv needs --prior and --out")
        rows = list(csv.DictReader(open(a.fit_csv)))
        cmds = np.array([[float(r["cmd_vx"]), float(r["cmd_vy"]), float(r["cmd_w"])] for r in rows])
        v = np.array([[float(r["v_vx"]), float(r["v_vy"]), float(r["v_w"])] for r in rows])
        yaw = np.array([float(r.get("yaw_dps", "nan")) for r in rows])
        dt = np.array([float(r.get("dt_s", a.dt)) for r in rows])
        prof = json.load(open(a.prior))
        ref = ProfileRefitter.from_profile(prof)
        res = ref.fit(cmds, dt, v, yaw)
        json.dump(ref.apply_to_profile(prof, res), open(a.out, "w"), indent=1)
        print(
            f"dz={res.deadzone:.3f} drive={res.drive_tau:.3f} "
            f"brake={res.brake_tau:.3f} yaw_scale={res.yaw_scale:.3f} "
            f"rmse={res.steady_rmse:.4f} n={res.n_steady} -> {a.out}"
        )
    if not a.emit_csv and not a.demo_sim and not a.fit_csv:
        print(
            f"default schedule: {len(sched.segments)} segments, "
            f"{sched.total_s:.2f} s, max|cmd|={sched.max_abs_cmd():.2f}, "
            f"max pulse={sched.max_pulse_s():.2f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
