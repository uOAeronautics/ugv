"""Holonomic smooth-follow policy with velocity lead and coast/search fallback.

Legacy bang-bang pivots (FollowSteering) stay intact for compatibility.
This module blends simultaneous forward/strafe/turn with IMU yaw-rate
damping and a smooth deadzone, plus a CoastSearch state machine:

    TRACK -> COAST (<=0.6s exponential decay along lead)
          -> SEARCH (gentle bounded yaw scan <=1.2s)
          -> STOP
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_deadzone(error: float, deadzone: float = 0.04) -> float:
    """Continuous deadzone: 0 inside dz, linearly rescaled outside."""
    magnitude = abs(error)
    if magnitude <= deadzone:
        return 0.0
    span = max(1e-6, 1.0 - deadzone)
    return math.copysign((magnitude - deadzone) / span, error)


@dataclass
class TargetState:
    x: float = 0.0  # lateral error, normalized [-1, 1] (camera frame)
    z: float = 0.0  # range in metres
    vx: float = 0.0  # lateral velocity (norm units / s)
    vz: float = 0.0  # range rate (m / s)
    sigma: float = 0.05  # position uncertainty (m)
    updated_at: float = 0.0


@dataclass
class AlphaBetaLead:
    """Constant-velocity alpha-beta filter with lead prediction + sigma growth."""

    alpha: float = 0.70
    beta: float = 0.12
    predict_s: float = 0.15
    base_sigma_m: float = 0.02
    sigma_growth_per_s: float = 0.10
    max_sigma_m: float = 0.50
    state: TargetState = field(default_factory=TargetState)
    _init: bool = False

    def reset(self) -> None:
        self.state = TargetState()
        self._init = False

    def update(self, x: float, z: float, now: float) -> TargetState:
        st = self.state
        if not self._init:
            self.state = TargetState(
                x=x, z=z, vx=0.0, vz=0.0, sigma=self.base_sigma_m, updated_at=now
            )
            self._init = True
            return self.state
        dt = now - st.updated_at
        if dt <= 0.0 or dt > 0.5:
            self.state = TargetState(
                x=x, z=z, vx=0.0, vz=0.0, sigma=self.base_sigma_m, updated_at=now
            )
            return self.state
        predicted_x = st.x + st.vx * dt
        predicted_z = st.z + st.vz * dt
        residual_x = x - predicted_x
        residual_z = z - predicted_z
        new_x = predicted_x + self.alpha * residual_x
        new_z = predicted_z + self.alpha * residual_z
        new_vx = _clamp(st.vx + self.beta * residual_x / dt, -3.0, 3.0)
        new_vz = _clamp(st.vz + self.beta * residual_z / dt, -3.0, 3.0)
        self.state = TargetState(
            x=new_x, z=new_z, vx=new_vx, vz=new_vz, sigma=self.base_sigma_m, updated_at=now
        )
        return self.state

    def lead(self, now: float) -> tuple[float, float, float]:
        """Return (x_lead, z_lead, sigma_m) with velocity lead + covariance growth."""
        st = self.state
        if not self._init:
            return 0.0, 0.0, self.max_sigma_m
        age = max(0.0, now - st.updated_at)
        x_lead = st.x + st.vx * self.predict_s
        z_lead = st.z + st.vz * self.predict_s
        sigma = min(self.max_sigma_m, st.sigma + self.sigma_growth_per_s * age)
        return x_lead, z_lead, sigma


class HolonomicFollowPolicy:
    """Blend forward + strafe + turn with gyro damping and coast/search."""

    TRACK = "TRACK"
    COAST = "COAST"
    SEARCH = "SEARCH"
    STOP = "STOP"

    COAST_MAX_S = 0.6
    SEARCH_MAX_S = 1.2

    def __init__(
        self,
        target_distance_m: float = 0.30,
        predict_s: float = 0.15,
        kp_forward: float = 0.9,
        kp_strafe: float = 0.8,
        kp_turn: float = 1.0,
        kd_yaw: float = 0.006,
        deadzone: float = 0.04,
        max_forward: float = 0.48,
        max_strafe: float = 0.5,
        max_turn: float = 1.0,
        coast_decay_tau: float = 0.25,
        search_scan_amp: float = 0.35,
        search_scan_hz: float = 0.4,
    ) -> None:
        self.target_distance_m = target_distance_m
        self.kp_forward = kp_forward
        self.kp_strafe = kp_strafe
        self.kp_turn = kp_turn
        self.kd_yaw = kd_yaw
        self.deadzone = deadzone
        self.max_forward = max_forward
        self.max_strafe = max_strafe
        self.max_turn = max_turn
        self.coast_decay_tau = coast_decay_tau
        self.search_scan_amp = search_scan_amp
        self.search_scan_hz = search_scan_hz
        self.lead = AlphaBetaLead(predict_s=predict_s)
        self.reset()

    def reset(self) -> None:
        self.lead.reset()
        self.mode = self.TRACK
        self.lost_at: float | None = None
        self.coast_at: float | None = None
        self.search_at: float | None = None
        self._coast_vec = (0.0, 0.0, 0.0)
        self._last_debug: dict = {
            "mode": self.TRACK,
            "err_lead": 0.0,
            "z_lead": 0.0,
            "vx": 0.0,
            "vz": 0.0,
            "sigma_m": 0.05,
            "coast_age_ms": 0,
            "cmd_vec": [0.0, 0.0, 0.0],
        }

    @property
    def debug(self) -> dict:
        return dict(self._last_debug)

    def _track_command(
        self, error_x: float, distance_m: float, now: float, yaw_rate_dps: float
    ) -> tuple[float, float, float, dict]:
        self.lead.update(error_x, distance_m, now)
        st = self.lead.state
        x_lead = st.x + st.vx * self.lead.predict_s
        z_lead = st.z + st.vz * self.lead.predict_s
        sigma = st.sigma
        err_smooth = smooth_deadzone(x_lead, self.deadzone)
        fwd_err = smooth_deadzone(z_lead - self.target_distance_m, 0.02)
        forward = _clamp(self.kp_forward * fwd_err, -self.max_forward, self.max_forward)
        if z_lead <= self.target_distance_m + 0.035:
            forward = min(forward, 0.0)
        strafe = _clamp(-self.kp_strafe * err_smooth, -self.max_strafe, self.max_strafe)
        turn = _clamp(
            -self.kp_turn * err_smooth - self.kd_yaw * yaw_rate_dps, -self.max_turn, self.max_turn
        )
        info = {"err_lead": x_lead, "z_lead": z_lead, "vx": st.vx, "vz": st.vz, "sigma_m": sigma}
        return forward, strafe, turn, info

    def step(
        self,
        error_x: float | None,
        distance_m: float | None,
        now: float,
        yaw_rate_dps: float = 0.0,
        visible: bool = True,
    ):
        """Compute one holonomic command. Returns (forward, strafe, turn, info).

        When visible is False (or inputs None) the CoastSearch machine runs:
        TRACK -> COAST (exp decay, <=0.6s) -> SEARCH (bounded scan, <=1.2s) -> STOP.
        """
        from .aruco import FollowCommand  # local import: avoid cycle at module load

        yaw_rate_dps = yaw_rate_dps if math.isfinite(yaw_rate_dps or 0.0) else 0.0
        if (
            visible
            and error_x is not None
            and distance_m is not None
            and math.isfinite(error_x)
            and math.isfinite(distance_m)
        ):
            if self.mode != self.TRACK:
                self.mode = self.TRACK
            self.lost_at = None
            self.coast_at = None
            self.search_at = None
            forward, strafe, turn, info = self._track_command(
                _clamp(error_x, -1.0, 1.0), max(0.0, distance_m), now, yaw_rate_dps
            )
            close = (
                info["z_lead"] <= self.target_distance_m + 0.035 and abs(info["err_lead"]) <= 0.04
            )
            reason = "smooth-target-reached" if close else "smooth-track"
            if close:
                forward = strafe = turn = 0.0
            else:
                self._coast_vec = (forward, strafe, turn)
            self._last_debug = {
                "mode": self.mode,
                "err_lead": round(info["err_lead"], 4),
                "z_lead": round(info["z_lead"], 4),
                "vx": round(info["vx"], 4),
                "vz": round(info["vz"], 4),
                "sigma_m": round(info["sigma_m"], 4),
                "coast_age_ms": 0,
                "cmd_vec": [round(forward, 4), round(strafe, 4), round(turn, 4)],
            }
            cmd = FollowCommand(forward=forward, turn=turn, duration=None, reason=reason)
            cmd.strafe = strafe
            return cmd

        # --- target lost: CoastSearch ---
        if self.lost_at is None:
            self.lost_at = now
            self.coast_at = now
            self.mode = self.COAST
        age = now - self.lost_at
        coast_age = now - self.coast_at if self.coast_at is not None else age
        x_lead, z_lead, sigma = self.lead.lead(now)
        if age <= self.COAST_MAX_S:
            self.mode = self.COAST
            decay = math.exp(-coast_age / max(1e-6, self.coast_decay_tau))
            forward, strafe, turn = (v * decay for v in self._coast_vec)
            reason = "smooth-coast"
        elif age <= self.COAST_MAX_S + self.SEARCH_MAX_S:
            if self.search_at is None:
                self.search_at = now
            self.mode = self.SEARCH
            phase = 2.0 * math.pi * self.search_scan_hz * (now - self.search_at)
            turn = _clamp(self.search_scan_amp * math.sin(phase), -self.max_turn, self.max_turn)
            forward = strafe = 0.0
            reason = "smooth-search"
        else:
            self.mode = self.STOP
            forward = strafe = turn = 0.0
            reason = "smooth-stop"
        self._last_debug = {
            "mode": self.mode,
            "err_lead": round(x_lead, 4),
            "z_lead": round(z_lead, 4),
            "vx": round(self.lead.state.vx, 4),
            "vz": round(self.lead.state.vz, 4),
            "sigma_m": round(sigma, 4),
            "coast_age_ms": round(coast_age * 1000),
            "cmd_vec": [round(forward, 4), round(strafe, 4), round(turn, 4)],
        }
        cmd = FollowCommand(forward=forward, turn=turn, duration=None, reason=reason)
        cmd.strafe = strafe
        return cmd
