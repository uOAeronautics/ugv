"""The parking behaviour as a declared state machine over the estimated pose.

All geometry is in the airframe frame (x nose, y left, floor plane). The target is the parked
pose from `geometry.bay()`: centred on the fuselage line, facing the nose, between the main
gear and the nose wheel. The car reaches it by

    SEARCH   spin in place until the airframe is measured
    ACQUIRE  hold still until the airframe estimate has settled
    ORBIT    move on a circle around the airframe while yawing to keep the camera on it,
             until the car is on the fuselage line behind the tail
    APPROACH drive along the fuselage line to the staging point, correcting lateral offset
             and heading while the airframe is still in full view
    ENTER    the careful routine: stop, measure while still, then one small move (turn by the
             gyro, strafe, or creep forward part of the remaining distance), repeat. Each
             move's measured result calibrates the next move's duration, so the last move,
             made blind when the airframe has left the image, uses a gain measured at the
             same speed moments earlier
    PARKED   stop
    BACKOFF  reverse out along the line and retry APPROACH (lateral or heading error, shield)
    ABORT    give up: too many retries or timeout

Failure exits are explicit (`TABLE`): losing the airframe sends ORBIT, ACQUIRE and the far part
of APPROACH back to SEARCH; a pose jump (estimator re-initialised) sends the car to ACQUIRE; an
error at the bay mouth or a predicted contact sends it to BACKOFF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ugv.fsm import Machine
from ugv.park.estimator import Estimator
from ugv.park.geometry import Bay, wrap

TABLE = {
    "SEARCH": {"ACQUIRE", "ABORT"},
    "ACQUIRE": {"ORBIT", "SEARCH", "ABORT"},
    "ORBIT": {"APPROACH", "SEARCH", "ACQUIRE", "ABORT"},
    "APPROACH": {"ENTER", "SEARCH", "ACQUIRE", "BACKOFF", "ABORT"},
    "ENTER": {"PARKED", "BACKOFF", "ABORT"},
    "BACKOFF": {"APPROACH", "SEARCH", "ABORT"},
    "PARKED": set(),
    "ABORT": set(),
}


@dataclass(frozen=True)
class ParkParams:
    """Every number the behaviour uses; `tools/park/search.py` tunes them."""

    search_w: float = 0.9  # rad/s
    acquire_s: float = 0.6
    acquire_samples: int = 4
    orbit_r: float = 1.0  # m, around the airframe origin
    orbit_v: float = 0.25  # m/s tangential
    k_radial: float = 1.5
    k_face: float = 3.0
    orbit_tol_deg: float = 6.0
    approach_v: float = 0.18
    k_lat: float = 2.5
    k_head: float = 3.0
    enter_d: float = 0.35  # distance before the target where ENTER starts
    align_lat_m: float = 0.03  # alignment needed to start ENTER (ENTER refines it)
    align_yaw_deg: float = 8.0
    abort_lat_m: float = 0.03  # misalignment inside the bay that forces BACKOFF
    abort_yaw_deg: float = 10.0
    settle_s: float = 0.3  # ENTER: wait after a move before measuring
    measure_s: float = 0.5  # ENTER: measuring window while still
    step_v: float = 0.08  # ENTER: forward and strafe move speed (commanded)
    step_w: float = 0.5  # ENTER: turn rate
    step_frac: float = 0.5  # ENTER: fraction of the remaining distance per forward move
    max_step_m: float = 0.12
    lat_tol_m: float = 0.006
    yaw_tol_deg: float = 1.5
    stop_tol_m: float = 0.006
    lost_s: float = 1.5
    shield_inflate_m: float = 0.008
    lookahead_s: float = 0.4
    max_retries: int = 3
    timeout_s: float = 120.0
    k_hold: float = (
        0.0  # ENTER moves: hold the heading the move started with, by the gyro (0 = off)
    )
    shield_backoff: bool = False  # also shield the reverse out of the bay
    backoff_extra_m: float = 0.25  # distance behind stage_x to reverse to in BACKOFF
    backoff_straight_in_bay: bool = False  # while x > stage_x, reverse straight out (no strafe/yaw)
    approach_align_first: bool = False  # pause forward advance in APPROACH while misaligned
    calib_min_points: int = 0  # estimator: keypoints a fix needs to calibrate gains and gyro bias
    blind_max_m: float = 1.0  # ENTER: farther than this from the target, never move unseen
    blind_waits: int = 3  # ENTER: empty measure windows to wait out before backing off
    misalign_confirm: int = 1  # ENTER: consecutive misaligned windows before backing off
    still_gain: float = 0.0  # estimator: sample weight while still (0 = its default 0.3)
    # Health (0/False = off): stop safely when a sensor or the drive cannot be trusted.
    gyro_check: bool = False  # estimator replaces a flat-lined gyro by the motor model
    camera_silent_s: float = 0.0  # abort when no camera frame has arrived for this long
    health_innov_m: float = 0.0  # abort when fixes land this far from odometry (EMA, metres)


def _body(v_air: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v_air[0] + s * v_air[1], -s * v_air[0] + c * v_air[1]])


@dataclass
class ParkController:
    bay: Bay
    params: ParkParams = field(default_factory=ParkParams)
    clock_t: float = 0.0
    retries: int = 0
    reason: str = ""
    shield_hits: int = 0
    # ENTER sub-state: ("measure", t0, samples0) or ("move", t_end, axis, cmd, start) or
    # ("turn", target_yaw_odom, cmd)
    _step: tuple | None = None
    _k: list = field(default_factory=lambda: [1.0, 1.0])  # measured/commanded, forward and strafe
    _last_move: tuple | None = None  # (axis, commanded distance, car pose before)
    steps: int = 0
    _unseen: int = 0
    _misaligned: int = 0

    def __post_init__(self) -> None:
        self.fsm = Machine("park", "SEARCH", TABLE, clock=lambda: self.clock_t)

    @property
    def state(self) -> str:
        return self.fsm.state

    def _go(self, state: str, why: str) -> None:
        self.fsm.go(state, why)
        self.reason = why

    def step(self, t: float, est: Estimator, jumped: bool) -> np.ndarray:
        """Body velocity command (vx, vy, w) for this tick; also advances the state machine."""
        p, b = self.params, self.bay
        self.clock_t = t
        s = self.fsm.state
        if s in ("PARKED", "ABORT"):
            return np.zeros(3)
        if t > p.timeout_s:
            self._go("ABORT", "timeout")
            return np.zeros(3)
        if p.camera_silent_s and t - est.last_frame_t > p.camera_silent_s:
            self._go("ABORT", "camera silent")
            return np.zeros(3)
        if (
            p.health_innov_m
            and est.innov > p.health_innov_m
            and s in ("APPROACH", "ENTER", "BACKOFF")
        ):
            self._go("ABORT", "odometry disagrees with the camera")
            return np.zeros(3)
        fresh = t - est.last_fix_t < p.lost_s
        q = est.car_in_airframe()
        if jumped and s in ("ORBIT", "APPROACH"):
            self._go("ACQUIRE", "pose jump")
            return np.zeros(3)

        if s == "SEARCH":
            if q is not None and t - est.last_fix_t < 0.3:
                self._go("ACQUIRE", "airframe seen")
                return np.zeros(3)
            return np.array([0.0, 0.0, p.search_w])

        if s == "ACQUIRE":
            if not fresh:
                self._go("SEARCH", "lost while acquiring")
            elif self.fsm.age() >= p.acquire_s and est.samples >= p.acquire_samples:
                self._go("ORBIT", "estimate settled")
            return np.zeros(3)

        assert q is not None
        x, y, yaw = q
        xt = b.target[0]

        if s == "ORBIT":
            if not fresh:
                self._go("SEARCH", "lost in orbit")
                return np.zeros(3)
            r, phi = np.hypot(x, y), np.arctan2(y, x)
            e_phi = wrap(np.pi - phi)
            if abs(e_phi) < np.radians(p.orbit_tol_deg) and abs(p.orbit_r - r) < 0.08:
                self._go("APPROACH", "on the fuselage line")
            rhat = np.array([np.cos(phi), np.sin(phi)])
            that = np.array([-np.sin(phi), np.cos(phi)])
            v_air = (
                p.k_radial * (p.orbit_r - r) * rhat
                + np.sign(e_phi) * min(p.orbit_v, 0.6 * abs(e_phi) + 0.05) * that
            )
            w = p.k_face * wrap(np.arctan2(-y, -x) - yaw)
            return self._shielded(q, np.array([*_body(v_air, yaw), w]))

        if s == "APPROACH":
            stage_x = xt - p.enter_d
            if not fresh and x < stage_x - 0.1:
                self._go("SEARCH", "lost on approach")
                return np.zeros(3)
            aligned = abs(y) < p.align_lat_m and abs(yaw) < np.radians(p.align_yaw_deg)
            if x >= stage_x - 0.01 and aligned:
                self._go("ENTER", "aligned at the staging point")
                return np.zeros(3)
            vx = (
                0.0
                if x >= stage_x
                or (p.approach_align_first and (abs(y) > 0.08 or abs(yaw) > np.radians(10.0)))
                else p.approach_v * float(np.clip((stage_x - x) / 0.25, 0.25, 1))
            )
            v_air = np.array([vx, float(np.clip(-p.k_lat * y, -0.12, 0.12))])
            w = p.k_head * wrap(-yaw)
            return self._shielded(q, np.array([*_body(v_air, yaw), w]))

        if s == "ENTER":
            return self._enter(t, est, q)

        if s == "BACKOFF":
            self._step = None
            stage_x = xt - p.enter_d
            if x < stage_x - p.backoff_extra_m:
                if not fresh:
                    self._go("SEARCH", "lost after backing off")
                else:
                    self._go("APPROACH", "backed off, retry")
                return np.zeros(3)
            if p.backoff_straight_in_bay and x > stage_x:
                cmd = np.array([-p.approach_v, 0.0, 0.0])
            else:
                v_air = np.array([-p.approach_v, float(np.clip(-p.k_lat * y, -0.05, 0.05))])
                cmd = np.array([*_body(v_air, yaw), p.k_head * wrap(-yaw)])
            if p.shield_backoff and self._would_hit(q, cmd):
                self.shield_hits += 1
                return np.array([0.0, 0.0, cmd[2]])  # turn in place to straighten, no translation
            return cmd
        raise AssertionError(s)

    def _enter(self, t: float, est: Estimator, q: np.ndarray) -> np.ndarray:
        p, xt = self.params, self.bay.target[0]
        if self._step is None:
            self._step = ("measure", t, est.samples)
        kind = self._step[0]
        if kind == "move":
            _, t_end, axis, cmd, hold = self._step
            if t < t_end:
                if p.k_hold > 0:
                    cmd = cmd.copy()
                    cmd[2] = float(np.clip(p.k_hold * wrap(hold - est.odom[2]), -0.3, 0.3))
                return cmd
            self._step = ("measure", t, est.samples)
            return np.zeros(3)
        if kind == "turn":
            _, target, w = self._step
            if abs(wrap(target - est.odom[2])) < np.radians(0.5) or np.sign(
                wrap(target - est.odom[2])
            ) != np.sign(w):
                self._step = ("measure", t, est.samples)
                return np.zeros(3)
            return np.array([0.0, 0.0, w])
        # measure: still, collect fixes
        _, t0, n0 = self._step
        if t - t0 < p.settle_s + p.measure_s:
            return np.zeros(3)
        seen = est.samples - n0 >= 2 and est.last_fix_t > t0 + p.settle_s
        x, y, yaw = q
        if self._last_move is not None and seen:
            axis, dist, before = self._last_move
            got = (q[:2] - before[:2])[axis]  # airframe axes ~ car axes (yaw small)
            # A move shorter than the distance covered while the motors spin up (step_v times
            # the motor lag, ~1.6 cm) measures the lag, not the gain; calibrating on it makes
            # the next move overshoot.
            if abs(dist) > 0.02 and np.sign(got) == np.sign(dist):
                ratio = float(np.clip(got / dist, 0.3, 3.0))
                self._k[axis] = float(np.clip(self._k[axis] * ratio, 0.3, 3.0))
        self._last_move = None
        if abs(y) > p.abort_lat_m or abs(yaw) > np.radians(p.abort_yaw_deg):
            # A single few-point fix couples lateral and yaw errors: confirm before giving up.
            self._misaligned += 1
            if self._misaligned < p.misalign_confirm:
                self._step = ("measure", t, est.samples)
                return np.zeros(3)
            self._misaligned = 0
            self._step = None
            return self._backoff("misaligned in the bay")
        self._misaligned = 0
        remaining = xt - x
        if not seen and remaining > p.blind_max_m:
            # Nothing seen (exposure, lost frames, occlusion) and too far to go blind: look again.
            self._unseen += 1
            if self._unseen > p.blind_waits:
                self._unseen = 0
                self._step = None
                return self._backoff("lost in the bay")
            self._step = ("measure", t, est.samples)
            return np.zeros(3)
        if seen:
            self._unseen = 0
        if abs(yaw) > np.radians(p.yaw_tol_deg) and seen:
            self._step = ("turn", wrap(est.odom[2] - yaw), -np.sign(yaw) * p.step_w)
            self.steps += 1
            return np.zeros(3)
        if abs(y) > p.lat_tol_m and seen:
            return self._move(t, 1, -y, q, est)
        if remaining < p.stop_tol_m:
            self._step = None
            self._go("PARKED", "at the target" if seen else "at the target (dead reckoned)")
            return np.zeros(3)
        dist = remaining if not seen else min(p.max_step_m, max(0.01, p.step_frac * remaining))
        if not seen or remaining < 0.02:
            dist = remaining
        return self._move(t, 0, dist, q, est)

    def _move(self, t: float, axis: int, dist: float, q: np.ndarray, est: Estimator) -> np.ndarray:
        p = self.params
        v = p.step_v
        cmd = np.zeros(3)
        cmd[axis] = np.sign(dist) * v
        duration = abs(dist) / (v * self._k[axis])
        end = q.copy()
        end[axis] += dist
        if self.bay.collides(end, p.shield_inflate_m):
            self.shield_hits += 1
            self._step = None
            return self._backoff("shield")
        self._step = ("move", t + duration, axis, cmd, float(est.odom[2]))
        self._last_move = (axis, dist, q.copy())
        self.steps += 1
        return cmd

    def _backoff(self, why: str) -> np.ndarray:
        self.retries += 1
        if self.retries > self.params.max_retries:
            self._go("ABORT", f"{why}; out of retries")
        else:
            self._go("BACKOFF", why)
        return np.zeros(3)

    def _would_hit(self, q: np.ndarray, cmd: np.ndarray) -> bool:
        """Would the estimated footprint touch the airframe within the lookahead?"""
        p = self.params
        c, s = np.cos(q[2]), np.sin(q[2])
        dx = (c * cmd[0] - s * cmd[1]) * p.lookahead_s
        dy = (s * cmd[0] + c * cmd[1]) * p.lookahead_s
        pred = np.array([q[0] + dx, q[1] + dy, wrap(q[2] + cmd[2] * p.lookahead_s)])
        return bool(self.bay.collides(pred, p.shield_inflate_m))

    def _shielded(self, q: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        """Stop translation if the estimated footprint would touch the airframe soon."""
        if self._would_hit(q, cmd):
            self.shield_hits += 1
            if self.fsm.state in ("APPROACH", "ENTER"):
                return self._backoff("shield")
            return np.array([0.0, 0.0, 0.0])
        return cmd
