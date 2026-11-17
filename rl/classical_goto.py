"""Classical non-privileged baseline for H-0002: stage P go-to + dead-reckoned finale.

Laws mirror ``ugv.autopark.GoTo``: ``STAGE`` is ``StageGoto`` (target (0, 0.55, 0), 5 cm / 3.5 deg)
and ``ENTRY`` is ``enter_goto`` (target (0, 0.02, 0), face-the-drone-off). Vectorised over N envs,
numpy only. The controller sees only the delayed tag pose + valid flag, the IMU yaw rate and its
own previous duties. It never reads env.pose / env.vel.

Phases per env (drone at the origin, the car enters from +y):
  search  no tag fix yet: spin in place until one arrives
  stage   P go-to on the tag pose to the stage point (0, 0.55, 0)
  entry   P go-to on the tag pose toward the park point; runs until the tag drops out of view
          (about 0.40 m out)
  dr      tag lost after a fix: dead-reckon from the last fix, integrating the body speed implied
          by the controller's own duties and the IMU yaw rate; PD on yaw, the rate nulled
          (zero-yawrate turn correction); P go-to on the dead-reckoned pose

Output: (vx, vy, w) in the sim action space, clamped to [-1, 1]. Duties are divided by KDUTY
(the dashboard scale, PWM_LIMIT / 4095). ``AXIS_SIGN`` comes from ``exp_classical_l1 --probe-axes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

KDUTY = 1800 / 4095  # ugv.autopark.PWM_LIMIT / 4095
AXIS_SIGN = (1.0, -1.0, 1.0)  # probe: ch0 = left (+), ch1 = backward (so -fwd), ch2 = CCW (+)
PARK_Y = 0.02  # ugv.autopark enter_goto target


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _duty(e, kp, vcap, gain, dz, tol, vmin=0.0):
    """Vector form of ugv.autopark.GoTo.command_for.duty."""
    a = np.abs(e)
    v = np.minimum(vcap, np.maximum(np.where(a > tol * 2.0, vmin, 0.0), kp * a))
    return np.where(a < tol, 0.0, np.sign(e) * (dz + v / gain))


@dataclass(frozen=True)
class GoToLaw:
    """The fields of ugv.autopark.GoTo that the duty law uses. Target is (0, target_y, 0)."""

    target_y: float = 0.0
    kp_pos: float = 1.2
    kp_yaw: float = 1.5
    vcap_fwd: float = 0.30
    vcap_left: float = 0.25
    vcap_yaw: float = 0.50
    gain_fwd: float = 2.3
    gain_left: float = 1.6
    gain_yaw: float = 15.7
    dz_fwd: float = 0.05
    dz_left: float = 0.13
    dz_yaw: float = 0.16
    tol_pos: float = 0.05
    tol_yaw_deg: float = 3.5
    face_origin_dist: float = 0.25
    min_fwd_speed: float = 0.0


# ugv.autopark.StageGoto: GoTo((0.0, 0.55, 0.0), tol_pos=0.05, tol_yaw_deg=3.5, settle_s=0.1)
STAGE = GoToLaw(target_y=0.55)
# ugv.autopark.enter_goto: GoTo((0.0, 0.02, 0.0), kp_pos=1.8, vcap_fwd=0.35, vcap_left=0.22,
#   tol_pos=0.04, tol_yaw_deg=4.0, settle_s=0.0, face_origin_dist=1.0, min_fwd_speed=0.22)
ENTRY = GoToLaw(
    target_y=PARK_Y,
    kp_pos=1.8,
    vcap_fwd=0.35,
    vcap_left=0.22,
    tol_pos=0.04,
    tol_yaw_deg=4.0,
    face_origin_dist=1.0,
    min_fwd_speed=0.22,
)


# Tuned on dev seed 3 (128 episodes; the gate seed 11 was not used): lower P on position and
# yaw, a 0.15 cap on the entry approach, and yaw-rate damping. Other fields stay the field values.
STAGE_TUNED = replace(STAGE, kp_pos=0.6, kp_yaw=0.5)
ENTRY_TUNED = replace(ENTRY, kp_pos=0.6, kp_yaw=0.5, vcap_fwd=0.15)


@dataclass(frozen=True)
class ControllerConfig:
    stage: GoToLaw = STAGE_TUNED
    entry: GoToLaw = ENTRY_TUNED
    kd_yaw: float = 0.1  # yaw-rate damping (duty per rad/s, IMU); nulls the rate in dr
    lost_s: float = 0.20  # tag invalid for this long -> dead-reckon
    search_w: float = 0.35  # in-place spin before the first fix (sim units)


PH_SEARCH, PH_STAGE, PH_ENTRY, PH_DR, PH_HOLD = 0, 1, 2, 3, 4


@dataclass
class StageGotoController:
    n: int
    dt: float = 0.02
    cfg: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self):
        n = self.n
        self.has_fix = np.zeros(n, bool)
        self.lost = np.zeros(n)  # seconds since the last valid tag
        self.pose = np.zeros((n, 3))  # dead-reckoned pose (== tag pose while valid)
        self.duty = np.zeros((n, 3))  # last duties (fwd, left, yaw), dashboard units
        self.phase = np.full(n, PH_SEARCH)
        self.staged = np.zeros(n, bool)  # reached the stage point once
        self.dr_entered = np.zeros(n, bool)  # first tick in dr (reported once)

    def step(self, tag, valid, imu_w):
        """tag [n,3] (x, y, yaw), valid [n] bool, imu_w [n] rad/s -> cmd [n,3] in [-1, 1]."""
        c, dt = self.cfg, self.dt
        tag = np.asarray(tag, float)
        valid = np.asarray(valid, bool)
        imu_w = np.asarray(imu_w, float)
        # 1. dead reckoning over the last dt, from the duties this controller issued
        x, y, yaw = self.pose[:, 0], self.pose[:, 1], self.pose[:, 2]
        v_f = (
            np.sign(self.duty[:, 0])
            * c.entry.gain_fwd
            * np.maximum(np.abs(self.duty[:, 0]) - c.entry.dz_fwd, 0.0)
        )
        v_l = (
            np.sign(self.duty[:, 1])
            * c.entry.gain_left
            * np.maximum(np.abs(self.duty[:, 1]) - c.entry.dz_left, 0.0)
        )
        dyaw = imu_w * dt
        mid = yaw + 0.5 * dyaw  # forward unit (sin, -cos), left unit (cos, sin): GoTo's frame
        nx = x + dt * (v_f * np.sin(mid) + v_l * np.cos(mid))
        ny = y + dt * (-v_f * np.cos(mid) + v_l * np.sin(mid))
        self.pose = np.stack([nx, ny, wrap(yaw + dyaw)], axis=1)
        # 2. a valid tag fix resets the dead-reckoned pose
        self.pose[valid] = tag[valid]
        self.has_fix |= valid
        self.lost = np.where(valid, 0.0, self.lost + dt)
        # 3. phase: search until a first fix; stage until the stage point is reached (tag
        #    fresh); entry while the tag is fresh after that; dr only once a staged car loses
        #    the tag (about 0.40 m out). A tag lost before staging holds still until it returns.
        prev_dr = self.phase == PH_DR
        fresh = self.lost < c.lost_s
        at_stage = self._dist(self.pose, c.stage) < c.stage.tol_pos
        self.staged |= at_stage & fresh & self.has_fix
        phase = np.where(
            ~self.has_fix,
            PH_SEARCH,
            np.where(
                fresh,
                np.where(self.staged, PH_ENTRY, PH_STAGE),
                np.where(self.staged, PH_DR, PH_HOLD),
            ),
        )
        self.phase = phase
        self.dr_entered = (self.phase == PH_DR) & ~prev_dr
        # 4. commands
        duty = np.zeros((self.n, 3))
        m = self.phase == PH_STAGE
        if m.any():
            duty[m] = self._goto(self.pose[m], c.stage, imu_w[m], rate=False)
        m = (self.phase == PH_ENTRY) | (self.phase == PH_DR)
        if m.any():
            duty[m] = self._goto(self.pose[m], c.entry, imu_w[m], rate=True)
        self.duty = duty
        cmd = (
            np.stack(
                [AXIS_SIGN[0] * duty[:, 1], AXIS_SIGN[1] * duty[:, 0], AXIS_SIGN[2] * duty[:, 2]],
                axis=1,
            )
            / KDUTY
        )
        search = self.phase == PH_SEARCH
        cmd[search] = [0.0, 0.0, c.search_w]
        hold = self.phase == PH_HOLD
        cmd[hold] = 0.0
        self.duty[search | hold] = 0.0
        return np.clip(cmd, -1.0, 1.0)

    @staticmethod
    def _dist(pose, law):
        return np.hypot(pose[:, 0], pose[:, 1] - law.target_y)

    def _goto(self, pose, law, imu_w, rate):
        """Duties (fwd, left, yaw) toward (0, law.target_y, 0): GoTo.command_for, vectorised."""
        x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
        tol_yaw = math.radians(law.tol_yaw_deg)
        ex, ey = -x, law.target_y - y
        dist = np.hypot(ex, ey)
        yaw_goal = np.where(dist < law.face_origin_dist, 0.0, wrap(-np.arctan2(x, y)))
        eyaw = wrap(yaw_goal - yaw)
        th = yaw - np.pi / 2.0
        e_f = ex * np.cos(th) + ey * np.sin(th)
        e_l = -ex * np.sin(th) + ey * np.cos(th)
        d_f = _duty(
            e_f,
            law.kp_pos,
            law.vcap_fwd,
            law.gain_fwd,
            law.dz_fwd,
            law.tol_pos * 0.5,
            law.min_fwd_speed,
        )
        d_l = _duty(e_l, law.kp_pos, law.vcap_left, law.gain_left, law.dz_left, law.tol_pos * 0.5)
        d_y = _duty(eyaw, law.kp_yaw, law.vcap_yaw, law.gain_yaw, law.dz_yaw, tol_yaw * 0.5)
        if rate:  # PD on yaw: the gyro rate damps the turn (zero-yawrate correction in dr)
            d_y = d_y - self.cfg.kd_yaw * imu_w
        out = np.stack([d_f, d_l, d_y], axis=1)
        at_goal = (dist < law.tol_pos) & (np.abs(eyaw) < tol_yaw)
        out[at_goal] = 0.0
        return out
