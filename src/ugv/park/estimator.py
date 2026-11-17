"""Where the airframe is, from odometry plus intermittent, delayed pose measurements.

The car integrates its own pose in an odometry frame from the commanded velocity (through the
nominal plant model) and the gyro. The airframe is static, so each measurement, composed with
the odometry pose at the time the image was taken, is one sample of the airframe pose in the
odometry frame. Samples are fused by a gated running average; a run of samples that disagree
with the estimate replaces it (the airframe was mis-identified, or odometry jumped).

Control reads the car's pose in the airframe frame: `inverse(airframe_odom) o car_odom`. While
the airframe is out of view that pose comes from odometry alone, so its error grows with the
distance driven since the last fix; `since_fix_m` reports that distance.

Odometry integrates the duty the car sent through its own motor model (`actuator.Actuator`),
which the estimator identifies online: yaw from the gyro, forward and strafe from the motion
measured between fixes, the gyro bias from the heading change between fixes. The real car's
gains and deadzones differ from the nominal model by tens of percent; identifying them while
the airframe is in view is what keeps the blind finale honest.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ugv.park.actuator import Actuator
from ugv.park.geometry import compose, inverse, wrap


@dataclass
class Estimator:
    gain: float = 0.3  # weight of a new sample once initialised
    gate_m: float = 0.25
    gate_rad: float = np.radians(20)
    reinit_after: int = 4  # consecutive gated-out samples that replace the estimate
    odom: np.ndarray = field(default_factory=lambda: np.zeros(3))
    airframe: np.ndarray | None = None
    samples: int = 0
    rejected_run: int = 0
    last_fix_t: float = -1e9
    since_fix_m: float = 0.0
    jumps: int = 0
    adapt: bool = True
    actuator: Actuator = field(default_factory=Actuator)
    gyro_bias: float = 0.0
    pitch: float = 0.0  # camera pitch, self-calibrated from fixes with 4+ points
    tau: tuple[float, float] = (0.12, 0.2)  # nominal motor lag (plant.NOMINAL.tau), s
    _v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _anchor: tuple | None = None  # (t, odom at t, car-in-airframe measured at t)
    _history: deque = field(default_factory=lambda: deque(maxlen=400))  # (t, odom, duty)
    _yaw_win: list = field(default_factory=list)  # (duty_w, gyro) over the current window

    def predict(self, t: float, duty: np.ndarray, gyro: float, dt: float) -> None:
        """Advance odometry by one tick in which `duty` was applied and `gyro` was measured.
        The velocity the duty should give (identified motor model) passes the motor lag."""
        alpha = 1 - np.exp(-dt / np.array(self.tau))
        self._v = self._v + alpha * (self.actuator.velocity(duty)[:2] - self._v)
        v_body = self._v
        if self.adapt:
            self._observe_yaw(float(duty[2]), gyro)
        gyro = gyro - self.gyro_bias
        c, s = np.cos(self.odom[2]), np.sin(self.odom[2])
        self.odom = np.array(
            [
                self.odom[0] + (c * v_body[0] - s * v_body[1]) * dt,
                self.odom[1] + (s * v_body[0] + c * v_body[1]) * dt,
                wrap(self.odom[2] + gyro * dt),
            ]
        )
        self.since_fix_m += float(np.hypot(v_body[0], v_body[1]) * dt)
        self._history.append((t, self.odom.copy(), np.asarray(duty, float).copy()))

    def _observe_yaw(self, d: float, gyro: float) -> None:
        w = self._yaw_win
        if w and (np.sign(d) != np.sign(w[0][0]) or abs(d - w[0][0]) > 0.02):
            if len(w) >= 15:  # 0.3 s of steady duty: skip the first third (motor lag)
                arr = np.array(w[len(w) // 3 :])
                self.actuator.axes[2].observe(
                    float(arr[:, 0].mean()), float(arr[:, 1].mean() - self.gyro_bias)
                )
            w.clear()
        w.append((d, gyro))

    def odom_at(self, t: float) -> np.ndarray:
        for ts, pose, _ in reversed(self._history):
            if ts <= t:
                return pose
        return self._history[0][1] if self._history else self.odom

    def _duties(self, t0: float, t1: float) -> np.ndarray:
        return np.array([d for ts, _, d in self._history if t0 <= ts < t1]).reshape(-1, 3)

    def update(self, t_capture: float, meas: np.ndarray, now: float) -> str:
        """Fuse a measurement (x, y, yaw, pitch) of the airframe in the car frame, taken at
        `t_capture`."""
        rel = meas[:3]
        points = int(meas[4]) if len(meas) > 4 else 12
        if len(meas) > 3 and points >= 6:
            self.pitch += 0.1 * (float(meas[3]) - self.pitch)
        sample = compose(self.odom_at(t_capture), rel)
        if points < 3 and self.airframe is not None:
            # Few points: yaw is barely observable (often taken from the prior). Keep the
            # airframe heading; re-solve the position with it.
            odom_c = self.odom_at(t_capture)
            yaw_rel = wrap(self.airframe[2] - odom_c[2])
            sample = compose(odom_c, np.array([rel[0], rel[1], yaw_rel]))
        if self.airframe is None:
            self.airframe, self.samples, self.rejected_run = sample, 1, 0
            self.last_fix_t, self.since_fix_m = now, 0.0
            return "init"
        d = sample - self.airframe
        d[2] = wrap(d[2])
        if np.hypot(d[0], d[1]) > self.gate_m or abs(d[2]) > self.gate_rad:
            self.rejected_run += 1
            if self.rejected_run >= self.reinit_after:
                self.airframe, self.samples, self.rejected_run = sample, 1, 0
                self.jumps += 1
                self.last_fix_t, self.since_fix_m = now, 0.0
                return "reinit"
            return "rejected"
        if self.adapt:
            self._calibrate(t_capture, rel)
        k = max(self.gain, 1.0 / (self.samples + 1))
        self.airframe = self.airframe + k * d
        self.airframe[2] = wrap(self.airframe[2])
        self.samples += 1
        self.rejected_run = 0
        self.last_fix_t, self.since_fix_m = now, 0.0
        return "fused"

    def _calibrate(self, t: float, rel: np.ndarray) -> None:
        """Compare measured and odometry motion since the anchor fix; update scale and bias."""
        odom = self.odom_at(t)
        car = inverse(rel)  # car in the airframe frame, measured
        if self._anchor is None:
            self._anchor = (t, odom, car)
            return
        t0, odom0, car0 = self._anchor
        d_odom = compose(inverse(odom0), odom)  # motion in the anchor's body frame
        d_meas = compose(inverse(car0), car)
        moved = np.hypot(d_odom[0], d_odom[1])
        if t - t0 < 0.8 and moved < 0.12:
            return
        duties = self._duties(t0, t)
        if len(duties) and abs(d_meas[2]) < 0.15:  # little rotation: body axes stay comparable
            for i in range(2):
                col = duties[:, i]
                steady = col.std() < 0.01 and ((col > 0).all() or (col < 0).all())
                if steady:
                    self.actuator.axes[i].observe(
                        float(col.mean()), float(d_meas[i] / (t - t0)), stall_ok=False
                    )
        if t - t0 > 0.8:
            yaw_err = wrap(d_odom[2] - d_meas[2])
            self.gyro_bias = float(np.clip(self.gyro_bias + 0.05 * yaw_err / (t - t0), -0.03, 0.03))
        self._anchor = (t, odom, car)

    def predicted_airframe_in_car(self) -> np.ndarray | None:
        """The prior for the floor-constrained solve."""
        if self.airframe is None:
            return None
        return np.array([*compose(inverse(self.odom), self.airframe), self.pitch])

    def car_in_airframe(self) -> np.ndarray | None:
        if self.airframe is None:
            return None
        return compose(inverse(self.airframe), self.odom)
