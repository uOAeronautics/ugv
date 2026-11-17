"""The car's model of its own motors, identified online: duty -> body velocity per axis.

Each axis follows `v = g * (|d| - dz) * sign(d)` beyond a deadzone `dz`. The real car's gains
and deadzones differ from the nominal ones by tens of percent and change with battery and
floor, so the car fits them while it drives:

- yaw from the gyro, continuously (the gyro measures the yaw rate directly);
- forward and strafe from the motion measured between airframe fixes, against the mean duty
  applied over the same interval.

The fit is a recursive least squares on `v = a * d + b * sign(d)` (so `g = a`,
`dz = -b / a`), started from the nominal model as a prior (gain within 30%, deadzone within
0.05 duty), with the axis's `still` speed as the observation noise. A steady window in which a non-zero duty
left the gyro still raises the yaw deadzone to at least that duty (translation windows, measured
through the camera, are too noisy for that). This is the small,
mechanistic world model the controller and the odometry share: commands are inverted through
it, and odometry integrates through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ugv.park.plant import DUTY_CAP, NOMINAL

PRIOR_REL = 0.3  # prior std of the gain, relative to nominal
PRIOR_DZ = 0.05  # prior std of the deadzone, in duty


@dataclass
class Axis:
    gain: float
    deadzone: float
    still: float  # |speed| below this counts as not moving; also the speed noise std
    _P: np.ndarray | None = None
    _theta: np.ndarray | None = None
    stall_floor: float = 0.0
    samples: int = 0

    def __post_init__(self) -> None:
        self.gain0 = self.gain
        self._theta = np.array([self.gain, -self.gain * self.deadzone])
        # Prior covariance of (a, b) = (g, -g dz), in the units of the observations.
        self._P = np.diag([(PRIOR_REL * self.gain) ** 2, (self.gain * PRIOR_DZ) ** 2])

    def observe(
        self, duty: float, speed: float, weight: float = 1.0, stall_ok: bool = True
    ) -> None:
        """One (mean duty, measured mean speed) pair over a window of steady duty. `stall_ok`:
        whether a still reading may raise the deadzone (only for clean, direct measurements)."""
        if abs(duty) < 1e-3:
            return
        if abs(speed) < self.still:
            if not stall_ok:
                return
            # Commanded but no motion: the deadzone is at least this duty.
            self.stall_floor = max(self.stall_floor, abs(duty) + 0.005)
            self.deadzone = max(self.deadzone, self.stall_floor)
            return
        if np.sign(speed) != np.sign(duty):
            return
        x = np.array([duty, np.sign(duty)])
        P = self._P
        k = P @ x / (self.still**2 / weight + x @ P @ x)
        self._theta = self._theta + k * (speed - x @ self._theta)
        self._P = P - np.outer(k, x @ P)
        a, b = self._theta
        if 0.2 * self.gain0 < a < 5 * self.gain0:
            self.gain = float(a)
            self.deadzone = float(np.clip(-b / a, self.stall_floor, 0.35))
        self.samples += 1

    def speed(self, duty: float) -> float:
        mag = max(abs(duty) - self.deadzone, 0.0)
        return float(np.sign(duty) * mag * self.gain)

    def duty(self, speed: float) -> float:
        if abs(speed) < 1e-4:
            return 0.0
        return float(np.sign(speed) * (self.deadzone + abs(speed) / self.gain))


@dataclass
class Actuator:
    axes: list[Axis] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.axes:
            stills = (0.01, 0.01, 0.04)
            self.axes = [
                Axis(g, dz, st)
                for g, dz, st in zip(NOMINAL.gain, NOMINAL.deadzone, stills, strict=True)
            ]

    def velocity(self, duty: np.ndarray) -> np.ndarray:
        return np.array([ax.speed(d) for ax, d in zip(self.axes, duty, strict=True)])

    def to_duty(self, v: np.ndarray) -> np.ndarray:
        return np.clip(
            np.array([ax.duty(s) for ax, s in zip(self.axes, v, strict=True)]), -DUTY_CAP, DUTY_CAP
        )
