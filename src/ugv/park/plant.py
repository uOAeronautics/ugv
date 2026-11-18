"""Mecanum car plant on the floor, with the response measured on the real car.

Measured on 2026-11-13 (`docs/L7_FIELD_NOTES.md`, probe run `20261113-181040-probe-axes`):
body speed per unit duty beyond a deadzone, forward 2.3 m/s (deadzone 0.05, onset 0.18 s),
strafe 1.6 m/s (0.13, 0.25-0.37 s), yaw 15.7 rad/s (0.10, 0.07 s). The dashboard caps duty at
1800/4095 = 0.44. The raw gyro rate on the moving car had 0.63 rad/s std at 100 Hz, but that
includes real chassis oscillation from the old controller; the heading the car integrates comes
from the IMU's fused orientation. Scored episodes draw 0.02-0.22 rad/s of white noise per 50 Hz
sample at full duty, scaled by the duty (motor vibration), and 0.004 rad/s when still.

`PlantParams.sample` draws a different car per episode (gains, deadzones, lags, strafe-to-yaw
coupling, gyro noise) so a controller is scored on the spread of plausible cars, not on one.
The controller never sees these values: it uses `NOMINAL`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ugv.park.faults import FaultState

DUTY_CAP = 1800 / 4095


@dataclass(frozen=True)
class PlantParams:
    gain: tuple[float, float, float] = (2.3, 1.6, 15.7)  # m/s, m/s, rad/s per duty
    deadzone: tuple[float, float, float] = (0.05, 0.13, 0.10)
    tau: tuple[float, float, float] = (0.12, 0.2, 0.05)  # first-order lag, s
    strafe_yaw: float = 0.0  # rad/s of yaw per m/s of strafe (roller slip)
    speed_noise: float = 0.05  # multiplicative, per step
    gyro_std: float = 0.1  # rad/s white noise per 50 Hz sample (see module docstring)
    gyro_bias: float = 0.0  # rad/s

    @staticmethod
    def sample(rng: np.random.Generator, scale: float = 1.0) -> PlantParams:
        """A plausible car. `scale` = 0 gives `NOMINAL`; 1 the spread used for scoring."""

        def jitter(v, rel):
            return tuple(float(x * (1 + scale * rng.uniform(-rel, rel))) for x in v)

        return PlantParams(
            gain=jitter(NOMINAL.gain, 0.25),
            deadzone=jitter(NOMINAL.deadzone, 0.4),
            tau=jitter(NOMINAL.tau, 0.5),
            strafe_yaw=float(scale * rng.uniform(-0.6, 0.6)),
            speed_noise=0.05 * scale + 0.01,
            gyro_std=float(0.02 + scale * rng.uniform(0.0, 0.2)),
            gyro_bias=float(scale * rng.normal(0, 0.01)),
        )


NOMINAL = PlantParams()


def velocity_to_duty(v: np.ndarray, p: PlantParams = NOMINAL) -> np.ndarray:
    """Body velocity (vx, vy, w) -> duty, inverting the nominal deadzone model. Capped."""
    out = np.zeros(3)
    for i in range(3):
        if abs(v[i]) > 1e-4:
            out[i] = np.sign(v[i]) * (p.deadzone[i] + abs(v[i]) / p.gain[i])
    return np.clip(out, -DUTY_CAP, DUTY_CAP)


def duty_to_velocity(d: np.ndarray, p: PlantParams) -> np.ndarray:
    mag = np.maximum(np.abs(d) - np.array(p.deadzone), 0.0)
    return np.sign(d) * mag * np.array(p.gain)


@dataclass
class Plant:
    """True car state in the world frame. `step` applies a duty for `dt` seconds."""

    params: PlantParams
    pose: np.ndarray  # (x, y, yaw) world
    rng: np.random.Generator
    vel: np.ndarray | None = None  # body (vx, vy, w)
    faults: FaultState | None = None  # see faults.py; None = a healthy car

    def __post_init__(self) -> None:
        self.vel = np.zeros(3) if self.vel is None else self.vel

    def step(self, duty: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """Advance; returns (true body velocity, measured gyro rate)."""
        p = self.params
        if self.faults is not None:
            duty = self.faults.command(duty, dt)
        target = duty_to_velocity(np.clip(duty, -DUTY_CAP, DUTY_CAP), p)
        target = target * (1 + p.speed_noise * self.rng.normal(size=3))
        alpha = 1 - np.exp(-dt / np.array(p.tau))
        self.vel = self.vel + alpha * (target - self.vel)
        v = self.vel if self.faults is None else self.faults.wheels(self.vel, dt)
        w = v[2] + p.strafe_yaw * v[1]
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose = np.array(
            [
                self.pose[0] + (c * v[0] - s * v[1]) * dt,
                self.pose[1] + (s * v[0] + c * v[1]) * dt,
                (self.pose[2] + w * dt + np.pi) % (2 * np.pi) - np.pi,
            ]
        )
        # Vibration noise comes from the motors: it scales with the duty; a still car's gyro is quiet.
        shake = float(np.max(np.abs(duty))) / DUTY_CAP
        gyro = w + p.gyro_bias + (0.004 + p.gyro_std * shake) * self.rng.normal()
        if self.faults is not None:
            gyro = self.faults.gyro(gyro, dt)
        return np.array([v[0], v[1], w]), float(gyro)
