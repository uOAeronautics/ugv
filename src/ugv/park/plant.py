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
The controller never sees these values: it uses `NOMINAL`. `Floor` adds the surface: a clean
floor is exact, a slippery one loses authority and lets the ground velocity lag the body.
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


@dataclass(frozen=True)
class Floor:
    """Traction. A clean floor is the nominal case; a slippery one loses strafe and yaw
    authority, lets the ground velocity lag the body through turns (the car skids sideways),
    and changes grip from patch to patch so no single calibration holds everywhere.

    `slip` is the lag at walking speed, s; `patch` the fractional grip variation over ~1 m.
    The controller never sees these numbers: it sees the gyro and the camera.
    """

    grip: float = 1.0
    slip: float = 0.0
    patch: float = 0.0
    phase: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def sample(rng: np.random.Generator, scale: float = 0.0) -> Floor:
        if scale <= 0:
            return Floor()
        return Floor(
            grip=float(1 - scale * rng.uniform(0.05, 0.4)),
            slip=float(scale * rng.uniform(0.2, 1.0)),
            patch=float(scale * rng.uniform(0.0, 0.5)),
            phase=(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))),
        )

    def grip_at(self, x: float, y: float) -> float:
        """Local grip: the sampled floor plus a smooth field, 1-2 patches across the room."""
        if not self.patch:
            return self.grip
        a, b = self.phase
        field = (
            np.sin(1.9 * x + a) * np.cos(1.4 * y + b) + 0.5 * np.sin(3.3 * x - 2.1 * y + a + b)
        ) / 1.5
        return float(np.clip(self.grip * (1 + self.patch * field), 0.15, 1.0))


NOMINAL_FLOOR = Floor()


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
    floor: Floor = NOMINAL_FLOOR
    ground_v: np.ndarray | None = None  # world (vx, vy) the tyres actually hold

    def __post_init__(self) -> None:
        self.vel = np.zeros(3) if self.vel is None else self.vel
        self.ground_v = np.zeros(2) if self.ground_v is None else self.ground_v

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
        g = self.floor.grip_at(float(self.pose[0]), float(self.pose[1]))
        v = v * np.array([1.0, g, g])
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        # The tyres can hold only so much of the commanded motion: on a low-grip patch the ground
        # velocity lags the body, so a spin slides the car sideways instead of rotating it on the
        # spot. `slip` grows with speed; a clean floor has no lag and integrates exactly.
        world = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        speed = float(np.hypot(*world))
        tau = self.floor.slip * (0.15 + 0.35 * min(speed, 1.0))
        if tau > 1e-6:
            beta = 1 - np.exp(-dt / tau)
            self.ground_v = self.ground_v + beta * (world - self.ground_v)
        else:
            self.ground_v = world
        w = v[2] + p.strafe_yaw * v[1]
        self.pose = np.array(
            [
                self.pose[0] + self.ground_v[0] * dt,
                self.pose[1] + self.ground_v[1] * dt,
                (self.pose[2] + w * dt + np.pi) % (2 * np.pi) - np.pi,
            ]
        )
        # Vibration noise comes from the motors: it scales with the duty; a still car's gyro is quiet.
        shake = float(np.max(np.abs(duty))) / DUTY_CAP
        gyro = w + p.gyro_bias + (0.004 + p.gyro_std * shake) * self.rng.normal()
        if self.faults is not None:
            gyro = self.faults.gyro(gyro, dt)
        true_body = np.array(
            [
                c * self.ground_v[0] + s * self.ground_v[1],
                -s * self.ground_v[0] + c * self.ground_v[1],
                w,
            ]
        )
        return true_body, float(gyro)
