"""What can be wrong with the car on the day: per-wheel faults, battery, timing, sensors, and things
in the room that look like the airframe.

The plant in `plant.py` is a body-level model (gain, deadzone, lag per axis) that describes a
healthy car up to tens of percent. Faults are drawn on top of it, each episode its own set, from
their own random stream (so a fault-free run is unchanged):

- `wheel_gain`: one or two wheels weaker or stronger (worn gearbox, binding axle, rollers).
  Applied through the mecanum kinematics, so a weak wheel makes a straight drive curve.
- `battery`, `sag_per_min`: supply voltage relative to the calibration day; it scales the duty
  *before* the deadzone (a flat battery also widens the deadzone) and sags while driving.
- `stall_*`: one wheel cuts out for 50-300 ms at random (loose connector, brown-out).
- `slip_std`: per-wheel random traction loss each tick (dusty floor, cable under a wheel).
- `delay_ticks`: extra command latency (Pi scheduling, Wi-Fi), on top of the motor lag.
- `drop_rate`, `burst`: camera frames that never arrive (blur, exposure, USB), sometimes in runs.
- `phantom_rate`: a frame in which the detector reports the airframe somewhere it is not
  (another drone, a chair, a reflection), at a plausible range.
- `gyro_scale`, `gyro_walk`: gyro scale error and a bias that wanders.
- Camera (what the mono image suffers, as its effect on the keypoint detector):
  `exposure_rate`: spells of over/under-exposure (0.3-2 s) that hide most keypoints;
  `exposure_stops`: exposure stuck fixed N stops over (auto-exposure off, set too high). Permanent,
  the detector drowns gradually as the level rises — the challenge's fixed-high-exposure defect.
  `blur_px`: motion blur, pixel error growing with the turn rate; `noise_px`: sensor noise;
  `occluder`: a fixed region of the image blocked (a cable, a smudge, the car's own wire);
  `shake_deg`: mount jitter per frame; `bump_deg`: the mount knocked to a new angle once,
  at a random moment, for the rest of the run.
- `focus`: the lens is fixed-focus at `focus_m` with an aperture `blur_coc`: the airframe is
  sharp near that distance and blurred nearer or farther (thin-lens circle of confusion, in px,
  grows with |1/range - 1/focus|), so the detector's pixel error depends on range.
- Shake and knocks also smear the frame: motion blur in proportion to the mount's angular jump.
- `broken`: a sensor or actuator stops working mid-run: the camera goes dark, the gyro sticks
  at its last value or reads zero, or a wheel dies. The car must notice and end safely.
- `lens`: the real lens is not the calibrated one: focal ±6%, principal point off by up to
  25 px, barrel or pincushion distortion (k1 -0.2..0.1, k2). Permanent for the car.

`Faults.sample(rng, severity)` draws a car: each fault is present with probability ~severity/2
and has a magnitude that grows with severity. `only` keeps a single fault family, for a sweep
that says which fault hurts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FAMILIES = (
    "wheel",
    "battery",
    "stall",
    "slip",
    "delay",
    "drops",
    "phantom",
    "gyro",
    "exposure",
    "blur",
    "occlusion",
    "shake",
    "bump",
    "lens",
    "focus",
)
# Mecanum wheel speeds from body velocity (vx, vy, w): FL, FR, RL, RR. K = half wheelbase + half track.
_K = 0.16
J = np.array([[1, -1, -_K], [1, 1, _K], [1, 1, -_K], [1, -1, _K]], dtype=float)
J_PINV = np.linalg.pinv(J)


@dataclass(frozen=True)
class Faults:
    wheel_gain: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    battery: float = 1.0
    sag_per_min: float = 0.0
    stall_wheel: int = -1
    stall_rate_hz: float = 0.0
    slip_std: float = 0.0
    delay_ticks: int = 0
    drop_rate: float = 0.0
    burst: float = 0.0  # probability that a dropped frame starts a run of drops
    phantom_rate: float = 0.0
    gyro_scale: float = 1.0
    gyro_walk: float = 0.0  # rad/s per sqrt(s)
    exposure_rate: float = 0.0  # bad-exposure spells per second
    # fixed over-exposure, stops over correct (0 = auto); negative = under
    exposure_stops: float = 0.0
    blur_px: float = 0.0  # extra pixel error per rad/s of turn
    noise_px: float = 0.0
    occluder: tuple[float, float, float, float] | None = None  # u0, v0, u1, v1 at 1280x720
    shake_deg: float = 0.0
    bump_deg: float = 0.0
    bump_at_s: float = 0.0
    lens: tuple[float, float, float, float, float] | None = None  # focal scale, cx, cy, k1, k2
    broken: str = ""  # "", "camera", "gyro-stuck", "gyro-zero" or "wheel"
    broken_at_s: float = 0.0
    focus_m: float = 0.0  # 0 = always sharp
    blur_coc: float = 0.0  # px of blur per (1/m) of defocus

    @staticmethod
    def sample(rng: np.random.Generator, severity: float = 1.0, only: str | None = None) -> Faults:
        def on(name: str) -> bool:
            return (only is None and rng.random() < 0.5 * severity) or only == name

        s = max(severity, 1e-9)
        f: dict = {}
        if on("wheel"):
            g = np.ones(4)
            for w in rng.choice(4, size=rng.integers(1, 3), replace=False):
                g[w] = 1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.4) * s
            f["wheel_gain"] = tuple(float(x) for x in np.clip(g, 0.3, 1.6))
        if on("battery"):
            f["battery"] = float(rng.uniform(1 - 0.2 * s, 1 + 0.08 * s))
            f["sag_per_min"] = float(rng.uniform(0, 0.1 * s))
        if on("stall"):
            f["stall_wheel"] = int(rng.integers(4))
            f["stall_rate_hz"] = float(rng.uniform(0.1, 1.0) * s)
        if on("slip"):
            f["slip_std"] = float(rng.uniform(0.05, 0.2) * s)
        if on("delay"):
            f["delay_ticks"] = int(rng.integers(1, 1 + round(3 * s)))  # 20-60 ms at 50 Hz
        if on("drops"):
            f["drop_rate"] = float(rng.uniform(0.05, 0.3) * s)
            f["burst"] = float(rng.uniform(0, 0.7))
        if on("phantom"):
            f["phantom_rate"] = float(rng.uniform(0.01, 0.05) * s)
        if on("gyro"):
            f["gyro_scale"] = float(1 + rng.uniform(-0.05, 0.05) * s)
            f["gyro_walk"] = float(
                rng.uniform(0, 0.001) * s
            )  # ~0.3°/s after 30 s: a poor MEMS part
        if on("exposure"):
            f["exposure_rate"] = float(rng.uniform(0.05, 0.3) * s)
        if on("blur"):
            f["blur_px"] = float(rng.uniform(2, 10) * s)
        if on("blur") and rng.random() < 0.5:
            f["noise_px"] = float(rng.uniform(0.5, 2) * s)
        if on("occlusion"):
            w, h = rng.uniform(0.15, 0.4) * 1280 * min(s, 1.5), rng.uniform(0.15, 0.5) * 720
            u0, v0 = rng.uniform(0, 1280 - w), rng.uniform(0, 720 - h)
            f["occluder"] = (float(u0), float(v0), float(u0 + w), float(v0 + h))
        if on("shake"):
            f["shake_deg"] = float(rng.uniform(0.2, 1.0) * s)
        if on("bump"):
            f["bump_deg"] = float(rng.uniform(1, 4) * s)
            f["bump_at_s"] = float(rng.uniform(3, 40))
        if on("focus"):
            f["focus_m"] = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))  # near, mid or far
            f["blur_coc"] = float(rng.uniform(2, 8) * s)
        if on("broken"):
            f["broken"] = str(rng.choice(["camera", "gyro-stuck", "gyro-zero", "wheel"]))
            f["broken_at_s"] = float(rng.uniform(2, 40))
        if on("lens"):
            f["lens"] = (
                float(1 + rng.uniform(-0.06, 0.06) * s),
                float(rng.uniform(-25, 25) * s),
                float(rng.uniform(-25, 25) * s),
                float(rng.uniform(-0.2, 0.1) * s),
                float(rng.uniform(-0.05, 0.05) * s),
            )
        return Faults(**f)

    def describe(self) -> list[str]:
        out = []
        names = ("front-left", "front-right", "rear-left", "rear-right")
        for n, g in zip(names, self.wheel_gain, strict=True):
            if abs(g - 1) > 1e-6:
                out.append(f"{n} wheel {100 * (g - 1):+.0f}%")
        if self.battery != 1.0 or self.sag_per_min:
            out.append(f"battery {100 * self.battery:.0f}%, sags {100 * self.sag_per_min:.0f}%/min")
        if self.stall_wheel >= 0:
            out.append(f"{names[self.stall_wheel]} wheel cuts out {self.stall_rate_hz:.1f}/s")
        if self.slip_std:
            out.append(f"wheel slip {100 * self.slip_std:.0f}%")
        if self.delay_ticks:
            out.append(f"command delay +{20 * self.delay_ticks} ms")
        if self.drop_rate:
            out.append(f"{100 * self.drop_rate:.0f}% camera frames lost")
        if self.phantom_rate:
            out.append(f"false airframe in {100 * self.phantom_rate:.1f}% of frames")
        if self.gyro_scale != 1.0 or self.gyro_walk:
            out.append(f"gyro scale {100 * (self.gyro_scale - 1):+.1f}%, bias drifts")
        if self.exposure_rate:
            out.append(f"bad exposure spells {self.exposure_rate:.2f}/s")
        if self.exposure_stops:
            out.append(f"exposure stuck at {self.exposure_stops:+.1f} stops")
        if self.blur_px:
            out.append(f"motion blur {self.blur_px:.0f} px per rad/s")
        if self.noise_px:
            out.append(f"sensor noise +{self.noise_px:.1f} px")
        if self.occluder:
            u0, v0, u1, v1 = self.occluder
            out.append(f"{100 * (u1 - u0) * (v1 - v0) / (1280 * 720):.0f}% of the image blocked")
        if self.shake_deg:
            out.append(f"camera shake {self.shake_deg:.1f}°")
        if self.bump_deg:
            out.append(f"camera knocked {self.bump_deg:.1f}° at {self.bump_at_s:.0f} s")
        if self.broken:
            what = {
                "camera": "camera goes dark",
                "gyro-stuck": "gyro sticks",
                "gyro-zero": "gyro reads zero",
                "wheel": "a wheel dies",
            }
            out.append(f"{what[self.broken]} at {self.broken_at_s:.0f} s")
        if self.focus_m:
            out.append(
                f"fixed focus at {self.focus_m:.1f} m, blur {self.blur_coc:.0f} px per dioptre"
            )
        if self.lens:
            fs, cx, cy, k1, _ = self.lens
            out.append(
                f"lens: focal {100 * (fs - 1):+.0f}%, centre {cx:+.0f}/{cy:+.0f} px, k1 {k1:+.2f}"
            )
        return out


@dataclass
class FaultState:
    """The faults of one episode as they unfold in time (own RNG stream)."""

    f: Faults
    rng: np.random.Generator
    t: float = 0.0
    stall_left: float = 0.0
    dropping: bool = False
    bias: float = 0.0
    exposure_left: float = 0.0
    knocked: bool = False
    stuck: float | None = None
    exposure_sign: float = 1.0
    _queue: deque = field(default_factory=deque)

    def command(self, duty: np.ndarray, dt: float) -> np.ndarray:
        """The duty that reaches the motors this tick: delayed, then scaled by the battery."""
        self.t += dt
        self._queue.append(np.asarray(duty, dtype=float))
        d = self._queue.popleft() if len(self._queue) > self.f.delay_ticks else np.zeros(3)
        return d * max(0.0, self.f.battery * (1 - self.f.sag_per_min * self.t / 60))

    def wheels(self, v: np.ndarray, dt: float) -> np.ndarray:
        """Body velocity after per-wheel gain, stalls and slip (mecanum kinematics)."""
        g = np.array(self.f.wheel_gain)
        if self.f.stall_wheel >= 0:
            if self.stall_left <= 0 and self.rng.random() < self.f.stall_rate_hz * dt:
                self.stall_left = self.rng.uniform(0.05, 0.3)
            if self.stall_left > 0:
                self.stall_left -= dt
                g[self.f.stall_wheel] *= 0.1
        if self.f.broken == "wheel" and self.t >= self.f.broken_at_s:
            g[0] = 0.0
        if self.f.slip_std:
            g = g * np.clip(1 - np.abs(self.rng.normal(0, self.f.slip_std, 4)), 0, 1)
        return J_PINV @ (g * (J @ v))

    def gyro(self, rate: float, dt: float) -> float:
        self.bias += self.f.gyro_walk * np.sqrt(dt) * self.rng.normal()
        out = rate * self.f.gyro_scale + self.bias
        if self.f.broken.startswith("gyro") and self.t >= self.f.broken_at_s:
            if self.stuck is None:
                self.stuck = out
            return 0.0 if self.f.broken == "gyro-zero" else self.stuck
        return out

    def frame_lost(self) -> bool:
        if self.f.broken == "camera" and self.t >= self.f.broken_at_s:
            return True
        if self.dropping:
            self.dropping = self.rng.random() < self.f.burst
            return True
        if self.rng.random() < self.f.drop_rate:
            self.dropping = self.rng.random() < self.f.burst
            return True
        return False

    def frame(self, w: float, range_m: float = 1.0) -> dict:
        """This frame's camera faults, given the turn rate w and the airframe's range: extra pixel
        error, extra miss rate, mount offsets (rad) and the blocked region. Also what the
        dashboard shows."""
        f = self.f
        if (
            f.exposure_rate
            and self.exposure_left <= 0
            and self.rng.random() < f.exposure_rate * 0.1
        ):
            self.exposure_left = self.rng.uniform(0.3, 2.0)
            self.exposure_sign = self.rng.choice([-1.0, 1.0])
        bad = self.exposure_left > 0
        self.exposure_left -= 0.1
        # A fixed setting never clears, so it reports its own stop count and swamps the spells
        fixed, strength = f.exposure_stops, abs(f.exposure_stops)
        exposure = float(fixed) if strength else (float(self.exposure_sign) if bad else 0.0)
        shake = np.radians(f.shake_deg) * self.rng.normal(size=2) if f.shake_deg else np.zeros(2)
        bump = np.radians(f.bump_deg) if f.bump_deg and self.t >= f.bump_at_s else 0.0
        knock = bump if bump and not self.knocked else 0.0  # the frame in which it happens
        self.knocked = self.knocked or bool(bump)
        # smear: turning, the mount jumping (shake, the knock itself), and defocus at this range
        jump = float(np.hypot(*shake)) + knock
        smear = f.blur_px * abs(w) + 968 * jump * 0.5
        defocus = f.blur_coc * abs(1 / max(range_m, 0.1) - 1 / f.focus_m) if f.focus_m else 0.0
        blur = smear + defocus
        return {
            "px": f.noise_px + 0.5 * blur + 0.5 * strength,
            "miss": (0.6 if bad else 0.0) + min(0.95, 0.3 * strength) + min(0.5, 0.02 * blur),
            "exposure": exposure,
            "yaw": float(shake[0] + 0.6 * bump),
            "pitch": float(shake[1] - bump),
            "occluder": f.occluder,
            "blur": float(blur),
        }

    def phantom(self, pitch: float) -> np.ndarray | None:
        """A false detection: an airframe-like pose 0.6-3 m ahead, within the field of view."""
        if self.rng.random() >= self.f.phantom_rate:
            return None
        r, b = self.rng.uniform(0.6, 3.0), self.rng.uniform(-0.5, 0.5)
        return np.array([r * np.cos(b), r * np.sin(b), self.rng.uniform(-np.pi, np.pi), pitch, 6])
