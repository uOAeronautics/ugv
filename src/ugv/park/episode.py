"""One simulated parking attempt, and the score it gets.

A scenario places the airframe somewhere in a 4 m x 4 m room at a random heading and the car
1.3-3 m from it at a random heading (`start="ring"`, the airframe may start out of view), or
0.4-1.3 m from it on any side, nose and wingtips included (`start="close"`). The car runs the
controller at 50 Hz; the camera produces a measurement at 10 Hz that arrives 0.1 s late. Each
episode draws its own car (`PlantParams.sample`) and camera calibration error, and with
`faults` > 0 its own faults (`faults.py`: weak wheels, battery, stalls, lost frames, false
detections ...).

`Episode` advances one 20 ms tick at a time, so a person can drive (`tick(duty)`) and hand over
to the controller (`engage()`) mid-run; `run` is the batch loop.

Success is judged on the true pose, never the estimate: the car ends in PARKED, inside the
tolerance of the target, and never touched the airframe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ugv.park.controller import ParkController, ParkParams
from ugv.park.estimator import Estimator
from ugv.park.faults import Faults, FaultState
from ugv.park.geometry import bay, compose, inverse, wrap
from ugv.park.perception import CameraError, DetectorModel, measure
from ugv.park.plant import Plant, PlantParams

DT = 0.02
CAMERA_PERIOD = 5  # ticks: 10 Hz
LATENCY_TICKS = 5  # 0.1 s
TOL = (0.02, 0.02, np.radians(5))  # x, y, yaw of the parked pose
ROOM = 2.0  # half-width of the room, m


@dataclass(frozen=True)
class Scenario:
    airframe: tuple[float, float, float]
    car: tuple[float, float, float]
    plant_scale: float = 1.0
    detector: DetectorModel = field(default_factory=DetectorModel)
    faults: float = 0.0  # severity; 0 = a healthy car
    only: str | None = None  # a single fault family (faults.FAMILIES)

    @staticmethod
    def sample(
        rng: np.random.Generator,
        plant_scale: float = 1.0,
        det: DetectorModel | None = None,
        faults: float = 0.0,
        only: str | None = None,
        start: str = "ring",
    ) -> Scenario:
        a = (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-np.pi, np.pi))
        b = bay() if start == "close" else None
        while True:
            if b is None:
                d, bearing = rng.uniform(1.3, 3.0), rng.uniform(-np.pi, np.pi)
                c = (
                    a[0] + d * np.cos(bearing),
                    a[1] + d * np.sin(bearing),
                    rng.uniform(-np.pi, np.pi),
                )
            else:  # close to the airframe on any side, in the airframe frame, then into the room
                d, bearing = rng.uniform(0.4, 1.3), rng.uniform(-np.pi, np.pi)
                rel = np.array(
                    [d * np.cos(bearing), d * np.sin(bearing), rng.uniform(-np.pi, np.pi)]
                )
                if b.collides(rel, 0.02):
                    continue
                c = tuple(float(v) for v in compose(np.array(a), rel))
            if abs(c[0]) < ROOM - 0.2 and abs(c[1]) < ROOM - 0.2:
                return Scenario(a, c, plant_scale, det or DetectorModel(), faults, only)


@dataclass
class Result:
    success: bool
    outcome: str  # parked | parked-off-target | collision | abort:<why> | geofence | timeout
    time_s: float
    final_error: tuple[float, float, float]  # true car pose minus target, airframe frame
    retries: int
    shield_hits: int
    jumps: int
    fixes: int
    states: dict[str, float]  # seconds per state
    trace: list | None = None
    faults: list[str] = field(default_factory=list)


class Episode:
    """The world, the car's estimator and (when engaged) the controller, one tick at a time."""

    def __init__(
        self,
        scenario: Scenario,
        params: ParkParams | None = None,
        seed: int = 0,
        keep_trace: bool = False,
        engaged: bool = True,
    ) -> None:
        self.rng = rng = np.random.default_rng(seed)
        self.b = bay()
        self.scenario = scenario
        plant_p = PlantParams.sample(rng, scenario.plant_scale)
        self.cam = (
            CameraError.sample(rng, scenario.detector)
            if scenario.plant_scale > 0
            else CameraError()
        )
        self.A = np.array(scenario.airframe)
        self.plant = Plant(plant_p, np.array(scenario.car, dtype=float), rng)
        self.faults = None
        if scenario.faults > 0 or scenario.only:
            f = Faults.sample(
                np.random.default_rng([seed, 7]), scenario.faults or 1.0, scenario.only
            )
            self.faults = FaultState(f, np.random.default_rng([seed, 8]))
            self.plant.faults = self.faults
            if f.lens:
                fs, cx, cy, k1, k2 = f.lens
                c = self.cam
                self.cam = replace(
                    c, focal_scale=c.focal_scale * fs, cx_px=cx, cy_px=cy, k1=k1, k2=k2
                )
        self.params = params or ParkParams()
        self.est = Estimator(
            calib_min_points=self.params.calib_min_points, still_gain=self.params.still_gain
        )
        self.ctl = ParkController(self.b, self.params) if engaged else None
        self.pending: list[tuple[int, float, np.ndarray]] = []
        self.states: dict[str, float] = {}
        self.trace: list | None = [] if keep_trace else None
        self.seen: list | None = [] if keep_trace else None
        self.fixes, self.outcome, self.k = 0, "", 0
        self.max_k = int((self.params.timeout_s + 1) / DT)
        self.last_kp: list = []
        self.cam_fx: dict | None = None  # this frame's camera faults, for the dashboard

    @property
    def t(self) -> float:
        return self.k * DT

    @property
    def state(self) -> str:
        return self.ctl.state if self.ctl is not None else "MANUAL"

    def true_rel(self) -> np.ndarray:
        """The car's true pose in the airframe frame."""
        return compose(inverse(self.A), self.plant.pose)

    def engage(self, on: bool) -> None:
        """Hand the car to a fresh controller (it starts from SEARCH) or take it back."""
        self.ctl = ParkController(self.b, self.params) if on else None

    def tick(self, duty: np.ndarray | None = None) -> bool:
        """One 20 ms step. `duty` drives by hand; None lets the controller drive. True when over."""
        if self.outcome:
            return True
        k, t = self.k, self.t
        rel = self.true_rel()
        if self.b.collides(rel):
            self.outcome = "collision"
            return True
        if max(abs(self.plant.pose[0]), abs(self.plant.pose[1])) > ROOM:
            self.outcome = "geofence"
            return True
        if k % CAMERA_PERIOD == 0:
            self._camera(t, k)
        jumped = False
        while self.pending and self.pending[0][0] <= k:
            _, t_cap, m = self.pending.pop(0)
            r = self.est.update(t_cap, m, t)
            self.fixes += r in ("init", "fused", "reinit")
            jumped |= r == "reinit"
        if duty is None and self.ctl is not None:
            cmd = self.ctl.step(t, self.est, jumped)
            self.states[self.ctl.state] = self.states.get(self.ctl.state, 0.0) + DT
            if self.ctl.state in ("PARKED", "ABORT"):
                self.outcome = (
                    "parked" if self.ctl.state == "PARKED" else f"abort:{self.ctl.reason}"
                )
                return True
            duty = self.est.actuator.to_duty(cmd)
        elif duty is None:
            duty = np.zeros(3)
        duty = np.asarray(duty, dtype=float)
        _, gyro = self.plant.step(duty, DT)
        self.est.predict(t, duty, gyro, DT)
        if self.trace is not None and k % 5 == 0:
            q = self.est.car_in_airframe()
            self.trace.append(
                {
                    "t": round(t, 2),
                    "car": [round(float(v), 4) for v in rel],
                    "est": None if q is None else [round(float(v), 4) for v in q],
                    "state": self.state,
                    "fix": bool(t - self.est.last_fix_t < 0.15),
                    "kp": self.last_kp,
                }
            )
        if k + 1 >= self.max_k:
            self.outcome = "timeout"
            return True
        self.k += 1
        return False

    def _camera(self, t: float, k: int) -> None:
        if self.faults is not None and self.faults.frame_lost():
            self.last_kp = []
            return
        prior = self.est.predicted_airframe_in_car()
        seen: list = []
        det, cam, occ = self.scenario.detector, self.cam, None
        if self.faults is not None:
            rel = compose(inverse(self.plant.pose), self.A)
            fx = self.cam_fx = self.faults.frame(
                float(self.plant.vel[2]), float(np.hypot(*rel[:2]))
            )
            det = replace(
                det,
                pixel_std=det.pixel_std + fx["px"],
                miss_rate=min(0.95, det.miss_rate + fx["miss"]),
            )
            cam = replace(cam, yaw=cam.yaw + fx["yaw"], pitch=cam.pitch + fx["pitch"])
            occ = fx["occluder"]
        m = measure(compose(inverse(self.plant.pose), self.A), self.rng, det, cam, prior, seen, occ)
        self.last_kp = detections(seen[0])
        if self.faults is not None:
            ghost = self.faults.phantom(self.est.pitch)
            if ghost is not None:
                m = ghost
        if m is not None:
            self.pending.append((k + LATENCY_TICKS, t, m))

    def result(self) -> Result:
        final = self.true_rel() - self.b.target
        final[2] = wrap(final[2])
        outcome = self.outcome or "timeout"
        if outcome == "parked" and not all(abs(final[i]) <= TOL[i] for i in range(3)):
            outcome = "parked-off-target"
        ctl = self.ctl
        return Result(
            success=outcome == "parked",
            outcome=outcome,
            time_s=round(self.k * DT, 2),
            final_error=tuple(round(float(v), 4) for v in final),
            retries=ctl.retries if ctl else 0,
            shield_hits=ctl.shield_hits if ctl else 0,
            jumps=self.est.jumps,
            fixes=self.fixes,
            states={s: round(v, 2) for s, v in self.states.items()},
            trace=self.trace,
            faults=self.faults.f.describe() if self.faults else [],
        )


def run(
    scenario: Scenario,
    params: ParkParams | None = None,
    seed: int = 0,
    keep_trace: bool = False,
) -> Result:
    ep = Episode(scenario, params, seed, keep_trace)
    while not ep.tick():
        pass
    return ep.result()


def detections(o) -> list:
    """Detector output of one frame: [[keypoint index, u, v], ...]."""
    if o is None:
        return []
    keep, obs = o
    return [
        [int(i), round(float(u), 1), round(float(v), 1)]
        for i, (u, v) in zip(np.flatnonzero(keep), obs, strict=True)
    ]


def as_dict(r: Result) -> dict:
    d = asdict(r)
    d.pop("trace")
    return d
