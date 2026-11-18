"""One simulated parking attempt from a random start, and the score it gets.

A scenario places the airframe somewhere in a 4 m x 4 m room at a random heading and the car
1.3-3 m from it at a random heading (the airframe may start out of view). The car runs the
controller at 50 Hz; the camera produces a measurement at 10 Hz that arrives 0.1 s late. Each
episode also draws its own car (`PlantParams.sample`) and camera calibration error.

Success is judged on the true pose, never the estimate: the car ends in PARKED, inside the
tolerance of the target, and never touched the airframe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ugv.park.controller import ParkController, ParkParams
from ugv.park.estimator import Estimator
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

    @staticmethod
    def sample(
        rng: np.random.Generator, plant_scale: float = 1.0, det: DetectorModel | None = None
    ) -> Scenario:
        a = (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-np.pi, np.pi))
        while True:
            d, bearing = rng.uniform(1.3, 3.0), rng.uniform(-np.pi, np.pi)
            c = (a[0] + d * np.cos(bearing), a[1] + d * np.sin(bearing), rng.uniform(-np.pi, np.pi))
            if abs(c[0]) < ROOM - 0.2 and abs(c[1]) < ROOM - 0.2:
                return Scenario(a, c, plant_scale, det or DetectorModel())


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


def run(
    scenario: Scenario,
    params: ParkParams | None = None,
    seed: int = 0,
    keep_trace: bool = False,
) -> Result:
    rng = np.random.default_rng(seed)
    b = bay()
    plant_p = PlantParams.sample(rng, scenario.plant_scale)
    cam = CameraError.sample(rng, scenario.detector) if scenario.plant_scale > 0 else CameraError()
    A = np.array(scenario.airframe)
    plant = Plant(plant_p, np.array(scenario.car, dtype=float), rng)
    est = Estimator()
    ctl = ParkController(b, params or ParkParams())
    pending: list[tuple[int, float, np.ndarray]] = []
    states: dict[str, float] = {}
    trace = [] if keep_trace else None
    fixes, outcome = 0, ""
    seen: list | None = [] if keep_trace else None
    max_t = ctl.params.timeout_s + 1
    for k in range(int(max_t / DT)):
        t = k * DT
        true_rel_car = compose(inverse(A), plant.pose)  # car in airframe frame
        if b.collides(true_rel_car):
            outcome = "collision"
            break
        if max(abs(plant.pose[0]), abs(plant.pose[1])) > ROOM:
            outcome = "geofence"
            break
        if k % CAMERA_PERIOD == 0:
            prior = est.predicted_airframe_in_car()
            m = measure(compose(inverse(plant.pose), A), rng, scenario.detector, cam, prior, seen)
            if m is not None:
                pending.append((k + LATENCY_TICKS, t, m))
        jumped = False
        while pending and pending[0][0] <= k:
            _, t_cap, m = pending.pop(0)
            r = est.update(t_cap, m, t)
            fixes += r in ("init", "fused", "reinit")
            jumped |= r == "reinit"
        cmd = ctl.step(t, est, jumped)
        states[ctl.state] = states.get(ctl.state, 0.0) + DT
        if ctl.state in ("PARKED", "ABORT"):
            outcome = "parked" if ctl.state == "PARKED" else f"abort:{ctl.reason}"
            break
        duty = est.actuator.to_duty(cmd)
        _, gyro = plant.step(duty, DT)
        est.predict(t, duty, gyro, DT)
        if trace is not None and k % 5 == 0:
            q = est.car_in_airframe()
            trace.append(
                {
                    "t": round(t, 2),
                    "car": [round(float(v), 4) for v in true_rel_car],
                    "est": None if q is None else [round(float(v), 4) for v in q],
                    "state": ctl.state,
                    "fix": bool(t - est.last_fix_t < 0.15),
                    "kp": detections(seen[-1] if seen else None),
                }
            )
    else:
        outcome = "timeout"
    final = compose(inverse(A), plant.pose) - b.target
    final[2] = wrap(final[2])
    on_target = all(abs(final[i]) <= TOL[i] for i in range(3))
    if outcome == "parked" and not on_target:
        outcome = "parked-off-target"
    return Result(
        success=outcome == "parked",
        outcome=outcome,
        time_s=round(k * DT, 2),
        final_error=tuple(round(float(v), 4) for v in final),
        retries=ctl.retries,
        shield_hits=ctl.shield_hits,
        jumps=est.jumps,
        fixes=fixes,
        states={s: round(v, 2) for s, v in states.items()},
        trace=trace,
    )


def detections(o) -> list:
    """Detector output of the frame captured at this tick: [[keypoint index, u, v], ...]."""
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
