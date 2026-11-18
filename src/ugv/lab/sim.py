"""The parking challenge's server side: rounds, the real controller as autopilot, the detector.

A round is a seed. The car the player drives (gains, deadzones, lags, roller slip) and its camera
calibration error are drawn exactly as `ugv.park.episode.run` draws them for that seed, so the
autopilot started from the player's start pose drives the very same car with the tuned
controller, the estimator and the simulated keypoint detector. Everything is in the airframe
frame (airframe at the origin, nose along +x). Simulation only; nothing here talks to the car.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from ugv.park.episode import Scenario, detections, run
from ugv.park.geometry import bay, wrap
from ugv.park.perception import CameraError, DetectorModel, _model_points, observe, solve
from ugv.park.plant import PlantParams

LEVELS = {"nominal": 0.0, "random": 1.0, "blind": 1.0}  # plant/camera spread per level
PARAMS = Path(__file__).resolve().parents[3] / "config" / "park_params.json"
HALF_ROOM = 1.8  # start inside the episode geofence (2 m) with margin


@cache
def _params():
    from ugv.park.search import load_params

    return load_params(PARAMS)


def _car(seed: int, scale: float) -> tuple[PlantParams, CameraError]:
    """The plant and camera error `episode.run` draws for this seed (same RNG order)."""
    rng = np.random.default_rng(seed)
    plant = PlantParams.sample(rng, scale)
    cam = CameraError.sample(rng, DetectorModel()) if scale > 0 else CameraError()
    return plant, cam


def new_round(seed: int, level: str) -> dict:
    scale = LEVELS[level]
    b = bay()
    rng = np.random.default_rng([seed, 1])
    while True:
        d, bearing = rng.uniform(1.3, 2.2), rng.uniform(-np.pi, np.pi)
        q = np.array([d * np.cos(bearing), d * np.sin(bearing), rng.uniform(-np.pi, np.pi)])
        if max(abs(q[0]), abs(q[1])) < HALF_ROOM and not b.collides(q, inflate=0.05):
            break
    plant, cam = _car(seed, scale)
    return {
        "seed": seed,
        "level": level,
        "start": [round(float(v), 4) for v in q],
        "plant": {
            "gain": plant.gain,
            "deadzone": plant.deadzone,
            "tau": plant.tau,
            "strafe_yaw": plant.strafe_yaw,
            "speed_noise": plant.speed_noise,
        },
        "camera_error": {"focal_scale": cam.focal_scale, "yaw": cam.yaw, "pitch": cam.pitch},
    }


def autopilot(seed: int, level: str, start: list[float]) -> dict:
    sc = Scenario(
        airframe=(0.0, 0.0, 0.0), car=tuple(float(v) for v in start), plant_scale=LEVELS[level]
    )
    r = run(sc, _params(), seed=seed, keep_trace=True)
    return {
        "outcome": r.outcome,
        "success": r.success,
        "time_s": r.time_s,
        "final_error": r.final_error,
        "retries": r.retries,
        "trace": r.trace,
    }


def perceive(seed: int, level: str, car: list[float], tick: int) -> dict:
    """One camera frame for a car the player drives: detected keypoints and, with 6 or more, the
    6-DoF PnP pose of the airframe in the car frame. No odometry prior: the player has none."""
    _, cam = _car(seed, LEVELS[level])
    q = np.asarray(car, dtype=float)
    c, s = np.cos(q[2]), np.sin(q[2])
    rel = np.array([-(c * q[0] + s * q[1]), s * q[0] - c * q[1], wrap(-q[2])])  # airframe in car
    det = DetectorModel()
    o = observe(rel, np.random.default_rng([seed, 2, tick]), det, cam)
    pose = None
    if o is not None:
        _, model = _model_points()
        m = solve(model[o[0]], o[1], det, None)
        pose = None if m is None else [round(float(v), 4) for v in m[:4]]
    return {"kp": detections(o), "pose": pose}


@cache
def skeleton() -> dict:
    names, pts = _model_points()
    return {"names": list(names), "points": np.round(pts, 4).tolist()}
