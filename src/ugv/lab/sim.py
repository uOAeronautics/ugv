"""The parking challenge's server side: live simulation sessions a person and the controller share.

A session is one `ugv.park.episode.Episode` with the airframe at the origin (airframe frame, nose
along +x), a car drawn like a scored episode with faults (weak wheels, battery, stalls, lost
frames, false detections ...), the tuned controller parameters, and the simulated keypoint
detector. The browser advances it in real time: with a duty when a person drives, without one
when the controller is engaged, and can switch at any tick. Simulation only; nothing here talks
to the car.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from functools import cache
from pathlib import Path

import numpy as np

from ugv.park.episode import DT, Episode, Scenario
from ugv.park.geometry import bay
from ugv.park.perception import _model_points

PARAMS = Path(__file__).resolve().parents[3] / "config" / "park_params.json"
# Sensor quality: fault severity and family. "broken" = one sensor or wheel dies mid-run.
QUALITY = {
    "healthy": (0.0, None),
    "noisy": (0.5, None),
    "bad": (1.0, None),
    "broken": (1.0, "broken"),
}
# What each playground environment does to the sensors and the floor, on top of the car's own
# faults. Floor severity: 0 clean, 1 = polished concrete or a wet bay (grip, skid, patches).
ENVIRONMENTS = {
    "studio": ((), 0.0),
    "lab": ((), 0.15),
    "hangar": ((("exposure_rate", 0.25),), 0.55),  # skylights, polished concrete
    "workshop": (
        (("phantom_rate", 0.04), ("occluder", (420.0, 470.0, 760.0, 720.0))),
        0.35,
    ),  # clutter, a cable, dust
    "wet": ((("noise_px", 1.0),), 1.0),  # a wet loading bay: the tyres let go
    "night": (
        (("noise_px", 2.0), ("exposure_rate", 0.2), ("blur_px", 6.0)),
        0.2,
    ),  # dark: gain noise, long exposure
}
MAX_SESSIONS = 8
MAX_TICKS = 25  # per request: 0.5 s of simulation

# Named starts in the airframe frame (x, y, yaw); the awkward ones are near the nose and gear.
PRESETS = {
    "tail": (-1.4, 0.0, 0.0),
    "nose-on": (0.75, 0.0, np.pi),
    "nose-oblique": (0.6, 0.45, -2.4),
    "beside-nose-gear": (0.15, 0.42, -np.pi / 2),
    "under-wingtip": (-0.1, 0.75, np.pi),
    "side": (-0.1, -1.2, np.pi / 2),
}


@cache
def _params():
    from ugv.park.search import load_params

    return load_params(PARAMS)


def start_pose(seed: int, start: list[float] | None, preset: str | None) -> np.ndarray:
    b = bay()
    if preset:
        q = np.array(PRESETS[preset], dtype=float)
    elif start is not None:
        q = np.array(start, dtype=float)
    else:  # random, close or far, any side
        rng = np.random.default_rng([seed, 1])
        while True:
            d, bearing = rng.uniform(0.5, 2.0), rng.uniform(-np.pi, np.pi)
            q = np.array([d * np.cos(bearing), d * np.sin(bearing), rng.uniform(-np.pi, np.pi)])
            if not b.collides(q, 0.03) and max(abs(q[:2])) < 1.75:
                break
    if b.collides(q, 0.01) or max(abs(q[:2])) > 1.8:
        raise ValueError("that start touches the airframe or leaves the room")
    return q


class Sessions:
    def __init__(self) -> None:
        self._s: OrderedDict[str, tuple[Episode, threading.Lock]] = OrderedDict()
        self._lock = threading.Lock()

    def create(
        self,
        seed: int,
        start: list[float] | None = None,
        preset: str | None = None,
        quality: str = "bad",
        env: str = "lab",
    ) -> dict:
        q = start_pose(seed, start, preset)
        sev, only = QUALITY[quality]
        car = tuple(float(v) for v in q)
        overrides, floor = ENVIRONMENTS.get(env, ((), 0.25))  # a real panorama: neutral effects
        sc = Scenario(
            (0.0, 0.0, 0.0),
            car,
            1.0,
            faults=sev,
            only=only,
            overrides=overrides,
            floor=floor,
        )
        ep = Episode(sc, _params(), seed=seed, engaged=False)
        sid = f"{seed}-{time.monotonic_ns()}"
        with self._lock:
            self._s[sid] = (ep, threading.Lock())
            while len(self._s) > MAX_SESSIONS:
                self._s.popitem(last=False)
        faults = ep.faults.f.describe() if ep.faults else []
        cam = {"focal_scale": ep.cam.focal_scale, "yaw": ep.cam.yaw, "pitch": ep.cam.pitch}
        return {
            "id": sid,
            "seed": seed,
            "start": q.round(4).tolist(),
            "faults": faults,
            "camera_error": cam,
            **snapshot(ep),
        }

    def step(self, sid: str, ticks: int, duty: list[float] | None, auto: bool) -> dict:
        with self._lock:
            if sid not in self._s:
                raise KeyError(sid)
            self._s.move_to_end(sid)
            ep, lock = self._s[sid]
        with lock:
            if auto != (ep.ctl is not None):
                ep.engage(auto)
            d = None if auto else np.clip(np.asarray(duty or [0, 0, 0], dtype=float), -1, 1)
            for _ in range(max(0, min(ticks, MAX_TICKS))):
                if ep.tick(d):
                    break
            return snapshot(ep)


def snapshot(ep: Episode) -> dict:
    est = ep.est.car_in_airframe()
    out = {
        "t": round(ep.t, 2),
        "car": ep.true_rel().round(4).tolist(),
        "vel": np.asarray(ep.plant.vel).round(3).tolist(),
        "est": None if est is None else np.round(est, 4).tolist(),
        "state": ep.state,
        "reason": ep.ctl.reason if ep.ctl else "",
        "fix": bool(ep.t - ep.est.last_fix_t < 0.3),
        "kp": ep.last_kp,
        "outcome": ep.outcome,
        "cam_fx": ep.cam_fx,
        "floor": [round(ep.floor.grip, 2), round(ep.floor.slip, 2), round(ep.floor.patch, 2)],
    }
    if ep.outcome:
        res = ep.result()
        out |= {"result": res.outcome, "final_error": res.final_error}
    return out


@cache
def skeleton() -> dict:
    names, pts = _model_points()
    return {
        "names": list(names),
        "points": np.round(pts, 4).tolist(),
        "presets": {k: [float(x) for x in v] for k, v in PRESETS.items()},
        "dt": DT,
    }


SESSIONS = Sessions()
