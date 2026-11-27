"""The parking challenge's server side: live simulation sessions a person and the controller share.

A session is one `ugv.park.episode.Episode` with the airframe at the origin (airframe frame, nose
along +x), a car drawn like a scored episode with faults (weak wheels, battery, stalls, lost
frames, false detections ...), the tuned controller parameters, and a strict camera: the browser
renders the frame, analyses the same pixels, and posts the detections back (see
docs/PARK_SIM.md, "The strict camera protocol"). The browser advances it in real time: with a
duty when a person drives, without one when the controller is engaged, and can switch at any
tick. Simulation only; nothing here talks to the car.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np

from ugv.lab import record
from ugv.park.episode import DT, PRESETS, Episode, Scenario
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
        exposure: float = 0.0,
    ) -> dict:
        q = start_pose(seed, start, preset)
        sev, only = QUALITY[quality]
        car = tuple(float(v) for v in q)
        env_overrides, floor = ENVIRONMENTS.get(env, ((), 0.25))  # a real panorama: neutral effects
        # A fixed over-exposure setting on top of everything else (0 = auto-exposure as normal)
        overrides = (*env_overrides, *((("exposure_stops", float(exposure)),) if exposure else ()))
        sc = Scenario(
            (0.0, 0.0, 0.0),
            car,
            1.0,
            faults=sev,
            only=only,
            overrides=overrides,
            floor=floor,
        )
        ep = Episode(sc, _params(), seed=seed, engaged=False, external_camera=True)
        if ep.faults is not None:
            # the phantom (a drone-looking impostor the detector cannot see) is a batch-surrogate
            # artifact — the strict protocol never hallucinates what the picture does not contain
            ep.faults.f = replace(ep.faults.f, phantom_rate=0.0)
        sid = f"{seed}-{time.monotonic_ns()}"
        with self._lock:
            self._s[sid] = (ep, threading.Lock())
            while len(self._s) > MAX_SESSIONS:
                self._s.popitem(last=False)
        faults = ep.faults.f.describe() if ep.faults else []
        cam = {"focal_scale": ep.cam.focal_scale, "yaw": ep.cam.yaw, "pitch": ep.cam.pitch}
        # What the browser needs to render the frame exactly as this car's camera would: the
        # mount error plus the round's fixed camera faults. The per-frame dice (shake, spells,
        # drops) are drawn browser-side from the round's seed; the server draws none of them.
        cfs = ep.faults.f if ep.faults else None
        camera_sim = {
            "seed": seed,
            "mount": cam,
            "fx": {
                "cx_px": ep.cam.cx_px,
                "cy_px": ep.cam.cy_px,
                "k1": ep.cam.k1,
                "k2": ep.cam.k2,
            },
            "exposure_stops": cfs.exposure_stops if cfs else 0.0,
            "exposure_rate": cfs.exposure_rate if cfs else 0.0,
            "blur_px": cfs.blur_px if cfs else 0.0,
            "noise_px": cfs.noise_px if cfs else 0.0,
            "focus_m": cfs.focus_m if cfs else 0.0,
            "blur_coc": cfs.blur_coc if cfs else 0.0,
            "shake_deg": cfs.shake_deg if cfs else 0.0,
            "bump_deg": cfs.bump_deg if cfs else 0.0,
            "bump_at_s": cfs.bump_at_s if cfs else 0.0,
            "occluder": list(cfs.occluder) if cfs and cfs.occluder else None,
            "drop_rate": cfs.drop_rate if cfs else 0.0,
            "burst": cfs.burst if cfs else 0.0,
        }
        return {
            "id": sid,
            "seed": seed,
            "start": q.round(4).tolist(),
            "faults": faults,
            "camera_error": cam,
            "camera_sim": camera_sim,
            "exposure": exposure,
            **snapshot(ep),
        }

    def set_exposure(self, sid: str, stops: float) -> dict:
        """Pin (or release) the fixed over-exposure defect on a live session; no restart."""
        with self._lock:
            if sid not in self._s:
                raise KeyError(sid)
            self._s.move_to_end(sid)
            ep, lock = self._s[sid]
        with lock:
            ep.set_exposure_stops(stops)
        return {"ok": True, "exposure": stops}

    def camera_frame(self, sid: str, detections: list | None, lost: bool, fx: dict | None) -> dict:
        """One camera frame from the challenge's pixel-accurate sensor simulation: the estimator
        fuses the detections (the same picture the human sees). The server samples nothing."""
        with self._lock:
            if sid not in self._s:
                raise KeyError(sid)
            self._s.move_to_end(sid)
            ep, lock = self._s[sid]
        with lock:
            ep.add_camera_frame(detections, fx, lost)
        return snapshot(ep)

    def camera_image(
        self, sid: str, frame: np.ndarray | None, fx: dict, rendered: tuple | None = None
    ) -> dict:
        """One rendered camera frame (720x1280 mono) through the trained keypoint detector, then
        the estimator; `None` is a frame that never arrived. The detector reads pixels only.
        `rendered` (car pose, mount yaw/pitch the browser drew) is used only by the recorder."""
        if sid not in self._s:
            raise KeyError(sid)
        if frame is None:
            return self.camera_frame(sid, None, True, None)
        det = detector()(frame)
        rec = record.RECORDER
        if rec is not None and rendered is not None:
            ep = self._s[sid][0]
            rec.save(sid, ep.t, frame, ep.cam, np.asarray(rendered[0]), rendered[1], det)
        out = self.camera_frame(sid, det.points, False, fx | {"ms": det.ms})
        return out | {"det": {"points": det.points, "box": det.box, "ms": det.ms}}

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
def detector():
    """The trained two-stage keypoint detector (runs/pose_train, ONNX). Raises
    FileNotFoundError when the model files are not on this machine; a failure is not cached."""
    from ugv.pose.detector import load_default

    return load_default()


@cache
def skeleton() -> dict:
    from ugv.park.perception import DetectorModel, model_points

    pts = model_points(DetectorModel())  # the 12-point skeleton plus the 3 near-field tyre points
    names, _ = _model_points()
    return {
        "names": [*names, "tyre_left", "tyre_right", "tyre_top"],
        "points": np.round(pts, 4).tolist(),
        "presets": {k: [float(x) for x in v] for k, v in PRESETS.items()},
        "dt": DT,
    }


SESSIONS = Sessions()
