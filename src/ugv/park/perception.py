"""Simulated airframe pose measurement from the car camera, through the real geometry.

Each measurement projects the 12 skeleton keypoints of the 3D model into the car camera
(1280x720, fx = 968, 0.064 m off the floor, looking along the car's +x), keeps those inside the
image, perturbs them like a keypoint detector would (pixel noise, misses, gross outliers) and
solves PnP + RANSAC exactly as the car would. The planar part of the result is the
measurement. So range, viewing angle, keypoint count and the PnP conditioning all come from
the model; nothing is a hand-set noise level on the pose itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from ugv.park.geometry import CAMERA_X_M, CAMERA_Z_M, bay, wrap
from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.solve import solve_pnp

WIDTH, HEIGHT, F = 1280, 720, 968.0
K = np.array([[F, 0, WIDTH / 2], [0, F, HEIGHT / 2], [0, 0, 1.0]])


@dataclass(frozen=True)
class DetectorModel:
    """How the keypoint detector errs. Defaults: a good detector, by the smoke-test budget."""

    pixel_std: float = 2.0  # px at 1280x720
    miss_rate: float = 0.1  # per visible keypoint
    outlier_rate: float = 0.03  # per keypoint: placed 20-80 px away
    min_points: int = 6  # for the full 6-DoF PnP
    min_planar_points: int = 2  # floor-constrained solve (needs a prior); 2 points: yaw from prior
    min_span_px: float = 40.0  # airframe smaller than this in the image: not detected
    max_range_m: float = 4.0
    # Car-side solver: with 4-5 points whose fit misses the gate, retry with the camera pitch
    # free (a knocked mount) before dropping points (attempt 2026-11-18-park-pitch-refit).
    refit_pitch: bool = True
    # Near-field keypoints on the nose wheel's tyre (left, right, top), visible from the bay
    # where the 12-point skeleton is mostly out of view (attempt 2026-11-18-park-near-field).
    near_field: bool = True
    # Calibration errors drawn once per episode (the same bias on every frame).
    focal_rel_std: float = 0.02
    mount_yaw_std: float = np.radians(1.0)
    mount_pitch_std: float = np.radians(1.5)


@dataclass(frozen=True)
class CameraError:
    """The true camera differs from the one the car assumes: focal scale and mount angles, and
    (as a fault, `faults.py`) the lens: principal-point offset and radial distortion k1, k2 in
    normalised coordinates. The car's solver always assumes the nominal pinhole."""

    focal_scale: float = 1.0
    yaw: float = 0.0
    pitch: float = 0.0
    cx_px: float = 0.0
    cy_px: float = 0.0
    k1: float = 0.0
    k2: float = 0.0

    @staticmethod
    def sample(rng: np.random.Generator, det: DetectorModel) -> CameraError:
        return CameraError(
            float(1 + rng.normal(0, det.focal_rel_std)),
            float(rng.normal(0, det.mount_yaw_std)),
            float(rng.normal(0, det.mount_pitch_std)),
        )


@cache
def _model_points() -> tuple[tuple[str, ...], np.ndarray]:
    kp = keypoints(load_airframe())
    names = tuple(kp)
    pts = np.array([kp[n] for n in names])
    pts[:, 2] -= bay().floor_z  # heights above the floor
    return names, pts


@cache
def _near_points() -> np.ndarray:
    """The nose wheel's tyre seen from behind: its left, right and top extremes (floor heights)."""
    frame = load_airframe()
    v = frame.tris.reshape(-1, 3).copy()
    v[:, 2] -= bay().floor_z
    g = keypoints(frame)["gear_nose"]
    wheel = v[(np.linalg.norm(v[:, :2] - g[:2], axis=1) < 0.05) & (v[:, 2] < 0.06)]
    return np.array(
        [wheel[wheel[:, 1].argmax()], wheel[wheel[:, 1].argmin()], wheel[wheel[:, 2].argmax()]]
    )


def model_points(det: DetectorModel | None = None) -> np.ndarray:
    """The points the detector looks for, heights above the floor: the 12-point skeleton, plus
    the near-field tyre points when `det.near_field`."""
    _, pts = _model_points()
    return np.vstack([pts, _near_points()]) if det is not None and det.near_field else pts


def camera_points(rel: np.ndarray, det: DetectorModel | None = None) -> np.ndarray:
    """Keypoints in the OpenCV camera frame, for the airframe at planar pose `rel` in the car
    frame (airframe origin on the floor)."""
    pts = model_points(det)
    c, s = np.cos(rel[2]), np.sin(rel[2])
    x = rel[0] + c * pts[:, 0] - s * pts[:, 1] - CAMERA_X_M
    y = rel[1] + s * pts[:, 0] + c * pts[:, 1]
    z = pts[:, 2] - CAMERA_Z_M
    return np.stack([-y, -z, x], axis=1)  # camera: x right, y down, z forward


def _true_projection(pc: np.ndarray, cam: CameraError) -> tuple[np.ndarray, np.ndarray]:
    cy, sy = np.cos(cam.yaw), np.sin(cam.yaw)
    cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)
    yaw = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])  # about camera y (down)
    pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])  # about camera x
    q = pc @ (pitch @ yaw).T
    k = K.copy()
    k[0, 0] *= cam.focal_scale
    k[1, 1] *= cam.focal_scale
    k[0, 2] += cam.cx_px
    k[1, 2] += cam.cy_px
    front = q[:, 2] > 0.05
    xy = q[:, :2] / np.where(front, q[:, 2], 1.0)[:, None]
    if cam.k1 or cam.k2:
        r2 = (xy**2).sum(axis=1, keepdims=True)
        xy = xy * (1 + cam.k1 * r2 + cam.k2 * r2 * r2)
    uv = xy * k[[0, 1], [0, 1]] + k[:2, 2]
    return uv, front


def measure(
    rel_true: np.ndarray,
    rng: np.random.Generator,
    det: DetectorModel,
    cam: CameraError | None = None,
    prior: np.ndarray | None = None,
    seen: list | None = None,
    occluder: tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """One noisy planar measurement (x, y, yaw, camera pitch) of the airframe pose in the car frame, or None.

    The image is formed by the true camera (`cam` errors included); the pose is solved with the
    nominal camera, as on the car. `prior` is the car's predicted airframe pose in the car frame
    plus its camera-pitch estimate (from its estimator); it lets the floor-constrained solve run
    with only 2 keypoints. `seen`, if given, receives the detector output (or None).
    """
    o = observe(rel_true, rng, det, cam, occluder)
    if seen is not None:
        seen.append(o)
    if o is None:
        return None
    keep, obs = o
    return solve(model_points(det)[keep], obs, det, prior)


def observe(
    rel_true: np.ndarray,
    rng: np.random.Generator,
    det: DetectorModel,
    cam: CameraError | None = None,
    occluder: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """What the keypoint detector reports: (mask of detected keypoints, their pixels with noise
    and outliers), or None when the airframe is out of range, out of view or too small.
    Keypoints inside `occluder` (u0, v0, u1, v1) are hidden."""
    if np.hypot(rel_true[0], rel_true[1]) > det.max_range_m:
        return None
    uv, front = _true_projection(camera_points(rel_true, det), cam or CameraError())
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
    if occluder is not None:
        u0, v0, u1, v1 = occluder
        inside &= ~((uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1))
    if inside.sum() < 2 or np.hypot(*np.ptp(uv[inside], axis=0)) < det.min_span_px:
        return None
    keep = inside & (rng.random(len(uv)) > det.miss_rate)
    if keep.sum() < det.min_planar_points:
        return None
    obs = uv[keep] + rng.normal(0, det.pixel_std, (keep.sum(), 2))
    bad = rng.random(len(obs)) < det.outlier_rate
    obs[bad] += rng.uniform(20, 80, (bad.sum(), 1)) * rng.choice([-1, 1], (bad.sum(), 2))
    return keep, obs


def project_planar(model: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pixels of model points (heights above floor) for the airframe at planar pose q[:3] in
    the car frame, seen by a camera pitched by q[3] (rad, positive = looking up)."""
    c, s = np.cos(q[2]), np.sin(q[2])
    x = q[0] + c * model[:, 0] - s * model[:, 1] - CAMERA_X_M
    y = q[1] + s * model[:, 0] + c * model[:, 1]
    z = model[:, 2] - CAMERA_Z_M
    cp, sp = np.cos(q[3]), np.sin(q[3])
    fwd, up = cp * x + sp * z, -sp * x + cp * z
    pc = np.stack([-y, -up, fwd], axis=1)
    return (pc @ K.T)[:, :2] / np.maximum(pc[:, 2:3], 1e-3)


def solve_planar(
    model: np.ndarray,
    obs: np.ndarray,
    init: np.ndarray,
    iters: int = 12,
    free_pitch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on (x, y, yaw, camera pitch). Both bodies stand on the floor, so the pose
    has 3 unknowns; the camera pitch is a 4th, shared by every frame, so the car calibrates its
    own mount as it goes. With 6 or more points all four are solved; with 3-5, pitch stays at
    `init` (freeing it with so few points washes out yaw); with 2 (often both on the
    centreline, where yaw is unobservable) yaw stays too.
    Returns (x, y, yaw, pitch) and per-point residuals (px)."""
    q = init.astype(float).copy()
    free = [0, 1, 2, 3] if len(obs) >= 6 else [0, 1, 2] if len(obs) >= 3 else [0, 1]
    if free_pitch and len(obs) >= 4:
        free = [0, 1, 2, 3]
    eps = 1e-4
    for _ in range(iters):
        r = (project_planar(model, q) - obs).ravel()
        J = np.stack(
            [
                ((project_planar(model, q + np.eye(4)[j] * eps) - obs).ravel() - r) / eps
                for j in free
            ],
            axis=1,
        )
        sub, *_ = np.linalg.lstsq(J, -r, rcond=None)
        step = np.zeros(4)
        step[free] = sub
        q += step
        if np.abs(step).max() < 1e-6:
            break
    res = np.linalg.norm(project_planar(model, q) - obs, axis=1)
    return q, res


def solve(
    model: np.ndarray, obs: np.ndarray, det: DetectorModel, prior: np.ndarray | None
) -> np.ndarray | None:
    """The car's solver: 6-DoF PnP when enough points are seen (no prior needed), then a
    floor-constrained refinement; with fewer points, the floor-constrained solve from `prior`
    (x, y, yaw, pitch). Points with residual over `gate` px are dropped one at a time.
    Returns (x, y, yaw, pitch, points used) or None."""
    pitch0 = 0.0 if prior is None else float(prior[3])
    init = None
    if len(obs) >= det.min_points:
        sol = solve_pnp(model, obs, K, reproj_px=max(4.0, 3 * det.pixel_std))
        if sol is not None:
            R, t, _ = sol
            if abs(-t[1] + CAMERA_Z_M) < 0.25:  # the airframe origin sits near the floor
                init = np.array([t[2] + CAMERA_X_M, -t[0], np.arctan2(-R[0, 0], R[2, 0]), pitch0])
    if init is None:
        init = prior
    if init is None:
        return None
    gate = max(6.0, 4 * det.pixel_std)
    m, o = model, obs
    while len(o) >= det.min_planar_points:
        q, res = solve_planar(m, o, init)
        if not np.isfinite(q).all():
            return None
        worst = int(np.argmax(res))
        if res[worst] <= gate:
            return np.array([q[0], q[1], wrap(q[2]), q[3], len(o)])
        if det.refit_pitch and 4 <= len(o) < 6:
            q2, res2 = solve_planar(m, o, init, free_pitch=True)
            if np.isfinite(q2).all() and res2.max() <= gate and abs(q2[3] - init[3]) < 0.12:
                return np.array([q2[0], q2[1], wrap(q2[2]), q2[3], len(o), 1.0])
        m, o = np.delete(m, worst, axis=0), np.delete(o, worst, axis=0)
    return None
