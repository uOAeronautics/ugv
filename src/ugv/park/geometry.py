"""Floor-plane geometry of the parking task, derived from the 3D airframe mesh.

Frames: planar poses are `(x, y, yaw)` on the floor, yaw counter-clockwise. The airframe frame
is the mesh frame (x nose, y left) dropped to the floor: its origin is under the model origin.
The car frame is x forward, y left. A car pose given "in the airframe frame" is the car's
centre and heading in that frame.

The car is two boxes: the body (tyre envelope, 0.044 m tall) and the camera stand (40 x 40 mm,
0.084 m tall). Obstacles are the airframe surface points lower than each box's top plus a
margin, so the car may pass under the wings and the tail boom but not through gear legs,
wheels, struts or low props. They are sampled from the mesh triangles, so they follow the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from ugv.pose.mesh import load_airframe

CAR_LENGTH_M = 0.203  # 8 in tyre envelope (docs/HARDWARE.md)
CAR_WIDTH_M = 0.146  # 5.75 in
CAR_BODY_TOP_M = 0.044  # main board top, 1.75 in
CAR_TOP_M = 0.084  # top of the 40 x 40 mm camera stand
TOWER_X_M = (0.015, 0.055)  # camera stand footprint in the car frame (40 mm behind the front)
TOWER_Y_M = (-0.02, 0.02)
CAMERA_X_M = 0.055  # lens 40 mm behind the PCB front edge, PCB 7.5 in long
CAMERA_Z_M = 0.064


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """Pose `b` expressed in frame `a`, returned in `a`'s parent frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def inverse(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2])])


def _sample_triangles(tris: np.ndarray, spacing: float) -> np.ndarray:
    out = [tris.reshape(-1, 3)]
    for t in tris:
        e = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[0]), 1e-9)
        n = int(np.ceil(e / spacing))
        if n <= 1:
            continue
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        keep = (u + v) <= 1
        u, v = u[keep], v[keep]
        out.append(t[0] + u[:, None] * (t[1] - t[0]) + v[:, None] * (t[2] - t[0]))
    return np.concatenate(out)


@dataclass(frozen=True)
class Bay:
    """Obstacles (N, 2) in the airframe frame, the floor height of the mesh, and the target."""

    obstacles: np.ndarray  # lower than the car body top: hit by the whole footprint
    tall: np.ndarray  # between body top and stand top: hit only by the camera stand
    floor_z: float
    target: np.ndarray  # car pose in the airframe frame when parked
    clearance_m: float  # largest outline inflation for which the straight entry stays clear
    free_x: tuple[float, float]  # main-gear axle x and nose-wheel x: the target lies between

    def collides(self, car_in_airframe: np.ndarray, inflate: float = 0.0) -> bool:
        return collision_count(self, car_in_airframe, inflate) > 0


def _local(points: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = points - pose[:2]
    return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]


def collision_count(b: Bay, pose: np.ndarray, inflate: float = 0.0) -> int:
    """Obstacle points inside the car (body box, then camera stand), grown by `inflate`."""
    lx, ly = _local(b.obstacles, pose)
    n = int(
        (
            (np.abs(lx) <= CAR_LENGTH_M / 2 + inflate) & (np.abs(ly) <= CAR_WIDTH_M / 2 + inflate)
        ).sum()
    )
    tx, ty = _local(b.tall, pose)
    n += int(
        (
            (tx >= TOWER_X_M[0] - inflate)
            & (tx <= TOWER_X_M[1] + inflate)
            & (ty >= TOWER_Y_M[0] - inflate)
            & (ty <= TOWER_Y_M[1] + inflate)
        ).sum()
    )
    return n


def _points(margin_m: float, spacing_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    frame = load_airframe()
    floor = float(frame.tris[:, :, 2].min())
    low = (frame.tris[:, :, 2] - floor).min(axis=1) < CAR_TOP_M + margin_m
    pts = _sample_triangles(frame.tris[low], spacing_m)
    h = pts[:, 2] - floor
    body = np.unique(np.round(pts[h < CAR_BODY_TOP_M + margin_m, :2], 3), axis=0)
    tall = np.unique(np.round(pts[h < CAR_TOP_M + margin_m, :2], 3), axis=0)
    return body, tall, floor


def entry_clear(b: Bay, target: np.ndarray, inflate: float, length: float = 0.8) -> bool:
    """The straight reverse of the final approach (backing out along the car's -x) is free."""
    back = np.array([-np.cos(target[2]), -np.sin(target[2])])
    for d in np.arange(0.0, length, 0.02):
        p = np.array([*(target[:2] + d * back), target[2]])
        if b.collides(p, inflate):
            return False
    return True


def _clearance(b: Bay, target: np.ndarray) -> float:
    if not entry_clear(b, target, 0.0):
        return -1.0
    lo, hi = 0.0, 0.08
    for _ in range(7):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if entry_clear(b, target, mid) else (lo, mid)
    return lo


@cache
def bay(margin_m: float = 0.005, spacing_m: float = 0.004) -> Bay:
    """Build the bay from the mesh and pick the parking pose.

    The car must end between the main-gear axle and the nose wheel (under the fuselage).
    Candidates: centred on the fuselage line, facing the nose (tail entry) or across it (side
    entry from either side). The entry with the largest clearance (growth of the car outline
    that keeps the straight 0.8 m entry free) wins; on it, the pose closest to the middle of
    the bay within 5 mm of that clearance.
    """
    from ugv.pose.mesh import keypoints

    kp = keypoints(load_airframe())
    main_x = float(kp["gear_main_left"][0])
    nose_x = float(kp["gear_nose"][0])
    body, tall, floor = _points(margin_m, spacing_m)
    probe = Bay(body, tall, floor, np.zeros(3), 0.0, (main_x, nose_x))
    mid = 0.5 * (main_x + nose_x)
    best, best_c = None, -1.0
    for yaw in (0.0, -np.pi / 2, np.pi / 2):
        xs = np.arange(main_x, nose_x, 0.01)
        cs = np.array([_clearance(probe, np.array([x, 0.0, yaw])) for x in xs])
        if cs.max() < 0:
            continue
        # Deepest pose (closest to the middle of the bay) on this entry's clearance plateau.
        ok = xs[cs >= cs.max() - 0.005]
        x = ok[np.argmin(np.abs(ok - mid))]
        if cs.max() > best_c + 0.005:
            best, best_c = np.array([x, 0.0, yaw]), float(cs[xs == x][0])
    return Bay(body, tall, floor, best, best_c, (main_x, nose_x))
