"""Synthetic views of the airframe with ground-truth pose, keypoints and a triangle-id map.

Camera convention is OpenCV: x right, y down, z forward. A pose `(R, t)` maps model to camera,
`X_c = R @ X_m + t`. Rendering is a painter's-algorithm rasterizer (OpenCV polygon fill, back to
front): no GPU, deterministic for a seed, about 20 ms per 640x480 view. The id map gives, for
each pixel, the triangle drawn there (-1 = background), which is what makes 2D-3D
correspondences for feature matching exact up to triangle size.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ugv.pose.mesh import Airframe


@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 360
    hfov_deg: float = 66.9  # car camera: fx = 968 px at 1280x720 (docs/PARKING.md)
    intrinsics: tuple[float, float, float] | None = None  # (f, cx, cy) overrides hfov

    def scaled(self, width: int, height: int) -> Camera:
        return Camera(width, height, self.hfov_deg)

    def crop(self, x0: float, y0: float, size: float, out: int) -> Camera:
        """The camera that sees the square crop [x0, x0+size) x [y0, y0+size) of this one,
        resampled to out x out pixels. A crop of a pinhole image is a pinhole image."""
        K = self.K
        s = out / size
        return Camera(
            out, out, self.hfov_deg, (K[0, 0] * s, (K[0, 2] - x0) * s, (K[1, 2] - y0) * s)
        )

    @property
    def K(self) -> np.ndarray:
        if self.intrinsics is not None:
            f, cx, cy = self.intrinsics
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        f = 0.5 * self.width / np.tan(np.radians(self.hfov_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])


@dataclass(frozen=True)
class ViewSpec:
    """Ranges the pose sampler draws from.

    With `eye_z_m` set, the camera height relative to the airframe origin is drawn from it
    instead of from `elevation_deg`. `CAR_VIEWS` uses this: the car camera sits 0.064 m off the
    floor and the airframe origin about 0.15 m above its gear contacts, so the camera looks
    slightly up at the belly.
    """

    distance_m: tuple[float, float] = (0.6, 4.0)
    elevation_deg: tuple[float, float] = (-5.0, 35.0)
    eye_z_m: tuple[float, float] | None = None
    roll_deg: float = 8.0  # max camera roll
    off_centre: float = 0.35  # max aim offset as a fraction of the image half-width


CAR_VIEWS = ViewSpec(distance_m=(0.4, 3.5), eye_z_m=(-0.12, -0.06), roll_deg=3.0)


@dataclass
class View:
    image: np.ndarray  # (H, W, 3) uint8 BGR
    ids: np.ndarray  # (H, W) int32 triangle index, -1 background
    R: np.ndarray  # (3, 3) model -> camera
    t: np.ndarray  # (3,)
    kp2d: dict[str, np.ndarray]  # pixel (u, v)
    kp_visible: dict[str, bool]  # in frame and not occluded


def look_at(eye: np.ndarray, target: np.ndarray, roll: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of a camera at `eye` looking at `target`, model z up."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    c, s = np.cos(roll), np.sin(roll)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R, -R @ eye


def sample_pose(
    rng: np.random.Generator, cam: Camera, spec: ViewSpec
) -> tuple[np.ndarray, np.ndarray]:
    d = rng.uniform(*spec.distance_m)
    az = rng.uniform(-np.pi, np.pi)
    if spec.eye_z_m is not None:
        el = np.arcsin(np.clip(rng.uniform(*spec.eye_z_m) / d, -1, 1))
    else:
        el = np.radians(rng.uniform(*spec.elevation_deg))
    eye = d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    R, t = look_at(eye, np.zeros(3), np.radians(rng.uniform(-spec.roll_deg, spec.roll_deg)))
    # Shift the airframe off the optical axis so the detector cannot assume it is centred.
    shift = (
        rng.uniform(-spec.off_centre, spec.off_centre, 2) * np.array([cam.width, cam.height]) / 2
    )
    t = t + np.array([*(shift * t[2] / cam.K[0, 0]), 0.0])
    return R, t


@dataclass(frozen=True)
class CarLevelSpec:
    """The real car's camera: level, `cam_z_m` above the floor, facing wherever the car faces.

    The car stands `distance_m` from the airframe origin on a random side; its heading points at
    the airframe plus a bearing offset. Offsets within `in_view_deg` keep the airframe in or near
    the picture, `partial_deg` lets it leave the frame, and `away` (a probability) turns the car
    away entirely: a picture with no airframe, which the detector must learn to call empty.
    """

    floor_z: float  # the floor's height in the model frame (the gear contacts)
    cam_z_m: float = 0.064
    distance_m: tuple[float, float] = (0.25, 3.5)
    in_view_deg: float = 25.0
    partial_deg: float = 70.0
    partial: float = 0.3
    away: float = 0.12
    pitch_std_deg: float = 2.5  # mount pitch error plus the one-time knock
    roll_std_deg: float = 1.0


def sample_car_level_pose(
    rng: np.random.Generator, spec: CarLevelSpec
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (R, t, away): model -> camera, and whether the car faces away from the airframe."""
    d = rng.uniform(*spec.distance_m)
    az = rng.uniform(-np.pi, np.pi)
    eye = np.array([d * np.cos(az), d * np.sin(az), spec.floor_z + spec.cam_z_m])
    u = rng.random()
    if u < spec.away:
        off = np.radians(rng.uniform(spec.partial_deg, 180.0)) * rng.choice([-1, 1])
    elif u < spec.away + spec.partial:
        off = np.radians(rng.uniform(spec.in_view_deg, spec.partial_deg)) * rng.choice([-1, 1])
    else:
        off = np.radians(rng.uniform(-spec.in_view_deg, spec.in_view_deg))
    heading = az + np.pi + off  # facing the airframe origin, then turned by `off`
    pitch = np.radians(rng.normal(0, spec.pitch_std_deg))
    fwd = np.array([np.cos(heading), np.sin(heading), np.tan(pitch)])
    R, t = look_at(eye, eye + fwd, np.radians(rng.normal(0, spec.roll_std_deg)))
    return R, t, u < spec.away


def sample_bay_pose(
    rng: np.random.Generator,
    floor_z: float,
    target: np.ndarray,
    cam_x: float,
    cam_z: float,
    run_in: tuple[float, float] = (-1.0, 0.25),
) -> tuple[np.ndarray, np.ndarray]:
    """The car's camera on its way into the bay: the car stands at the parking `target`
    (x, y, yaw in the airframe frame) shifted `run_in` metres along its own heading (negative =
    still outside), a few cm and degrees off the line, camera mount shake included. The airframe
    is then mostly out of the picture: the nose gear and the tyre points are what is in view.
    Returns (R, t): model -> camera."""
    x, y, yaw = (float(v) for v in target)
    yaw += np.radians(rng.normal(0, 8))
    run = rng.uniform(*run_in)
    side = rng.normal(0, 0.06)
    fwd2 = np.array([np.cos(yaw), np.sin(yaw)])
    left2 = np.array([-fwd2[1], fwd2[0]])
    car = np.array([x, y]) + run * fwd2 + side * left2
    eye = np.array([*(car + cam_x * fwd2), floor_z + cam_z])
    heading = yaw + np.radians(rng.normal(0, 3))
    pitch = np.radians(rng.normal(0, 2.5))
    d = np.array([np.cos(heading), np.sin(heading), np.tan(pitch)])
    return look_at(eye, eye + d, np.radians(rng.normal(0, 1.0)))


def room_background(rng: np.random.Generator, cam: Camera, R: np.ndarray) -> np.ndarray:
    """A clean, bright indoor render: flat wall and floor greys split at the horizon, with a few
    dark blocks (cabinets, tables) on the wall. Covers flat-shaded simulators and plain rooms."""
    h, w = cam.height, cam.width
    fwd = R[2].copy()  # optical axis in the model frame
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    v_h = h / 2
    if n > 1e-6:  # the horizon is where level directions project
        pc = R @ (fwd / n)
        if pc[2] > 1e-3:
            v_h = float((cam.K @ pc)[1] / pc[2])
    wall, floor = rng.uniform(120, 235), rng.uniform(100, 215)
    img = np.empty((h, w), np.float32)
    rows = np.arange(h)[:, None]
    img[:] = np.where(rows < v_h, wall, floor)
    img += np.linspace(rng.uniform(-25, 25), rng.uniform(-25, 25), w)[None, :]
    for _ in range(rng.integers(0, 5)):
        x0 = rng.integers(-w // 4, w)
        bw, bh = rng.integers(w // 20, w // 3), rng.integers(h // 10, h // 2)
        y1 = int(np.clip(v_h + rng.uniform(-0.05, 0.05) * h, 0, h))
        img[max(0, y1 - bh) : y1, max(0, x0) : max(0, x0 + bw)] = rng.uniform(10, 90)
    img += rng.normal(0, rng.uniform(0, 4), img.shape)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def procedural_texture(centroids: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fixed per-triangle albedo (N, 3) in [0, 1]: a sum of random plane waves over 3D position.

    The same seed gives the same texture: it stands in for a pattern printed on the airframe.
    """
    rng = np.random.default_rng(seed)
    freq = rng.normal(0, 25.0, (4, 3))  # cycles per metre ~ 4 cm features
    phase = rng.uniform(0, 2 * np.pi, (4, 3))
    w = np.sin((centroids @ freq.T)[:, :, None] + phase[None])  # (N, 4 waves, 3 channels)
    return np.clip(0.5 + 0.25 * w.sum(axis=1), 0, 1)


def noise_background(rng: np.random.Generator, cam: Camera) -> np.ndarray:
    """Band-limited colour noise at a random scale plus a random gradient."""
    h, w = cam.height, cam.width
    scale = rng.integers(4, 64)
    small = rng.uniform(0, 255, (h // scale + 2, w // scale + 2, 3)).astype(np.float32)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    grad = np.linspace(rng.uniform(-60, 60), rng.uniform(-60, 60), h)[:, None, None]
    return np.clip(img + grad + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)


def render(
    frame: Airframe,
    kp3d: dict[str, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    cam: Camera,
    rng: np.random.Generator,
    albedo: np.ndarray,
    background: np.ndarray | None = None,
) -> View:
    """Draw the airframe over `background` (BGR, camera size) or a random noise field."""
    K = cam.K
    pts = frame.tris.reshape(-1, 3) @ R.T + t  # (3N, 3) camera frame
    z = pts[:, 2].reshape(-1, 3)
    keep = (z > 0.05).all(axis=1)
    uv = (pts @ K.T)[:, :2] / pts[:, 2:3]
    uv = uv.reshape(-1, 3, 2)

    normals = np.cross(frame.tris[:, 1] - frame.tris[:, 0], frame.tris[:, 2] - frame.tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = rng.normal(size=3)
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ (R.T @ light))  # two-sided: STL winding is unreliable
    colour = (albedo * shade[:, None] * rng.uniform(0.6, 1.2) * 255).clip(0, 255)

    image = noise_background(rng, cam) if background is None else background.copy()
    ids = np.full((cam.height, cam.width), -1, np.int32)
    order = np.flatnonzero(keep)[np.argsort(-z[keep].mean(axis=1))]
    for i in order:
        poly = np.round(uv[i] * 16).astype(np.int32)  # 4 fractional bits
        cv2.fillConvexPoly(image, poly, colour[i][::-1].tolist(), cv2.LINE_8, 4)
        cv2.fillConvexPoly(ids, poly, int(i), cv2.LINE_8, 4)

    depth_of_tri = (frame.centroids @ R.T + t)[:, 2]
    kp2d, vis = {}, {}
    for name, p in kp3d.items():
        pc = R @ p + t
        u, v = (K @ pc)[:2] / pc[2]
        kp2d[name] = np.array([u, v])
        inside = pc[2] > 0.05 and 0 <= u < cam.width and 0 <= v < cam.height
        hit = ids[int(v), int(u)] if inside else -1
        vis[name] = bool(inside and (hit < 0 or depth_of_tri[hit] >= pc[2] - 0.03))
    return View(image, ids, R, t, kp2d, vis)
