"""Synthetic camera-in-the-loop for Gazebo L6 lockstep evals.

Why synthetic instead of live Ogre2 frames: `camera_validation.json` proves the
Ogre2 camera sensor renders headless (`gz sim -s`, 1280x720 L8 @30 Hz, <1.1 px
target error), but `lockstep_env.headless_world` strips every camera/IMU sensor
for physics speed and `lockstep_bridge` has no image transport, so piping live
Ogre2 frames into the 50 Hz lockstep loop needs new async transport plumbing.
This module keeps Gazebo physics (poses, mecanum dynamics) and renders the 3D
ArUco constellation (#12, #9 leg cards; #13, #14 payload cards, per
`drone_model.py`) through a pinhole with the real camera intrinsics, then
decodes every frame with the real `src/ugv/aruco.py:detect_gray` + solvePnP and
inverts the PnP transform into a car pose. That tests genuine OpenCV pixel
decoding + PnP pose estimation inside the 3D physics loop, deterministically,
at ~ms/frame, with per-episode photometric domain randomization
(brightness/contrast/gamma, shadow gradient, glare, sensor noise, blur).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from lockstep_env import DT, Sensors  # noqa: E402
from ugv.aruco import FRAME_WIDTH, FRAME_HEIGHT, FOCAL_PX, TAG_SIZE_M  # noqa: E402
from ugv.aruco import ArucoFollower  # noqa: E402

CX, CY = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
MARKER_IDS = (12, 9, 13, 14)
CARD_SIZE_M = 0.068  # 50 mm tag + 9 mm white quiet zone per side

# Camera mount on the car (camera_model.py + dimensions.json): stand 4 cm back
# from the 7.5 in board front edge, lens 15.8 mm ahead of the stand face.
BOARD_FRONT_M = 7.5 * 0.0254 / 2
CAM_FWD_M = (BOARD_FRONT_M - 0.04) + 0.0158
CAM_Z_M = 1.75 * 0.0254 + 0.04 / 2


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def marker_world_poses() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Marker id -> (center xyz in drone frame, local-to-drone rotation).

    Mirrors the card() placements in drone_model.py (static photo_quadcopter at
    the world origin of the drone frame used by lockstep_env.to_drone).
    """
    out = {}
    halves = TAG_SIZE_M / 2
    _ = halves
    out[12] = (np.array([0.0, 0.168, 0.092]), _rx(-math.pi / 2))
    out[9] = (np.array([0.0, -0.168, 0.092]), _rx(math.pi / 2))
    out[13] = (np.array([-0.058, -0.005, 0.202]), _rz(-math.pi / 2) @ _rx(math.pi / 2))
    out[14] = (np.array([0.058, -0.005, 0.202]), _rz(math.pi / 2) @ _rx(math.pi / 2))
    return out


MARKERS = marker_world_poses()


def camera_pose(x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Car pose (drone frame) -> (camera center, right vec, forward vec)."""
    # Facing f with up +z, the right hand is f x up (e.g. yaw 0: f=(0,-1,0) -> r=(-1,0,0)).
    r = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
    f = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    c = np.array([x + CAM_FWD_M * f[0], y + CAM_FWD_M * f[1], CAM_Z_M])
    return c, r, f


def project(corners_w: np.ndarray, cam_c: np.ndarray, r: np.ndarray,
            f: np.ndarray) -> np.ndarray | None:
    """World xyz corners -> pixel uv. None if any corner behind the near clip."""
    d = corners_w - cam_c
    depth = d @ f
    if np.any(depth < 0.05):
        return None
    u = CX + FOCAL_PX * (d @ r) / depth
    v = CY - FOCAL_PX * d[:, 2] / depth
    return np.stack([u, v], axis=1)


def _marker_templates() -> dict[int, np.ndarray]:
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return {i: cv2.aruco.generateImageMarker(dictionary, i, 120, borderBits=1)
            for i in MARKER_IDS}


_TEMPLATES: dict[int, np.ndarray] | None = None


def render_frame(pose: tuple[float, float, float],
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Render the 68 mm-white-card constellation as 1280x720 gray.

    Far-to-near painter's order; back-facing and behind-camera markers culled.
    """
    import cv2
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _marker_templates()
    assert _TEMPLATES is not None
    x, y, yaw = (float(v) for v in pose)
    cam_c, r, f = camera_pose(x, y, yaw)
    img = np.full((FRAME_HEIGHT, FRAME_WIDTH), 128, dtype=np.uint8)
    h = TAG_SIZE_M / 2
    q = CARD_SIZE_M / 2
    items = []
    for mid, (center, R) in MARKERS.items():
        facing = float(R[:, 2] @ (cam_c - center))
        if facing <= 0:
            continue
        card_local = np.array([[-q, -q, 0], [q, -q, 0], [q, q, 0], [-q, q, 0]])
        tag_local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        card_w = center + card_local @ R.T
        tag_w = center + tag_local @ R.T
        depth = float((center - cam_c) @ f)
        if depth < 0.05:
            continue
        items.append((depth, mid, card_w, tag_w))
    for _, mid, card_w, tag_w in sorted(items, reverse=True):
        card_px = project(card_w, cam_c, r, f)
        tag_px = project(tag_w, cam_c, r, f)
        if card_px is None or tag_px is None:
            continue
        cv2.fillConvexPoly(img, card_px.astype(np.int32), 255)
        src = _TEMPLATES[mid]  # uint8; warp requires dst depth == src depth
        M = cv2.getPerspectiveTransform(
            np.array([[0, 0], [src.shape[1], 0], [src.shape[1], src.shape[0]],
                      [0, src.shape[0]]], dtype=np.float32),
            tag_px.astype(np.float32))
        cv2.warpPerspective(src, M, (FRAME_WIDTH, FRAME_HEIGHT), img,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_TRANSPARENT)
    _ = rng  # frame noise comes from apply_photometric_dr, not the renderer
    return img


@dataclass
class DRParams:
    brightness: float = 1.0   # multiplicative, sampled [0.25, 1.8]
    contrast: float = 1.0     # around mid-gray, sampled [0.7, 1.4]
    gamma: float = 1.0        # sampled [0.7, 1.4]
    shadow_strength: float = 0.0  # linear gradient darkening, [0, 0.6]
    shadow_angle: float = 0.0
    glare_gain: float = 0.0   # saturating gaussian blob, [0, 1.2]
    glare_xy: tuple = (0.5, 0.5)
    glare_sigma: float = 90.0
    noise_sigma: float = 0.0  # gaussian sensor noise, [0, 12]
    blur_k: int = 1           # motion/defocus proxy in {1, 3, 5}


def sample_dr(rng: np.random.Generator) -> DRParams:
    gx, gy = float(rng.uniform()), float(rng.uniform())
    return DRParams(
        brightness=float(rng.uniform(0.25, 1.8)),
        contrast=float(rng.uniform(0.7, 1.4)),
        gamma=float(rng.uniform(0.7, 1.4)),
        shadow_strength=float(rng.uniform(0.0, 0.6)),
        shadow_angle=float(rng.uniform(0.0, 2 * math.pi)),
        glare_gain=float(rng.uniform(0.0, 1.2)) if rng.random() < 0.5 else 0.0,
        glare_xy=(gx, gy),
        glare_sigma=float(rng.uniform(40.0, 160.0)),
        noise_sigma=float(rng.uniform(0.0, 12.0)),
        blur_k=int(rng.choice([1, 1, 3, 3, 5])),
    )


def apply_photometric_dr(img: np.ndarray, p: DRParams,
                         rng: np.random.Generator) -> np.ndarray:
    """Photometric domain randomization on a gray frame (all in float32)."""
    import cv2
    f = img.astype(np.float32)
    f = (f - 128.0) * p.contrast + 128.0
    f *= p.brightness
    if p.shadow_strength > 0:
        yy, xx = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
        proj = (xx * math.cos(p.shadow_angle) + yy * math.sin(p.shadow_angle))
        proj /= math.hypot(FRAME_WIDTH, FRAME_HEIGHT)
        f *= (1.0 - p.shadow_strength * proj)
    if p.glare_gain > 0:
        yy, xx = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
        gx = (p.glare_xy[0] * FRAME_WIDTH - xx) / p.glare_sigma
        gy = (p.glare_xy[1] * FRAME_HEIGHT - yy) / p.glare_sigma
        f += p.glare_gain * 255.0 * np.exp(-0.5 * (gx * gx + gy * gy))
    if p.gamma != 1.0:
        f = np.clip(f, 0, 255) / 255.0
        f = np.power(f, p.gamma) * 255.0
    if p.noise_sigma > 0:
        f += rng.normal(0.0, p.noise_sigma, size=f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if p.blur_k > 1:
        out = cv2.GaussianBlur(out, (p.blur_k, p.blur_k), 0)
    return out


def decode_pose(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Real aruco.py pixel decode + PnP inversion -> car (x, y, yaw) estimate."""
    import cv2
    dets = ArucoFollower.detect_gray(img)
    cands = [d for d in dets if d.tag_id in MARKERS and d.rotation_vector
             and d.translation_vector]
    if not cands:
        return np.zeros(3), False
    d = max(cands, key=lambda c: c.size_px)
    center, R_mw = MARKERS[d.tag_id]
    R_cm, _ = cv2.Rodrigues(np.asarray(d.rotation_vector, dtype=np.float64))
    t = np.asarray(d.translation_vector, dtype=np.float64).reshape(3)
    R_cw = R_cm @ R_mw.T
    cam_c = center - R_mw @ R_cm.T @ t
    f_w = R_cw @ np.array([0.0, 0.0, 1.0])
    n = math.hypot(f_w[0], f_w[1])
    if n < 1e-6 or cam_c[2] < -0.5 or cam_c[2] > 1.5:
        return np.zeros(3), False
    f_w = f_w / n
    yaw = math.atan2(f_w[0], -f_w[1])
    est = np.array([cam_c[0] - CAM_FWD_M * f_w[0],
                    cam_c[1] - CAM_FWD_M * f_w[1], yaw])
    est[2] = math.atan2(math.sin(est[2]), math.cos(est[2]))
    return est, True


class VisionSensors:
    """Latency/dropout/blackout timing from Sensors; tag values from vision.

    Each step renders the constellation at the true Gazebo pose, applies the
    per-episode photometric DR, and decodes via real detect_gray + PnP. The
    delayed estimate runs through the same latency ring as the emulated path,
    so decode failures surface as invalid tags (age grows), exactly like a
    lost camera frame on the car.
    """

    def __init__(self, level: dict, rng: np.random.Generator, max_time: float,
                 init_pose: tuple[float, float, float] | None = None,
                 photometric: bool = True):
        self.base = Sensors(level, rng, max_time, init_pose=init_pose)
        self.enabled = True
        self.dr = sample_dr(rng) if photometric else DRParams()
        self.rng = rng
        self.vring: list[tuple[np.ndarray, bool]] = []
        self.frames = 0
        self.decodes = 0

    @property
    def latency(self) -> float:
        return self.base.latency

    @property
    def age(self) -> float:
        return self.base.age

    @property
    def imu_yaw(self) -> float:
        return self.base.imu_yaw

    def tag(self, pose: tuple[float, float, float]) -> tuple[np.ndarray, bool]:
        img = render_frame(pose)
        img = apply_photometric_dr(img, self.dr, self.rng)
        est, ok = decode_pose(img)
        self.frames += 1
        self.decodes += int(ok)
        _, seen = self.base.tag(pose)  # advances latency ring / age / FOV / drops
        self.vring.append((est, ok))
        self.vring = self.vring[-40:]
        k = int(round(self.base.latency / DT))
        d_est, d_ok = self.vring[max(0, len(self.vring) - 1 - k)]
        if not seen or not d_ok:
            if seen and not d_ok:
                self.base.age += DT  # decoded drop: frame arrived, tag missing
            return np.zeros(3), False
        return d_est, True

    def gyro(self, w_true: float) -> float:
        return self.base.gyro(w_true)


def selftest() -> dict:
    """Render/decode round-trip at known poses (no DR) + DR decode smoke."""
    rng = np.random.default_rng(0)
    rep = []
    for pose in [(0.0, 0.8, 0.0), (0.3, 1.0, 0.2), (-0.25, 0.6, -0.3),
                 (0.1, 0.5, 0.9)]:
        est, ok = decode_pose(render_frame(pose))
        err = (float(np.hypot(est[0] - pose[0], est[1] - pose[1])),
               float(abs(math.atan2(math.sin(est[2] - pose[2]),
                                    math.cos(est[2] - pose[2]))))) if ok else None
        rep.append({"pose": list(pose), "ok": ok, "err_xy_yaw": err})
    hits = 0
    trials = 40
    for i in range(trials):
        rng2 = np.random.default_rng(1000 + i)
        img = apply_photometric_dr(render_frame((0.15, 0.9, 0.1)),
                                  sample_dr(rng2), rng2)
        _, ok = decode_pose(img)
        hits += int(ok)
    return {"roundtrip": rep, "dr_decode_rate": hits / trials, "dr_trials": trials}
