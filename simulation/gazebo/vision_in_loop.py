"""Synthetic camera-in-the-loop for Gazebo L6 lockstep evals.

Why synthetic instead of live Ogre2 frames: `camera_validation.json` proves the
Ogre2 camera sensor renders headless (`gz sim -s`, 1280x720 L8 @30 Hz, <1.1 px
target error), but `lockstep_env.headless_world` strips every camera/IMU sensor
for physics speed and `lockstep_bridge` has no image transport, so piping live
Ogre2 frames into the 50 Hz lockstep loop needs new async transport plumbing.
This module keeps Gazebo physics (poses, mecanum dynamics) and renders the 3D
ArUco constellation (#14 entry, #6 +x side, #3 -x side in the 2026-11-13 real
layout; #12/#9/#13/#14 selectable via layout='legacy', per `drone_model.py`)
through a pinhole with the real camera intrinsics, then
decodes every frame with the real `src/ugv/aruco.py:detect_gray` + solvePnP and
inverts the PnP transform into a car pose. That tests genuine OpenCV pixel
decoding + PnP pose estimation inside the 3D physics loop, deterministically,
at ~ms/frame, with per-episode photometric domain randomization
(brightness/contrast/gamma, shadow gradient, glare, sensor noise, blur).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
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
MARKER_LAYOUTS = ("real", "legacy")
DEFAULT_LAYOUT = "real"
MARKER_IDS_BY_LAYOUT = {
    "real": (14, 6, 3),
    "legacy": (12, 9, 13, 14),
}
ALL_MARKER_IDS = (3, 6, 9, 12, 13, 14)
MARKER_IDS = MARKER_IDS_BY_LAYOUT[DEFAULT_LAYOUT]
CARD_SIZE_M = 0.068  # legacy symmetric 50 mm tag + 9 mm quiet zone per side
# Real loose paper cards (observed 2026-11-13): ~1.5 mm bottom quiet zone,
# ~3 mm sides, ~18 mm top hanging margin (left, right, bottom, top in card frame).
REAL_PAPER_BOUNDS = (-0.028, 0.028, -0.0265, 0.043)

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


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def marker_world_poses(
    layout: str = DEFAULT_LAYOUT,
    randomize_tilt: bool = False,
    rng: np.random.Generator | None = None,
    tilt_deg: tuple[float, float] = (10.0, 25.0),
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Marker id -> (center xyz in rl/sim.py drone frame, local-to-drone rotation).

    layout='real' (2026-11-13): 14 (+y entry, 25 deg pitch), 6 (+x side), 3 (-x side).
    A fourth -y tag is omitted because wires obscured its ID.
    layout='legacy': original 4-card placement (12, 9, 13, 14) for reproducibility.
    When randomize_tilt=True, adds +/-10..25 deg roll/pitch jitter per card.
    """
    if layout not in MARKER_LAYOUTS:
        raise ValueError(f"layout must be one of {MARKER_LAYOUTS}, got {layout!r}")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if layout == "legacy":
        out[12] = (np.array([0.0, 0.168, 0.092]), _rx(-math.pi / 2))
        out[9] = (np.array([0.0, -0.168, 0.092]), _rx(math.pi / 2))
        out[13] = (np.array([-0.058, -0.005, 0.202]), _rz(-math.pi / 2) @ _rx(math.pi / 2))
        out[14] = (np.array([0.058, -0.005, 0.202]), _rz(math.pi / 2) @ _rx(math.pi / 2))
    else:
        # Real constellation in rl/sim.py drone frame (skids along y, car enters from +y).
        # Tag 14 pitched ~25 deg; 6 & 3 from online co-observation estimates; -y tag omitted.
        out[14] = (np.array([-0.028, 0.0, 0.14]),
                   _rz(math.pi) @ _rx(math.pi / 2) @ _rx(math.radians(-25.0)))
        out[6] = (np.array([0.14, -0.09, 0.15]), _rz(math.pi / 2) @ _rx(math.pi / 2))
        out[3] = (np.array([-0.13, 0.0, 0.16]), _rz(-math.pi / 2) @ _rx(math.pi / 2))
    if randomize_tilt:
        g = rng if rng is not None else np.random.default_rng(0)
        lo, hi = tilt_deg
        for mid, (center, R) in list(out.items()):
            rx_j = math.radians(float(g.choice((-1.0, 1.0)) * g.uniform(lo, hi)))
            rz_j = math.radians(float(g.choice((-1.0, 1.0)) * g.uniform(lo, hi)))
            out[mid] = (center.copy(), R @ _rz(rz_j) @ _rx(rx_j))
    return out


MARKERS = marker_world_poses(DEFAULT_LAYOUT)


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
            for i in ALL_MARKER_IDS}


_TEMPLATES: dict[int, np.ndarray] | None = None


def render_frame(
    pose: tuple[float, float, float],
    layout: str = DEFAULT_LAYOUT,
    markers: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Render the ArUco card constellation as 1280x720 gray.

    Far-to-near painter's order; back-facing and behind-camera markers culled.
    """
    import cv2
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _marker_templates()
    assert _TEMPLATES is not None
    active = markers if markers is not None else (
        MARKERS if layout == DEFAULT_LAYOUT else marker_world_poses(layout)
    )
    x, y, yaw = (float(v) for v in pose)
    cam_c, r, f = camera_pose(x, y, yaw)
    img = np.full((FRAME_HEIGHT, FRAME_WIDTH), 128, dtype=np.uint8)
    h = TAG_SIZE_M / 2
    if layout == "legacy":
        q = CARD_SIZE_M / 2
        card_local = np.array([[-q, -q, 0], [q, -q, 0], [q, q, 0], [-q, q, 0]])
    else:
        cl, cr, cb, ct = REAL_PAPER_BOUNDS
        card_local = np.array([[cl, cb, 0], [cr, cb, 0], [cr, ct, 0], [cl, ct, 0]])
    tag_local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    items = []
    for mid, (center, R) in active.items():
        facing = float(R[:, 2] @ (cam_c - center))
        if facing <= 0:
            continue
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


def _local_ippe(
    d, markers: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Both IPPE_SQUARE solutions for one detection, as world-frame (R_cw, t).

    Solved in marker-local coordinates (where IPPE is stable) and converted
    with the known marker placement. Sorted by reprojection error.
    """
    import cv2
    h = TAG_SIZE_M / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                   dtype=np.float64)
    img_pts = np.asarray(d.corners, dtype=np.float64)
    K = np.array([[FOCAL_PX, 0, CX], [0, FOCAL_PX, CY], [0, 0, 1]], dtype=np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img_pts, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    active = markers if markers is not None else MARKERS
    center, R_mw = active[d.tag_id]
    out = []
    for rv, tv, err in zip(rvecs, tvecs, errs.ravel()):
        R_cm, _ = cv2.Rodrigues(rv)
        R_cw = R_cm @ R_mw.T
        t_w = tv.reshape(3) - R_cw @ center
        out.append((R_cw, t_w, float(err)))
    out.sort(key=lambda s: s[2])
    return out


def _pose_from_rt(R_cw: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, bool]:
    cam_c = (-R_cw.T @ t.reshape(3))
    if not (-0.2 < cam_c[2] < 1.0):
        return np.zeros(3), False
    f_w = R_cw.T @ np.array([0.0, 0.0, 1.0])  # camera +z axis in world
    n = math.hypot(f_w[0], f_w[1])
    if n < 1e-6:
        return np.zeros(3), False
    f_w = f_w / n
    yaw = math.atan2(f_w[0], -f_w[1])
    est = np.array([cam_c[0] - CAM_FWD_M * f_w[0],
                    cam_c[1] - CAM_FWD_M * f_w[1], yaw])
    est[2] = math.atan2(math.sin(est[2]), math.cos(est[2]))
    return est, True


def _ang_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def decode_pose(
    img: np.ndarray,
    prev_yaw: float | None = None,
    markers: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    layout: str = DEFAULT_LAYOUT,
) -> tuple[np.ndarray, bool]:
    """Real aruco.py pixel decode + PnP -> car (x, y, yaw).

    Per-marker local IPPE_SQUARE gives both ambiguous solutions; the winner
    is picked by cross-marker camera consistency (>=2 markers, then a joint
    world-frame ITERATIVE refine) or by reprojection error with a temporal
    yaw prior as tie-break (1 marker with near-equal errors).
    """
    import cv2
    active = markers if markers is not None else (
        MARKERS if layout == DEFAULT_LAYOUT else marker_world_poses(layout)
    )
    dets = [d for d in ArucoFollower.detect_gray(img) if d.tag_id in active]
    if not dets:
        return np.zeros(3), False
    sols = [(d, _local_ippe(d, active)) for d in dets]
    sols = [(d, s) for d, s in sols if s]
    if not sols:
        return np.zeros(3), False

    def yaw_of(R_cw: np.ndarray) -> float:
        f_w = R_cw.T @ np.array([0.0, 0.0, 1.0])  # camera +z axis in world
        return math.atan2(f_w[0], -f_w[1])

    def cam_c(R_cw: np.ndarray, t_w: np.ndarray) -> np.ndarray:
        return -R_cw.T @ t_w

    if len(sols) == 1:
        s = sols[0][1]
        if len(s) == 2 and prev_yaw is not None and s[0][2] > 0 and s[1][2] / s[0][2] < 3.0:
            s = sorted(s, key=lambda c: _ang_diff(yaw_of(c[0]), prev_yaw))
        R_cw, t_w, _ = s[0]
        return _pose_from_rt(R_cw, t_w)

    import itertools
    best: tuple | None = None
    for combo in itertools.product(*[[(d, c) for c in s] for d, s in sols]):
        centers = [cam_c(c[0], c[1]) for _, c in combo]
        spread = sum(float(np.linalg.norm(a - b))
                     for i, a in enumerate(centers) for b in centers[i + 1:])
        err = sum(c[2] for _, c in combo)
        key = (spread, err)
        if best is None or key < best[0]:
            best = (key, combo)
    assert best is not None
    combo = best[1]
    R0, t0, _ = min((c for _, c in combo), key=lambda c: c[2])
    h = TAG_SIZE_M / 2
    local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                     dtype=np.float64)
    obj, img_pts = [], []
    for d, _ in combo:
        center, R_mw = active[d.tag_id]
        obj.append(center + local @ R_mw.T)
        img_pts.append(np.asarray(d.corners, dtype=np.float64))
    obj = np.vstack(obj)
    img_pts = np.vstack(img_pts)
    K = np.array([[FOCAL_PX, 0, CX], [0, FOCAL_PX, CY], [0, 0, 1]], dtype=np.float64)
    rvec0, _ = cv2.Rodrigues(R0)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, np.zeros(5),
                                  rvec=rvec0, tvec=t0.reshape(3, 1),
                                  useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        R_cw, _ = cv2.Rodrigues(rvec)
        return _pose_from_rt(R_cw, tvec)
    return _pose_from_rt(R0, t0)


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
                 photometric: bool = True, layout: str = DEFAULT_LAYOUT,
                 randomize_tilt: bool = False):
        self.base = Sensors(level, rng, max_time, init_pose=init_pose)
        self.enabled = True
        self.layout = layout
        self.randomize_tilt = bool(randomize_tilt)
        self.decode_markers = marker_world_poses(layout)
        self.render_markers = (
            marker_world_poses(layout, randomize_tilt=True, rng=rng)
            if self.randomize_tilt else self.decode_markers
        )
        self.dr = sample_dr(rng) if photometric else DRParams()
        self.rng = rng
        self.ring: list[tuple[np.ndarray, bool]] = []
        self.frames = 0
        self.decodes = 0
        init_yaw = float(init_pose[2]) if init_pose is not None else 0.0
        self._prev_yaw: float | None = init_yaw

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
        img = render_frame(pose, layout=self.layout, markers=self.render_markers)
        img = apply_photometric_dr(img, self.dr, self.rng)
        est, ok = decode_pose(img, self._prev_yaw, markers=self.decode_markers)
        if ok:
            self._prev_yaw = float(est[2])
        self.frames += 1
        self.decodes += int(ok)
        _, seen = self.base.tag(pose)  # advances latency ring / age / FOV / drops
        self.ring.append((est, ok))
        self.ring = self.ring[-40:]
        k = int(round(self.base.latency / DT))
        d_est, d_ok = self.ring[max(0, len(self.ring) - 1 - k)]
        if not seen or not d_ok:
            if seen and not d_ok:
                self.base.age += DT  # decoded drop: frame arrived, tag missing
            return np.zeros(3), False
        return d_est, True

    def gyro(self, w_true: float) -> float:
        return self.base.gyro(w_true)


def selftest(layout: str = DEFAULT_LAYOUT, randomize_tilt: bool = False,
             seed: int = 0) -> dict:
    """Render/decode round-trip at known poses (no DR) + DR decode smoke."""
    decode_m = marker_world_poses(layout)
    render_m = (marker_world_poses(layout, randomize_tilt=True, rng=np.random.default_rng(seed))
                if randomize_tilt else decode_m)
    poses = (
        [(0.0, 0.8, 0.0), (0.3, 1.0, 0.2), (-0.25, 0.9, -0.2),
         (0.7, 0.0, -1.5), (-0.7, 0.0, 1.5)]
        if layout == "real"
        else [(0.0, 0.8, 0.0), (0.3, 1.0, 0.2), (-0.25, 0.6, -0.3), (0.1, 0.5, 0.9)]
    )
    rep = []
    for pose in poses:
        est, ok = decode_pose(render_frame(pose, layout=layout, markers=render_m),
                              markers=decode_m, layout=layout)
        err = (float(np.hypot(est[0] - pose[0], est[1] - pose[1])),
               float(abs(math.atan2(math.sin(est[2] - pose[2]),
                                    math.cos(est[2] - pose[2]))))) if ok else None
        rep.append({"pose": list(pose), "ok": ok, "err_xy_yaw": err})
    hits = 0
    trials = 40
    for i in range(trials):
        rng2 = np.random.default_rng(1000 + i)
        img = apply_photometric_dr(
            render_frame((0.15, 0.9, 0.1), layout=layout, markers=render_m),
            sample_dr(rng2), rng2)
        _, ok = decode_pose(img, markers=decode_m, layout=layout)
        hits += int(ok)
    return {"layout": layout, "randomize_tilt": randomize_tilt,
            "marker_ids": list(MARKER_IDS_BY_LAYOUT[layout]),
            "roundtrip": rep, "dr_decode_rate": hits / trials, "dr_trials": trials}
