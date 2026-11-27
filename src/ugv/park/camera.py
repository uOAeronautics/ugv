"""Image-in-the-loop perception for closed-loop parking evaluation.

Each camera tick renders the true 1280x720 scene with `ugv.pose.render`, applies image-level
sensor faults (radial distortion, Gaussian blur, exposure stops/spells, sensor noise, occluder)
directly to the pixels, runs `ugv.pose.detector.KeypointDetector`, and feeds its detections into
`ugv.park.perception.measurement_from_detections`. No ground-truth pose leaks to the detector or
solver, and no fallback to the sampled `DetectorModel` exists.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ugv.park.dagger import EpisodeLog
from ugv.park.geometry import CAMERA_X_M, CAMERA_Z_M, bay
from ugv.park.perception import (
    HEIGHT,
    WIDTH,
    CameraError,
    DetectorModel,
    F,
    _true_projection,
    camera_points,
    measurement_from_detections,
    model_points,
)
from ugv.pose.detector import Detection, load_default
from ugv.pose.mesh import Airframe, keypoints, load_airframe
from ugv.pose.render import SDF_PART_RGB, Camera, View, part_albedo, render, room_background


def sdf_albedo(frame: Airframe) -> np.ndarray:
    """Per-triangle RGB albedo matching `simulation/gazebo/strl_001/model.sdf` and `stl.js`."""
    return part_albedo(frame, **SDF_PART_RGB)


def camera_extrinsics(
    rel: np.ndarray, cam: CameraError, floor_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Model-to-camera `(R, t)` matching `camera_points(rel)` followed by `CameraError` angles."""
    c, s = np.cos(rel[2]), np.sin(rel[2])
    R_nom = np.array([[-s, -c, 0.0], [0.0, 0.0, -1.0], [c, -s, 0.0]])
    t_nom = np.array([-rel[1], floor_z + CAMERA_Z_M, rel[0] - CAMERA_X_M])
    cy, sy = np.cos(cam.yaw), np.sin(cam.yaw)
    cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)
    M = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]) @ np.array(
        [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]]
    )
    return M @ R_nom, M @ t_nom


def _lens_maps(cam: CameraError) -> tuple[np.ndarray, np.ndarray]:
    """Inverse radial distortion maps matching `scene3d.js` `unwarp()` and `_true_projection`."""
    fx, cx, cy = F * cam.focal_scale, WIDTH / 2 + cam.cx_px, HEIGHT / 2 + cam.cy_px
    xs, ys = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
    nx, ny = (xs - cx) / fx, (ys - cy) / fx
    mx, my = nx.copy(), ny.copy()
    for _ in range(4):
        r2 = mx * mx + my * my
        scale = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
        mx, my = nx / scale, ny / scale
    return (cx + mx * fx).astype(np.float32), (cy + my * fx).astype(np.float32)


def apply_image_faults(
    gray: np.ndarray,
    cam: CameraError,
    fx: dict | None,
    noise_px: float,
    rng: np.random.Generator,
    remap_cache: dict,
    occluder: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Apply lens distortion, blur, exposure, sensor noise, and occluder to a 1280x720 frame."""
    if cam.k1 or cam.k2:
        key = (
            round(cam.focal_scale, 6),
            round(cam.cx_px, 3),
            round(cam.cy_px, 3),
            round(cam.k1, 6),
            round(cam.k2, 6),
        )
        if key not in remap_cache:
            remap_cache[key] = _lens_maps(cam)
        gray = cv2.remap(gray, *remap_cache[key], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    blur = float(fx.get("blur", 0.0)) if fx else 0.0
    if blur > 0.35:
        gray = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=min(blur, 25.0), borderType=cv2.BORDER_REPLICATE
        )
    stops = float(fx.get("exposure", 0.0)) if fx else 0.0
    if stops != 0.0:
        lin = np.power(gray.astype(np.float32) / 255.0, 2.2) * (2.0**stops)
        lin = np.where(lin <= 0.6, lin, np.minimum(1.0, 1.0 - 0.4 * np.exp(-(lin - 0.6) / 0.4)))
        gray = np.clip(np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    if noise_px > 0.0:
        gray = np.clip(
            gray.astype(np.float32) + rng.normal(0.0, 4.0 * noise_px, gray.shape), 0, 255
        ).astype(np.uint8)
    occ = fx.get("occluder") if fx is not None else occluder
    if occ is not None:
        h, w = gray.shape[:2]
        u0, v0, u1, v1 = (int(round(v)) for v in occ)
        gray = gray.copy()
        gray[max(0, v0) : min(h, v1), max(0, u0) : min(w, u1)] = 0
    return gray


class NetworkPerception:
    """Closed-loop image renderer + real ONNX `KeypointDetector` adapter."""

    def __init__(
        self,
        detector=None,
        *,
        threads: int = 1,
        dump_frames: bool = False,
        dump_err_px: float = 10.0,
        keep_overlay: bool = False,
    ) -> None:
        try:
            self.frame = load_airframe()
            self.kp3d = keypoints(self.frame, near=True)
            self.kp_names = tuple(self.kp3d)
            self.albedo = sdf_albedo(self.frame)
            self.floor_z = bay().floor_z
        except Exception as e:
            raise RuntimeError(f"cannot initialize airframe renderer/mesh: {e}") from e
        self.detector = load_default(threads=threads) if detector is None else detector
        self.dump_frames, self.dump_err_px, self.keep_overlay = (
            dump_frames,
            dump_err_px,
            keep_overlay,
        )
        self._remap_cache: dict = {}
        self.ep = EpisodeLog()

    def reset_episode(self) -> None:
        self.ep = EpisodeLog()

    def render_frame(
        self,
        rel_true: np.ndarray,
        cam: CameraError,
        fx: dict | None,
        noise_px: float,
        scene_seed: int,
        rng: np.random.Generator,
        occluder: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, Camera, View, np.ndarray, np.ndarray]:
        R, t = camera_extrinsics(rel_true, cam, self.floor_z)
        cam_model = Camera(
            WIDTH,
            HEIGHT,
            intrinsics=(F * cam.focal_scale, WIDTH / 2 + cam.cx_px, HEIGHT / 2 + cam.cy_px),
        )
        bg = room_background(np.random.default_rng(scene_seed), cam_model, R)
        view = render(
            self.frame,
            self.kp3d,
            R,
            t,
            cam_model,
            np.random.default_rng(scene_seed),
            self.albedo,
            bg,
        )
        gray = apply_image_faults(
            cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY),
            cam,
            fx,
            noise_px,
            rng,
            self._remap_cache,
            occluder,
        )
        true_uv, front = _true_projection(
            camera_points(rel_true, DetectorModel(near_field=True)), cam
        )
        self_ok = np.array([view.kp_visible[n] for n in self.kp_names], dtype=bool)
        true_vis = (
            front
            & self_ok
            & (true_uv[:, 0] >= 0)
            & (true_uv[:, 0] < WIDTH)
            & (true_uv[:, 1] >= 0)
            & (true_uv[:, 1] < HEIGHT)
        )
        return gray, cam_model, view, true_uv, true_vis

    def measure(
        self,
        rel_true: np.ndarray,
        rng: np.random.Generator,
        det: DetectorModel,
        cam: CameraError | None = None,
        prior: np.ndarray | None = None,
        seen: list | None = None,
        occluder: tuple[float, float, float, float] | None = None,
        *,
        fx: dict | None = None,
        noise_px: float = 0.0,
        scene_seed: int = 0,
        fsm_state: str = "",
        t: float = 0.0,
    ) -> np.ndarray | None:
        cam = cam or CameraError()
        t0 = time.perf_counter()
        gray, cam_model, view, true_uv, true_vis = self.render_frame(
            rel_true, cam, fx, noise_px, scene_seed, rng, occluder
        )
        t1 = time.perf_counter()
        det_res: Detection = self.detector(gray)
        t2 = time.perf_counter()
        n_pts = len(model_points(det))
        pts = sorted((p for p in det_res.points if 0 <= int(p[0]) < n_pts), key=lambda p: int(p[0]))
        if seen is not None:
            if pts:
                keep = np.zeros(n_pts, dtype=bool)
                keep[[int(p[0]) for p in pts]] = True
                seen.append((keep, np.array([[float(p[1]), float(p[2])] for p in pts])))
            else:
                seen.append(None)
        m = measurement_from_detections(pts, det, prior)
        pose = np.concatenate([view.R.ravel(), view.t]).astype(np.float32)
        self.ep.record(
            gray,
            cam_model,
            pose,
            true_uv,
            true_vis,
            det_res,
            pts,
            m,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            fsm_state,
            t,
            rng,
            dump_frames=self.dump_frames,
            dump_err_px=self.dump_err_px,
            keep_overlay=self.keep_overlay,
        )
        return m
