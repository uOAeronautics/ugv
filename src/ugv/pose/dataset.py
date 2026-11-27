"""Synthetic keypoint training samples: the airframe over real car-camera frames, domain-randomised.

Used by `tools/pose/make_dataset.py` (a fixed set on disk) and `train/pose_kp.py --pool` (a
rolling pool that never repeats a set). One sample per integer seed, reproducible; worker
processes call `init_worker` once, then `sample(seed)`. See `tools/pose/make_dataset.py` for the
mix of views.
"""

from __future__ import annotations

import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ugv.park.geometry import CAMERA_X_M, CAMERA_Z_M, bay
from ugv.park.perception import CameraError
from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.render import (
    CAR_VIEWS,
    Camera,
    CarLevelSpec,
    ViewSpec,
    procedural_texture,
    random_part_albedo,
    render,
    room_background,
    sample_bay_pose,
    sample_car_level_pose,
    sample_pose,
)

REPO = Path(__file__).resolve().parents[3]
W, H = 320, 180
CROP = 256
FULL = Camera().scaled(1280, 720)
_STATE: dict = {}
LEVEL = 0.55  # share of generic level car-camera views; `--bay` takes its extra share


def init_worker(frames: list[str], crop: bool, bay_frac: float) -> None:
    from ugv.park.camera import apply_image_faults

    frame = load_airframe()
    _STATE.update(
        frame=frame,
        kp=keypoints(frame, near=True),
        frames=frames,
        cam=Camera().scaled(W, H),
        crop=crop,
        bay=bay_frac,
        apply_faults=apply_image_faults,
    )
    _STATE["fixed"] = procedural_texture(frame.centroids, seed=0)
    _STATE["level"] = CarLevelSpec(floor_z=bay().floor_z, away=0.2)
    _STATE["corners"] = frame.tris.reshape(-1, 3)[::25]


def _crop_camera(rng: np.random.Generator, R: np.ndarray, t: np.ndarray) -> tuple[Camera, tuple]:
    """The crop a first stage would propose around the airframe in the full camera image."""
    p = _STATE["corners"] @ R.T + t
    p = p[p[:, 2] > 0.05]
    uv = (p @ FULL.K.T)[:, :2] / p[:, 2:3]
    lo, hi = np.percentile(uv, 2, axis=0), np.percentile(uv, 98, axis=0)
    centre = (lo + hi) / 2 + rng.normal(0, 0.08, 2) * (hi - lo)
    size = float(np.clip(max(hi - lo) * rng.uniform(1.1, 1.6), 64, 1400))
    x0, y0 = centre - size / 2
    return FULL.crop(x0, y0, size, CROP), (x0, y0, size)


def _keypoint_crop(rng: np.random.Generator, R: np.ndarray, t: np.ndarray) -> tuple[Camera, tuple]:
    """The crop the deployed first stage proposes (`ugv.pose.detector`): the box of the
    keypoints inside the full 1280x720 picture, 1.1-1.6x, 64-1400 px. With fewer than two in
    view (the car faces away, or the airframe is mostly out of frame), a random box: what stage
    1 hands over after a false alarm, which stage 2 must call empty."""
    kp = np.array(list(_STATE["kp"].values())) @ R.T + t
    front = kp[:, 2] > 0.05
    uv = (kp @ FULL.K.T)[:, :2] / np.where(front, kp[:, 2], 1.0)[:, None]
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
    if inside.sum() >= 2:
        lo, hi = uv[inside].min(0), uv[inside].max(0)
        centre = (lo + hi) / 2 + rng.normal(0, 0.05, 2) * (hi - lo)
        size = float(np.clip(max(hi - lo) * rng.uniform(1.1, 1.6), 64, 1400))
    else:
        size = float(rng.uniform(64, 700))
        centre = rng.uniform([0, 0], [1280, 720])
    x0, y0 = centre - size / 2
    return FULL.crop(x0, y0, size, CROP), (x0, y0, size)


def _background(rng: np.random.Generator, box: tuple | None = None) -> np.ndarray | None:
    if not _STATE["frames"] or rng.random() < 0.3:
        return None
    img = cv2.imread(rng.choice(_STATE["frames"]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if box is not None:  # the same crop of a random real frame (padded where it leaves the image)
        x0, y0, size = box
        M = np.array([[CROP / size, 0, -x0 * CROP / size], [0, CROP / size, -y0 * CROP / size]])
        img = cv2.warpAffine(
            img, M, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT
        )
        return np.clip(
            img.astype(np.float32) * rng.uniform(0.5, 1.5) + rng.uniform(-30, 30), 0, 255
        ).astype(np.uint8)
    s = rng.uniform(0.6, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
    return np.clip(
        img.astype(np.float32) * rng.uniform(0.5, 1.5) + rng.uniform(-30, 30), 0, 255
    ).astype(np.uint8)


def sample(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    st = _STATE
    # 55% the real car's level camera anywhere around the airframe (partial views and pictures
    # without it included), 25% aimed low views, 20% anywhere on the sphere
    u = rng.random()
    level = u < LEVEL + st["bay"]
    if u < LEVEL:
        R, t, _ = sample_car_level_pose(rng, st["level"])
    elif level:  # on the way into the bay: the airframe mostly out of the picture
        b = bay()
        R, t = sample_bay_pose(rng, b.floor_z, b.target, CAMERA_X_M, CAMERA_Z_M)
    else:
        spec = CAR_VIEWS if u < 0.8 else ViewSpec(distance_m=(0.4, 4.0))
        R, t = sample_pose(rng, st["cam"], spec)
    cam, box = (st["cam"], None)
    if st["crop"]:
        cam, box = _keypoint_crop(rng, R, t) if level else _crop_camera(rng, R, t)
    u_alb = rng.random()
    if u_alb < 0.5:
        albedo = random_part_albedo(st["frame"], rng)
    elif u_alb < 0.7:  # plain airframe paint, as the real one may be
        albedo = np.full_like(st["fixed"], rng.uniform(0.4, 0.95))
    elif u_alb < 0.8:
        albedo = st["fixed"]
    else:
        albedo = procedural_texture(st["frame"].centroids, int(rng.integers(1 << 30)))
    bg = room_background(rng, cam, R) if level and rng.random() < 0.35 else _background(rng, box)
    v = render(st["frame"], st["kp"], R, t, cam, rng, albedo, bg)
    w_, h_ = cam.width, cam.height
    g = cv2.cvtColor(v.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if rng.random() < 0.25:  # directional motion blur
        klen = int(rng.choice([5, 7, 9, 11]))
        k = np.zeros((klen, klen), np.float32)
        k[klen // 2] = 1.0 / klen
        rot = cv2.getRotationMatrix2D((klen / 2 - 0.5, klen / 2 - 0.5), rng.uniform(0, 180), 1)
        g = cv2.filter2D(g, -1, cv2.warpAffine(k, rot, (klen, klen)))
    g = 255 * (np.clip(g, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
    gray = np.clip(g, 0, 255).astype(np.uint8)
    scale = float(cam.K[0, 0] / FULL.K[0, 0])
    faulty = rng.random() < 0.45
    fx: dict = {}
    if faulty and rng.random() < 0.65:  # defocus / isotropic smear matching faults.py (up to 25 px)
        blur_1280 = float(np.exp(rng.uniform(np.log(1.5), np.log(25.0))))
        fx["blur"] = blur_1280 * scale
    elif rng.random() < 0.5:
        fx["blur"] = float(rng.uniform(0.4, 1.4))
    if faulty and rng.random() < 0.55:  # over/under-exposure spells & fixed high exposure
        fx["exposure"] = float(
            rng.uniform(0.7, 3.5) if rng.random() < 0.6 else rng.uniform(-2.5, -0.6)
        )
    elif rng.random() < 0.5:
        fx["exposure"] = float(rng.uniform(-0.4, 0.4))
    if faulty and rng.random() < 0.55:  # black occluder matching faults.py (15-40% W x 15-50% H)
        bw, bh = rng.uniform(0.15, 0.40) * w_, rng.uniform(0.15, 0.50) * h_
        u0, v0 = rng.uniform(0, w_ - bw), rng.uniform(0, h_ - bh)
        fx["occluder"] = (float(u0), float(v0), float(u0 + bw), float(v0 + bh))
    noise_px = float(rng.uniform(0.5, 2.5)) if faulty else float(rng.uniform(0.2, 1.0))
    gray = st["apply_faults"](gray, CameraError(), fx, noise_px, rng, {})
    for _ in range(rng.integers(1, 3) if rng.random() < 0.15 else 0):  # non-black occluders
        x, y = rng.integers(0, w_), rng.integers(0, h_)
        cv2.rectangle(
            gray,
            (x, y),
            (x + rng.integers(10, max(11, w_ // 3)), y + rng.integers(10, max(11, h_ // 3))),
            int(rng.integers(20, 240)),
            -1,
        )
    names = list(st["kp"])
    kp = np.array([v.kp2d[n] for n in names], np.float32)
    vis = np.array([v.kp_visible[n] for n in names], bool)
    pose = np.concatenate([R.ravel(), t]).astype(np.float32)
    return gray, kp, vis, pose, cam.K.astype(np.float32)


def sample_chunk(seeds: list[int]) -> list[tuple]:
    return [sample(s) for s in seeds]


class RollingPool:
    """A training pool that keeps changing: `n` samples in memory, and worker processes replace
    random slots with freshly rendered ones the whole time, so a long run never sees one fixed
    set. Seeds come from `seed0` upward, so choose a range that no validation set uses.
    Readers take `lock` while they copy a batch; the refresher takes it to write a chunk."""

    CHUNK = 64

    def __init__(
        self, n: int, *, crop: bool, bay: float, workers: int, seed0: int, shape: tuple[int, int]
    ) -> None:
        self.n, self.next_seed, self.replaced = n, seed0, 0
        h, w = shape
        self.images = np.zeros((n, h, w), np.uint8)
        self.kp = np.zeros((n, 15, 2), np.float32)
        self.vis = np.zeros((n, 15), bool)
        self.pose = np.zeros((n, 12), np.float32)
        self.K = np.zeros((n, 3, 3), np.float32)
        self.lock = threading.Lock()
        self._rng = np.random.default_rng(seed0)
        frames = sorted(str(p) for p in (REPO / "calibration" / "runs").glob("**/*.jpg"))
        self._ex = ProcessPoolExecutor(
            workers, initializer=init_worker, initargs=(frames, crop, bay)
        )
        self._workers = workers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fill()

    def _seeds(self) -> list[int]:
        s = list(range(self.next_seed, self.next_seed + self.CHUNK))
        self.next_seed += self.CHUNK
        return s

    def _put(self, slots, out) -> None:
        with self.lock:
            for i, (img, kp, v, pose, K) in zip(slots, out, strict=True):
                self.images[i], self.kp[i], self.vis[i], self.pose[i], self.K[i] = (
                    img,
                    kp,
                    v,
                    pose,
                    K,
                )

    def _fill(self) -> None:
        """The first fill: every slot, all workers busy."""
        chunks = [list(range(i, min(i + self.CHUNK, self.n))) for i in range(0, self.n, self.CHUNK)]
        futs = [self._ex.submit(sample_chunk, self._seeds()[: len(c)]) for c in chunks]
        for c, f in zip(chunks, futs, strict=True):
            self._put(c, f.result())

    def start(self) -> None:
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()

    def _refresh(self) -> None:
        pending: dict = {}
        while not self._stop.is_set():
            while len(pending) < 2 * self._workers:
                slots = self._rng.choice(self.n, self.CHUNK, replace=False)
                pending[self._ex.submit(sample_chunk, self._seeds())] = slots
            done = [f for f in pending if f.done()]
            if not done:
                self._stop.wait(0.05)
            for f in done:
                self._put(pending.pop(f), f.result())
                self.replaced += self.CHUNK
        for f in pending:
            f.cancel()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._ex.shutdown(cancel_futures=True)
