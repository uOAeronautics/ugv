from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ugv.parking import Capsule, Pose2, swept_clear

cv2.setNumThreads(1)

K_DEFAULT = np.array([[968.0, 0.0, 640.0], [0.0, 968.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
TAG_SIZE_M = 0.050
CAM_FWD_M = 0.071
DEFAULT_TAG_Y = -0.08
PWM_LIMIT = 1800
KDUTY = PWM_LIMIT / 4095.0
HALF_BASE = (0.21 + 0.149) / 2.0
CROP_Y1 = 440

DEFAULT_TAGMAP_PATH = (
    Path(__file__).resolve().parents[2] / "calibration" / "runs" / "l7" / "tagmap.json"
)

_H = TAG_SIZE_M / 2.0
_OBJ = np.array([[-_H, _H, 0.0], [_H, _H, 0.0], [_H, -_H, 0.0], [-_H, -_H, 0.0]], dtype=np.float64)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_tagmap(
    path: Path | str | None = None,
    tag_y: float = 0.0,
    *,
    include_uncalibrated: bool = False,
) -> dict[int, list[float]]:
    p = Path(path) if path is not None else DEFAULT_TAGMAP_PATH
    raw = json.loads(p.read_text())
    out: dict[int, list[float]] = {}
    for k, v in raw.items():
        obs = int(v[3])
        if obs == 0 and not include_uncalibrated:
            continue
        entry = [float(v[0]), float(v[1]), float(v[2]), obs]
        if tag_y and obs != -2:
            entry[1] += tag_y
        out[int(k)] = entry
    return out


def tag_solutions(corners: np.ndarray, K: np.ndarray = K_DEFAULT) -> list[list[float]]:
    """IPPE planar PnP returning [tx, ty, tz, nx, ny, nz, reproj_px] with camera-Z > 0 verified."""
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            _OBJ, corners.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return []
    out: list[list[float]] = []
    for i in range(n):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = np.asarray(tvecs[i], dtype=np.float64).ravel()
        if not np.all(np.isfinite(t)) or t[2] <= 0.0:
            continue
        err = float(np.ravel(errs)[i])
        if not math.isfinite(err):
            continue
        out.append(
            [
                float(t[0]),
                float(t[1]),
                float(t[2]),
                float(R[0, 2]),
                float(R[1, 2]),
                float(R[2, 2]),
                err,
            ]
        )
    return out


def make_detector() -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 3
    p.minMarkerPerimeterRate = 0.03
    p.polygonalApproxAccuracyRate = 0.06
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    return cv2.aruco.ArucoDetector(d, p)


class TagBits:
    """Validates a detected quad by re-reading its 6x6 cells against DICT_4X4_50."""

    def __init__(self):
        self.d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.tpl: dict[int, np.ndarray] = {}
        self.dst = np.array([[0, 0], [48, 0], [48, 48], [0, 48]], dtype=np.float32)

    def errors(self, gray: np.ndarray, tag_id: int, pts: np.ndarray) -> int:
        if tag_id not in self.tpl:
            self.tpl[tag_id] = cv2.aruco.generateImageMarker(self.d, tag_id, 6, borderBits=1) > 127
        H = cv2.getPerspectiveTransform(pts.astype(np.float32), self.dst)
        cells = cv2.resize(
            cv2.warpPerspective(gray, H, (48, 48)), (6, 6), interpolation=cv2.INTER_AREA
        )
        lo, hi = float(cells.min()), float(cells.max())
        if hi - lo < 25.0:
            return 36
        thr, _ = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return int(np.count_nonzero((cells > thr) != self.tpl[tag_id]))


class TagDetector:
    """Thread-safe OpenCV detector running on JPEG bytes or grayscale frames."""

    def __init__(self, valid_ids: Iterable[int] = (3, 6, 14), K: np.ndarray = K_DEFAULT):
        cv2.setNumThreads(1)
        self.detector = make_detector()
        self.bits = TagBits()
        self.valid_ids = set(int(i) for i in valid_ids)
        self.K = np.asarray(K, dtype=np.float64)
        self._subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.02)
        self.last_timings: dict[str, float] = {}

    def detect_gray(self, gray: np.ndarray, cap: float) -> dict:
        t_det0 = time.perf_counter()
        h, w = gray.shape[:2]
        crop_h = min(CROP_Y1, h)
        crop = gray[:crop_h, :]
        small = cv2.resize(
            crop,
            (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        corners_list, ids, _ = self.detector.detectMarkers(small)
        detect_ms = (time.perf_counter() - t_det0) * 1000.0
        subpix_ms = 0.0
        pnp_ms = 0.0
        tags: list[dict] = []
        if ids is None:
            self.last_timings = {"detect_ms": detect_ms, "subpix_ms": subpix_ms, "pnp_ms": pnp_ms}
            return {"cap": cap, "tags": tags}
        for c, tid_val in zip(corners_list, np.ravel(ids)):
            tid = int(tid_val)
            if self.valid_ids and tid not in self.valid_ids:
                continue
            pts = c.reshape(4, 2).astype(np.float32) * 2.0
            pts[:, 0] = np.clip(pts[:, 0], 1.0, w - 2.0)
            pts[:, 1] = np.clip(pts[:, 1], 1.0, h - 2.0)
            t_sp0 = time.perf_counter()
            cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (3, 3), (-1, -1), self._subpix_crit)
            subpix_ms += (time.perf_counter() - t_sp0) * 1000.0
            if self.bits.errors(gray, tid, pts) > 1:
                continue
            t_pnp0 = time.perf_counter()
            sols = tag_solutions(pts, self.K)
            pnp_ms += (time.perf_counter() - t_pnp0) * 1000.0
            if not sols:
                continue
            px = float(cv2.contourArea(pts))
            tags.append({"id": tid, "px": px, "sol": sols})
        self.last_timings = {"detect_ms": detect_ms, "subpix_ms": subpix_ms, "pnp_ms": pnp_ms}
        return {"cap": cap, "tags": tags}

    def detect_jpeg(self, jpeg_bytes: bytes, cap: float) -> dict:
        t_dec0 = time.perf_counter()
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        if gray is None:
            self.last_timings = {"decode_ms": decode_ms}
            return {"cap": cap, "tags": []}
        res = self.detect_gray(gray, cap)
        self.last_timings["decode_ms"] = decode_ms
        return res


class Localizer:
    """Fuses camera tag fixes with gyro yaw and propagates pose from capture time to now."""

    def __init__(
        self,
        tagmap: dict[int | str, Sequence[float]] | None = None,
        tag_y: float = DEFAULT_TAG_Y,
        primary: int = 14,
        blend: float = 0.35,
        gain_fwd: float = 2.3,
        gain_left: float = 1.6,
        dz_fwd: float = 0.05,
        dz_left: float = 0.13,
        tau: float = 0.2,
    ):
        if tagmap is None:
            full_map = load_tagmap(DEFAULT_TAGMAP_PATH, tag_y=tag_y, include_uncalibrated=True)
        else:
            full_map = {}
            for k, v in tagmap.items():
                obs = int(v[3])
                entry = [float(v[0]), float(v[1]), float(v[2]), obs]
                if tag_y and obs != -2:
                    entry[1] += tag_y
                full_map[int(k)] = entry
        self.cal_map: dict[int, list[float]] = {k: v for k, v in full_map.items() if int(v[3]) != 0}
        self.uncal_map: dict[int, list[float]] = {
            k: v for k, v in full_map.items() if int(v[3]) == 0
        }
        self.map = self.cal_map
        self.cal_ids: set[int] = set(self.cal_map)
        self.tag_y = tag_y
        self.primary = primary
        self.off: float | None = None
        self.outliers = 0
        self.last_used_calibrated: bool = False
        self._has_calibrated_fix: bool = False
        self.pose: tuple[float, float, float] | None = None
        self.pose_t = 0.0
        self.dr_pose: list[float] | None = None
        self.dr_t = 0.0
        self.dr_path = 0.0
        self.v = np.zeros(2, dtype=np.float64)
        self.gain = np.array([gain_fwd, gain_left], dtype=np.float64)
        self.dz = np.array([dz_fwd, dz_left], dtype=np.float64)
        self.tau = tau
        self.blend = blend
        self.last: dict = {}

    def fused_yaw(self, gyro_yaw: float) -> float | None:
        if self.off is None:
            return None
        return wrap(gyro_yaw + self.off)

    @staticmethod
    def _tag_yaw(phi: float, sol: Sequence[float]) -> float:
        nx, nz = sol[3], sol[5]
        return wrap(phi - math.atan2(-nx, nz) + math.pi / 2.0)

    def _pick_cands(
        self, tags: Sequence[dict], tag_map: dict[int, list[float]], pred: float | None
    ) -> list[tuple[dict, list[float], list[float]]]:
        cands = []
        for t in tags:
            m = tag_map.get(int(t["id"]))
            sols = t.get("sol") or []
            if m is None or not sols:
                continue
            best = sols[0]
            if len(sols) > 1 and pred is not None:
                # Hardened IPPE branch disambiguation: reject secondary branch if error >= 1.2*e0 + 0.15
                if sols[1][6] < 1.2 * sols[0][6] + 0.15:
                    d0 = abs(wrap(self._tag_yaw(m[2], sols[0]) - pred))
                    d1 = abs(wrap(self._tag_yaw(m[2], sols[1]) - pred))
                    if d1 < d0:
                        best = sols[1]
            cands.append((t, m, best))
        return cands

    def update(
        self,
        rec: dict,
        gyro_at_cap: float,
        now: float | None = None,
        gyro_now: float | None = None,
        duty_fwd: float = 0.0,
        duty_left: float = 0.0,
        allow_uncalibrated: bool = False,
    ) -> bool:
        cap = float(rec["cap"])
        pred = None if self.off is None else wrap(gyro_at_cap + self.off)
        raw_tags = rec.get("tags", [])
        cands = self._pick_cands(raw_tags, self.cal_map, pred if self._has_calibrated_fix else None)
        used_calibrated = True
        if not cands and allow_uncalibrated and self.uncal_map:
            cands = self._pick_cands(raw_tags, self.uncal_map, pred)
            used_calibrated = False
        if not cands:
            return False

        trusted = [c for c in cands if c[1][3] == -1 or c[1][3] >= 10] or cands
        wsum = sum(max(1.0, float(c[0].get("px", 1.0))) for c in trusted)
        ys = sum(
            max(1.0, float(c[0].get("px", 1.0))) * math.sin(self._tag_yaw(c[1][2], c[2]))
            for c in trusted
        )
        yc = sum(
            max(1.0, float(c[0].get("px", 1.0))) * math.cos(self._tag_yaw(c[1][2], c[2]))
            for c in trusted
        )
        tag_yaw = math.atan2(ys / wsum, yc / wsum)

        if self.off is None or (used_calibrated and not self._has_calibrated_fix):
            self.off = wrap(tag_yaw - gyro_at_cap)
            self.outliers = 0
        else:
            e = wrap(tag_yaw - gyro_at_cap - self.off)
            if abs(e) >= math.radians(20.0):
                self.outliers += 1
                return False
            self.off = wrap(self.off + 0.12 * e)
            self.outliers = 0

        yaw = wrap(gyro_at_cap + self.off)
        th = yaw - math.pi / 2.0
        f = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
        r = np.array([math.sin(th), -math.cos(th)], dtype=np.float64)

        pick = next((c for c in cands if int(c[0]["id"]) == self.primary), None)
        if pick is None:
            pick = max(trusted, key=lambda c: float(c[0].get("px", 0.0)))
        t, m, s = pick
        cam = np.array(m[:2], dtype=np.float64) - (s[2] * f + s[0] * r)
        car = cam - CAM_FWD_M * f

        self.pose = (float(car[0]), float(car[1]), yaw)
        self.pose_t = cap

        if (
            self.dr_pose is None
            or (used_calibrated and not self._has_calibrated_fix)
            or math.hypot(car[0] - self.dr_pose[0], car[1] - self.dr_pose[1]) > 0.25
        ):
            self.dr_pose = [float(car[0]), float(car[1]), yaw]
            self.dr_t = cap
        else:
            if cap > self.dr_t:
                self.propagate(cap, yaw, duty_fwd, duty_left)
            self.dr_pose[0] += self.blend * (float(car[0]) - self.dr_pose[0])
            self.dr_pose[1] += self.blend * (float(car[1]) - self.dr_pose[1])
            self.dr_pose[2] = yaw
            self.dr_t = cap
        self.dr_path = 0.0

        if now is not None and now > cap:
            yaw_now = wrap((gyro_now if gyro_now is not None else gyro_at_cap) + self.off)
            self.propagate(now, yaw_now, duty_fwd, duty_left)

        if used_calibrated:
            self._has_calibrated_fix = True
        self.last_used_calibrated = used_calibrated
        self.last = {
            "id": int(t["id"]),
            "tag_yaw": tag_yaw,
            "range": float(math.hypot(s[0], s[2])),
            "h": float(-s[1]),
            "n": len(cands),
            "calibrated": used_calibrated,
        }
        return True

    def propagate(
        self,
        now: float,
        gyro_or_fused_yaw: float,
        duty_fwd: float,
        duty_left: float,
        *,
        is_gyro: bool = False,
    ):
        if self.dr_pose is None:
            return
        dt = min(0.1, max(0.0, now - self.dr_t))
        self.dr_t = now
        u = np.array([duty_fwd, duty_left], dtype=np.float64)
        target = self.gain * np.sign(u) * np.maximum(np.abs(u) - self.dz, 0.0)
        self.v += (1.0 - math.exp(-dt / self.tau)) * (target - self.v)
        yaw = (
            wrap(gyro_or_fused_yaw + self.off)
            if (is_gyro and self.off is not None)
            else wrap(gyro_or_fused_yaw)
        )
        th = yaw - math.pi / 2.0
        f = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
        left = np.array([-math.sin(th), math.cos(th)], dtype=np.float64)
        d = (self.v[0] * f + self.v[1] * left) * dt
        self.dr_pose[0] += float(d[0])
        self.dr_pose[1] += float(d[1])
        self.dr_pose[2] = yaw
        self.dr_path += float(np.linalg.norm(d))


def duty_to_dashboard(
    duty_fwd: float, duty_left: float, duty_ccw: float
) -> tuple[float, float, float, float]:
    F, S, T = duty_fwd / KDUTY, -duty_left / KDUTY, duty_ccw / KDUTY
    wheels = (-F - S + T, -F + S + T, -F + S - T, -F - S - T)
    s = max(1.0, abs(F), abs(S), abs(T), *(abs(w) for w in wheels))
    return F / s, S / s, T / s, s


class GoTo:
    """Holonomic P controller to a policy-frame pose (x, y, yaw)."""

    def __init__(
        self,
        target: tuple[float, float, float],
        *,
        kp_pos: float = 1.2,
        kp_yaw: float = 1.5,
        vcap_fwd: float = 0.30,
        vcap_left: float = 0.25,
        vcap_yaw: float = 0.50,
        gain_fwd: float = 2.3,
        gain_left: float = 1.6,
        gain_yaw: float = 15.7,
        dz_fwd: float = 0.05,
        dz_left: float = 0.13,
        dz_yaw: float = 0.16,
        tol_pos: float = 0.04,
        tol_yaw_deg: float = 4.0,
        settle_s: float = 0.2,
        face_origin_dist: float = 0.25,
        min_fwd_speed: float = 0.0,
    ):
        self.t = target
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw
        self.vcap_fwd = vcap_fwd
        self.vcap_left = vcap_left
        self.vcap_yaw = vcap_yaw
        self.gain_fwd = gain_fwd
        self.gain_left = gain_left
        self.gain_yaw = gain_yaw
        self.dz_fwd = dz_fwd
        self.dz_left = dz_left
        self.dz_yaw = dz_yaw
        self.tol_pos = tol_pos
        self.tol_yaw = math.radians(tol_yaw_deg)
        self.settle_s = settle_s
        self.face_origin_dist = face_origin_dist
        self.min_fwd_speed = min_fwd_speed
        self.ok_since: float | None = None

    def command_for(
        self, pose: Sequence[float], target: Sequence[float]
    ) -> tuple[float, float, float]:
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        ex, ey = float(target[0]) - x, float(target[1]) - y
        dist = math.hypot(ex, ey)
        yaw_goal = float(target[2]) if dist < self.face_origin_dist else wrap(-math.atan2(x, y))
        eyaw = wrap(yaw_goal - yaw)
        th = yaw - math.pi / 2.0
        e_f = ex * math.cos(th) + ey * math.sin(th)
        e_l = -ex * math.sin(th) + ey * math.cos(th)

        def duty(
            e: float, kp: float, vcap: float, gain: float, dz: float, tol: float, vmin: float = 0.0
        ) -> float:
            if abs(e) < tol:
                return 0.0
            v = min(vcap, max(vmin if abs(e) > tol * 2.0 else 0.0, kp * abs(e)))
            return math.copysign(dz + v / gain, e)

        return (
            duty(
                e_f,
                self.kp_pos,
                self.vcap_fwd,
                self.gain_fwd,
                self.dz_fwd,
                self.tol_pos * 0.5,
                self.min_fwd_speed,
            ),
            duty(
                e_l, self.kp_pos, self.vcap_left, self.gain_left, self.dz_left, self.tol_pos * 0.5
            ),
            duty(eyaw, self.kp_yaw, self.vcap_yaw, self.gain_yaw, self.dz_yaw, self.tol_yaw * 0.5),
        )

    def step(self, pose: Sequence[float], now: float) -> tuple[tuple[float, float, float], bool]:
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        dist = math.hypot(self.t[0] - x, self.t[1] - y)
        if dist < self.tol_pos and abs(wrap(self.t[2] - yaw)) < self.tol_yaw:
            self.ok_since = self.ok_since if self.ok_since is not None else now
            return (0.0, 0.0, 0.0), (now - self.ok_since) >= self.settle_s
        self.ok_since = None
        return self.command_for(pose, self.t), False


@dataclass(frozen=True)
class AutoParkCommand:
    stage: str
    status: str
    forward: float = 0.0
    strafe: float = 0.0
    turn: float = 0.0
    stop: bool = False
    done: bool = False
    reason: str = ""
    pose: tuple[float, float, float] | None = None


class AutoParkRunner:
    """2-phase docking sequence: acquiring -> staging (0, 0.55, 0) -> entering (0, 0.02, 0) -> parked."""

    def __init__(
        self,
        localizer: Localizer | None = None,
        obstacles: Sequence[Capsule] = (),
        vehicle_width: float = 0.21,
        vehicle_length: float = 0.26,
        inflation: float = 0.03,
        acquire_dps: float = 25.0,
        max_blind_dr_s: float = 1.5,
        corridor_x_limit: float = 0.14,
        stationary_dps: float = 2.0,
        imu_stale_s: float = 0.25,
        frame_stale_s: float = 0.45,
        lookahead_m: float = 0.35,
        acquire_timeout_s: float = 16.0,
    ):
        self.localizer = localizer or Localizer()
        self.loc = self.localizer
        self.obstacles = list(obstacles)
        self.vehicle_width = float(vehicle_width)
        self.vehicle_length = float(vehicle_length)
        self.inflation = float(inflation)
        self.acquire_dps = max(-25.0, min(25.0, float(acquire_dps)))
        self.max_blind_dr_s = float(max_blind_dr_s)
        self.corridor_x_limit = float(corridor_x_limit)
        self.stationary_dps = float(stationary_dps)
        self.imu_stale_s = float(imu_stale_s)
        self.frame_stale_s = float(frame_stale_s)
        self.lookahead_m = float(lookahead_m)
        self.acquire_timeout_s = float(acquire_timeout_s)

        self.stage = "acquiring"
        self.acquire_started_at: float | None = None
        self.stage_goto = GoTo((0.0, 0.55, 0.0), tol_pos=0.05, tol_yaw_deg=3.5, settle_s=0.1)
        self.enter_goto = GoTo(
            (0.0, 0.02, 0.0),
            kp_pos=1.8,
            vcap_fwd=0.35,
            vcap_left=0.22,
            tol_pos=0.04,
            tol_yaw_deg=4.0,
            settle_s=0.0,
            face_origin_dist=1.0,
            min_fwd_speed=0.22,
        )
        self.last_duty = (0.0, 0.0)
        self.brake_at: float | None = None
        self.gyro_hist: list[tuple[float, float]] = []
        self.stall_s: float = 0.0
        self._prev_step_t: float | None = None
        self._acquire_moving: bool = False

    def _boost_turn_duty(self, dc: float, dt: float, imu_w_dps: float, *, acquiring: bool) -> float:
        if acquiring:
            w_rad = math.radians(self.acquire_dps)
            dc_base = (
                math.copysign(self.stage_goto.dz_yaw + abs(w_rad) / self.stage_goto.gain_yaw, w_rad)
                if w_rad
                else 0.0
            )
            if not dc_base:
                self.stall_s = 0.0
                return 0.0
            sign = math.copysign(1.0, dc_base)
            w_abs = abs(imu_w_dps)
            if w_abs >= 6.0 or (self._acquire_moving and w_abs >= 4.0):
                self._acquire_moving = True
                self.stall_s = 0.0
                dc_mag = max(
                    0.05, min(0.26, abs(dc_base) + 0.002 * (abs(self.acquire_dps) - w_abs))
                )
            else:
                self._acquire_moving = False
                self.stall_s += dt
                ramp = self.stage_goto.dz_yaw + 0.08 * min(self.stall_s / 0.8, 1.0)
                dc_mag = min(0.26, ramp + abs(w_rad) / self.stage_goto.gain_yaw)
            return sign * dc_mag
        self._acquire_moving = False
        if abs(dc) > 1e-6 and abs(imu_w_dps) < 3.0:
            self.stall_s += dt
            ramp = self.stage_goto.dz_yaw + 0.08 * min(self.stall_s / 0.8, 1.0)
            return math.copysign(min(0.26, max(abs(dc), ramp)), dc)
        self.stall_s = 0.0
        return dc

    def record_gyro(self, t: float, yaw_rad: float):
        if not self.gyro_hist or t > self.gyro_hist[-1][0]:
            self.gyro_hist.append((t, yaw_rad))
            if len(self.gyro_hist) > 600:
                self.gyro_hist = self.gyro_hist[-600:]

    def gyro_yaw_at(self, t: float, fallback: float) -> float:
        for ts, y in reversed(self.gyro_hist):
            if ts <= t:
                return y
        return self.gyro_hist[0][1] if self.gyro_hist else fallback

    def _check_swept(self, pose: Sequence[float], df: float, dl: float, dc: float) -> bool:
        if not self.obstacles:
            return True
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        vf = (
            self.stage_goto.gain_fwd * math.copysign(max(0.0, abs(df) - self.stage_goto.dz_fwd), df)
            if df
            else 0.0
        )
        vl = (
            self.stage_goto.gain_left
            * math.copysign(max(0.0, abs(dl) - self.stage_goto.dz_left), dl)
            if dl
            else 0.0
        )
        w = (
            self.stage_goto.gain_yaw * math.copysign(max(0.0, abs(dc) - self.stage_goto.dz_yaw), dc)
            if dc
            else 0.0
        )
        th = yaw - math.pi / 2.0
        vx = vf * math.cos(th) - vl * math.sin(th)
        vy = vf * math.sin(th) + vl * math.cos(th)
        speed = math.hypot(vx, vy)
        if speed > 1e-4:
            dt = min(1.0, self.lookahead_m / speed)
            nx, ny = x + vx * dt, y + vy * dt
            nyaw = wrap(yaw + w * min(0.4, dt))
        else:
            nx, ny = x, y
            nyaw = wrap(yaw + w * 0.4)
        # parking.py uses CW+ Pose2(x, z, yaw); negate CCW+ yaw.
        return swept_clear(
            Pose2(x, y, -yaw),
            Pose2(nx, ny, -nyaw),
            self.vehicle_width,
            self.vehicle_length,
            self.obstacles,
            self.inflation,
        )

    def _staging_orbit_waypoints(self, x: float, y: float) -> list[tuple[float, float, float]]:
        r = math.hypot(x, y)
        r_orbit = min(1.15, max(0.72, max(0.75, r)))
        theta = math.atan2(y, x) if r > 1e-6 else -math.pi / 2.0
        dtheta = wrap(math.pi / 2.0 - theta)
        arc_dir = 1.0 if dtheta >= 0.0 else -1.0
        step = arc_dir * min(0.6, abs(dtheta))

        candidates: list[tuple[float, float]] = [
            (r_orbit * math.cos(theta + step), r_orbit * math.sin(theta + step)),
        ]
        if r < 0.72:
            r_out = max(0.75, r + 0.30)
            for frac in (0.4, 0.0, 0.7):
                ang = theta + arc_dir * min(0.6 * frac, abs(dtheta))
                candidates.append((r_out * math.cos(ang), r_out * math.sin(ang)))
        for ang_off in (arc_dir * 0.35, arc_dir * 0.9, -arc_dir * 0.6, -arc_dir * 0.35):
            candidates.append(
                (r_orbit * math.cos(theta + ang_off), r_orbit * math.sin(theta + ang_off))
            )
        if r >= 0.72:
            r_out = min(1.15, r + 0.30)
            candidates.append((r_out * math.cos(theta), r_out * math.sin(theta)))

        return [(wx, wy, wrap(-math.atan2(wx, wy))) for wx, wy in candidates]

    def _to_cmd(
        self,
        stage: str,
        status: str,
        df: float,
        dl: float,
        dc: float,
        pose: tuple[float, float, float] | None,
    ) -> AutoParkCommand:
        F, S, T, scale = duty_to_dashboard(df, dl, dc)
        self.last_duty = (F * KDUTY, -S * KDUTY)
        return AutoParkCommand(
            stage=stage,
            status=status,
            forward=F,
            strafe=S,
            turn=T,
            stop=False,
            done=False,
            pose=pose,
        )

    def step(
        self,
        now: float,
        *,
        imu_age_s: float | None,
        frame_age_s: float | None,
        imu_w_dps: float,
        gyro_yaw_rad: float,
        rec: dict | None = None,
    ) -> AutoParkCommand:
        dt = min(0.25, max(0.0, now - self._prev_step_t)) if self._prev_step_t is not None else 0.0
        self._prev_step_t = now
        self.record_gyro(now, gyro_yaw_rad)
        cur_pose = tuple(self.localizer.dr_pose) if self.localizer.dr_pose is not None else None

        if imu_age_s is None or imu_age_s > self.imu_stale_s:
            self.stage = "error"
            self.last_duty = (0.0, 0.0)
            return AutoParkCommand(
                stage="error",
                status="autopark-imu-stale",
                stop=True,
                reason="autopark-imu-stale",
                pose=cur_pose,
            )
        if frame_age_s is None or frame_age_s > self.frame_stale_s:
            self.stage = "error"
            self.last_duty = (0.0, 0.0)
            return AutoParkCommand(
                stage="error",
                status="autopark-frame-stale",
                stop=True,
                reason="autopark-frame-stale",
                pose=cur_pose,
            )

        if rec is not None and rec.get("tags"):
            cap = float(rec["cap"])
            g_cap = self.gyro_yaw_at(cap, gyro_yaw_rad)
            self.localizer.update(
                rec,
                g_cap,
                now=now,
                gyro_now=gyro_yaw_rad,
                duty_fwd=self.last_duty[0],
                duty_left=self.last_duty[1],
                allow_uncalibrated=(self.stage in ("acquiring", "staging")),
            )
        elif self.localizer.dr_pose is not None:
            self.localizer.propagate(
                now, gyro_yaw_rad, self.last_duty[0], self.last_duty[1], is_gyro=True
            )

        cur_pose = tuple(self.localizer.dr_pose) if self.localizer.dr_pose is not None else None

        if self.stage == "acquiring":
            if cur_pose is None:
                if self.acquire_started_at is None:
                    self.acquire_started_at = now
                elif now - self.acquire_started_at >= self.acquire_timeout_s:
                    self.stage = "aborted"
                    self.last_duty = (0.0, 0.0)
                    self.stall_s = 0.0
                    self._acquire_moving = False
                    return AutoParkCommand(
                        stage="aborted",
                        status="No tag found after 360 deg scan",
                        stop=True,
                        reason="acquire-timeout",
                        pose=None,
                    )
                dc = self._boost_turn_duty(0.0, dt, imu_w_dps, acquiring=True)
                return self._to_cmd("acquiring", "searching for drone tags", 0.0, 0.0, dc, None)
            self.stage = "staging"
            self.stall_s = 0.0
            self._acquire_moving = False

        if self.stage == "staging":
            assert cur_pose is not None
            x, y, yaw = cur_pose
            if (
                abs(x) < 0.06
                and abs(y - 0.55) < 0.06
                and abs(wrap(yaw)) < math.radians(4.0)
                and self.loc.last_used_calibrated
            ):
                self.stage = "entering"
                self.stall_s = 0.0
            else:
                saved_stall = self.stall_s
                (df_dir, dl_dir, dc_dir), _ = self.stage_goto.step(cur_pose, now)
                dc_dir = self._boost_turn_duty(dc_dir, dt, imu_w_dps, acquiring=False)
                stall_after_dir = self.stall_s
                direct_ok = self._check_swept(cur_pose, df_dir, dl_dir, dc_dir)

                outside_corridor = y < 0.42 and (abs(x) > 0.20 or y < 0.10)
                chosen: tuple[float, float, float] | None = None
                chosen_stall = stall_after_dir
                if not outside_corridor and direct_ok:
                    chosen = (df_dir, dl_dir, dc_dir)
                else:
                    for wp in self._staging_orbit_waypoints(x, y):
                        self.stall_s = saved_stall
                        df_w, dl_w, dc_w = self.stage_goto.command_for(cur_pose, wp)
                        dc_w = self._boost_turn_duty(dc_w, dt, imu_w_dps, acquiring=False)
                        if self._check_swept(cur_pose, df_w, dl_w, dc_w):
                            chosen = (df_w, dl_w, dc_w)
                            chosen_stall = self.stall_s
                            break
                    if chosen is None and direct_ok:
                        chosen = (df_dir, dl_dir, dc_dir)
                        chosen_stall = stall_after_dir

                if chosen is None:
                    self.stage = "blocked"
                    self.last_duty = (0.0, 0.0)
                    self.stall_s = 0.0
                    return AutoParkCommand(
                        stage="blocked",
                        status="swept-clear-blocked",
                        stop=True,
                        reason="swept-clear-blocked",
                        pose=cur_pose,
                    )

                self.stall_s = chosen_stall
                df, dl, dc = chosen
                status = (
                    "staging to (0.00, 0.55)"
                    if self.loc.last_used_calibrated
                    else "staging (awaiting calibrated tag)"
                )
                return self._to_cmd("staging", status, df, dl, dc, cur_pose)

        if self.stage == "entering":
            assert cur_pose is not None
            x, y, yaw = cur_pose
            if abs(x) >= self.corridor_x_limit:
                self.stage = "aborted"
                self.last_duty = (0.0, 0.0)
                self.stall_s = 0.0
                return AutoParkCommand(
                    stage="aborted",
                    status="corridor-x-limit",
                    stop=True,
                    reason="corridor-x-limit",
                    pose=cur_pose,
                )
            blind_dt = max(0.0, now - self.localizer.pose_t)
            if y < 0.40 and blind_dt > self.max_blind_dr_s:
                self.stage = "aborted"
                self.last_duty = (0.0, 0.0)
                self.stall_s = 0.0
                return AutoParkCommand(
                    stage="aborted",
                    status="blind-dr-timeout",
                    stop=True,
                    reason="blind-dr-timeout",
                    pose=cur_pose,
                )
            if abs(x) < 0.06 and abs(y - 0.02) < 0.05 and abs(wrap(yaw)) < math.radians(5.0):
                self.stage = "braking"
                self.brake_at = now
                self.last_duty = (0.0, 0.0)
                self.stall_s = 0.0
                return AutoParkCommand(
                    stage="braking",
                    status="braking at dock",
                    stop=True,
                    reason="entering-arrived",
                    pose=cur_pose,
                )
            (df, dl, dc), _ = self.enter_goto.step(cur_pose, now)
            dc = self._boost_turn_duty(dc, dt, imu_w_dps, acquiring=False)
            return self._to_cmd("entering", "entering dock corridor", df, dl, dc, cur_pose)

        if self.stage == "braking":
            self.last_duty = (0.0, 0.0)
            if abs(imu_w_dps) < self.stationary_dps:
                self.stage = "parked"
                return AutoParkCommand(
                    stage="parked",
                    status="parked",
                    stop=True,
                    done=True,
                    reason="parked",
                    pose=cur_pose,
                )
            if self.brake_at is not None and (now - self.brake_at) > 2.0:
                self.stage = "aborted"
                return AutoParkCommand(
                    stage="aborted",
                    status="brake-stationary-timeout",
                    stop=True,
                    reason="brake-stationary-timeout",
                    pose=cur_pose,
                )
            return AutoParkCommand(
                stage="braking",
                status="verifying stationary",
                stop=True,
                reason="verifying-stationary",
                pose=cur_pose,
            )

        return AutoParkCommand(
            stage=self.stage,
            status=self.stage,
            stop=True,
            done=(self.stage == "parked"),
            reason=self.stage,
            pose=cur_pose,
        )
