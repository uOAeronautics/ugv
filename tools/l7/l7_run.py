#!/usr/bin/env python3
"""L7 real-robot harness: run a controller on the car through the dashboard and record everything.

Runs ON THE PI against the live dashboard service (ws://127.0.0.1:8765). The dashboard keeps
motor authority, its 600 ms watchdog and the 1800/4095 PWM cap; this process only sends
`drive` messages, so killing it brakes the car.

Frames (same as rl/sim.py): drone at the origin, skids along y, car pose (x, y, yaw) with
yaw 0 = camera facing the drone along -y, yaw CCW positive. The frame is anchored on the
primary ArUco tag (default 14, assumed centred in the corridor, facing +y); other tags are
mapped online by co-observation and saved to the tag map for the next run.

Modes
  probe    open-loop axis pulses; prints measured body displacement per pulse (signs, gains)
  stage    go-to-pose P controller (reset between episodes; also a classical reference)
  policy   StudentPolicy (rl_sim axes remapped to the physical car as in
           simulation/gazebo/run_policy_episodes.py)
  watch    no motion; localize and record only

Outputs (one directory per run): meta.json, ticks.jsonl (50 Hz control record),
frames.jsonl (every processed camera frame: corners + both PnP solutions), imu.jsonl
(100 Hz gyro/accel + applied wheel PWM from /api/dynamics), frames/*.jpg (subsampled raw
JPEG), summary.json.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HOST, PORT = "127.0.0.1", 8765
K = np.array([[968.0, 0, 640.0], [0, 968.0, 360.0], [0, 0, 1.0]])  # nominal, uncalibrated
TAG_SIZE_M = 0.050
CAM_FWD_M = 0.071          # lens ahead of car centre (simulation/gazebo/vision_in_loop.py)
HALF_BASE = (0.21 + 0.149) / 2
PWM_LIMIT = 1800           # dashboard hard cap
KDUTY = PWM_LIMIT / 4095.0
DT = 0.02
V_MAX = (1.27, 0.63, 2.20)  # policy.py V_X_MAX, V_Y_MAX, W_MAX


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def should_abort_corridor(
    mode: str,
    est: tuple[float, float, float] | None,
    pose_t: float | None,
    now: float,
    max_tag_age: float = 0.15,
) -> bool:
    if mode != "stage" or est is None:
        return False
    pose_stale = pose_t is None or (now - pose_t) >= max_tag_age
    return pose_stale and est[1] < 0.55 and abs(est[0]) > 0.14


class JsonlWriter:
    def __init__(self, path: Path):
        self.f = open(path, "w", buffering=1 << 16)
        self.lock = threading.Lock()

    def write(self, rec: dict):
        line = json.dumps(rec, separators=(",", ":"))
        with self.lock:
            self.f.write(line + "\n")

    def close(self):
        with self.lock:
            self.f.close()


def http_conn():
    return http.client.HTTPConnection(HOST, PORT, timeout=3)


# ----------------------------------------------------------------------------- IMU
class Imu(threading.Thread):
    """Polls /api/dynamics, integrates gyro z into a CCW yaw (rad), logs raw samples."""

    def __init__(self, log: JsonlWriter, gyro_sign: float):
        super().__init__(daemon=True, name="imu")
        self.log, self.sign = log, gyro_sign
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.hist: deque[tuple[float, float]] = deque(maxlen=600)  # (t, yaw)
        self.yaw = 0.0
        self.rate = 0.0
        self.bias = 0.0
        self.bias_n = 0
        self.last_t: float | None = None
        self.sample_at = 0.0
        self.wheels = (0, 0, 0, 0)

    def run(self):
        c = http_conn()
        c.request("GET", "/api/dynamics?after=999999999")
        seq = json.loads(c.getresponse().read())["latest_seq"]
        while not self.stop_evt.is_set():
            try:
                c.request("GET", f"/api/dynamics?after={seq}")
                j = json.loads(c.getresponse().read())
            except Exception:
                c.close()
                c = http_conn()
                time.sleep(0.05)
                continue
            for s in j["samples"]:
                seq = s["seq"]
                gz = s["gyro_dps"][2]
                w = s["wheels"]
                wh = (w["front-left"], w["rear-left"], w["front-right"], w["rear-right"])
                still = s["stopped"] and abs(gz - self.bias) < 2.0
                with self.lock:
                    if still and (self.last_t is None or s["t"] - self._moved_at > 0.6):
                        # Rest: track gyro bias (fast at first, slow afterwards).
                        self.bias_n += 1
                        a = max(0.005, 1.0 / self.bias_n)
                        self.bias += a * (gz - self.bias)
                    if not s["stopped"]:
                        self._moved_at = s["t"]
                    rate = self.sign * math.radians(gz - self.bias)
                    if self.last_t is not None:
                        self.yaw += rate * min(0.05, max(0.0, s["t"] - self.last_t))
                    self.last_t, self.rate, self.sample_at, self.wheels = s["t"], rate, s["t"], wh
                    self.hist.append((s["t"], self.yaw))
                self.log.write({"q": s["seq"], "t": s["t"], "g": s["gyro_dps"], "a": s["accel_g"],
                                "w": wh, "st": s["stopped"]})
            time.sleep(0.012)

    _moved_at = -1e9

    def yaw_at(self, t: float) -> float:
        with self.lock:
            if not self.hist:
                return self.yaw
            for i in range(len(self.hist) - 1, -1, -1):
                if self.hist[i][0] <= t:
                    return self.hist[i][1]
            return self.hist[0][1]

    def snapshot(self):
        with self.lock:
            return self.yaw, self.rate, self.sample_at, self.wheels


# ----------------------------------------------------------------------------- vision
_H = TAG_SIZE_M / 2
_OBJ = np.array([[-_H, _H, 0], [_H, _H, 0], [_H, -_H, 0], [-_H, -_H, 0]], dtype=np.float64)


def tag_solutions(corners: np.ndarray) -> list[list[float]]:
    """IPPE planar PnP. Each solution: [tx, ty, tz, nx, ny, nz, reproj_px] in the camera
    frame (x right, y down, z forward); n is the tag's outward normal."""
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(_OBJ, corners.astype(np.float64), K, None,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    out = []
    for i in range(n):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i].ravel()
        if t[2] <= 0:
            continue
        out.append([*(float(v) for v in t), *(float(v) for v in R[:, 2]), float(np.ravel(errs)[i])])
    return out


CROP_Y1 = 440   # tags hang above the lens height, so they never appear in the floor half


def make_detector():
    """Runs on the 2x-decimated crop: twice the recall of full resolution on these cards
    (thin quiet zone) at a quarter of the cost; corners are refined at full resolution."""
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
    """Validates a tracked quad by re-reading its 6x6 cells against the dictionary."""

    def __init__(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.d = d
        self.tpl: dict[int, np.ndarray] = {}
        self.dst = np.array([[0, 0], [48, 0], [48, 48], [0, 48]], dtype=np.float32)

    def errors(self, gray: np.ndarray, tag_id: int, pts: np.ndarray) -> int:
        if tag_id not in self.tpl:
            self.tpl[tag_id] = cv2.aruco.generateImageMarker(self.d, tag_id, 6, borderBits=1) > 127
        H = cv2.getPerspectiveTransform(pts.astype(np.float32), self.dst)
        cells = cv2.resize(cv2.warpPerspective(gray, H, (48, 48)), (6, 6), interpolation=cv2.INTER_AREA)
        lo, hi = float(cells.min()), float(cells.max())
        if hi - lo < 25:
            return 36
        thr, _ = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return int(np.count_nonzero((cells > thr) != self.tpl[tag_id]))


def vision_worker(q, stop_evt, out_dir: str, ids: set, save_every: int):
    """Separate process: latest JPEG -> half-res ArUco detect -> full-res corner refine,
    KLT fallback from the previous frame, bit check, PnP. Writes frames.jsonl + JPEGs."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    out = Path(out_dir)
    log = JsonlWriter(out / "frames.jsonl")
    det, bits = make_detector(), TagBits()
    c = http_conn()
    last_seq, n, body = -1, 0, None
    state: dict = {"prev": None, "held": {}}
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    sub = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05)
    while not stop_evt.is_set():
        try:
            c.request("GET", "/api/calibration/frame")
            r = c.getresponse()
            data = r.read()
            if r.status != 200:
                time.sleep(0.05)
                continue
            seq = int(r.getheader("X-Frame-Sequence"))
            cap = float(r.getheader("X-Pi-Monotonic"))
        except Exception:
            c.close()
            c = http_conn()
            time.sleep(0.05)
            continue
        if seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq, body = seq, data
        t0 = time.monotonic()
        try:
            rec = _process_frame(body, seq, cap, t0, n, det, bits, state, ids, save_every, out, lk, sub)
        except Exception as e:   # keep the stream alive; the main loop brakes on stale frames
            log.write({"seq": seq, "cap": cap, "error": f"{type(e).__name__}: {e}"})
            state["prev"], state["held"] = None, {}
            continue
        log.write(rec)
        q.put(rec)
        n += 1
    if body:
        (out / "end.jpg").write_bytes(body)
    log.close()


def _process_frame(body, seq, cap, t0, n, det, bits, state, ids, save_every, out, lk, sub):
    prev, held = state["prev"], state["held"]
    if True:
        full = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
        if full is None:
            raise ValueError("undecodable JPEG")
        gray = full[:CROP_Y1]
        half = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        found: dict[int, tuple[np.ndarray, str]] = {}
        quads, qids, _ = det.detectMarkers(half)
        if qids is not None:
            for qd, i in zip(quads, qids.ravel()):
                i = int(i)
                if i not in ids:
                    continue
                pts = (qd[0] * 2.0 + 0.5).astype(np.float32)
                if np.linalg.norm(pts[1] - pts[0]) < 14:
                    continue
                if pts.min() < 7 or pts[:, 0].max() > gray.shape[1] - 8 or pts[:, 1].max() > gray.shape[0] - 8:
                    continue   # cornerSubPix needs its window inside the image
                pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2).copy(), (4, 4), (-1, -1), sub).reshape(4, 2)
                if bits.errors(gray, i, pts) <= 1:
                    found[i] = (pts, "det")
        lost = [i for i in held if i not in found]
        if prev is not None and lost:
            p0 = np.concatenate([held[i] for i in lost]).reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **lk)
            fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
            for j, i in enumerate(lost):
                sl = slice(4 * j, 4 * j + 4)
                pts = p1[sl].reshape(4, 2)
                if st[sl].all() and st2[sl].all() and fb[sl].max() < 1.5 and bits.errors(gray, i, pts) <= 1:
                    found[i] = (pts, "trk")
        tags = []
        held = {}
        for i, (pts, how) in found.items():
            side = float(np.mean([np.linalg.norm(pts[(k + 1) % 4] - pts[k]) for k in range(4)]))
            if side < 11 or pts[:, 1].max() > CROP_Y1 - 3 or pts.min() < 2 or pts[:, 0].max() > gray.shape[1] - 3:
                continue
            held[i] = pts.astype(np.float32)
            tags.append({"id": i, "px": round(side, 1), "how": how,
                         "c": [round(float(v), 2) for v in pts.ravel()], "sol": tag_solutions(pts)})
        state["prev"], state["held"] = gray, held
        done = time.monotonic()
        saved = None
        if save_every and n % save_every == 0:
            saved = f"{seq:08d}.jpg"
            (out / "frames" / saved).write_bytes(body)
        return {"seq": seq, "cap": cap, "done": done, "ms": round((done - t0) * 1e3, 1),
                "tags": tags, "jpg": saved}


class Vision:
    """Main-process handle on the vision worker; newest record wins."""

    def __init__(self, out: Path, ids: set[int], save_every: int):
        import multiprocessing
        # spawn, not fork: forking after the IMU/websocket threads exist can deadlock the child.
        mp = multiprocessing.get_context("spawn")
        self.q = mp.Queue()
        self.stop_evt = mp.Event()
        self.proc = mp.Process(target=vision_worker, args=(self.q, self.stop_evt, str(out), ids, save_every),
                               daemon=True)
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.serial = 0
        self.frame_at = 0.0
        self.proc_ms: deque[float] = deque(maxlen=400)

    def start(self):
        self.proc.start()
        threading.Thread(target=self._drain, daemon=True, name="vision-drain").start()

    def _drain(self):
        while True:
            try:
                rec = self.q.get(timeout=0.5)
            except Exception:
                if not self.proc.is_alive():
                    return
                continue
            with self.lock:
                self.latest, self.frame_at = rec, rec["cap"]
                self.serial += 1
                self.proc_ms.append(rec["ms"])

    def join(self, timeout=3):
        self.proc.join(timeout)

    def poll(self, seen: int):
        with self.lock:
            if self.serial == seen:
                return None, seen
            return self.latest, self.serial


# ----------------------------------------------------------------------------- localizer
DEFAULT_MAP = {
    # id: [x, y, phi(normal direction, rad), observations]. 14 defines the frame.
    # 6 and 3 are priors (side faces, CCW order A=3, B=?, C=6, D=14); replaced by co-observation.
    "14": [-0.028, 0.0, math.pi / 2, -1],   # 28 mm off the skid midline (20261113-181622 end.jpg)
    "6": [0.06, -0.05, 0.0, 0],
    "3": [-0.06, -0.05, math.pi, 0],
}


class Localizer:
    """Camera/tag measurements + gyro yaw -> car pose in the drone (policy) frame."""

    def __init__(self, imu: Imu, tagmap: dict, tag_y: float, primary: int, args):
        self.imu = imu
        self.map = {int(k): list(v) for k, v in tagmap.items()}
        for v in self.map.values():
            v[1] += tag_y if v[3] != -2 else 0.0   # shift along corridor: policy y of the tag plane
        self.tag_y = tag_y
        self.primary = primary
        self.off: float | None = None      # fused_yaw = gyro_yaw + off
        self.outliers = 0
        self.pose: tuple[float, float, float] | None = None
        self.pose_t = 0.0                  # capture time of last tag pose
        self.dr_pose: list[float] | None = None
        self.dr_t = 0.0
        self.dr_path = 0.0
        self.v = np.zeros(2)               # modelled body velocity (forward, left)
        self.gain = np.array([args.gain_fwd, args.gain_left])
        self.dz = np.array([args.dz_fwd, args.dz_left])
        self.tau = 0.2
        self.blend = args.blend
        self.calibrate_tagmap = bool(getattr(args, "calibrate_tagmap", False))
        self.allow_uncalibrated_tags = bool(getattr(args, "allow_uncalibrated_tags", False))
        self.last: dict = {}

    def fused_yaw(self, t: float | None = None) -> float | None:
        if self.off is None:
            return None
        g = self.imu.yaw_at(t) if t is not None else self.imu.snapshot()[0]
        return wrap(g + self.off)

    @staticmethod
    def _tag_yaw(phi: float, sol: list[float]) -> float:
        nx, nz = sol[3], sol[5]
        return wrap(phi - math.atan2(-nx, nz) + math.pi / 2)

    def update(self, rec: dict) -> bool:
        cap = rec["cap"]
        gyro = self.imu.yaw_at(cap)
        pred = None if self.off is None else wrap(gyro + self.off)
        cands = []
        map_cands = []
        for t in rec["tags"]:
            m = self.map.get(t["id"])
            if m is None or not t.get("sol"):
                continue
            sols = sorted((s for s in t["sol"] if len(s) >= 7 and s[2] > 0 and s[5] < -0.2), key=lambda s: s[6])
            if not sols:
                continue
            is_clear = len(sols) == 1 or sols[1][6] >= 1.2 * sols[0][6] + 0.15
            best = sols[0]
            if pred is None:
                if not is_clear:
                    continue
            elif len(sols) > 1 and not is_clear:
                d0 = abs(wrap(self._tag_yaw(m[2], sols[0]) - pred))
                d1 = abs(wrap(self._tag_yaw(m[2], sols[1]) - pred))
                if d1 < d0:
                    best = sols[1]
            map_cands.append((t, m, best, is_clear))
            if m[3] != 0 or self.allow_uncalibrated_tags:
                cands.append((t, m, best, is_clear))
        if not cands:
            return False
        # Yaw: weighted tag yaw -> slow correction of the gyro offset.
        trusted = [c for c in cands if c[1][3] < 0 or c[1][3] >= 10] or cands
        wsum = sum(c[0]["px"] for c in trusted)
        ys = sum(c[0]["px"] * math.sin(self._tag_yaw(c[1][2], c[2])) for c in trusted)
        yc = sum(c[0]["px"] * math.cos(self._tag_yaw(c[1][2], c[2])) for c in trusted)
        tag_yaw = math.atan2(ys / wsum, yc / wsum)
        if self.off is None:
            self.off = wrap(tag_yaw - gyro)
            self.outliers = 0
            e = 0.0
        else:
            e = wrap(tag_yaw - gyro - self.off)
            if abs(e) < math.radians(20):
                self.off = wrap(self.off + 0.12 * e)
                self.outliers = 0
            else:
                self.outliers += 1
                return False
        yaw = wrap(gyro + self.off)
        th = yaw - math.pi / 2
        f = np.array([math.cos(th), math.sin(th)])
        r = np.array([math.sin(th), -math.cos(th)])
        # Position: primary tag if visible, else the best-mapped largest tag.
        pick = next((c for c in cands if c[0]["id"] == self.primary), None)
        if pick is None:
            pick = max(trusted, key=lambda c: c[0]["px"])
        t, m, s, pick_clear = pick
        cam = np.array(m[:2]) - (s[2] * f + s[0] * r)
        car = cam - CAM_FWD_M * f
        # Online map: only calibrate non-fixed tags when explicitly enabled, primary is clear, and |e| < 8 deg.
        if self.calibrate_tagmap and pick[0]["id"] == self.primary and pick_clear and abs(e) < math.radians(8):
            for t2, m2, s2, clear2 in map_cands:
                if m2[3] < 0 or not clear2:
                    continue
                p = cam + s2[2] * f + s2[0] * r
                phi = wrap(th + math.atan2(-s2[3], s2[5]))
                if m2[3] == 0:
                    m2[0], m2[1], m2[2], m2[3] = float(p[0]), float(p[1]), phi, 1
                else:
                    a = max(0.02, 1.0 / (m2[3] + 1))
                    m2[0] += a * (float(p[0]) - m2[0])
                    m2[1] += a * (float(p[1]) - m2[1])
                    m2[2] = wrap(m2[2] + a * wrap(phi - m2[2]))
                    m2[3] += 1
        self.pose = (float(car[0]), float(car[1]), yaw)
        self.pose_t = cap
        now = time.monotonic()
        dt_lat = min(0.5, max(0.0, now - cap))
        yaw_now = self.fused_yaw()
        if yaw_now is None:
            yaw_now = yaw
        yaw_mid = wrap(yaw + 0.5 * wrap(yaw_now - yaw))
        th_mid = yaw_mid - math.pi / 2
        f_mid = np.array([math.cos(th_mid), math.sin(th_mid)])
        left_mid = np.array([-math.sin(th_mid), math.cos(th_mid)])
        d_lat = (self.v[0] * f_mid + self.v[1] * left_mid) * dt_lat
        car_now = car + d_lat
        if self.dr_pose is None or math.hypot(car_now[0] - self.dr_pose[0], car_now[1] - self.dr_pose[1]) > 0.25:
            self.dr_pose, self.dr_t = [float(car_now[0]), float(car_now[1]), yaw_now], now
        else:
            # Blend: the estimate is model-propagated every tick and pulled toward each tag fix,
            # so the policy sees a continuous pose instead of 10-20 Hz jumps.
            self.dr_pose[0] += self.blend * (float(car_now[0]) - self.dr_pose[0])
            self.dr_pose[1] += self.blend * (float(car_now[1]) - self.dr_pose[1])
            self.dr_pose[2] = yaw_now
            self.dr_t = now
        self.dr_path = float(np.linalg.norm(d_lat))
        self.last = {"id": t["id"], "tag_yaw": tag_yaw, "range": float(math.hypot(s[0], s[2])),
                     "h": float(-s[1]), "n": len(cands)}
        return True

    def propagate(self, now: float, duty_fwd: float, duty_left: float):
        """Dead-reckon the pose with the modelled body velocity and the gyro yaw."""
        if self.dr_pose is None:
            return
        dt = min(0.1, max(0.0, now - self.dr_t))
        self.dr_t = now
        u = np.array([duty_fwd, duty_left])
        target = self.gain * np.sign(u) * np.maximum(np.abs(u) - self.dz, 0.0)
        self.v += (1 - math.exp(-dt / self.tau)) * (target - self.v)
        yaw = self.fused_yaw()
        th = yaw - math.pi / 2
        f = np.array([math.cos(th), math.sin(th)])
        left = np.array([-math.sin(th), math.cos(th)])
        d = (self.v[0] * f + self.v[1] * left) * dt
        self.dr_pose[0] += float(d[0])
        self.dr_pose[1] += float(d[1])
        self.dr_pose[2] = yaw
        self.dr_path += float(np.linalg.norm(d))

    def export_map(self) -> dict:
        return {str(k): [v[0], v[1] - self.tag_y, v[2], v[3]] for k, v in self.map.items()}


# ----------------------------------------------------------------------------- drive
class Drive:
    def __init__(self, ws_log: JsonlWriter | None = None):
        from websockets.sync.client import connect
        self.ws_log = ws_log
        self.ws = connect(f"ws://{HOST}:{PORT}/ws", open_timeout=5)
        self.state: dict = {}
        self.errors: deque[str] = deque(maxlen=20)
        self.alive = True
        threading.Thread(target=self._reader, daemon=True, name="ws-reader").start()

    def _reader(self):
        try:
            for raw in self.ws:
                t_rx = time.monotonic()
                m = json.loads(raw)
                if m.get("type") == "state":
                    self.state = m["state"]
                    if self.ws_log is not None:
                        self.ws_log.write({"t": t_rx, "dir": "rx", "type": "state", "state": m["state"]})
                elif m.get("type") == "error":
                    err = m.get("error", "?")
                    self.errors.append(err)
                    if self.ws_log is not None:
                        self.ws_log.write({"t": t_rx, "dir": "rx", "type": "error", "error": err})
        except Exception:
            pass
        self.alive = False

    def send(self, **m):
        t_tx = time.monotonic()
        self.ws.send(json.dumps(m))
        if self.ws_log is not None:
            self.ws_log.write({"t": t_tx, "dir": "tx", **m})

    def arm(self):
        self.send(type="aruco", enabled=False, follow=False)   # free the dashboard's detector CPU
        self.send(type="arm")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            if self.state.get("armed") and self.state.get("you_are_owner"):
                return
            time.sleep(0.02)
        raise RuntimeError(f"arm failed: {list(self.errors)} owner={self.state.get('owner')}")

    def drive(self, forward: float, strafe: float, turn: float):
        self.send(type="drive", forward=forward, strafe=strafe, turn=turn, speed_limit=PWM_LIMIT)

    def brake(self):
        self.send(type="drive", forward=0, strafe=0, turn=0, speed_limit=PWM_LIMIT)

    def release(self):
        try:
            self.send(type="stop", reason="disarm")
            time.sleep(0.1)
            self.ws.close()
        except Exception:
            pass


def duty_to_dashboard(duty_fwd: float, duty_left: float, duty_ccw: float):
    """Wheel-level duties (fraction of 4095; forward, left, CCW) -> dashboard (F, S, T).

    Identical wheel pattern to rl/sim.py + staged_baseline.mix for cmd_phys=(fwd, left, w):
    FL=-F-S+T, RL=-F+S+T, FR=-F+S-T, RR=-F-S-T with F=fwd, S=-left, T=+ccw. Scaled down
    uniformly when a wheel would exceed the 1800/4095 cap. Returns (F, S, T, scale)."""
    F, S, T = duty_fwd / KDUTY, -duty_left / KDUTY, duty_ccw / KDUTY
    wheels = (-F - S + T, -F + S + T, -F + S - T, -F - S - T)
    s = max(1.0, abs(F), abs(S), abs(T), *(abs(w) for w in wheels))
    return F / s, S / s, T / s, s


def action_to_duty(act, gain: float):
    """Student action (rl_sim axes) -> physical wheel-level duties (forward, left, CCW).

    rl/sim.py: act[0] drives +x_drone (car-left at yaw 0), act[1] drives +y_drone
    (backward), act[2] is +yaw (CCW). The sim's turn wheel term is w * HALF_BASE * 4."""
    return (-float(act[1]) * gain, float(act[0]) * gain, float(act[2]) * HALF_BASE * 4.0 * gain)


# ----------------------------------------------------------------------------- controllers
_STAGE_GEAR = {"clear_width_m": 0.31, "bar_length_m": 0.48, "bar_radius_m": 0.015, "inward_slope": 0.0}
_STAGE_VEHICLE = {"width_m": 0.172, "length_m": 0.203}


def _import_parking():
    try:
        from ugv.parking import Pose2, gear_obstacles, swept_clear
        return Pose2, gear_obstacles, swept_clear
    except ImportError:
        for p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2] / "src"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from ugv.parking import Pose2, gear_obstacles, swept_clear
        return Pose2, gear_obstacles, swept_clear


def validate_stage_arrival(outcome: str, target: tuple[float, float, float],
                           end: tuple[float, float, float] | None) -> str:
    if outcome == "arrived" and target[1] >= 0.42:
        if (end is None
                or math.hypot(end[0] - target[0], end[1] - target[1]) > 0.15
                or abs(wrap(end[2] - target[2])) > math.radians(12)):
            return "false_arrive"
    return outcome


class GoTo:
    """Holonomic P controller to a pose; faces the drone while travelling."""

    def __init__(self, target, args):
        self.t = target
        self.a = args
        self.ok_since = None
        self.last_reason = "goto"
        Pose2, gear_obstacles, swept_clear = _import_parking()
        self._Pose2 = Pose2
        self._swept_clear = swept_clear
        self._obstacles = gear_obstacles(_STAGE_GEAR)

    def _duty_to_vel(self, d_f: float, d_l: float, d_w: float) -> tuple[float, float, float]:
        a = self.a
        vf = 0.0 if d_f == 0.0 else math.copysign(max(0.0, abs(d_f) - a.dz_fwd) * a.gain_fwd, d_f)
        vl = 0.0 if d_l == 0.0 else math.copysign(max(0.0, abs(d_l) - a.dz_left) * a.gain_left, d_l)
        wz = 0.0 if d_w == 0.0 else math.copysign(max(0.0, abs(d_w) - a.dz_yaw) * a.gain_yaw, d_w)
        return vf, vl, wz

    def _is_clear(self, x: float, y: float, yaw: float, d_f: float, d_l: float, d_w: float,
                  horizon: float = 0.25) -> bool:
        vf, vl, wz = self._duty_to_vel(d_f, d_l, d_w)
        th = yaw - math.pi / 2
        dx = (vf * math.cos(th) - vl * math.sin(th)) * horizon
        dy = (vf * math.sin(th) + vl * math.cos(th)) * horizon
        dyaw = wz * horizon
        start = self._Pose2(x, y, -yaw)
        end = self._Pose2(x + dx, y + dy, -(yaw + dyaw))
        return self._swept_clear(start, end, _STAGE_VEHICLE["width_m"], _STAGE_VEHICLE["length_m"],
                                 self._obstacles, inflation=0.005)

    def step(self, pose, now):
        x, y, yaw = pose
        ex, ey = self.t[0] - x, self.t[1] - y
        dist = math.hypot(ex, ey)
        yaw_goal = self.t[2] if (dist < 0.25 or y < 0.25) else wrap(-math.atan2(x, y))
        eyaw = wrap(yaw_goal - yaw)
        th = yaw - math.pi / 2
        e_f = ex * math.cos(th) + ey * math.sin(th)
        e_l = -ex * math.sin(th) + ey * math.cos(th)
        a = self.a
        if dist < a.tol_pos and abs(wrap(self.t[2] - yaw)) < math.radians(a.tol_yaw):
            self.ok_since = self.ok_since or now
            self.last_reason = "goto"
            return (0.0, 0.0, 0.0), now - self.ok_since > 0.6
        self.ok_since = None

        def duty(e, kp, vcap, gain, dz, tol):
            # P velocity command, inverted through the fitted plant v = gain * (|duty| - dz)
            if abs(e) < tol:
                return 0.0
            return math.copysign(dz + min(vcap, kp * abs(e)) / gain, e)

        cmd = (duty(e_f, a.kp_pos, a.vcap_fwd, a.gain_fwd, a.dz_fwd, a.tol_pos * 0.5),
               duty(e_l, a.kp_pos, a.vcap_left, a.gain_left, a.dz_left, a.tol_pos * 0.5),
               duty(eyaw, a.kp_yaw, a.vcap_yaw, a.gain_yaw, a.dz_yaw, math.radians(a.tol_yaw) * 0.5))
        if self._is_clear(x, y, yaw, *cmd):
            self.last_reason = "goto"
            return cmd, False
        # Blocked by landing gear: try pure lateral centering, pure yaw alignment, or backing out.
        c_f = -x * math.cos(th)
        c_l = x * math.sin(th)
        b_f = 0.2 * math.sin(th)
        b_l = 0.2 * math.cos(th)
        for cand in (
            (duty(c_f, a.kp_pos, a.vcap_fwd, a.gain_fwd, a.dz_fwd, 0.005),
             duty(c_l, a.kp_pos, a.vcap_left, a.gain_left, a.dz_left, 0.005), 0.0),
            (0.0, 0.0, duty(wrap(self.t[2] - yaw), a.kp_yaw, a.vcap_yaw, a.gain_yaw, a.dz_yaw, 0.02)),
            (duty(b_f, a.kp_pos, a.vcap_fwd, a.gain_fwd, a.dz_fwd, 0.005),
             duty(b_l, a.kp_pos, a.vcap_left, a.gain_left, a.dz_left, 0.005), 0.0),
        ):
            if any(cand) and self._is_clear(x, y, yaw, *cand):
                self.last_reason = "goto:steer_clear"
                return cand, False
        self.last_reason = "brake: blocked"
        return (0.0, 0.0, 0.0), False


# ----------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "stage", "policy", "watch"])
    ap.add_argument("--out", type=Path, default=Path.home() / "l7_runs")
    ap.add_argument("--name", default="")
    ap.add_argument("--weights", type=Path, default=Path.home() / "ugv/runs/student.npz")
    ap.add_argument("--ids", default="3,6,14")
    ap.add_argument("--primary", type=int, default=14)
    ap.add_argument("--tagmap", type=Path, default=Path.home() / "l7_runs/tagmap.json")
    ap.add_argument("--tag-y", type=float, default=-0.08,
                    help="policy-frame y of the primary tag plane (goal is y=+0.04)")
    ap.add_argument("--gyro-sign", type=float, default=1.0)
    ap.add_argument("--time", type=float, default=45.0)
    ap.add_argument("--save-every", type=int, default=4, help="save every Nth processed JPEG (0=off)")
    ap.add_argument("--target", default="0,0.9,0", help="stage: x,y,yaw_deg in the policy frame")
    ap.add_argument("--act-gain", type=float, default=1.0, help="policy: action -> duty scale")
    ap.add_argument("--imu-w-mode", choices=["raw", "zero", "lp"], default="raw",
                    help="policy: gyro rate passed to policy.step (lp = EMA alpha 0.1)")
    ap.add_argument("--shield", choices=["real", "sim", "off"], default="real")
    ap.add_argument("--dr-max", type=float, default=1.0, help="s of dead reckoning allowed outside the corridor")
    ap.add_argument("--dr-max-corridor", type=float, default=8.0)
    ap.add_argument("--stage-dr-max", type=float, default=1.5, help="stage: max blind DR time (s) without a tag fix")
    ap.add_argument("--dr-path", type=float, default=0.9)
    ap.add_argument("--blend", type=float, default=0.35, help="per-fix position correction of the estimate")
    ap.add_argument("--calibrate-tagmap", action="store_true", help="update non-fixed tagmap entries online")
    ap.add_argument("--allow-uncalibrated-tags", action="store_true",
                    help="allow obs==0 prior tags (e.g. tag 6) in localization")
    # body-velocity model for dead reckoning (m/s per unit duty beyond dz); set from probe data
    # real plant, fitted by fit_probe.py on 20261113-181040-probe-axes: v = gain * (|duty| - dz)
    ap.add_argument("--gain-fwd", type=float, default=2.3)
    ap.add_argument("--gain-left", type=float, default=1.6)
    ap.add_argument("--gain-yaw", type=float, default=15.7, help="rad/s per duty")
    ap.add_argument("--dz-fwd", type=float, default=0.05)
    ap.add_argument("--dz-left", type=float, default=0.13)
    ap.add_argument("--dz-yaw", type=float, default=0.10)
    ap.add_argument("--adapter", choices=["vel", "raw"], default="vel",
                    help="policy: vel = reproduce the sim plant's body velocity with the fitted real "
                         "plant; raw = identical wheel pattern to rl/sim.py")
    ap.add_argument("--acquire-dps", type=float, default=25.0, help="search spin rate; sign = direction (+CCW)")
    ap.add_argument("--sim-plant", choices=["mean", "p0"], default="p0")
    ap.add_argument("--yaw-ki", type=float, default=0.8, help="duty per rad of integrated rate error")
    ap.add_argument("--yaw-kp", type=float, default=0.03, help="duty per rad/s")
    ap.add_argument("--profile", type=Path, default=Path(__file__).resolve().parent / "mecanum_bayesian_profile.json")
    # go-to controller (duty units)
    ap.add_argument("--kp-pos", type=float, default=1.2, help="1/s")
    ap.add_argument("--kp-yaw", type=float, default=1.5, help="1/s")
    ap.add_argument("--vcap-fwd", type=float, default=0.30)
    ap.add_argument("--vcap-left", type=float, default=0.25)
    ap.add_argument("--vcap-yaw", type=float, default=0.5, help="rad/s; faster yaw blurs the tags")
    ap.add_argument("--tol-pos", type=float, default=0.04)
    ap.add_argument("--tol-yaw", type=float, default=5.0)
    # probe
    ap.add_argument("--probe-duty", default="0.2,0.3")
    ap.add_argument("--probe-s", type=float, default=0.4)
    ap.add_argument("--probe-axes", default="fwd,left,ccw")
    ap.add_argument("--baseline-s", type=float, default=0.0,
                    help="stationary pre-arm baseline logging duration (s)")
    ap.add_argument("--coast-s", type=float, default=0.8,
                    help="probe: post-brake coast logging duration (s)")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{args.mode}" + (f"-{args.name}" if args.name else "")
    out = args.out / run_id
    (out / "frames").mkdir(parents=True)
    ids = {int(v) for v in args.ids.split(",")}
    tagmap = json.loads(args.tagmap.read_text()) if args.tagmap.is_file() else DEFAULT_MAP
    for k, v in DEFAULT_MAP.items():
        tagmap.setdefault(k, v)
    meta = {"run_id": run_id, "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
            "K": K.tolist(), "tag_size_m": TAG_SIZE_M, "cam_fwd_m": CAM_FWD_M, "pwm_limit": PWM_LIMIT,
            "tagmap_in": tagmap, "host": os.uname().nodename, "started_unix": time.time(),
            "started_monotonic": time.monotonic(),
            "frame": "drone/policy frame: skids along y, yaw 0 faces -y, CCW+; anchored on primary tag"}

    ticks, imulog, wslog = (JsonlWriter(out / n) for n in ("ticks.jsonl", "imu.jsonl", "ws_state.jsonl"))
    imu = Imu(imulog, args.gyro_sign)
    vis = Vision(out, ids, args.save_every)
    loc = Localizer(imu, tagmap, args.tag_y, args.primary, args)
    vis.start()
    imu.start()

    policy = None
    if args.mode == "policy":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ugv.policy import StudentPolicy
        from ugv.parking import Pose2
        kw = {}
        if args.shield == "real":   # simulation/gazebo/drone_dimensions.json + Gazebo car footprint
            kw = dict(gear={"clear_width_m": 0.31, "bar_length_m": 0.48, "bar_radius_m": 0.015,
                            "inward_slope": 0.0},
                      vehicle={"width_m": 0.172, "length_m": 0.203})
        policy = StudentPolicy.load(args.weights, **kw)
        if args.shield == "off":
            policy.obstacles = []
        meta["policy"] = {"sha": policy.sha, "stateful": policy.stateful, "weights": str(args.weights)}
        sim_model = None
        if args.adapter == "vel":
            # rl/sim.py EnvConfig.use_mean_dynamics=True is the L0/L1 training plant: posterior-mean
            # coefficients, in which a full turn command yields only ~0.03-0.2 rad/s.
            from ugv.staged_baseline import mix, wheel_features
            prof = json.loads(args.profile.read_text())
            if args.sim_plant == "mean":
                coef0 = np.mean([q["coefficients"] for q in prof["posterior_particles"]], axis=0)
                dz0 = prof["deadzone"]
            else:
                coef0 = np.array(prof["posterior_particles"][0]["coefficients"])
                dz0 = prof["posterior_particles"][0]["deadzone"]

            def sim_model(a):
                return wheel_features(mix(a), dz0) @ coef0
        yaw_i = [0.0]

        def adapt(act):
            """Sim steady-state body velocity for this action -> real duties that produce it."""
            if sim_model is None:
                return action_to_duty(act, args.act_gain), None
            v = sim_model(np.asarray(act[:3], dtype=float))
            want = (-float(v[1]), float(v[0]), float(v[2]))   # forward, left, ccw

            def inv(x, gain, dz, eps):
                return 0.0 if abs(x) < eps else math.copysign(dz + abs(x) / gain, x) * args.act_gain
            # Yaw: gyro rate loop (PI on duty) so small commanded rates survive the turn deadzone
            # and translation-induced yaw is cancelled.
            e_w = want[2] - imu.snapshot()[1]
            yaw_i[0] = max(-0.22, min(0.22, yaw_i[0] + args.yaw_ki * e_w * DT))
            d_yaw = max(-0.25, min(0.25, yaw_i[0] + args.yaw_kp * e_w))
            return (inv(want[0], args.gain_fwd, args.dz_fwd, 0.02), inv(want[1], args.gain_left, args.dz_left, 0.02),
                    d_yaw), want
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    drv = None
    stop = {"flag": False, "why": ""}

    def on_sig(signum, _):
        stop["flag"], stop["why"] = True, f"signal-{signum}"
        if drv:
            try:
                drv.brake()
            except Exception:
                pass
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    summary: dict = {"run_id": run_id, "mode": args.mode}
    seen = 0

    def pump() -> bool:
        """Consume the newest vision record. True if it produced a tag pose."""
        nonlocal seen
        rec, seen2 = vis.poll(seen)
        if rec is None:
            return False
        seen = seen2
        return loc.update(rec)

    def rest_pose(seconds=0.7, need=4):
        """Median tag pose while stationary (waits for the camera to settle)."""
        time.sleep(0.35)
        ps, t_end = [], time.monotonic() + seconds + 2.0
        t_min = time.monotonic() + seconds
        while time.monotonic() < t_end and (len(ps) < need or time.monotonic() < t_min):
            if pump():
                ps.append(loc.pose)
            time.sleep(0.01)
        if len(ps) < 2:
            return None
        a = np.array(ps)
        return (float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                math.atan2(np.median(np.sin(a[:, 2])), np.median(np.cos(a[:, 2]))))

    try:
        # ---- acquire a first fix (no motion) ----
        t_wait = time.monotonic() + 4.0
        while time.monotonic() < t_wait and loc.pose is None:
            pump()
            time.sleep(0.01)
        p0 = rest_pose() if loc.pose is not None else None
        print(f"[{run_id}] start pose={None if p0 is None else [round(p0[0],3), round(p0[1],3), round(math.degrees(p0[2]),1)]} "
              f"last={loc.last} vision_ms={np.median(vis.proc_ms) if vis.proc_ms else None} gyro_bias={imu.bias:.3f}", flush=True)
        summary["start_pose"] = p0
        if args.baseline_s > 0:
            t_base = time.monotonic()
            while time.monotonic() - t_base < args.baseline_s and not stop["flag"]:
                got = pump()
                yaw, rate, _, wh = imu.snapshot()
                ticks.write({"t": time.monotonic(), "phase": "baseline", "tag": got, "pose": loc.pose,
                             "pose_t": loc.pose_t, "last": loc.last, "imu": [yaw, rate], "wh": wh})
                time.sleep(DT)
        if args.mode == "watch":
            t_end = time.monotonic() + args.time
            while time.monotonic() < t_end and not stop["flag"]:
                got = pump()
                yaw, rate, _, _ = imu.snapshot()
                ticks.write({"t": time.monotonic(), "tag": got, "pose": loc.pose, "last": loc.last,
                             "imu": [yaw, rate]})
                time.sleep(DT)
        else:
            drv = Drive(ws_log=wslog)
            drv.arm()
        if drv and p0 is None and args.mode != "probe":
            # Acquire: gyro-rate-controlled spin (slow enough not to blur the tags) until a fix.
            integ, hits, g0 = 0.0, 0, imu.snapshot()[0]
            rate = math.radians(args.acquire_dps)
            t_a = time.monotonic()
            while not stop["flag"] and time.monotonic() - t_a < 30 and abs(imu.snapshot()[0] - g0) < 2 * math.pi:
                if not drv.alive or not vis.proc.is_alive():
                    break
                hits = hits + 1 if pump() else hits
                if hits >= 3:
                    break
                now_a = time.monotonic()
                _, imu_w_a, imu_at_a, _ = imu.snapshot()
                if now_a - vis.frame_at > 0.45 or now_a - imu_at_a > 0.25:
                    drv.brake()
                    time.sleep(DT)
                    continue
                e_w = rate - imu_w_a
                integ = max(-0.22, min(0.22, integ + 0.8 * e_w * DT))
                drv.drive(*duty_to_dashboard(0.0, 0.0, integ + 0.03 * e_w)[:3])
                time.sleep(DT)
            drv.brake()
            p0 = rest_pose() if hits >= 3 else None
            summary["acquired_pose"] = p0
            print("acquire ->", p0, flush=True)
            if p0 is None:
                raise RuntimeError("no tag found after acquisition spin; refusing to drive")

        # ---- probe ----
        if args.mode == "probe" and drv:
            rows = []
            for duty in (float(v) for v in args.probe_duty.split(",")):
                for axis in args.probe_axes.split(","):
                    for sgn in (1, -1):
                        if stop["flag"]:
                            break
                        d = {
                            "fwd": (duty * sgn, 0.0, 0.0),
                            "left": (0.0, duty * sgn, 0.0),
                            "ccw": (0.0, 0.0, duty * sgn),
                            "fwd+left": (duty * sgn, duty * sgn, 0.0),
                            "fwd+ccw": (duty * sgn, 0.0, duty * sgn),
                        }[axis]
                        F, S, T, sc = duty_to_dashboard(*d)
                        before = rest_pose(0.5)
                        g0 = imu.snapshot()[0]
                        t0 = time.monotonic()
                        while time.monotonic() - t0 < args.probe_s and not stop["flag"]:
                            drv.drive(F, S, T)
                            pump()
                            yaw, rate, _, wh = imu.snapshot()
                            ticks.write({"t": time.monotonic(), "phase": "drive", "probe": [axis, duty * sgn],
                                         "fst": [F, S, T], "pose": loc.pose, "pose_t": loc.pose_t,
                                         "imu": [yaw, rate], "wh": wh})
                            time.sleep(DT)
                        drv.brake()
                        if args.coast_s > 0:
                            t_c = time.monotonic()
                            while time.monotonic() - t_c < args.coast_s and not stop["flag"]:
                                pump()
                                yaw, rate, _, wh = imu.snapshot()
                                ticks.write({"t": time.monotonic(), "phase": "coast", "probe": [axis, duty * sgn],
                                             "fst": [0.0, 0.0, 0.0], "pose": loc.pose, "pose_t": loc.pose_t,
                                             "imu": [yaw, rate], "wh": wh})
                                time.sleep(DT)
                        after = rest_pose(0.5)
                        g1 = imu.snapshot()[0]
                        row = {"axis": axis, "duty": duty * sgn, "s": args.probe_s, "fst": [F, S, T],
                               "gyro_dyaw_deg": math.degrees(g1 - g0)}
                        if before and after:
                            th = before[2] - math.pi / 2
                            dx, dy = after[0] - before[0], after[1] - before[1]
                            row.update(d_fwd=dx * math.cos(th) + dy * math.sin(th),
                                       d_left=-dx * math.sin(th) + dy * math.cos(th),
                                       tag_dyaw_deg=math.degrees(wrap(after[2] - before[2])),
                                       before=before, after=after)
                        rows.append(row)
                        print("probe", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                   for k, v in row.items() if k not in ("before", "after", "fst")}), flush=True)
            summary["probe"] = rows

        # ---- closed loop: stage / policy ----
        if args.mode in ("stage", "policy") and drv:
            ctrl = None
            if args.mode == "stage":
                tx, ty, tyaw = (float(v) for v in args.target.split(","))
                ctrl = GoTo((tx, ty, math.radians(tyaw)), args)
            prev_act = np.zeros(3)
            duty = (0.0, 0.0, 0.0)
            applied = (0.0, 0.0)
            still_s, k = 0.0, 0
            last_valid_t = loc.pose_t if loc.pose_t > 0 else time.monotonic()
            reasons: dict[str, int] = {}
            loop_ms, infer_ms = [], []
            t_start = time.monotonic()
            t_next = t_start
            outcome = "timeout"
            prev_pose_for_speed = None
            brake_stale_since = None
            imu_w_lp = None
            while not stop["flag"]:
                now = time.monotonic()
                if now - t_start > args.time:
                    break
                got = pump()
                loc.propagate(now, applied[0], applied[1])
                frame_age = now - vis.frame_at
                yaw_g, imu_w, imu_at, wheels = imu.snapshot()
                fy = loc.fused_yaw()
                src, pose = "none", None
                if got:
                    last_valid_t = loc.pose_t
                if loc.dr_pose is not None:
                    # Observation = estimator pose at the sim camera cadence (30 Hz: 3 of 5 ticks),
                    # while a tag fix is recent enough (longer inside the corridor, where the tag
                    # leaves the vertical field of view).
                    age = now - loc.pose_t
                    in_corridor = abs(loc.dr_pose[0]) < 0.16 and loc.dr_pose[1] < 0.75
                    limit = args.dr_max_corridor if in_corridor else args.dr_max
                    if args.mode == "stage":
                        limit = min(limit, min(1.5, args.stage_dr_max))
                    if age < limit and loc.dr_path < args.dr_path and k % 5 in (0, 2, 3):
                        src = "tag" if age < 0.15 else "dr"
                        pose = tuple(loc.dr_pose)
                        if src == "tag":
                            last_valid_t = loc.pose_t
                est = tuple(loc.dr_pose) if loc.dr_pose is not None else None
                stale = max(0.0, now - loc.pose_t) if loc.pose_t > 0 else max(0.0, now - last_valid_t)
                rec = {"k": k, "t": now, "src": src, "pose": pose, "est": est, "cap": loc.pose_t if got else None,
                       "last": loc.last if got else None, "imu": [fy, imu_w], "gyro_yaw": yaw_g,
                       "frame_age": round(frame_age, 3), "stale": round(stale, 3), "wh": wheels}
                reason = ""
                if frame_age > 0.45 or now - imu_at > 0.2 or not drv.alive:
                    duty, reason = (0.0, 0.0, 0.0), "brake: sensors stale"
                    act = np.zeros(3)
                elif args.mode == "stage":
                    stage_dr_cap = min(1.5, args.stage_dr_max)
                    if should_abort_corridor(args.mode, est, loc.pose_t, now):
                        duty, reason, act = (0.0, 0.0, 0.0), "abort: blind_corridor_x", np.zeros(3)
                        outcome = "abort_corridor"
                        stop["flag"], stop["why"] = True, outcome
                    elif est is None or now - last_valid_t > stage_dr_cap:
                        duty, reason, act = (0.0, 0.0, 0.0), "brake: no pose", np.zeros(3)
                    else:
                        duty, done = ctrl.step(est, now)
                        act, reason = np.zeros(3), ctrl.last_reason
                        if done:
                            outcome = "arrived"
                            stop["flag"], stop["why"] = True, "arrived"
                else:
                    valid = src != "none"
                    tag = pose if valid else None
                    t_inf = time.perf_counter()
                    e = est if est is not None else (0.0, 2.0, 0.0)
                    imu_w_lp = imu_w if imu_w_lp is None else 0.1 * imu_w + 0.9 * imu_w_lp
                    imu_w_fed = 0.0 if args.imu_w_mode == "zero" else (imu_w_lp if args.imu_w_mode == "lp" else imu_w)
                    act, reason = policy.step(tag, stale, valid, imu_w_fed,
                                              fy if fy is not None else 0.0, prev_act,
                                              Pose2(e[0], e[1], e[2]), dt=DT, dt_stale=min(stale, 0.3))
                    infer_ms.append((time.perf_counter() - t_inf) * 1e3)
                    prev_act = np.asarray(act[:3], dtype=float)
                    duty, want = adapt(act)
                    rec["imu_w_fed"] = round(float(imu_w_fed), 6)
                    rec["act"] = [round(float(v), 4) for v in act[:3]]
                    rec["want"] = want
                F, S, T, sc = duty_to_dashboard(*duty)
                if any(duty):
                    drv.drive(F, S, T)
                else:
                    drv.brake()
                applied = (duty[0] / sc, duty[1] / sc)
                reasons[reason] = reasons.get(reason, 0) + 1
                rec.update(reason=reason, duty=[round(v, 4) for v in duty], fst=[round(F, 4), round(S, 4), round(T, 4)],
                           sat=round(sc, 3))
                # ---- termination bookkeeping (estimate-based) ----
                if est is not None:
                    if prev_pose_for_speed is not None:
                        sp = math.hypot(est[0] - prev_pose_for_speed[0], est[1] - prev_pose_for_speed[1]) / DT
                    else:
                        sp = 0.0
                    prev_pose_for_speed = est
                    near = abs(est[0]) < 0.03 and abs(est[1]) < 0.05 and abs(est[2]) < math.radians(5)
                    slow = abs(imu_w) < math.radians(2) and not any(wheels)
                    still_s = still_s + DT if (near and slow) else 0.0
                    if args.mode == "policy" and still_s >= 1.0:
                        outcome = "parked(est)"
                        stop["flag"], stop["why"] = True, outcome
                    if abs(est[0]) > 1.6 or est[1] > 2.6 or est[1] < -0.35:
                        outcome = "geofence"
                        stop["flag"], stop["why"] = True, outcome
                if args.mode == "policy":
                    if reason.startswith("brake: stale"):
                        brake_stale_since = brake_stale_since or now
                        if now - brake_stale_since > 4.0:
                            outcome = "lost"
                            stop["flag"], stop["why"] = True, outcome
                    else:
                        brake_stale_since = None
                t_done = time.monotonic()
                rec["loop_ms"] = round((t_done - now) * 1e3, 2)
                loop_ms.append((t_done - now) * 1e3)
                ticks.write(rec)
                k += 1
                t_next += DT
                sl = t_next - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
                else:
                    t_next = time.monotonic()
            drv.brake()
            end = rest_pose(0.6)
            if args.mode == "stage" and ctrl is not None:
                outcome = validate_stage_arrival(outcome, ctrl.t, end)
            summary.update(outcome=outcome if not stop["why"].startswith("signal") else stop["why"],
                           ticks=k, seconds=round(time.monotonic() - t_start, 2), reasons=reasons,
                           end_pose_tag=end, end_est=loc.dr_pose, dr_path=loc.dr_path,
                           loop_ms_p50=float(np.median(loop_ms)) if loop_ms else None,
                           loop_ms_p95=float(np.percentile(loop_ms, 95)) if loop_ms else None,
                           loop_ms_max=float(np.max(loop_ms)) if loop_ms else None,
                           infer_ms_p50=float(np.median(infer_ms)) if infer_ms else None,
                           hz=round(k / max(1e-6, time.monotonic() - t_start), 1))
    except Exception as e:  # always brake below
        summary["error"] = f"{type(e).__name__}: {e}"
        print("ERROR", summary["error"], flush=True)
    finally:
        if drv:
            try:
                drv.brake()
            except Exception:
                pass
            drv.release()
        time.sleep(0.15)
        vis.stop_evt.set()
        imu.stop_evt.set()
        vis.join(timeout=2)
        imu.join(timeout=2)
        summary["tagmap_out"] = loc.export_map()
        summary["vision_ms_p50"] = float(np.median(vis.proc_ms)) if vis.proc_ms else None
        if args.mode != "watch" or True:
            args.tagmap.parent.mkdir(parents=True, exist_ok=True)
            args.tagmap.write_text(json.dumps(summary["tagmap_out"], indent=1))
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for w in (ticks, imulog, wslog):
            w.close()
        print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k not in ("probe",)}, default=float), flush=True)
        print("OUT", out, flush=True)


if __name__ == "__main__":
    main()
