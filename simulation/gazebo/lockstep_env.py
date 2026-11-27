"""Shared Gazebo Harmonic lockstep environment, frames, geometry, and emulated sensors."""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as E
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from build_motor_plugin import plugin_environment  # noqa: E402
from drive_guard import require_drivable_car  # noqa: E402

from ugv.staged_baseline import GAZEBO_GEOMETRY, mix  # noqa: E402

DT = 0.02
PHYS = 0.001
WORLD, MODEL = "freenove_lab", "freenove"
DRONE_XY = (0.75, 0.0)  # drone model pose in the generated world (skids along X)
SPAWN_Z = 0.001
PREFIX = Path(os.environ.get("UGV_GZ_PREFIX", "/opt/gz-harmonic/usr"))
BRIDGE = HERE / "lockstep" / "build" / "lockstep_bridge"
FOV_HALF = math.radians(33.5)

LEVELS = {  # sensor + spawn subsets of rl/sim.py LEVELS (dynamics come from Gazebo)
    "L0": dict(
        dist=(0.75, 0.75),
        bearing=0.0,
        yaw=0.0,
        noise=0.0,
        latency=(0.0, 0.0),
        drops=(0.0, 0.0),
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        area=1.0,
    ),
    "L1": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.08, 0.08),
        drops=(0.0, 0.0),
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        area=1.5,
    ),
    "L2": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.08, 0.08),
        drops=(0.0, 0.0),
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        area=1.5,
    ),
    "L3": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.08, 0.20),
        drops=(0.25, 0.45),
        outliers=0.02,
        blackout=3.0,
        under_drone_loss=True,
        area=1.5,
    ),
}


# ---------------- frames ----------------
def to_drone(x: float, y: float, yaw: float) -> tuple[float, float, float]:
    """Gazebo world pose -> drone frame of rl/sim.py (drone at origin, skids along y,
    yaw 0 = camera facing the drone from +y)."""
    return y - DRONE_XY[1], -(x - DRONE_XY[0]), math.atan2(math.sin(yaw), math.cos(yaw))


def to_world(x: float, y: float, yaw: float) -> tuple[float, float, float]:
    return DRONE_XY[0] - y, DRONE_XY[1] + x, yaw


# ---------------- geometry check (scoring) ----------------
def collides(pose, g=GAZEBO_GEOMETRY) -> bool:
    """Car rectangle vs the two skid capsules (exact distance, drone frame)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    hw, hl = g.car_w / 2, g.car_l / 2
    corners = [
        (x + c * a - s * b, y + s * a + c * b)
        for a, b in ((-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl))
    ]

    def seg_dist(p, q, r, t):
        def pt_seg(pp, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            den = dx * dx + dy * dy
            u = (
                0.0
                if den == 0
                else max(0.0, min(1.0, ((pp[0] - a[0]) * dx + (pp[1] - a[1]) * dy) / den))
            )
            return math.hypot(pp[0] - a[0] - u * dx, pp[1] - a[1] - u * dy)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1, d2, d3, d4 = cross(r, t, p), cross(r, t, q), cross(p, q, r), cross(p, q, t)
        if d1 * d2 < 0 and d3 * d4 < 0:
            return 0.0
        return min(pt_seg(p, r, t), pt_seg(q, r, t), pt_seg(r, p, q), pt_seg(t, p, q))

    for sx in (-g.skid_x, g.skid_x):
        a, b = (sx, -g.skid_half_len), (sx, g.skid_half_len)
        # skid inside the rectangle
        lx, ly = c * (sx - x) + s * (0 - y), -s * (sx - x) + c * (0 - y)
        if abs(lx) <= hw and abs(ly) <= hl:
            return True
        if any(seg_dist(corners[i], corners[(i + 1) % 4], a, b) <= g.skid_r for i in range(4)):
            return True
    return False


# ---------------- sensors (rl/sim.py model) ----------------
class Sensors:
    def __init__(
        self,
        level: dict,
        rng: np.random.Generator,
        max_time: float,
        init_pose: tuple[float, float, float] | None = None,
    ):
        self.L, self.rng = level, rng
        self.latency = float(rng.uniform(*level["latency"]))
        self.drop_p = float(rng.uniform(*level["drops"]))
        self.blackout = (
            (float(rng.uniform(0, max_time - level["blackout"])), float(level["blackout"]))
            if level["blackout"]
            else None
        )
        self.ring = [tuple(init_pose)] if init_pose is not None else []
        self.bias = 0.002 * float(rng.standard_normal())
        self.t = 0.0
        self.age = 0.0
        self.imu_yaw = float(init_pose[2]) if init_pose is not None else 0.0

    def tag(self, pose: tuple[float, float, float]) -> tuple[np.ndarray, bool]:
        self.ring.append(tuple(pose))
        k = int(round(self.latency / DT))
        d_pose = self.ring[max(0, len(self.ring) - 1 - k)]
        self.ring = self.ring[-40:]
        self.t += DT
        x, y, _ = d_pose
        dist = math.hypot(x, y)
        bearing = math.atan2(x, max(y, 1e-6)) + pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        near = dist < 0.30
        in_fov = abs(bearing) < FOV_HALF and dist < 5.0
        seen = (in_fov and not near) if self.L["under_drone_loss"] else (in_fov or near)
        if self.blackout and self.blackout[0] < self.t < self.blackout[0] + self.blackout[1]:
            seen = False
        if self.rng.random() < self.drop_p:
            seen = False
        noise = float(self.L.get("noise", 1.0))
        ps = noise * (0.008 + 0.020 * dist)
        ys = noise * (math.radians(2) + math.radians(2) * dist)
        z = np.array(d_pose, dtype=float) + self.rng.standard_normal(3) * [ps, ps, ys]
        if self.L["outliers"] and self.rng.random() < self.L["outliers"]:
            z[:2] += 0.15 * self.rng.standard_normal(2)
        self.age = 0.0 if seen else self.age + DT
        return z, seen

    def gyro(self, w_true: float) -> float:
        noise = float(self.L.get("noise", 1.0))
        g = w_true + noise * 0.002 * float(self.rng.standard_normal()) + self.bias
        self.bias = self.bias * (1 - DT / 0.5) + math.sqrt(2 * DT / 0.5) * 0.001 * float(
            self.rng.standard_normal()
        )
        self.imu_yaw = math.atan2(math.sin(self.imu_yaw + g * DT), math.cos(self.imu_yaw + g * DT))
        return g


# ---------------- Gazebo process + bridge ----------------
def headless_world(src: Path, dst: Path):
    require_drivable_car(world=src)
    root = E.parse(src)
    for world in root.findall("world"):
        for plugin in list(world.findall("plugin")):
            if "Sensors" in plugin.get("name", ""):
                world.remove(plugin)
    for link in root.findall(".//link"):
        for sensor in list(link.findall("sensor")):
            link.remove(sensor)
    car = root.find(f".//model[@name='{MODEL}']")
    drive = car.find("plugin[@name='ugv::TTMotorSystem']")
    if drive is None or drive.findtext("mecanum") != "true":
        raise SystemExit(
            "world must use the physical mecanum motor plugin (generate.py --car freenove --drone)"
        )
    # The odometry publisher's TF also uses /model/<name>/pose (odom frame, unstamped);
    # remove it so the topic carries only stamped world poses.
    for plugin in list(car.findall("plugin")):
        if "OdometryPublisher" in plugin.get("name", ""):
            car.remove(plugin)
    pub = E.SubElement(
        car,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    # Publish every 1 ms physics step (1000 Hz) to eliminate 50 Hz boundary aliasing.
    for k, v in (
        ("publish_link_pose", "false"),
        ("publish_model_pose", "true"),
        ("use_pose_vector_msg", "true"),
        ("update_frequency", "1000"),
        ("static_publisher", "false"),
    ):
        E.SubElement(pub, k).text = v
    root.write(dst)


class Gazebo:
    def __init__(self, world: Path, log: Path):
        require_drivable_car(world=world)
        if not BRIDGE.exists():
            raise FileNotFoundError(f"build {BRIDGE} first (see lockstep/CMakeLists.txt)")
        part = "ugv-l6-" + uuid.uuid4().hex
        env = plugin_environment()
        env.update(GZ_PARTITION=part, GZ_IP="127.0.0.1")
        self.log_path = Path(log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb", buffering=0)
        gz_cmd = ["gz", "sim", "-s", "-v", "3", str(world)]
        if shutil.which("stdbuf"):
            gz_cmd = ["stdbuf", "-oL", "-eL", *gz_cmd]
        self.server = subprocess.Popen(
            gz_cmd,
            cwd=HERE,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        benv = dict(
            os.environ,
            GZ_PARTITION=part,
            GZ_IP="127.0.0.1",
            LD_LIBRARY_PATH=f"{PREFIX}/lib/x86_64-linux-gnu:{PREFIX}/lib",
        )
        self.bridge = subprocess.Popen(
            [str(BRIDGE), WORLD, MODEL, str(PHYS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=benv,
            bufsize=1,
        )
        deadline = time.monotonic() + 60
        while True:  # wait for the server to load (first step returns a pose)
            r = self.cmd("D 0 0 0 0 20")
            if r.startswith("P"):
                break
            if (
                time.monotonic() > deadline
                or self.server.poll() is not None
                or self.bridge.poll() is not None
            ):
                self.close()
                raise RuntimeError(f"Gazebo did not start; see {self.log_path}")
            time.sleep(0.5)

    def _check_alive(self):
        rc = self.server.poll()
        if rc is not None:
            raise RuntimeError(f"Gazebo server exited with code {rc}; see {self.log_path}")
        brc = self.bridge.poll()
        if brc is not None:
            raise RuntimeError(f"lockstep_bridge exited with code {brc}; see {self.log_path}")

    def cmd(self, line: str) -> str:
        assert self.bridge.stdin is not None and self.bridge.stdout is not None
        self.bridge.stdin.write(line + "\n")
        self.bridge.stdin.flush()
        return self.bridge.stdout.readline().strip()

    def set_pose(self, x: float, y: float, yaw: float, z: float = SPAWN_Z):
        self._check_alive()
        r = self.cmd(f"S {x} {y} {z} {yaw}")
        if r != "OK":
            self._check_alive()
            raise RuntimeError(f"set_pose failed ({r}); see {self.log_path}")

    def reset_pose(self, x: float, y: float, yaw: float, z: float = SPAWN_Z):
        """No world reset (plugins keep last-publish time). Brake 0.5 s, teleport, settle 0.3 s braked."""
        for _ in range(25):
            self.step([0.0, 0.0, 0.0, 0.0])
        self.set_pose(x, y, yaw, z=z)
        t = px = py = pyaw = 0.0
        for _ in range(15):
            t, px, py, pyaw = self.step([0.0, 0.0, 0.0, 0.0])
        return t, px, py, pyaw

    def step(self, duties, n: int = int(round(DT / PHYS))):
        self._check_alive()
        r = self.cmd("D " + " ".join(f"{d:.6f}" for d in duties) + f" {n}")
        if not r.startswith("P"):
            self._check_alive()
            raise RuntimeError(f"{r}; see {self.log_path}")
        _, t, x, y, yaw = r.split()
        return float(t), float(x), float(y), float(yaw)

    def close(self):
        try:
            if self.bridge.poll() is None:
                self.cmd("Q")
        except Exception:
            pass
        try:
            self.bridge.kill()
            self.bridge.wait(timeout=2)
        except Exception:
            pass
        try:
            os.killpg(self.server.pid, signal.SIGINT)
            self.server.wait(timeout=5)
        except Exception:
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
                self.server.wait(timeout=2)
            except Exception:
                pass
        try:
            self.log.close()
        except Exception:
            pass


def duties_for(cmd):
    """Controller (vx, vy, w) -> hardware wheel PWM (rl/sim.py mixer, order FL RL FR RR)
    -> Gazebo physical duties FL FR RL RR (polarity as in tools/calibration replay)."""
    w = mix(np.asarray(cmd, float)[None])[0]
    return (-w[[0, 2, 1, 3]]).tolist()
