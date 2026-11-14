"""L6 sim-to-sim harness for the exported NumPy student and the EKF + staged baseline
(docs/RL_PARKING.md).

--bridge fake: kinematic point-mass episodes that exercise the episode loop,
parked scoring and metrics with the real StudentPolicy. Runs anywhere; it
validates plumbing, not policy quality.

--bridge gazebo: closed-loop Gazebo Harmonic lockstep evaluation via lockstep_env.py
(passive-roller mecanum car, photo-model drone, 1 ms physics stepped at 50 Hz).
Supports --sensing emulated (rl/sim.py sensor model) and --sensing camera[_dr]
(synthetic 3D ArUco render at the true Gazebo pose decoded by real aruco.py
detect_gray + PnP, with optional --lighting-dr photometric DR).

Step rate is 50 Hz: the student history (50 frames = 1 s) and its tag-velocity
estimator are trained at dt=0.02 (rl/sim.py).
"""
from __future__ import annotations
import argparse
import inspect
import json
import math
import random
import subprocess
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

from lockstep_env import (  # noqa: E402
    BRIDGE,
    DT,
    Gazebo,
    LEVELS,
    PHYS,
    Sensors,
    collides,
    duties_for,
    headless_world,
    to_drone,
    to_world,
)
from ugv.parking import Pose2  # noqa: E402
from ugv.policy import StudentPolicy  # noqa: E402
from ugv.staged_baseline import GAZEBO_GEOMETRY, Geometry, MeanModel, StagedBaseline  # noqa: E402

LAT_TOL, LONG_TOL, YAW_TOL = 0.03, 0.05, math.radians(5)
STILL_V, STILL_W, HOLD_S = 0.01, math.radians(2), 1.0

GAZEBO_VEHICLE = {"width_m": GAZEBO_GEOMETRY.car_w, "length_m": GAZEBO_GEOMETRY.car_l}
GAZEBO_GEAR = {
    "clear_width_m": 2.0 * (GAZEBO_GEOMETRY.skid_x - GAZEBO_GEOMETRY.skid_r),
    "bar_length_m": 2.0 * GAZEBO_GEOMETRY.skid_half_len,
    "bar_radius_m": GAZEBO_GEOMETRY.skid_r,
    "inward_slope": 0.0,
}


@dataclass
class EpisodeResult:
    success: bool
    collision: bool
    timeout: bool
    time_s: float
    final_pos: float
    final_yaw_deg: float


class Scorer:
    """Parked criteria from docs/RL_PARKING.md. The bridge supplies collision."""

    def __init__(self, max_time=45.0):
        self.max_time = max_time
        self.reset()

    def reset(self):
        self.still = 0.0
        self.t = 0.0

    def update(self, x, y, yaw, vx, vy, w, collision, dt):
        self.t += dt
        near = abs(x) < LAT_TOL and abs(y) < LONG_TOL and abs(yaw) < YAW_TOL
        slow = abs(vx) < STILL_V and abs(vy) < STILL_V and abs(w) < STILL_W
        self.still = self.still + dt if (near and slow) else 0.0
        end = (self.still >= HOLD_S, collision, self.t >= self.max_time)
        if any(end):
            ok, hit = end[0], end[1] and not end[0]
            return EpisodeResult(ok, hit, not ok and not hit, self.t,
                                 math.hypot(x, y), math.degrees(abs(yaw)))
        return None


class FakeBridge:
    """Kinematic point mass with synthetic yaw coupling; not a dynamics model."""

    def reset(self, episode):
        rng = random.Random(episode)
        d, b = rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5)
        self.pose = [d * math.sin(b), d * math.cos(b), rng.uniform(-1.0, 1.0)]
        self.vel = [0.0, 0.0, 0.0]
        self.collision = False

    def observation(self):
        return list(self.pose), list(self.vel), self.collision

    def step(self, act, dt):
        vx, vy, w = (a * s for a, s in zip(act, (0.6, 0.6, 2.5)))
        w += 0.3 * vx
        self.vel = [vx, vy, w]
        th = self.pose[2] + w * dt / 2
        self.pose[0] += dt * (math.cos(th) * vx - math.sin(th) * vy)
        self.pose[1] += dt * (math.sin(th) * vx + math.cos(th) * vy)
        self.pose[2] += w * dt
        self.collision = abs(self.pose[0]) > 0.127 - 0.08 and abs(self.pose[1]) < 0.26


class GazeboBridge:
    """Lockstep Gazebo Harmonic bridge with emulated or vision-in-the-loop sensing."""

    def __init__(self, level: str, seed: int, max_time: float, world_sdf: Path,
                 out_dir: Path, axes: str = "rl_sim", sensing: str = "emulated",
                 log_name: str = "policy_gazebo", photometric: bool = False,
                 marker_layout: str = "real", randomize_tilt: bool = False):
        if sensing == "camera_aruco":
            sensing = "camera"
        if sensing == "camera_dr":
            sensing, photometric = "camera", True
        if sensing not in ("emulated", "camera"):
            raise ValueError(f"unsupported sensing mode: {sensing}")
        if sensing == "camera":
            global VisionSensors
            from vision_in_loop import VisionSensors  # noqa: E402 (cv2 only for camera modes)
        if axes not in ("rl_sim", "physical"):
            raise ValueError(f"axes must be rl_sim or physical, got {axes}")
        if not BRIDGE.exists():
            raise SystemExit(f"build {BRIDGE} first (see lockstep/CMakeLists.txt)")
        out_dir.mkdir(parents=True, exist_ok=True)
        if not world_sdf.exists():
            subprocess.run([sys.executable, str(HERE / "generate.py"), "--drone", "--output", str(world_sdf)],
                           check=True, cwd=HERE)
        headless_sdf = out_dir / "world_headless.sdf"
        headless_world(world_sdf, headless_sdf)
        self.gz = Gazebo(headless_sdf, out_dir / f"{log_name}.gz.log")
        self.level = level
        self.L = LEVELS[level]
        self.seed = seed
        self.max_time = max_time
        self.axes = axes
        self.sensing = sensing
        self.photometric = bool(photometric and sensing == "camera")
        self.marker_layout = marker_layout
        self.randomize_tilt = bool(randomize_tilt and sensing == "camera")
        self.pose = (0.0, 0.0, 0.0)
        self.vel = [0.0, 0.0, 0.0]
        self.collision = False
        self.out_of_bounds = False
        self.sensors: Sensors | VisionSensors | None = None
        self.tag: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.tag_valid = False
        self.tag_age = 0.0
        self.imu_w = 0.0
        self.imu_yaw = 0.0

    def reset(self, episode: int):
        rng = np.random.default_rng([self.seed, episode])
        d = rng.uniform(*self.L["dist"])
        b = math.radians(rng.uniform(-self.L["bearing"], self.L["bearing"]))
        yaw0 = math.radians(rng.uniform(-self.L["yaw"], self.L["yaw"]))
        start = (d * math.sin(b), d * math.cos(b), yaw0)
        wx, wy, wyaw = to_world(*start)
        _, x, y, yaw = self.gz.reset_pose(wx, wy, wyaw)
        self.pose = to_drone(x, y, yaw)
        self.vel = [0.0, 0.0, 0.0]
        self.collision = collides(self.pose, GAZEBO_GEOMETRY)
        area = float(self.L.get("area", 1.5))
        self.out_of_bounds = max(abs(self.pose[0]), abs(self.pose[1])) > area
        prev = getattr(self, "sensors", None)
        if prev is not None and self.sensing == "camera":
            self._acc_frames = getattr(self, "_acc_frames", 0) + int(prev.frames)
            self._acc_decodes = getattr(self, "_acc_decodes", 0) + int(prev.decodes)
        if self.sensing == "camera":
            assert VisionSensors is not None
            self.sensors = VisionSensors(self.L, rng, self.max_time,
                                         init_pose=self.pose,
                                         photometric=self.photometric,
                                         layout=self.marker_layout,
                                         randomize_tilt=self.randomize_tilt)
        else:
            self.sensors = Sensors(self.L, rng, self.max_time, init_pose=self.pose)
        z, seen = self.sensors.tag(self.pose)
        self.tag = tuple(float(v) for v in z) if seen else (0.0, 0.0, 0.0)
        self.tag_valid = bool(seen)
        self.tag_age = float(self.sensors.age)
        self.imu_w = self.sensors.gyro(0.0)
        self.imu_yaw = float(self.sensors.imu_yaw)

    def observation(self):
        return list(self.pose), list(self.vel), self.collision

    def sensor_observation(self):
        return (self.tag, self.tag_age, self.tag_valid,
                self.imu_w, self.imu_yaw, Pose2(*self.pose))

    @property
    def vision_stats(self) -> dict:
        acc_f = getattr(self, "_acc_frames", 0)
        acc_d = getattr(self, "_acc_decodes", 0)
        return {"frames": acc_f, "decodes": acc_d,
                "decode_rate": acc_d / max(acc_f, 1)}

    def step(self, act, dt=DT):
        assert self.sensors is not None
        if self.axes == "rl_sim":
            # rl_sim trained with act[0]=+x_drone (left), act[1]=+y_drone (-forward);
            # turn is mirrored (verified: unflipped w spins in place, flipped w parks).
            cmd_phys = (-float(act[1]), float(act[0]), -float(act[2]))
        else:
            cmd_phys = (float(act[0]), float(act[1]), float(act[2]))
        n_phys = max(1, int(round(dt / PHYS)))
        _, x, y, yaw = self.gz.step(duties_for(cmd_phys), n=n_phys)
        new_pose = to_drone(x, y, yaw)
        vx = (new_pose[0] - self.pose[0]) / dt
        vy = (new_pose[1] - self.pose[1]) / dt
        dyaw = math.atan2(math.sin(new_pose[2] - self.pose[2]),
                          math.cos(new_pose[2] - self.pose[2]))
        w_true = dyaw / dt
        self.pose = new_pose
        self.vel = [vx, vy, w_true]
        self.collision = collides(self.pose, GAZEBO_GEOMETRY)
        area = float(self.L.get("area", 1.5))
        self.out_of_bounds = max(abs(self.pose[0]), abs(self.pose[1])) > area
        z, seen = self.sensors.tag(self.pose)
        self.tag = tuple(float(v) for v in z) if seen else (0.0, 0.0, 0.0)
        self.tag_valid = bool(seen)
        self.tag_age = float(self.sensors.age)
        self.imu_w = self.sensors.gyro(w_true)
        self.imu_yaw = float(self.sensors.imu_yaw)

    def close(self):
        self.gz.close()


def pcontrol_step(tag, age, valid, imu_w, imu_yaw, prev, est):
    """Plumbing smoke controller; bypasses the student."""
    return np.clip([-est.x * 2.0, -est.z * 2.0, -est.yaw * 3.0], -1, 1), "smoke"


def baseline_controller(latency, axes="rl_sim", geometry=None):
    """EKF + staged baseline behind the harness step signature. Gets the tag, its valid
    flag and the gyro rate only; `latency` is the tag capture delay in seconds."""
    ctl = StagedBaseline(1, latency=latency, dt=DT, model=MeanModel(axes=axes),
                         geometry=geometry if geometry is not None else Geometry())
    names = ("search", "approach", "enter", "hold", "back")

    def step(tag, age, valid, imu_w, imu_yaw, prev, est):
        act = ctl.step(tag, valid, imu_w)[0]
        return act, names[int(ctl.phase[0])]

    def reset(ep_latency=None):
        if ep_latency is not None:
            ctl.latency = float(ep_latency)
        ctl.reset()

    return step, reset


def run_episodes(step_fn, reset_fn, bridge, episodes, dt=DT, max_time=45.0):
    scorer = Scorer(max_time)
    results = []
    accepts_lat = len(inspect.signature(reset_fn).parameters) >= 1
    has_sensor_obs = hasattr(bridge, "sensor_observation")
    for ep in range(episodes):
        bridge.reset(ep)
        scorer.reset()
        if accepts_lat and getattr(bridge, "sensors", None) is not None:
            reset_fn(bridge.sensors.latency)
        else:
            reset_fn()
        prev = np.zeros(3)
        while True:
            pose, vel, collision = bridge.observation()
            res = scorer.update(*pose, *vel, collision, dt)
            if res is None and getattr(bridge, "out_of_bounds", False):
                res = EpisodeResult(False, False, True, scorer.t,
                                    math.hypot(pose[0], pose[1]), math.degrees(abs(pose[2])))
            if res is not None:
                results.append(res)
                break
            if has_sensor_obs:
                tag, age, valid, imu_w, imu_yaw, est = bridge.sensor_observation()
                act, _ = step_fn(tag, age, valid, imu_w, imu_yaw, prev, est)
            else:
                act, _ = step_fn(tuple(pose), dt, True, vel[2], pose[2], prev, Pose2(*pose))
            prev = np.asarray(act, dtype=float)
            bridge.step(act, dt)
    return results


def summarize(results):
    n = len(results)
    ts = sorted(float(r.time_s) for r in results)
    s, c, t = (int(sum(bool(getattr(r, k)) for r in results))
               for k in ("success", "collision", "timeout"))
    return {"episodes": n, "success": s, "success_rate": s / max(n, 1),
            "collisions": c, "collision_rate": c / max(n, 1), "timeouts": t,
            "time_median": ts[n // 2] if n else float("nan"),
            "time_p95": ts[min(int(n * 0.95), n - 1)] if n else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help=".npz student from rl/export.py (required unless --smoke or --baseline)")
    ap.add_argument("--smoke", action="store_true", help="P-controller instead of the student")
    ap.add_argument("--baseline", action="store_true", help="EKF + staged baseline instead of the student")
    ap.add_argument("--latency", type=float, default=0.0, help="tag capture delay for --baseline on fake bridge (s)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--bridge", choices=["fake", "gazebo"], default="fake")
    ap.add_argument("--level", choices=list(LEVELS), default="L1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sensing", choices=["emulated", "camera", "camera_aruco", "camera_dr"],
                    default="emulated",
                    help="emulated: rl/sim.py sensor model; camera: synthetic 3D render + "
                         "real aruco.py detect_gray + PnP; camera_aruco: alias of camera; "
                         "camera_dr: camera + photometric DR")
    ap.add_argument("--lighting-dr", action="store_true",
                    help="per-episode photometric DR (brightness/contrast/gamma, shadow "
                         "gradient, glare, sensor noise, blur) on --sensing camera frames")
    ap.add_argument("--marker-layout", choices=["real", "legacy"], default="real",
                    help="ArUco constellation layout for --sensing camera (real=14/6/3, legacy=12/9/13/14)")
    ap.add_argument("--card-tilt-dr", action="store_true",
                    help="per-episode 10-25 deg card roll/pitch randomization on --sensing camera frames")
    ap.add_argument("--world", type=Path, default=REPO / "rl/runs/gazebo_l6/world_generated.sdf")
    ap.add_argument("--out", type=Path, default=REPO / "rl/runs/gazebo_l6")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-time", type=float, default=45.0)
    args = ap.parse_args()

    is_gz = args.bridge == "gazebo"
    if args.smoke:
        step_fn, reset_fn = pcontrol_step, lambda: None
        axes = "rl_sim"
        meta = {"controller": "pcontrol-smoke"}
    elif args.baseline:
        if is_gz:
            step_fn, reset_fn = baseline_controller(args.latency, axes="physical", geometry=GAZEBO_GEOMETRY)
            axes = "physical"
        else:
            step_fn, reset_fn = baseline_controller(args.latency)
            axes = "rl_sim"
        meta = {"controller": "ekf_staged_baseline"}
    elif args.weights:
        if is_gz:
            policy = StudentPolicy.load(args.weights, vehicle=GAZEBO_VEHICLE, gear=GAZEBO_GEAR)
        else:
            policy = StudentPolicy.load(args.weights)
        step_fn, reset_fn = policy.step, policy.reset
        axes = "rl_sim"
        meta = {"controller": "student", "weights": args.weights, "sha": policy.sha}
    else:
        ap.error("--weights is required unless --smoke or --baseline")

    if is_gz:
        log_tag = f"policy_gazebo_{meta['controller']}_{args.level}"
        bridge = GazeboBridge(args.level, args.seed, args.max_time, args.world, args.out,
                              axes=axes, sensing=args.sensing, log_name=log_tag,
                              photometric=args.lighting_dr,
                              marker_layout=args.marker_layout,
                              randomize_tilt=args.card_tilt_dr)
        meta.update(level=args.level, seed=args.seed, sensing=args.sensing,
                    photometric_dr=bool(args.lighting_dr and args.sensing.startswith("camera")))
    else:
        bridge = FakeBridge()

    try:
        summary = summarize(run_episodes(step_fn, reset_fn, bridge, args.episodes,
                                         max_time=args.max_time))
    finally:
        if hasattr(bridge, "close"):
            bridge.close()

    if is_gz and args.sensing.startswith("camera"):
        prev = getattr(bridge, "sensors", None)
        if prev is not None:
            bridge._acc_frames = getattr(bridge, "_acc_frames", 0) + int(prev.frames)
            bridge._acc_decodes = getattr(bridge, "_acc_decodes", 0) + int(prev.decodes)
        summary["vision"] = bridge.vision_stats

    summary.update(meta, bridge=args.bridge)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
