"""Closed-loop EKF + staged baseline in Gazebo (drone docking world, native mecanum rollers).

Gazebo provides the physics: passive-roller mecanum car with the fitted native motor
profile, the photo-model drone, 1 ms steps. The controller runs in lockstep at 50 Hz through
lockstep/lockstep_bridge (gz-transport, no Python bindings needed).

Sensing is EMULATED from the Gazebo car pose with the rl/sim.py sensor model (tag noise,
latency, FOV, drops, under-deck loss; gyro white noise + bias) by default; --sensing camera
renders the 3D ArUco constellation at the true pose and decodes it with the real
src/ugv/aruco.py detect_gray + PnP (see vision_in_loop.py), with optional --lighting-dr
photometric domain randomization. Scoring follows docs/RL_PARKING.md with the Gazebo drone
geometry.

Run (from the repo root): uv run python simulation/gazebo/run_baseline_gazebo.py --level L1 --episodes 20
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from lockstep_env import (  # noqa: E402
    BRIDGE,
    DT,
    Gazebo,
    LEVELS,
    Sensors,
    collides,
    duties_for,
    headless_world,
    to_drone,
    to_world,
)
from ugv.staged_baseline import GAZEBO_GEOMETRY, MeanModel, StagedBaseline  # noqa: E402


def episode(gz: Gazebo, level: str, ep: int, seed: int, max_time: float, trace=None,
            sensing: str = "emulated", photometric: bool = False, stats=None):
    rng = np.random.default_rng([seed, ep])
    L = LEVELS[level]
    d = rng.uniform(*L["dist"])
    b = math.radians(rng.uniform(-L["bearing"], L["bearing"]))
    yaw0 = math.radians(rng.uniform(-L["yaw"], L["yaw"]))
    start = (d * math.sin(b), d * math.cos(b), yaw0)
    wx, wy, wyaw = to_world(*start)
    t, x, y, yaw = gz.reset_pose(wx, wy, wyaw)
    if sensing == "camera":
        from vision_in_loop import VisionSensors
        sensors = VisionSensors(L, rng, max_time, photometric=photometric)
    else:
        sensors = Sensors(L, rng, max_time)
    ctl = StagedBaseline(1, latency=sensors.latency, dt=DT, model=MeanModel(axes="physical"),
                         geometry=GAZEBO_GEOMETRY)
    prev = to_drone(x, y, yaw)
    still = 0.0
    cmd = np.zeros(3)
    out = dict(start=start)
    area = L.get("area", 1.5)
    for k in range(int(max_time / DT)):
        t, x, y, yaw = gz.step(duties_for(cmd))
        pose = to_drone(x, y, yaw)
        dyaw = math.atan2(math.sin(pose[2] - prev[2]), math.cos(pose[2] - prev[2]))
        vxy = math.hypot(pose[0] - prev[0], pose[1] - prev[1]) / DT
        w_true = dyaw / DT
        prev = pose
        if trace is not None:
            trace.append([ep, k * DT, *pose, *cmd.tolist(), int(ctl.phase[0])])
        if collides(pose):
            out.update(result="collision", t=k * DT, pose=pose)
            return out
        if max(abs(pose[0]), abs(pose[1])) > area:
            out.update(result="out_of_bounds", t=k * DT, pose=pose)
            return out
        near = abs(pose[0]) <= 0.03 and abs(pose[1]) <= 0.05 and abs(pose[2]) <= math.radians(5)
        still = still + DT if (near and vxy < 0.01 and abs(w_true) < math.radians(2)) else 0.0
        if still >= 1.0:
            out.update(result="success", t=k * DT, pose=pose)
            return out
        z, seen = sensors.tag(pose)
        cmd = ctl.step(z, seen, sensors.gyro(w_true))[0]
    out.update(result="timeout", t=max_time, pose=pose, phase=int(ctl.phase[0]))
    if stats is not None and sensing == "camera":
        stats["frames"] += int(sensors.frames)
        stats["decodes"] += int(sensors.decodes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS), default="L1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-time", type=float, default=45.0)
    ap.add_argument("--world", type=Path, default=REPO / "rl/runs/gazebo_l6/world_generated.sdf",
                    help="output of generate.py --drone (regenerated if missing)")
    ap.add_argument("--out", type=Path, default=REPO / "rl/runs/gazebo_l6")
    ap.add_argument("--sensing", choices=["emulated", "camera", "camera_aruco", "camera_dr"],
                    default="emulated")
    ap.add_argument("--lighting-dr", action="store_true",
                    help="per-episode photometric DR on --sensing camera frames")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not BRIDGE.exists():
        raise SystemExit(f"build {BRIDGE} first (see lockstep/CMakeLists.txt)")
    if not a.world.exists():
        subprocess.run([sys.executable, str(HERE / "generate.py"), "--drone", "--output", str(a.world)],
                       check=True, cwd=HERE)
    world = a.out / "world_headless.sdf"
    headless_world(a.world, world)
    name = f"baseline_gazebo_{a.level}"
    if a.sensing == "camera_aruco":
        a.sensing = "camera"
    photometric = bool(a.lighting_dr or a.sensing == "camera_dr")
    sensing = "camera" if a.sensing == "camera_dr" else a.sensing
    if sensing == "camera":
        name += "_camdr" if photometric else "_cam"
    gz = Gazebo(world, a.out / f"{name}.gz.log")
    results, trace = [], []
    stats = {"frames": 0, "decodes": 0}
    wall = time.monotonic()
    try:
        for ep in range(a.episodes):
            r = episode(gz, a.level, ep, a.seed, a.max_time, trace,
                        sensing=sensing, photometric=photometric, stats=stats)
            results.append(r)
            print(json.dumps(dict(ep=ep, **{k: (round(v, 3) if isinstance(v, float) else
                                               [round(u, 3) for u in v] if isinstance(v, (tuple, list)) else v)
                                            for k, v in r.items()})), flush=True)
    finally:
        gz.close()
    n = len(results)
    count = lambda k: sum(r["result"] == k for r in results)
    ts = sorted(r["t"] for r in results)
    summary = dict(policy="ekf_staged", engine="gazebo-harmonic native mecanum rollers",
                   sensing=("vision-in-the-loop (synthetic 3D render + aruco.py detect_gray/PnP"
                            + (", photometric DR" if photometric else "")) if sensing == "camera"
                           else "emulated from Gazebo pose (rl/sim.py model)", level=a.level,
                   episodes=n, seed=a.seed, success=count("success"), collisions=count("collision"),
                   timeouts=count("timeout"), out_of_bounds=count("out_of_bounds"),
                   time_median=ts[n // 2] if n else None, wall_s=round(time.monotonic() - wall, 1))
    if sensing == "camera":
        summary["vision"] = dict(stats, decode_rate=stats["decodes"] / max(stats["frames"], 1))
    print(json.dumps(summary, indent=2))
    (a.out / f"{name}.json").write_text(json.dumps(dict(summary=summary, episodes=results), indent=2))
    np.save(a.out / f"{name}_trace.npy", np.array(trace))


if __name__ == "__main__":
    main()
