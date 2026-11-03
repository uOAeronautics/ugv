"""L6 sim-to-sim harness for the exported NumPy student (docs/RL_PARKING.md).

--bridge fake: kinematic point-mass episodes that exercise the episode loop,
parked scoring and metrics with the real StudentPolicy. Runs anywhere; it
validates plumbing, not policy quality.

--bridge gazebo: not implemented. Needs a Gazebo Harmonic host, the
ugv_tt_motor plugin (build_motor_plugin.py), a per-episode reset, the
estimator pose stream and a velocity-command seam.

Step rate is 50 Hz: the student history (50 frames = 1 s) and its tag-velocity
estimator are trained at dt=0.02 (rl/sim.py).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ugv.parking import Pose2  # noqa: E402
from ugv.policy import StudentPolicy  # noqa: E402

DT = 0.02
LAT_TOL, LONG_TOL, YAW_TOL = 0.03, 0.05, math.radians(5)
STILL_V, STILL_W, HOLD_S = 0.01, math.radians(2), 1.0


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


def pcontrol_step(tag, age, valid, imu_w, imu_yaw, prev, est):
    """Plumbing smoke controller; bypasses the student."""
    return np.clip([-est.x * 2.0, -est.z * 2.0, -est.yaw * 3.0], -1, 1), "smoke"


def run_episodes(step_fn, reset_fn, bridge, episodes, dt=DT, max_time=45.0):
    scorer = Scorer(max_time)
    results = []
    for ep in range(episodes):
        bridge.reset(ep)
        scorer.reset()
        reset_fn()
        prev = np.zeros(3)
        while True:
            pose, vel, collision = bridge.observation()
            res = scorer.update(*pose, *vel, collision, dt)
            if res is not None:
                results.append(res)
                break
            act, _ = step_fn(tuple(pose), dt, True, vel[2], pose[2], prev, Pose2(*pose))
            prev = act
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
    ap.add_argument("--weights", help=".npz student from rl/export.py (required unless --smoke)")
    ap.add_argument("--smoke", action="store_true", help="P-controller instead of the student")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--bridge", choices=["fake", "gazebo"], default="fake")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-time", type=float, default=45.0)
    args = ap.parse_args()
    if args.bridge == "gazebo":
        raise SystemExit("live Gazebo bridge not implemented; see module docstring")
    if args.smoke:
        step_fn, reset_fn, meta = pcontrol_step, lambda: None, {"controller": "pcontrol-smoke"}
    elif args.weights:
        policy = StudentPolicy.load(args.weights)
        step_fn, reset_fn = policy.step, policy.reset
        meta = {"controller": "student", "weights": args.weights, "sha": policy.sha}
    else:
        ap.error("--weights is required unless --smoke")
    summary = summarize(run_episodes(step_fn, reset_fn, FakeBridge(), args.episodes,
                                     max_time=args.max_time))
    summary.update(meta, bridge=args.bridge)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
