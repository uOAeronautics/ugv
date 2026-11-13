"""Benchmark the non-privileged classical baseline (EKF + staged, ugv.staged_baseline).

The controller sees only what the student sees: the delayed noisy tag pose + valid flag,
the IMU yaw rate and its own commands. The per-episode tag latency is passed as known
(camera timestamps). It never reads env.pose / env.vel / the dynamics draw. Metrics and
JSON match evaluate.py.

Usage: uv run python baselines.py --level L1 --episodes 1000 --seed 11 --json runs/baseline_L1.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from sim import MecanumVecEnv, EnvConfig
from evaluate import wilson

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ugv.staged_baseline import StagedBaseline, MeanModel  # noqa: E402  (numpy-only deploy module)


def run(episodes, level, device, seed=0, use_mean_dynamics=True):
    env = MecanumVecEnv(min(episodes, 512), EnvConfig(level=level, device=device, seed=seed,
                                                      use_mean_dynamics=use_mean_dynamics))
    model = MeanModel()
    max_steps = int(env.cfg.max_time / env.cfg.dt) + 5
    succ = coll = tout = 0
    times, epos, eyaw, jerk = [], [], [], []
    left = episodes
    while left > 0:
        n = env.n
        env.reset(torch.arange(n))
        ctl = StagedBaseline(n, latency=env.latency.cpu().numpy(), dt=env.dt, model=model)
        cmd = np.zeros((n, 3))
        prev_a = np.zeros((n, 3))
        done = np.zeros(n, bool)
        t = np.zeros(n)
        j = np.zeros(n)
        for _ in range(max_steps):
            obs, _, d, info = env.step(torch.as_tensor(cmd, dtype=torch.float32, device=env.device))
            cmd = ctl.step(obs["tag"].cpu().numpy(), obs["valid"].cpu().numpy(), env.imu_w.cpu().numpy())
            j += ((cmd - prev_a) ** 2).sum(1)
            prev_a = cmd
            t = np.where(~done, t + env.dt, t)
            done |= d.cpu().numpy()
            if done.all():
                break
        k = min(left, n)
        ok = info["success"][:k].cpu().numpy()
        succ += int((done[:k] & ok).sum())
        coll += int(info["collision"][:k].sum())
        tout += int((info["timeout"][:k].cpu().numpy() & ~ok).sum())
        times += t[:k].tolist()
        epos += env.pose[:k, :2].norm(dim=1).tolist()
        eyaw += env.pose[:k, 2].abs().tolist()
        jerk += (j[:k] / np.maximum(t[:k], 0.02)).tolist()
        left -= k
    lo, hi = wilson(succ, episodes)
    times.sort(); epos.sort(); eyaw.sort()
    q = lambda v, p: v[min(int(len(v) * p), len(v) - 1)] if v else float("nan")
    out = {"policy": "ekf_staged", "level": level, "episodes": episodes, "seed": seed,
           "success": succ, "success_lb": lo, "success_hi": hi,
           "collisions": coll, "collision_rate": coll / episodes,
           "false_parked": None, "false_parked_rate": None,
           "timeouts": tout, "timeout_rate": tout / episodes,
           "time_median": q(times, 0.5), "time_p95": q(times, 0.95),
           "pos_p95": q(epos, 0.95), "yaw_p95_deg": math.degrees(q(eyaw, 0.95)),
           "jerk_mean": sum(jerk) / len(jerk) if jerk else float("nan")}
    print(f"ekf_staged@{level}: success {succ}/{episodes} lb {lo:.3f} hi {hi:.3f} coll {coll} "
          f"timeout {tout} med {out['time_median']:.1f}s p95 {out['time_p95']:.1f}s "
          f"pos_p95 {out['pos_p95']*1000:.0f}mm yaw_p95 {out['yaw_p95_deg']:.1f}deg", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-mean-dynamics", action="store_true",
                    help="use particles[0] instead of mean_coef on L0/L1")
    a = ap.parse_args()
    out = run(a.episodes, a.level, "cuda" if torch.cuda.is_available() else "cpu", a.seed,
              use_mean_dynamics=not a.no_mean_dynamics)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
