"""H-0002 runner: non-privileged classical controller (stage P go-to + dead-reckoned finale).

Same loop, metrics and JSON shape as rl/baselines.py, with --controller stage_goto. The
controller sees only the delayed tag pose + valid flag, env.imu_w and its own commands. The
truth pose is read here, for metrics only (distance at finale entry), never by the controller.

Usage (CPU only, no GPU job):
  cd rl && CUDA_VISIBLE_DEVICES="" uv run python exp_classical_l1.py --level L1 \
      --episodes 1000 --seed 11 --no-mean-dynamics --json runs/H-0002_stage_goto.json
  cd rl && CUDA_VISIBLE_DEVICES="" uv run python exp_classical_l1.py --probe-axes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from classical_goto import ControllerConfig, StageGotoController
from evaluate import wilson
from sim import EnvConfig, MecanumVecEnv


def probe_axes(seed=0):
    """Which action channel drives forward / left / yaw in the sim (privileged, probe only)."""
    env = MecanumVecEnv(4, EnvConfig(level="L1", device="cpu", seed=seed, use_mean_dynamics=False))
    res = {}
    for ch in range(3):
        env.reset(torch.arange(env.n))
        p0 = env.pose.clone().numpy()
        cmd = np.zeros((env.n, 3))
        cmd[:, ch] = 0.5
        for _ in range(25):  # 0.5 s
            env.step(torch.as_tensor(cmd, dtype=torch.float32))
        p1 = env.pose.numpy()
        yaw = p0[:, 2]
        fwd = np.stack([np.sin(yaw), -np.cos(yaw)], 1)  # GoTo forward unit
        left = np.stack([np.cos(yaw), np.sin(yaw)], 1)  # GoTo left unit
        d = p1[:, :2] - p0[:, :2]
        res[f"ch{ch}"] = {
            "d_fwd_m": float((d * fwd).sum(1).mean()),
            "d_left_m": float((d * left).sum(1).mean()),
            "d_yaw_rad": float(np.mean(np.arctan2(np.sin(p1[:, 2] - yaw), np.cos(p1[:, 2] - yaw)))),
        }
    return res


def run(episodes, level, seed=11, use_mean_dynamics=False, cfg=None):
    env = MecanumVecEnv(
        min(episodes, 512),
        EnvConfig(level=level, device="cpu", seed=seed, use_mean_dynamics=use_mean_dynamics),
    )
    max_steps = int(env.cfg.max_time / env.cfg.dt) + 5
    succ = coll = tout = 0
    times, dr_dist, dr_fail, dr_n = [], [], 0, 0
    left = episodes
    while left > 0:
        n = env.n
        env.reset(torch.arange(n))
        ctl = StageGotoController(n, dt=env.dt, cfg=cfg or ControllerConfig())
        cmd = np.zeros((n, 3))
        done = np.zeros(n, bool)
        t = np.zeros(n)
        entered = np.zeros(n, bool)
        entry_dist = np.full(n, np.nan)
        for _ in range(max_steps):
            obs, _, d, info = env.step(torch.as_tensor(cmd, dtype=torch.float32))
            cmd = ctl.step(
                obs["tag"].cpu().numpy(), obs["valid"].cpu().numpy(), env.imu_w.cpu().numpy()
            )
            new = ctl.dr_entered & ~entered
            if new.any():
                truth = env.pose[:, :2].norm(dim=1).cpu().numpy()
                entry_dist[new] = truth[new]
                entered |= new
            t = np.where(~done, t + env.dt, t)
            done |= d.cpu().numpy()
            if done.all():
                break
        k = min(left, n)
        ok = info["success"][:k].cpu().numpy()
        succ += int((done[:k] & ok).sum())
        coll += int(info["collision"][:k].sum())
        tout += int((info["timeout"][:k].cpu().numpy() & ~ok).sum())
        times += t[:k][ok].tolist()
        e = entered[:k]
        dr_n += int(e.sum())
        dr_dist += entry_dist[:k][e].tolist()
        dr_fail += int((e & ~ok).sum())
        left -= k
    lo, hi = wilson(succ, episodes)
    times.sort()
    dr_dist.sort()

    def q(v, p):
        return v[min(int(len(v) * p), len(v) - 1)] if v else float("nan")

    return {
        "policy": "stage_goto",
        "level": level,
        "episodes": episodes,
        "seed": seed,
        "use_mean_dynamics": use_mean_dynamics,
        "device": "cpu",
        "success": succ,
        "success_lb": lo,
        "success_hi": hi,
        "collisions": coll,
        "collision_rate": coll / episodes,
        "timeouts": tout,
        "timeout_rate": tout / episodes,
        "time_median_s": q(times, 0.5),
        "time_p95_s": q(times, 0.95),
        "finale_entered": dr_n,
        "finale_entry_dist_median_m": q(dr_dist, 0.5),
        "finale_entry_dist_max_m": dr_dist[-1] if dr_dist else float("nan"),
        "finale_failures": dr_fail,
        "finale_failure_rate": dr_fail / dr_n if dr_n else float("nan"),
        "controller": asdict(cfg or ControllerConfig()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", default=None)
    ap.add_argument("--controller", default="stage_goto", choices=["stage_goto"])
    ap.add_argument("--no-mean-dynamics", action="store_true", help="p0 plant (gate setting)")
    ap.add_argument("--probe-axes", action="store_true", help="print sim channel directions")
    a = ap.parse_args()
    if a.probe_axes:
        print(json.dumps(probe_axes(a.seed), indent=2))
        return
    out = run(a.episodes, a.level, a.seed, use_mean_dynamics=not a.no_mean_dynamics)
    line = (
        f"stage_goto@{a.level}: success {out['success']}/{a.episodes} lb {out['success_lb']:.3f} "
        f"coll {out['collisions']} timeout {out['timeouts']} med {out['time_median_s']:.1f}s "
        f"finale {out['finale_entered']} fail {out['finale_failures']}"
    )
    print(line, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    sys.exit(main())
