"""Seeded benchmark per docs/RL_PARKING.md. Baseline + checkpoints, Wilson bounds, JSON."""
from __future__ import annotations
import argparse
import json
import math
import torch

from sim import MecanumVecEnv, EnvConfig
from models import Student, load_teacher, teacher_obs

BASELINES = ("random", "brake", "classical")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load_policy(spec: str, device):
    if spec in BASELINES:
        return spec, None
    ckpt = torch.load(spec, map_location=device)
    if "student" in ckpt:
        m = Student().to(device); m.load_state_dict(ckpt["student"]); m.eval()
        return "student", m
    return "teacher", load_teacher(spec, device)[0]


@torch.no_grad()
def policy_act(kind, model, obs, n, device):
    if kind == "brake":
        return torch.zeros(n, 3, device=device)
    if kind == "classical":
        # Classical baseline: staged PD + integral crab on true pose.
        # Documented result: 0% success, ~100% skid contact even on L0.
        # The turn command shoves x (-0.83 m/s per unit) and vx commands yaw
        # (+0.64 rad/s per unit), defeating decentralized PID. This measured
        # failure is the motivation for learned coordinated control.
        pose, vel = obs["pose"], obs["vel"]
        dist = pose[:, :2].norm(dim=1)
        stage = dist > 0.5
        w1 = (-pose[:, 2] * 4.0 - vel[:, 2] * 0.6).clamp(-1, 1)
        v1 = torch.stack([-pose[:, 0] * 0.6 - vel[:, 0] * 1.0,
                          -pose[:, 1] * 0.6 - vel[:, 1] * 1.0, w1], dim=1)
        w2 = (-pose[:, 2] * 4.0 - vel[:, 2] * 0.6).clamp(-1, 1)
        v2 = torch.stack([-pose[:, 0] * 0.8 - vel[:, 0] * 1.2,
                          -pose[:, 1] * 0.8 - vel[:, 1] * 1.2, w2], dim=1)
        a = torch.where(stage[:, None].expand_as(v1), v1, v2)
        return a.clamp(-0.5, 0.5)
    if kind == "random":
        return torch.empty(n, 3, device=device).uniform_(-0.5, 0.5)
    if kind == "teacher":
        return model(teacher_obs(obs)).clamp(-1, 1)
    return model(obs["student"]).clamp(-1, 1)  # student


def run(episodes, level, policy, device, seed=0):
    env = MecanumVecEnv(min(episodes, 512), EnvConfig(level=level, device=device, seed=seed))
    kind, model = load_policy(policy, env.device)
    succ = coll = tout = 0
    times, epos, eyaw, jerk = [], [], [], []
    left = episodes
    while left > 0:
        n = min(left, env.n)
        obs = env.reset(torch.arange(n))  # persistent stream: same seed, same JSON
        prev_a = torch.zeros(n, 3, device=env.device)
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        t = torch.zeros(n, device=env.device)
        j = torch.zeros(n, device=env.device)
        steps = 0
        while not bool(done.all()):
            a = policy_act(kind, model, {"pose": obs["pose"][:n], "vel": obs["vel"][:n],
                                         "student": obs["student"][:n]}, n, env.device)
            full = torch.zeros(env.n, 3, device=env.device)
            full[:n] = a
            obs_full, _, d, info = env.step(full)
            obs = {"pose": obs_full["pose"][:n], "vel": obs_full["vel"][:n],
                   "student": obs_full["student"][:n]}
            j += (a - prev_a).pow(2).sum(1)
            prev_a = a
            t = torch.where(~done, t + env.cfg.dt, t)
            done = done | d[:n].to(env.device)
            steps += 1
            if steps > 5000:
                break
        ok = info["success"][:n] if info["success"].shape[0] == env.n else info["success"]
        succ += int((done & ok[:n]).sum())
        coll += int(info["collision"][:n].sum())
        tout += int((info["timeout"][:n] & ~ok[:n]).sum())
        times += t.tolist()
        epos += obs["pose"][:, :2].norm(dim=1).tolist()
        eyaw += obs["pose"][:, 2].abs().tolist()
        jerk += (j / t.clamp_min(0.02)).tolist()
        left -= n
    lo, hi = wilson(succ, episodes)
    times.sort(); epos.sort(); eyaw.sort()
    q = lambda v, p: v[min(int(len(v) * p), len(v) - 1)] if v else float("nan")
    out = {"policy": policy, "level": level, "episodes": episodes,
           "success": succ, "success_lb": lo, "success_hi": hi,
           "collisions": coll, "collision_rate": coll / episodes,
           "false_parked": None, "false_parked_rate": None,  # not modelled yet
           "timeouts": tout, "timeout_rate": tout / episodes,
           "time_median": q(times, 0.5), "time_p95": q(times, 0.95),
           "pos_p95": q(epos, 0.95), "yaw_p95_deg": math.degrees(q(eyaw, 0.95)),
           "jerk_mean": sum(jerk) / len(jerk) if jerk else float("nan")}
    print(f"{policy}@{level}: success {succ}/{episodes} lb {lo:.3f} hi {hi:.3f} "
          f"coll {coll} timeout {tout} med {out['time_median']:.1f}s p95 {out['time_p95']:.1f}s "
          f"pos_p95 {out['pos_p95']*1000:.0f}mm yaw_p95 {out['yaw_p95_deg']:.1f}deg")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--policy", default="random",
                    help=f"checkpoint path or one of {BASELINES}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = run(args.episodes, args.level, args.policy,
              "cuda" if torch.cuda.is_available() else "cpu", args.seed)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
