"""Seeded benchmark per docs/RL_PARKING.md. Baseline + checkpoints, Wilson bounds, JSON."""

from __future__ import annotations

import argparse
import json
import math

import torch
from models import load_student, load_teacher, teacher_obs
from sim import EnvConfig, MecanumVecEnv

BASELINES = ("random", "brake", "oracle_pd")


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
        return spec, None, {"act_dim": 3, "stateful": False}
    ckpt = torch.load(spec, map_location=device)
    if "student" in ckpt:
        m, ckpt = load_student(spec, device)
        act_dim = ckpt.get("act_dim", 3)
        return (
            "student",
            m,
            {
                "act_dim": act_dim,
                "stateful": bool(ckpt.get("stateful", False)),
                "hidden": ckpt.get("hidden", 64),
            },
        )
    m, ck = load_teacher(spec, device)
    return "teacher", m, {"act_dim": ck.get("act_dim", 3), "stateful": False}


@torch.no_grad()
def policy_act(kind, model, obs, n, device, h_state=None):
    if kind == "brake":
        return torch.zeros(n, 3, device=device), h_state
    if kind == "oracle_pd":
        # Diagnostic only, not the baseline: staged PD crab on the TRUE pose and velocity
        # (privileged). Separates control failures from sensing failures. The baseline the
        # RL gates compare against is the non-privileged EKF + staged (baselines.py).
        pose, vel = obs["pose"], obs["vel"]
        dist = pose[:, :2].norm(dim=1)
        stage = dist > 0.5
        w1 = (-pose[:, 2] * 4.0 - vel[:, 2] * 0.6).clamp(-1, 1)
        v1 = torch.stack(
            [-pose[:, 0] * 0.6 - vel[:, 0] * 1.0, -pose[:, 1] * 0.6 - vel[:, 1] * 1.0, w1], dim=1
        )
        w2 = (-pose[:, 2] * 4.0 - vel[:, 2] * 0.6).clamp(-1, 1)
        v2 = torch.stack(
            [-pose[:, 0] * 0.8 - vel[:, 0] * 1.2, -pose[:, 1] * 0.8 - vel[:, 1] * 1.2, w2], dim=1
        )
        a = torch.where(stage[:, None].expand_as(v1), v1, v2)
        return a.clamp(-0.5, 0.5), h_state
    if kind == "random":
        return torch.empty(n, 3, device=device).uniform_(-0.5, 0.5), h_state
    if kind == "teacher":
        return model(teacher_obs(obs)).clamp(-1, 1), h_state
    if h_state is not None:
        out, h_next = model.forward_seq(obs["student"][:, -1:, :], h_state)
        return out[:, 0].clamp(-1, 1), h_next
    return model(obs["student"]).clamp(-1, 1), h_state  # student


def run(
    episodes,
    level,
    policy,
    device,
    seed=0,
    declare_parked=False,
    use_mean_dynamics=True,
    cam_hz=None,
    fov_search=False,
):
    env = MecanumVecEnv(
        min(episodes, 512),
        EnvConfig(
            level=level,
            device=device,
            seed=seed,
            declare_parked=declare_parked,
            use_mean_dynamics=use_mean_dynamics,
            cam_hz=cam_hz,
        ),
    )
    kind, model, meta = load_policy(policy, env.device)
    act_dim = meta["act_dim"]
    stateful = meta["stateful"]
    succ = coll = false_p = tout = 0
    times, epos, eyaw, jerk = [], [], [], []
    left = episodes
    while left > 0:
        n = min(left, env.n)
        obs = env.reset(torch.arange(n))  # persistent stream: same seed, same JSON
        h_state = torch.zeros(1, n, meta.get("hidden", 64), device=env.device) if stateful else None
        prev_a = torch.zeros(n, 3, device=env.device)
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        t = torch.zeros(n, device=env.device)
        j = torch.zeros(n, device=env.device)
        steps = 0
        while not bool(done.all()):
            a, h_state = policy_act(
                kind,
                model,
                {"pose": obs["pose"][:n], "vel": obs["vel"][:n], "student": obs["student"][:n]},
                n,
                env.device,
                h_state,
            )
            if fov_search and kind == "student":
                # deploy blind-search reflex (mirrors StudentPolicy.step): after
                # 1 s without a tag, spin toward imu_yaw = 0, zero translation.
                h = obs["student"][:n]
                blind = h[:, -1, 3] >= 1.0
                if bool(blind.any()):
                    u_turn = (-1.8 * h[:, -1, 6] - 0.3 * h[:, -1, 5]).clamp(-0.6, 0.6)
                    z = torch.zeros_like(u_turn)
                    a[blind] = torch.stack([z[blind], z[blind], u_turn[blind]], dim=1)[
                        :, : a.shape[1]
                    ]
            full = torch.zeros(env.n, act_dim, device=env.device)
            full[:n] = a
            obs_full, _, d, info = env.step(full)
            obs = {
                "pose": obs_full["pose"][:n],
                "vel": obs_full["vel"][:n],
                "student": obs_full["student"][:n],
            }
            j += (a[:, :3] - prev_a).pow(2).sum(1)
            prev_a = a[:, :3]
            t = torch.where(~done, t + env.cfg.dt, t)
            done = done | d[:n].to(env.device)
            steps += 1
            if steps > 5000:
                break
        ok = info["success"][:n] if info["success"].shape[0] == env.n else info["success"]
        co = info["collision"][:n]
        fp = info["false_park"][:n] & ~ok & ~co
        to = info["timeout"][:n] & ~ok & ~co & ~fp
        succ += int((done & ok).sum())
        coll += int(co.sum())
        false_p += int(fp.sum())
        tout += int(to.sum())
        times += t.tolist()
        epos += obs["pose"][:, :2].norm(dim=1).tolist()
        eyaw += obs["pose"][:, 2].abs().tolist()
        jerk += (j / t.clamp_min(0.02)).tolist()
        left -= n
    lo, hi = wilson(succ, episodes)
    times.sort()
    epos.sort()
    eyaw.sort()

    def q(v, p):
        return v[min(int(len(v) * p), len(v) - 1)] if v else float("nan")

    out = {
        "policy": policy,
        "level": level,
        "episodes": episodes,
        "use_mean_dynamics": use_mean_dynamics,
        "fov_search": fov_search,
        "success": succ,
        "success_lb": lo,
        "success_hi": hi,
        "collisions": coll,
        "collision_rate": coll / episodes,
        "false_parked": false_p,
        "false_parked_rate": false_p / episodes,
        "timeouts": tout,
        "timeout_rate": tout / episodes,
        "time_median": q(times, 0.5),
        "time_p95": q(times, 0.95),
        "pos_p95": q(epos, 0.95),
        "yaw_p95_deg": math.degrees(q(eyaw, 0.95)),
        "jerk_mean": sum(jerk) / len(jerk) if jerk else float("nan"),
    }
    print(
        f"{policy}@{level}: success {succ}/{episodes} lb {lo:.3f} hi {hi:.3f} "
        f"coll {coll} fp {false_p} timeout {tout} med {out['time_median']:.1f}s p95 {out['time_p95']:.1f}s "
        f"pos_p95 {out['pos_p95'] * 1000:.0f}mm yaw_p95 {out['yaw_p95_deg']:.1f}deg"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--policy", default="random", help=f"checkpoint path or one of {BASELINES}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--declare-parked", action="store_true")
    ap.add_argument(
        "--no-mean-dynamics",
        action="store_true",
        help="use particles[0] instead of mean_coef on L0/L1",
    )
    ap.add_argument("--cam-hz", type=float, default=None)
    ap.add_argument(
        "--fov-search",
        action="store_true",
        help="student blind-search spin reflex after 1 s without a tag",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = run(
        args.episodes,
        args.level,
        args.policy,
        "cuda" if torch.cuda.is_available() else "cpu",
        args.seed,
        declare_parked=args.declare_parked,
        use_mean_dynamics=not args.no_mean_dynamics,
        cam_hz=args.cam_hz,
        fov_search=args.fov_search,
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
