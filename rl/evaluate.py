"""Seeded benchmark per docs/RL_PARKING.md. Baseline + checkpoints, Wilson bounds, JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import load_student, load_teacher, teacher_obs
from sim import EnvConfig, MecanumVecEnv

from ugv.parking import pose2_from_ccw, pose2_to_ccw  # noqa: E402  (frame boundary)
from ugv.policy import (  # noqa: E402  (numpy-only deploy module)
    SEARCH_AGE,
    SEARCH_D,
    SEARCH_DWELL,
    SEARCH_MAX,
    SEARCH_P,
    SEARCH_SWEEP_AMP,
    SEARCH_SWEEP_RATE,
)

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


def search_yaw_target_vec(blind_s: torch.Tensor) -> torch.Tensor:
    """Vectorized mirror of ugv.policy.search_yaw_target (dwell then triangle sweep)."""
    u = (blind_s - SEARCH_DWELL).clamp_min(0.0) * SEARCH_SWEEP_RATE
    p = u % (4 * SEARCH_SWEEP_AMP)
    return torch.where(
        p < SEARCH_SWEEP_AMP,
        p,
        torch.where(p < 3 * SEARCH_SWEEP_AMP, 2 * SEARCH_SWEEP_AMP - p, p - 4 * SEARCH_SWEEP_AMP),
    )


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
    dr_scale=1.0,
    consistent_particles=False,
    save_traces="",
    max_trace_episodes=20,
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
            dr_scale=dr_scale,
            consistent_particles=consistent_particles,
        ),
    )
    kind, model, meta = load_policy(policy, env.device)
    act_dim = meta["act_dim"]
    stateful = meta["stateful"]
    succ = coll = false_p = tout = 0
    times, epos, eyaw, jerk = [], [], [], []
    # Replay trace recording (ugv_replay_v1): at most max_trace_episodes
    # complete episodes, per-step frames in the Observatory schema.
    recording = bool(save_traces)
    trace_episodes = []
    left = episodes
    while left > 0:
        n = min(left, env.n)
        obs = env.reset(torch.arange(n))  # persistent stream: same seed, same JSON
        h_state = torch.zeros(1, n, meta.get("hidden", 64), device=env.device) if stateful else None
        prev_a = torch.zeros(n, 3, device=env.device)
        blind_t = torch.zeros(n, device=env.device)
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        t = torch.zeros(n, device=env.device)
        j = torch.zeros(n, device=env.device)
        steps = 0
        m_tr = min(n, max_trace_episodes) if recording else 0
        frames = [[] for _ in range(m_tr)] if recording else None
        ep_active = [True] * m_tr if recording else None
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
                # deploy blind-search reflex (mirrors StudentPolicy.step): dwell
                # at yaw 0, then triangle-sweep the yaw setpoint, zero translation.
                h = obs["student"][:n]
                blind_t = torch.where(
                    h[:, -1, 4] > 0.5, torch.zeros_like(blind_t), blind_t + env.cfg.dt
                )
                blind = blind_t >= SEARCH_AGE
                if bool(blind.any()):
                    target = search_yaw_target_vec(blind_t[blind] - SEARCH_AGE)
                    yaw_err = (h[blind, -1, 6] - target + math.pi) % (2 * math.pi) - math.pi
                    u_turn = (-SEARCH_P * yaw_err - SEARCH_D * h[blind, -1, 5]).clamp(
                        -SEARCH_MAX, SEARCH_MAX
                    )
                    z = torch.zeros_like(u_turn)
                    a[blind] = torch.stack([z, z, u_turn], dim=1)[:, : a.shape[1]]
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
            if recording and len(trace_episodes) < max_trace_episodes and any(ep_active):
                delayed = env._delayed_pose()[:m_tr].detach().cpu().tolist()
                blk = (
                    (
                        (env.time[:m_tr] > env.blackout_start[:m_tr])
                        & (env.time[:m_tr] < env.blackout_start[:m_tr] + env.blackout_total[:m_tr])
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                poses_cpu = env.pose[:m_tr].detach().cpu().tolist()
                valid_cpu = obs_full["valid"][:m_tr].detach().cpu().tolist()
                age_cpu = (env.hist_age[:m_tr] / env.dt).round().long().detach().cpu().tolist()
                imu_w_cpu = env.imu_w[:m_tr].detach().cpu().tolist()
                imu_yaw_cpu = env.imu_yaw[:m_tr].detach().cpu().tolist()
                act_cpu = a[:m_tr].detach().cpu().tolist()
                d_cpu = d[:m_tr].detach().cpu().tolist()
                ok_cpu = info["success"][:m_tr].detach().cpu().tolist()
                co_cpu = info["collision"][:m_tr].detach().cpu().tolist()
                fp_cpu = info["false_park"][:m_tr].detach().cpu().tolist()
                if kind == "student" and hasattr(model, "aux_head"):
                    with torch.no_grad():
                        _, hh = model.gru(obs["student"][:m_tr])
                        aux_cpu = model.aux_head(hh.squeeze(0))[:, :3].detach().cpu().tolist()
                else:
                    aux_cpu = poses_cpu
                for i in range(m_tr):
                    if not ep_active[i]:
                        continue
                    if len(trace_episodes) >= max_trace_episodes:
                        break
                    fi = len(frames[i])
                    p_i = poses_cpu[i]
                    yaw_ccw = float(p_i[2])
                    ccw = pose2_to_ccw(pose2_from_ccw(float(p_i[0]), float(p_i[1]), yaw_ccw))
                    aux_i = aux_cpu[i]
                    frames[i].append(
                        {
                            "t": round(fi * env.dt, 3),
                            "true_pose": [
                                round(float(p_i[0]), 4),
                                round(float(p_i[1]), 4),
                                round(yaw_ccw, 4),
                            ],
                            "true_pose_ccw": [round(ccw[0], 4), round(ccw[1], 4), round(ccw[2], 4)],
                            "delayed_tag": [
                                round(float(delayed[i][0]), 4),
                                round(float(delayed[i][1]), 4),
                                round(float(delayed[i][2]), 4),
                            ],
                            "aux_pose": [round(float(v), 4) for v in aux_i],
                            "valid": bool(valid_cpu[i]),
                            "blackout": bool(blk[i]),
                            "steps_since_valid": int(age_cpu[i]),
                            "imu_w": float(imu_w_cpu[i]),
                            "imu_yaw": float(imu_yaw_cpu[i]),
                            "action": [
                                round(float(act_cpu[i][0]), 4),
                                round(float(act_cpu[i][1]), 4),
                                round(float(act_cpu[i][2]), 4),
                            ],
                            "shield_clear": True,
                        }
                    )
                    if bool(d_cpu[i]):
                        ep_active[i] = False
                        outcome = (
                            "success"
                            if ok_cpu[i]
                            else "collision"
                            if co_cpu[i]
                            else "false_park"
                            if fp_cpu[i]
                            else "timeout"
                        )
                        trace_episodes.append(
                            {
                                "episode_id": len(trace_episodes),
                                "seed": seed,
                                "outcome": outcome,
                                "steps": len(frames[i]),
                                "frames": frames[i],
                            }
                        )
                        frames[i] = []
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
    if recording:
        Path(save_traces).parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema": "ugv_replay_v1",
            "format": "ugv_replay_v1",
            "source": "sim_eval",
            "run_id": Path(policy).stem,
            "level": level,
            "dr_scale": dr_scale,
            "dt": env.dt,
            "episodes": trace_episodes,
        }
        with open(save_traces, "w") as f:
            json.dump(bundle, f)
        print(f"wrote {save_traces} ({len(trace_episodes)} episodes)")
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
        "--dr-scale",
        type=float,
        default=1.0,
        help="fixed DR intensity for every env (1.0 = current levels)",
    )
    ap.add_argument(
        "--consistent-particles",
        action="store_true",
        help="sample only posterior particles with sound turn columns",
    )
    ap.add_argument(
        "--save-traces",
        type=str,
        default="",
        help="write a ugv_replay_v1 JSON replay bundle here",
    )
    ap.add_argument(
        "--max-trace-episodes",
        type=int,
        default=20,
        help="max episodes recorded in the replay bundle",
    )
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
        dr_scale=args.dr_scale,
        consistent_particles=args.consistent_particles,
        save_traces=args.save_traces,
        max_trace_episodes=args.max_trace_episodes,
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
