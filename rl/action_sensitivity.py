"""Interventional 3x3 action-sensitivity audit at the slot mouth.

From identical snapshots, branch 7 fixed-action counterfactual rollouts
(zero, +-dx, +-dy, +-dw) and measure how each action axis moves progress,
collision risk and heading error in two regimes: 'mouth' (y 0.25-0.35 m, tag
visible) and 'blind' (y 0.35-0.60 m, tag blanked). Each branch restores the
full snapshot (plant, observation history and both env RNG states), so every
counterfactual sees the same process-noise draws (common random numbers).
Branches run sequentially rather than as one batched env: a shared generator
would give each branch different noise and break the paired contrast.

A closed-loop 'policy' reference branch rolls the student itself after HIST
zero-action warmup steps, so its GRU history reflects the regime rather than
the reset spawn; it is a reference, not part of the paired 3x3 matrix.

Vision is forced through the snapshot keys step() actually consumes:
env_min_tag_range (the near_deck geometry gate) and drop_p (camera drops).
tag_valid itself is recomputed from geometry every step and cannot be forced.
At L0-L2 (under_drone_loss=False) raising min_tag_range keeps the tag visible,
so drop_p=1.0 is the level-agnostic blanking lever; both are set for 'blind'.

CPU only. Thin main() + reusable functions; see test_rl.py for the smoke test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from models import RUNS, load_student
from sim import HIST, EnvConfig, MecanumVecEnv

AXES = ("x", "y", "w")
OUTCOMES = ("progress", "coll", "yaw")
BRANCHES = ("zero", "+dx", "-dx", "+dy", "-dy", "+dw", "-dw")
REGIME_Y = {"mouth": (0.25, 0.35), "blind": (0.35, 0.60)}


def branch_action(name: str, delta: float, n: int, device: str) -> torch.Tensor:
    a = torch.zeros(n, 3, device=device)
    if name != "zero":
        axis = AXES.index(name[-1])
        a[:, axis] = delta if name[0] == "+" else -delta
    return a


def place_regime(env: MecanumVecEnv, regime: str, gen: torch.Generator) -> dict:
    """Reset the batch, re-pose it into the regime and gate vision; snapshot it."""
    env.reset()
    n = env.n
    lo, hi = REGIME_Y[regime]
    y = lo + (hi - lo) * torch.rand(n, generator=gen)
    x = -0.12 + 0.24 * torch.rand(n, generator=gen)
    yaw = -0.15 + 0.30 * torch.rand(n, generator=gen)
    pose = torch.stack([x, y, yaw], dim=1).to(env.device)
    state = env.get_state()
    state["pose"][:] = pose
    state["vel"][:] = 0.0
    state["imu_yaw"][:] = pose[:, 2]
    state["pose_ring"][:] = pose.unsqueeze(1).expand(-1, state["pose_ring"].shape[1], -1)
    if regime == "mouth":
        state["env_min_tag_range"][:] = 0.0
    else:
        state["env_min_tag_range"][:] = 10.0
        state["drop_p"][:] = 1.0
    env.set_state(state)
    return env.get_state()


def rollout_branch(env: MecanumVecEnv, state: dict, action: torch.Tensor, steps: int) -> dict:
    """Restore the identical snapshot and roll a fixed action for `steps` steps.

    progress = y_start - y_end (m, positive = toward the drone); coll = ever
    collided within the horizon; yaw = |yaw_end| - |yaw_start| (rad). Done envs
    freeze in step(), so terminal values carry to the horizon end.
    """
    env.set_state(state)
    y0 = state["pose"][:, 1].clone()
    yaw0 = state["pose"][:, 2].abs().clone()
    coll = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    valid_seen = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    for _ in range(steps):
        obs, _, _, info = env.step(action)
        coll |= info["collision"]
        valid_seen |= obs["valid"]
    return {
        "progress": y0 - env.pose[:, 1],
        "coll": coll.float(),
        "yaw": env.pose[:, 2].abs() - yaw0,
        "mean_abs_action": float(action.abs().mean()),
        "valid_frac": float(valid_seen.float().mean()),
    }


def rollout_policy(env: MecanumVecEnv, state: dict, model, steps: int) -> dict:
    """Closed-loop policy reference from a warm snapshot (consistent history)."""
    env.set_state(state)
    y0 = state["pose"][:, 1].clone()
    yaw0 = state["pose"][:, 2].abs().clone()
    coll = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    valid_seen = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    abs_a = 0.0
    obs = env._privileged()
    with torch.no_grad():
        for _ in range(steps):
            a = model(obs["student"]).clamp(-1, 1)
            abs_a += float(a.abs().mean())
            obs, _, _, info = env.step(a)
            coll |= info["collision"]
            valid_seen |= obs["valid"]
    return {
        "progress": y0 - env.pose[:, 1],
        "coll": coll.float(),
        "yaw": env.pose[:, 2].abs() - yaw0,
        "mean_abs_action": abs_a / steps,
        "valid_frac": float(valid_seen.float().mean()),
    }


def audit(
    policy: str,
    level: str = "L_real",
    episodes: int = 32,
    seed: int = 11,
    horizon: float = 1.0,
    delta: float = 0.3,
    device: str = "cpu",
) -> dict:
    model, ckpt = load_student(policy, device)
    cfg = EnvConfig(level=level, device=device, seed=seed)
    env = MecanumVecEnv(episodes, cfg)
    steps = max(1, round(horizon / cfg.dt))
    gen = torch.Generator().manual_seed(seed)
    regimes = {}
    for regime in ("mouth", "blind"):
        state = place_regime(env, regime, gen)
        branches = {}
        for name in BRANCHES:
            a = branch_action(name, delta, episodes, device)
            branches[name] = rollout_branch(env, state, a, steps)
        # warm the observation history into the regime before rolling the
        # policy: set_state restores the spawn-time hist, not a regime one
        env.set_state(state)
        zero = torch.zeros(episodes, 3, device=device)
        for _ in range(HIST):
            env.step(zero)
        branches["policy"] = rollout_policy(env, env.get_state(), model, steps)
        policy_mean_abs_action = branches["policy"]["mean_abs_action"]
        if regime == "blind" and any(b["valid_frac"] > 0.0 for b in branches.values()):
            raise RuntimeError("blind regime saw a valid tag; vision forcing failed")
        matrix = [
            [
                float((branches[f"+d{ax}"][out] - branches[f"-d{ax}"][out]).mean())
                for out in OUTCOMES
            ]
            for ax in AXES
        ]
        regimes[regime] = {
            "n": episodes,
            "matrix": matrix,
            "policy_mean_abs_action": policy_mean_abs_action,
            "branches": {
                name: {k: float(v.mean()) if torch.is_tensor(v) else v for k, v in b.items()}
                for name, b in branches.items()
            },
        }
    return {
        "run_id": ckpt.get("run_id") or Path(policy).stem,
        "level": level,
        "seed": seed,
        "regimes": regimes,
        "config": {
            "policy": str(policy),
            "episodes": episodes,
            "horizon_s": horizon,
            "dt": cfg.dt,
            "steps": steps,
            "delta": delta,
            "axes": list(AXES),
            "outcomes": list(OUTCOMES),
            "branches": list(BRANCHES),
            "policy_branch": "closed-loop student after HIST zero-action warmup steps "
            "(reference, not part of the paired matrix)",
            "device": device,
            "matrix": "mean over snapshots of outcome(+axis) - outcome(-axis)",
            "progress": "y_start - y_end (m, positive = toward drone)",
            "coll": "collision-rate difference within horizon",
            "yaw": "|yaw_end| - |yaw_start| (rad)",
        },
    }


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [
        f"action sensitivity: {result['run_id']} @ {result['level']} "
        f"(N={cfg['episodes']}, seed {result['seed']}, delta={cfg['delta']}, "
        f"H={cfg['horizon_s']} s)",
        "",
        "matrix = mean[outcome(+axis) - outcome(-axis)] over snapshots; "
        "+dy moves away from the drone",
        "",
        "| regime | axis | dprogress (m) | dcoll | dyaw (rad) |",
        "|---|---|---|---|---|",
    ]
    for regime, data in result["regimes"].items():
        for ax, row in zip(AXES, data["matrix"], strict=True):
            lines.append(f"| {regime} | {ax} | {row[0]:+.4f} | {row[1]:+.4f} | {row[2]:+.4f} |")
    lines += [
        "",
        "| regime | branch | progress (m) | coll | yaw (rad) | mean abs action | valid frac |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime, data in result["regimes"].items():
        for name in ("zero", "policy"):
            b = data["branches"][name]
            lines.append(
                f"| {regime} | {name} | {b['progress']:+.4f} | {b['coll']:.3f} "
                f"| {b['yaw']:+.4f} | {b['mean_abs_action']:.4f} | {b['valid_frac']:.2f} |"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Interventional 3x3 action-sensitivity audit (CPU)")
    ap.add_argument("--policy", required=True, help="student checkpoint path")
    ap.add_argument("--level", default="L_real")
    ap.add_argument("--episodes", type=int, default=32, help="snapshots per regime")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--horizon", type=float, default=1.0, help="branch rollout horizon (s)")
    ap.add_argument("--delta", type=float, default=0.3, help="counterfactual action magnitude")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = audit(
        args.policy,
        level=args.level,
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        delta=args.delta,
        device="cpu",
    )
    out = args.out or str(RUNS / f"sensitivity_{Path(args.policy).stem}_{args.level}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(render_markdown(result))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
