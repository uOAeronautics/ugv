"""Top-down rollout visualization (Agg PNGs + optional mp4).

Records pose trajectories per episode and draws the slot (skid capsules),
goal box, path colored by time, and outcome. The vector sim has no pixels;
for 3D use the Gazebo drone world (simulation/gazebo/launch_drone.command).
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from evaluate import load_policy, policy_act
from matplotlib import animation
from matplotlib.patches import Rectangle
from sim import CAR_L, CAR_W, GEAR_LEN, SKID_X, EnvConfig, MecanumVecEnv


def rollout(policy, level, episodes, device, seed=0, max_steps=3000):
    env = MecanumVecEnv(min(episodes, 256), EnvConfig(level=level, device=device, seed=seed))
    kind, model = load_policy(policy, env.device)
    trajs = []
    left, batch = episodes, 0
    while left > 0:
        n = min(left, env.n)
        obs = env.reset(torch.arange(n))
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        paths = [[[], [], []] for _ in range(n)]  # x, y, yaw
        t = 0
        while not bool(done.all()) and t < max_steps:
            for i in range(n):
                p = obs["pose"][i].tolist()
                paths[i][0].append(p[0])
                paths[i][1].append(p[1])
                paths[i][2].append(p[2])
            a = policy_act(
                kind,
                model,
                {"pose": obs["pose"][:n], "vel": obs["vel"][:n], "student": obs["student"][:n]},
                n,
                env.device,
            )
            full = torch.zeros(env.n, 3, device=env.device)
            full[:n] = a
            obs, _, d, info = env.step(full)
            done = done | d[:n].to(env.device)
            t += 1
        for i in range(n):
            k = (
                "success"
                if info["success"][i]
                else ("collision" if info["collision"][i] else "timeout")
            )
            trajs.append({"path": paths[i], "outcome": k, "time": t * env.cfg.dt})
        left -= n
        batch += 1
    return trajs


def draw_episode(traj, path, show_vel=False):
    xs, ys, yaws = traj["path"]
    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(max(map(abs, xs + [1.0])), max(map(abs, ys + [1.0]))) * 1.15
    for sx in (-SKID_X, SKID_X):
        ax.plot(
            [sx, sx], [-GEAR_LEN / 2, GEAR_LEN / 2], "k-", lw=6, alpha=0.7, solid_capstyle="round"
        )
    ax.add_patch(
        Rectangle((-0.03, -0.05), 0.06, 0.10, fill=False, ec="green", lw=1.5, label="goal box")
    )
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis", s=8)
    ax.plot(xs[0], ys[0], "bo", ms=8, label="start")
    ax.plot(xs[-1], ys[-1], "rx", ms=10, mew=2, label="end")
    # car footprint at end pose
    c, s = math.cos(yaws[-1]), math.sin(yaws[-1])
    hw, hl = CAR_W / 2, CAR_L / 2
    corners = [
        (c * lx - s * ly + xs[-1], s * lx + c * ly + ys[-1])
        for lx, ly in [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl), (-hw, -hl)]
    ]
    ax.plot([p[0] for p in corners], [p[1] for p in corners], "r-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x right (m)")
    ax.set_ylabel("y fwd (m)")
    ax.set_title(f"{traj['outcome']}  t={traj['time']:.1f}s  n={len(xs)} steps")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="step")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def animate_episode(traj, path, fps=25):
    xs, ys, yaws = traj["path"]
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(max(map(abs, xs + [1.0])), max(map(abs, ys + [1.0]))) * 1.1
    for sx in (-SKID_X, SKID_X):
        ax.plot(
            [sx, sx], [-GEAR_LEN / 2, GEAR_LEN / 2], "k-", lw=6, alpha=0.7, solid_capstyle="round"
        )
    ax.add_patch(Rectangle((-0.03, -0.05), 0.06, 0.10, fill=False, ec="green", lw=1.5))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    (trail,) = ax.plot([], [], "b-", lw=1.5)
    (car,) = ax.plot([], [], "r-", lw=2)
    hw, hl = CAR_W / 2, CAR_L / 2

    def frame(i):
        j = max(0, min(i * 2, len(xs) - 1))
        trail.set_data(xs[: j + 1], ys[: j + 1])
        c, s = math.cos(yaws[j]), math.sin(yaws[j])
        corners = [
            (c * lx - s * ly + xs[j], s * lx + c * ly + ys[j])
            for lx, ly in [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl), (-hw, -hl)]
        ]
        car.set_data([p[0] for p in corners], [p[1] for p in corners])
        ax.set_title(f"{traj['outcome']}  t={j * 0.02:.1f}s")
        return trail, car

    animation.FuncAnimation(
        fig, frame, frames=len(xs) // 2 + 1, interval=1000 // fps, blit=False
    ).save(path, fps=fps)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--level", default="L0")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--animate", choices=["first", "none"], default="none")
    ap.add_argument("--outdir", default="runs/vis")
    args = ap.parse_args()
    import os

    os.makedirs(args.outdir, exist_ok=True)
    trajs = rollout(
        args.policy,
        args.level,
        args.episodes,
        "cuda" if torch.cuda.is_available() else "cpu",
        args.seed,
    )
    for i, tr in enumerate(trajs):
        p = f"{args.outdir}/ep{i:02d}_{tr['outcome']}.png"
        draw_episode(tr, p)
        print(p, f"t={tr['time']:.1f}s")
    if args.animate == "first":
        mp = f"{args.outdir}/ep00_{trajs[0]['outcome']}.mp4"
        animate_episode(trajs[0], mp)
        print(mp)


if __name__ == "__main__":
    main()
