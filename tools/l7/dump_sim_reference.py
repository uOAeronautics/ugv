#!/usr/bin/env python3
"""Dump per-tick L1 sim policy inputs and actions for offline contract auditing.

Runs on CPU with rl/.venv against the default rl/sim.py L1 configuration
(use_mean_dynamics=False, posterior particle 0) so audit_contract.py stays
NumPy-only in the root venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "rl" / "runs" / "gazebo_l6" / "student_L1_best.npz"
DEFAULT_OUT = Path(__file__).resolve().parent / "sim_reference_L1.npz"

CHANNEL_NAMES = [
    "tag_x",
    "tag_y",
    "tag_yaw",
    "tag_age",
    "valid",
    "imu_w",
    "imu_yaw",
    "prev_a0",
    "prev_a1",
    "prev_a2",
    "tvel_x",
    "tvel_y",
    "tvel_yaw",
]
ACTION_NAMES = ["act_0", "act_1", "act_2"]


def _import_rl():
    """Import rl/sim.py and rl/models.py with brief retry if sim.py is mid-edit."""
    rl_dir = str(REPO / "rl")
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)
    last_err: Exception | None = None
    for _ in range(5):
        try:
            from models import Student
            from sim import EnvConfig, MecanumVecEnv

            return EnvConfig, MecanumVecEnv, Student
        except Exception as exc:  # pragma: no cover - transient edit guard
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"Failed importing rl modules: {last_err}") from last_err


def _load_student_from_npz(weights_path: Path, student_cls):
    z = np.load(weights_path, allow_pickle=True)
    sd = {
        k: torch.from_numpy(np.array(z[k], copy=True))
        for k in z.files
        if k.startswith("gru.") or k.startswith("head.")
    }
    h_dim = int(sd["gru.weight_hh_l0"].shape[1])
    head_w = int(sd["head.0.weight"].shape[0])
    act_dim = int(sd["head.4.bias"].shape[0])
    model = student_cls(act_dim=act_dim, h=h_dim, head_w=head_w)
    model.load_state_dict(sd, strict=False)
    model.eval()
    sha = str(z["meta_sha"]) if "meta_sha" in z.files else "?"
    stateful = bool(z["meta_stateful"]) if "meta_stateful" in z.files else False
    return model, sha, stateful, h_dim, act_dim


def dump_reference(
    weights_path: Path,
    out_path: Path,
    episodes: int = 200,
    seed: int = 11,
    use_mean_dynamics: bool = False,
    level: str = "L1",
) -> dict:
    EnvConfig, MecanumVecEnv, Student = _import_rl()
    model, sha, stateful, h_dim, act_dim = _load_student_from_npz(weights_path, Student)

    env = MecanumVecEnv(
        episodes,
        EnvConfig(
            level=level,
            device="cpu",
            seed=seed,
            use_mean_dynamics=use_mean_dynamics,
        ),
    )
    obs = env.reset(torch.arange(episodes))
    done = torch.zeros(episodes, dtype=torch.bool)
    h_state = torch.zeros(1, episodes, h_dim) if stateful else None

    frames_list: list[np.ndarray] = []
    acts_list: list[np.ndarray] = []
    ep_list: list[np.ndarray] = []
    pose_list: list[np.ndarray] = []

    ep_ids = np.arange(episodes, dtype=np.int32)
    steps = 0
    max_steps = int(round(env.cfg.max_time / env.dt)) + 5

    with torch.no_grad():
        while not bool(done.all()) and steps < max_steps:
            active_mask = ~done
            active_np = active_mask.numpy()

            if h_state is not None:
                out, h_state = model.forward_seq(obs["student"][:, -1:, :], h_state)
                act = out[:, 0].clamp(-1, 1)
            else:
                act = model(obs["student"]).clamp(-1, 1)

            # Match StudentPolicy.step blind-search reflex (src/ugv/policy.py).
            h_hist = obs["student"]
            blind = h_hist[:, -1, 3] >= 1.0
            if bool(blind.any()):
                u_turn = (-1.8 * h_hist[:, -1, 6] - 0.3 * h_hist[:, -1, 5]).clamp(-0.6, 0.6)
                zz = torch.zeros_like(u_turn)
                act[blind] = torch.stack([zz[blind], zz[blind], u_turn[blind]], dim=1)[
                    :, : act.shape[1]
                ]

            obs_next, _, d, info = env.step(act)
            frames_list.append(obs_next["student"][active_mask, -1, :].numpy().astype(np.float32))
            acts_list.append(act[active_mask, :3].numpy().astype(np.float32))
            ep_list.append(ep_ids[active_np])
            pose_list.append(obs_next["pose"][active_mask].numpy().astype(np.float32))

            obs = obs_next
            done = done | d
            steps += 1

    frames = np.concatenate(frames_list, axis=0)
    acts = np.concatenate(acts_list, axis=0)
    ep_id = np.concatenate(ep_list, axis=0)
    pose = np.concatenate(pose_list, axis=0)

    meta = {
        "weights": str(weights_path),
        "sha": sha,
        "stateful": stateful,
        "level": str(level),
        "episodes": int(episodes),
        "seed": int(seed),
        "use_mean_dynamics": bool(use_mean_dynamics),
        "dt": float(env.dt),
        "total_ticks": int(frames.shape[0]),
        "success": int(info["success"].sum()),
        "collision": int(info["collision"].sum()),
        "timeout": int(info["timeout"].sum()),
        "channel_names": CHANNEL_NAMES,
        "action_names": ACTION_NAMES,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=frames,
        acts=acts,
        ep_id=ep_id,
        pose=pose,
        dt=np.float64(env.dt),
        meta_json=np.array(json.dumps(meta)),
    )
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--level", type=str, default="L1")
    ap.add_argument("--use-mean-dynamics", action="store_true")
    args = ap.parse_args()

    meta = dump_reference(
        weights_path=args.weights,
        out_path=args.out,
        episodes=args.episodes,
        seed=args.seed,
        use_mean_dynamics=args.use_mean_dynamics,
        level=args.level,
    )
    print(
        f"wrote {args.out}: {meta['total_ticks']} ticks over {meta['episodes']} {meta['level']} episodes "
        f"(success {meta['success']}/{meta['episodes']}, sha={meta['sha']})"
    )


if __name__ == "__main__":
    main()
