"""Regenerate tests/fixtures/policy_trace.npz for the 0.8 golden-trace test.

Rolls a short braked rollout (fixed seed) and records the exact per-step
inputs StudentPolicy.step needs plus the resulting env.hist. Rerun whenever
the sim frame builder or the deploy frame builder changes.
"""

from __future__ import annotations
import math
import numpy as np
import torch
from pathlib import Path

from sim import MecanumVecEnv, EnvConfig
from models import Student

STEPS = 25
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "policy_trace.npz"


def main():
    torch.manual_seed(0)
    env = MecanumVecEnv(1, EnvConfig(level="L1", device="cpu", seed=0))
    env.reset()
    student = Student()
    rec = {k: [] for k in ("tag", "valid", "age", "imu_w", "imu_yaw", "prev_act", "est")}
    for _ in range(STEPS):
        prev = env.prev_act[0].cpu().numpy()
        obs, _, _, _ = env.step(torch.zeros(1, 3))
        rec["tag"].append(env.tag_raw[0].cpu().numpy())
        rec["valid"].append(bool(env.tag_valid[0]))
        rec["age"].append(float(env.hist_age[0]))
        rec["imu_w"].append(float(env.imu_w[0]))
        rec["imu_yaw"].append(
            float(torch.atan2(torch.sin(env.imu_yaw[0]), torch.cos(env.imu_yaw[0])))
        )
        rec["prev_act"].append(prev)
        p = env.pose[0].cpu().numpy()
        rec["est"].append((float(p[0]), float(p[1]), float(p[2])))
    arrays = {k: np.array(v) for k, v in student.state_dict().items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        **{f"w_{k}": v for k, v in arrays.items()},
        **{f"in_{k}": np.array(v) for k, v in rec.items()},
        hist=np.array(obs["student"][0].cpu().numpy()),
    )
    print(f"wrote {OUT} ({STEPS} steps)")


if __name__ == "__main__":
    main()
