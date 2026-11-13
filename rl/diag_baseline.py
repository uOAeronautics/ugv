"""Failure-mode diagnostics for the staged baseline (temp scout artifact)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sim import MecanumVecEnv, EnvConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ugv.staged_baseline import StagedBaseline, MeanModel


def diag(level, episodes=300, seed=11, device="cuda"):
    env = MecanumVecEnv(min(episodes, 512), EnvConfig(level=level, device=device, seed=seed,
                                                      use_mean_dynamics=False))
    model = MeanModel()
    max_steps = int(env.cfg.max_time / env.cfg.dt) + 5
    recs = []
    left = episodes
    ep0 = 0
    while left > 0:
        n = env.n
        env.reset(torch.arange(n))
        ctl = StagedBaseline(n, latency=env.latency.cpu().numpy(), dt=env.dt, model=model)
        cmd = np.zeros((n, 3))
        done = np.zeros(n, bool)
        t = np.zeros(n)
        for _ in range(max_steps):
            obs, _, d, info = env.step(torch.as_tensor(cmd, dtype=torch.float32, device=env.device))
            cmd = ctl.step(obs["tag"].cpu().numpy(), obs["valid"].cpu().numpy(), env.imu_w.cpu().numpy())
            t = np.where(~done, t + env.dt, t)
            done |= d.cpu().numpy()
            if done.all():
                break
        k = min(left, n)
        pose = env.pose[:k].cpu().numpy()
        for i in range(k):
            recs.append(dict(
                ep=ep0 + i,
                succ=bool(info["success"][i]) and bool(done[i]),
                coll=bool(info["collision"][i]),
                tout=bool(info["timeout"][i].cpu().numpy()) if torch.is_tensor(info["timeout"][i]) else bool(info["timeout"][i]),
                time=float(t[i]),
                x=float(pose[i, 0]), y=float(pose[i, 1]), yaw=float(pose[i, 2]),
                phase=int(ctl.phase[i]),
            ))
        ep0 += k
        left -= k
    succ = sum(r["succ"] for r in recs)
    coll = sum(r["coll"] for r in recs)
    tout = sum(r["tout"] for r in recs)
    print(f"{level}: succ {succ}/{episodes} coll {coll} tout {tout}")
    print("  COLLISION final poses (x_mm, y_mm, yaw_deg, t):")
    for r in recs:
        if r["coll"]:
            print(f"    ep{r['ep']}: x={r['x']*1000:.0f} y={r['y']*1000:.0f} yaw={np.degrees(r['yaw']):.1f} t={r['time']:.1f} ph={r['phase']}")
    print("  TIMEOUT final poses (first 15):")
    shown = 0
    for r in recs:
        if r["tout"] and shown < 15:
            print(f"    ep{r['ep']}: x={r['x']*1000:.0f} y={r['y']*1000:.0f} yaw={np.degrees(r['yaw']):.1f} t={r['time']:.1f} ph={r['phase']}")
            shown += 1
    ts = sorted(r["time"] for r in recs if r["succ"])
    if ts:
        print(f"  succ time med {ts[len(ts)//2]:.1f} p95 {ts[int(len(ts)*0.95)]:.1f}")


if __name__ == "__main__":
    lvl = sys.argv[1] if len(sys.argv) > 1 else "L1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    diag(lvl, n)
