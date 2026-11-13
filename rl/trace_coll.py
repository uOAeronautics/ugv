"""Record full traces for collision episodes (temp scout artifact)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sim import MecanumVecEnv, EnvConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ugv.staged_baseline import StagedBaseline, MeanModel

level, seed, nep = sys.argv[1] if len(sys.argv) > 1 else "L1", 11, 200
env = MecanumVecEnv(min(nep, 512), EnvConfig(level=level, device="cuda", seed=seed,
                                             use_mean_dynamics=False))
model = MeanModel()
max_steps = int(env.cfg.max_time / env.cfg.dt) + 5
N = env.n
hist = {i: [] for i in range(N)}
env.reset(torch.arange(N))
ctl = StagedBaseline(N, latency=env.latency.cpu().numpy(), dt=env.dt, model=model)
cmd = np.zeros((N, 3))
done = np.zeros(N, bool)
for step in range(max_steps):
    obs, _, d, info = env.step(torch.as_tensor(cmd, dtype=torch.float32, device=env.device))
    tag = obs["tag"].cpu().numpy(); valid = obs["valid"].cpu().numpy()
    cmd = ctl.step(tag, valid, env.imu_w.cpu().numpy())
    pose = env.pose.cpu().numpy(); vel = env.vel.cpu().numpy()
    ekf = ctl.ekf.pose.copy()
    for i in range(N):
        if not done[i]:
            hist[i].append((round(step * env.dt, 2), np.round(pose[i], 3), np.round(vel[i], 3),
                            np.round(ekf[i], 3), int(ctl.phase[i]), np.round(cmd[i], 3),
                            bool(valid[i])))
    done |= d.cpu().numpy()
    if done.all():
        break
coll = info["collision"].cpu().numpy()
shown = 0
for i in range(N):
    if coll[i] and shown < 3:
        shown += 1
        h = hist[i]
        print(f"=== ep{i} coll, last {min(150, len(h))} steps (t,true,vel,ekf,ph,cmd,valid) ===")
        for row in h[-150:][::5]:
            print(f"t={row[0]:5.1f} true={row[1].tolist()} vel={row[2].tolist()} ekf={row[3].tolist()} ph={row[4]} cmd={row[5].tolist()} v={int(row[6])}")
