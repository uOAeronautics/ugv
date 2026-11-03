"""PPO directly on the history student with privileged critic.

Cloning a privileged teacher hits a POMDP floor at the slot mouth: the teacher
reacts to OU velocity kicks the history cannot show. Optimizing the deploy
architecture directly for reward sidesteps the clonability bottleneck while
keeping the privileged critic for variance reduction (asymmetric PPO).
Rollouts store the exact pre-step history [T,N,50,13].
"""
from __future__ import annotations
import argparse
import time
import torch
import torch.nn as nn

from sim import MecanumVecEnv, EnvConfig, HIST
from models import RUNS, Critic, Student, gae_advantages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L0")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--init", default=None, help="student ckpt to warm-start")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = MecanumVecEnv(args.envs, EnvConfig(level=args.level, device=dev))
    student = Student().to(dev)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location=dev)["student"])
        print(f"warm-started from {args.init}")
    critic = Critic().to(dev)
    opt = torch.optim.Adam(list(student.parameters()) + list(critic.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_iters=args.iters)
    n = args.envs

    for it in range(args.iters):
        t0 = time.time()
        obs = env.reset()
        HS, A, LP, R, D, V, PR = [], [], [], [], [], [], []
        ep_succ = ep_coll = 0
        for t in range(args.steps):
            h = obs["student"]
            mean = student(h)
            dist = torch.distributions.Normal(mean, 0.3)
            a = dist.sample()
            lp = dist.log_prob(a).sum(-1)
            o = torch.cat([obs["pose"], obs["vel"]], dim=1)
            priv = torch.cat([o, env.coef.reshape(n, -1)], dim=1)
            v = critic(priv)
            obs2, r, done, info = env.step(a)
            HS.append(h.detach())
            PR.append(priv.detach())
            A.append(a.detach()); LP.append(lp.detach())
            R.append(r.detach()); D.append(done.float()); V.append(v.detach())
            ep_succ += int(info["success"].sum()); ep_coll += int(info["collision"].sum())
            idx = torch.where(done)[0]
            if len(idx):
                env.reset(idx.cpu())
                obs2 = env._privileged()
            obs = obs2
        HSs = torch.stack(HS)
        PRs = torch.stack(PR)
        A = torch.stack(A); LP = torch.stack(LP); R = torch.stack(R)
        D = torch.stack(D); V = torch.stack(V)
        with torch.no_grad():
            o = torch.cat([obs["pose"], obs["vel"]], dim=1)
            last_v = critic(torch.cat([o, env.coef.reshape(n, -1)], dim=1))
            adv, ret = gae_advantages(R, V, D, last_v, args.gamma, args.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ent = 0.01 * (1 - 0.9 * it / max(args.iters - 1, 1))
        idx_all = torch.randperm(n, device=dev)
        mb = n // args.minibatches
        pl = vl = 0.0
        for e in range(args.epochs):
            for b in range(args.minibatches):
                sl = idx_all[b * mb:(b + 1) * mb]
                mean = student(HSs[:, sl].reshape(args.steps * len(sl), HIST, -1))
                dist = torch.distributions.Normal(mean, 0.3)
                nlp = dist.log_prob(A[:, sl].reshape(-1, 3)).sum(-1)
                ratio = torch.exp(nlp - LP[:, sl].reshape(-1))
                pg1 = ratio * adv[:, sl].reshape(-1)
                pg2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv[:, sl].reshape(-1)
                loss_p = -torch.min(pg1, pg2).mean()
                loss_v = 0.5 * (critic(PRs[:, sl].reshape(-1, PRs.shape[-1]))
                                - ret[:, sl].reshape(-1)).pow(2).mean()
                loss_e = -dist.entropy().sum(-1).mean()
                (loss_p + loss_v + ent * loss_e).backward()
                nn.utils.clip_grad_norm_(list(student.parameters()) + list(critic.parameters()), 0.5)
                opt.step(); opt.zero_grad()
                pl, vl = float(loss_p), float(loss_v)
        sched.step()
        print(f"iter {it} pol {pl:.4f} val {vl:.4f} succ {ep_succ} coll {ep_coll} {time.time()-t0:.1f}s", flush=True)
    RUNS.mkdir(exist_ok=True)
    out = args.out or str(RUNS / f"student_ppo_{args.level}.pt")
    torch.save({"student": student.state_dict(), "critic": critic.state_dict(),
                "level": args.level}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
