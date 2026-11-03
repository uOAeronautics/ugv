"""PPO teacher with asymmetric critic (privileged pose+dynamics in value only)."""
from __future__ import annotations
import argparse
import time
import torch
import torch.nn as nn

from sim import MecanumVecEnv, EnvConfig
from models import RUNS, TEACHER_OBS, Actor, Critic, gae_advantages, priv_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=1e-3,
                    help="constant entropy weight (status quo: runs/teacher_L0.pt used this)")
    ap.add_argument("--ent-anneal", action="store_true",
                    help="anneal entropy 0.01 -> 10%% over training instead of --ent")
    ap.add_argument("--curric", type=int, default=0,
                    help="first N iters spawn at 0.1-0.4m so the hold is discovered, "
                         "then spec spawns (final gate always uses spec spawns)")
    ap.add_argument("--init", default=None,
                    help="warm-start actor+critic from a checkpoint (curriculum/continuation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lat-weight", type=float, default=2.0,
                    help="extra lateral-error weight in progress potential")
    ap.add_argument("--rate-weight", type=float, default=0.0,
                    help="action-rate penalty weight (L0-gate run used 0.0)")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="overwrite --out every N iters so kills don't lose runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = MecanumVecEnv(args.envs, EnvConfig(level=args.level, device=dev,
                                         seed=args.seed, rate_w=args.rate_weight, lat_w=args.lat_weight))
    actor = Actor().to(dev)
    critic = Critic().to(dev)
    if args.init:
        ckpt = torch.load(args.init, map_location=dev)
        actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_iters=args.iters)
    if args.init:
        # Continue optimization where it left off (fresh schedule churns
        # converged policies; observed 911->881 and 976->720 on continuation).
        for key, obj in (("opt", opt), ("sched", sched)):
            if key in ckpt:
                try:
                    obj.load_state_dict(ckpt[key])
                except (RuntimeError, ValueError, KeyError):
                    pass
    RUNS.mkdir(exist_ok=True)
    out = args.out or str(RUNS / f"teacher_{args.level}.pt")

    def save_ckpt():
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "level": args.level, "obs_dim": TEACHER_OBS}, out)
    n = args.envs
    ent = args.ent

    for it in range(args.iters):
        t0 = time.time()
        if args.ent_anneal:
            ent = 0.01 * (1 - 0.9 * it / max(args.iters - 1, 1))
        env.cfg.spawn_dist = (0.1, 0.4) if it < args.curric else None
        obs = env.reset()
        o, c = priv_input(obs, env)
        O, C, A, LP, R, D, V = [], [], [], [], [], [], []
        ep_succ = ep_coll = 0
        for _ in range(args.steps):
            with torch.no_grad():
                mean = actor(o)
                dist = torch.distributions.Normal(mean, actor.logstd.exp())
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
                v = critic(c)
            obs2, r, done, info = env.step(a)
            O.append(o); C.append(c); A.append(a); LP.append(lp)
            R.append(r); D.append(done.float()); V.append(v)
            ep_succ += int(info["success"].sum()); ep_coll += int(info["collision"].sum())
            idx = torch.where(done)[0]
            if len(idx):
                env.reset(idx.cpu())
            o, c = priv_input(obs2 if not len(idx) else env._privileged(), env)
        O = torch.stack(O); C = torch.stack(C); A = torch.stack(A)
        LP = torch.stack(LP); R = torch.stack(R); D = torch.stack(D); V = torch.stack(V)
        with torch.no_grad():
            _, c_last = priv_input(env._privileged(), env)
            last_v = critic(c_last)
            adv, ret = gae_advantages(R, V, D, last_v, args.gamma, args.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape(args.steps * n, *x.shape[2:])
        O, C, A, LP, ADV, RET = flat(O), flat(C), flat(A), flat(LP), flat(adv), flat(ret)
        m = O.shape[0]
        idx_all = torch.randperm(m, device=dev)
        mb = m // args.minibatches
        pl = vl = el = 0.0
        for e in range(args.epochs):
            for b in range(args.minibatches):
                sl = idx_all[b * mb:(b + 1) * mb]
                mean = actor(O[sl])
                dist = torch.distributions.Normal(mean, actor.logstd.exp())
                nlp = dist.log_prob(A[sl]).sum(-1)
                ratio = torch.exp(nlp - LP[sl])
                pg1 = ratio * ADV[sl]; pg2 = ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[sl]
                loss_p = -torch.min(pg1, pg2).mean()
                loss_v = 0.5 * (critic(C[sl]) - RET[sl]).pow(2).mean()
                loss_e = -dist.entropy().sum(-1).mean()
                (loss_p + loss_v + ent * loss_e).backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step(); opt.zero_grad()
                pl, vl, el = float(loss_p.detach()), float(loss_v.detach()), float(-loss_e.detach())
        sched.step()
        print(f"iter {it} pol {pl:.4f} val {vl:.4f} ent {el:.3f} "
              f"succ {ep_succ} coll {ep_coll} {time.time()-t0:.1f}s", flush=True)
        if args.ckpt_every > 0 and (it + 1) % args.ckpt_every == 0:
            save_ckpt()
    save_ckpt()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
