"""PPO directly on the history student with privileged critic.

Cloning a privileged teacher hits a POMDP floor at the slot mouth: the teacher
reacts to OU velocity kicks the history cannot show. Optimizing the deploy
architecture directly for reward sidesteps the clonability bottleneck while
keeping the privileged critic for variance reduction (asymmetric PPO).
Rollouts store the exact pre-step history [T,N,50,13].

Envs persist across iterations (episodes are bootstrapped at the rollout cut,
not reset): a 128-step rollout is 2.56 s, shorter than approach + 1 s hold,
so per-iteration resets made success unreachable during training.
Exploration std is learned (log-param, saved as "logstd"); a fixed 0.3 std
cannot satisfy the <1 cm/s hold for 50 consecutive steps.
"""
from __future__ import annotations
import argparse
import math
import time
import torch
import torch.nn as nn

from sim import MecanumVecEnv, EnvConfig, HIST
from models import RUNS, Critic, Student, gae_advantages


def _parse_spawn(s: str | None):
    if s is None:
        return None
    parts = [float(x) for x in s.split(",")]
    return parts[0] if len(parts) == 1 else (parts[0], parts[1])


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
    ap.add_argument("--ent", type=float, default=1e-3,
                    help="constant entropy weight")
    ap.add_argument("--ent-anneal", action="store_true",
                    help="anneal entropy 0.01 -> 10%% over training instead of --ent")
    ap.add_argument("--curric", type=int, default=0,
                    help="first N iters spawn at 0.1-0.4m, then spec/spawn-dist")
    ap.add_argument("--spawn-dist", default=None,
                    help="override spawn distance, e.g. '0.5,1.5' or '0.75'")
    ap.add_argument("--act-dim", type=int, default=3)
    ap.add_argument("--stateful", action="store_true",
                    help="carry GRU hidden state across steps (for 3s L3+ blackouts)")
    ap.add_argument("--declare-parked", action="store_true")
    ap.add_argument("--no-mean-dynamics", action="store_true")
    ap.add_argument("--cam-hz", type=float, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--init", default=None,
                   help="warm-start student weights (fresh opt & LR schedule)")
    g.add_argument("--resume", default=None,
                   help="resume interrupted run (loads weights + opt + sched)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--head-w", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lat-weight", type=float, default=2.0,
                    help="extra lateral-error weight in progress potential")
    ap.add_argument("--rate-weight", type=float, default=0.0,
                    help="action-rate penalty weight (L0-gate teacher used 0.0)")
    ap.add_argument("--std0", type=float, default=0.3, help="initial exploration std")
    ap.add_argument("--reset-std", action="store_true",
                    help="with --init, reset logstd to log(std0) instead of inheriting converged std")
    ap.add_argument("--aux-weight", type=float, default=0.1,
                    help="LeWM/RMA aux supervision weight: MSE(aux_head(h), priv [pose,vel])")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="first N iters train the critic only (actor+logstd frozen) so a "
                         "fresh/mismatched V cannot destroy warm-started actor weights")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="overwrite --out every N iters so kills don't lose runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_spawn = _parse_spawn(args.spawn_dist)
    env = MecanumVecEnv(args.envs, EnvConfig(
        level=args.level, device=dev, seed=args.seed,
        spawn_dist=base_spawn, rate_w=args.rate_weight, lat_w=args.lat_weight,
        use_mean_dynamics=not args.no_mean_dynamics, declare_parked=args.declare_parked,
        cam_hz=args.cam_hz))
    student = Student(act_dim=args.act_dim, h=args.hidden, head_w=args.head_w).to(dev)
    logstd = nn.Parameter(torch.full((args.act_dim,), math.log(args.std0), device=dev))
    ckpt_path = args.resume or args.init
    ck = None
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=dev)
        try:
            student.load_state_dict(ck["student"])
        except RuntimeError:
            student.load_state_dict(ck["student"], strict=False)  # pre-aux-head ckpt
        if "logstd" in ck and ck["logstd"].shape == logstd.shape:
            logstd.data.copy_(ck["logstd"])
        if args.reset_std and args.init:
            logstd.data.fill_(math.log(args.std0))
        print(f"{'resumed' if args.resume else 'warm-started'} from {ckpt_path}")
    critic = Critic().to(dev)
    if ck is not None and "critic" in ck:
        try:
            critic.load_state_dict(ck["critic"])
        except RuntimeError:
            pass  # arch change: keep fresh critic
    params = list(student.parameters()) + [logstd] + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_iters=args.iters)
    if args.resume and ck is not None:
        for key, obj in (("opt", opt), ("sched", sched)):
            if key in ck:
                try:
                    obj.load_state_dict(ck[key])
                except (RuntimeError, ValueError, KeyError):
                    pass
    n = args.envs
    RUNS.mkdir(exist_ok=True)
    out = args.out or str(RUNS / f"student_ppo_{args.level}.pt")

    def save_ckpt():
        torch.save({"student": student.state_dict(), "critic": critic.state_dict(),
                    "logstd": logstd.detach().cpu(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "level": args.level,
                    "hidden": args.hidden, "head_w": args.head_w,
                    "act_dim": args.act_dim, "stateful": args.stateful}, out)

    def apply_curric(it):
        # alpha=0 starts L0-like (dist 0.55-0.85 straight ahead, no latency,
        # no sensor noise, 50 Hz camera), which is skid-safe: bearing 0 spawns
        # at x=0, clear of both skids. Every L0->L1 sensor change must ramp:
        # noise 1.0 alone collides 247/256 zero-shot at alpha=0, and 30 Hz
        # cam alone collides 254/256 (valid flickers 0101..., hist_age/tag_ema
        # pattern the L0 policy never saw) -- only noise=0 AND cam=50 together
        # park 256/256. Spawn-range widening is harmless (255/256 with full
        # L1 dist at alpha=0 otherwise), so bearing/yaw/latency/noise/camhz
        # carry the curriculum.
        if args.curric <= 0 or it >= args.curric:
            env.cfg.spawn_dist = base_spawn
            env.cfg.bearing_max = None
            env.cfg.yaw_max = None
            env.cfg.latency_override = None
            env.cfg.noise_override = None
            env.cfg.cam_hz = None
            env.cam_hz = float(env.level.get("cam_hz", 50.0))
            return
        alpha = it / max(args.curric, 1)
        tgt = base_spawn if base_spawn is not None else env.level["dist"]
        t_lo, t_hi = (float(tgt[0]), float(tgt[1])) if isinstance(tgt, tuple) else (float(tgt), float(tgt))
        env.cfg.spawn_dist = (0.55 + alpha * (t_lo - 0.55), 0.85 + alpha * (t_hi - 0.85))
        env.cfg.bearing_max = alpha * float(env.level["bearing"])
        env.cfg.yaw_max = alpha * float(env.level["yaw"])
        lat = env.level["latency"]
        env.cfg.latency_override = ((alpha * float(lat[0]), alpha * float(lat[1]))
                                    if isinstance(lat, tuple) else alpha * float(lat))
        env.cfg.noise_override = alpha * float(env.level["noise"])
        cam_tgt = float(env.level.get("cam_hz", 50.0))
        env.cfg.cam_hz = env.cam_hz = 50.0 + alpha * (cam_tgt - 50.0)

    apply_curric(0)
    obs = env.reset()
    h_state = torch.zeros(1, n, args.hidden, device=dev) if args.stateful else None
    # Critic warmup: the loaded critic was fitted to L0 returns, so its first
    # advantages on the noisy curriculum point the wrong way; 4 PPO epochs of
    # those gradients destroy the warm-started actor (iter1 coll 85->280 with
    # inherited std, 32->713 with reset std). Freeze actor+logstd while V
    # adapts. Adam skips params with grad None.
    if args.critic_warmup > 0:
        student.requires_grad_(False)
        logstd.requires_grad_(False)

    for it in range(args.iters):
        t0 = time.time()
        apply_curric(it)
        if it == args.critic_warmup and args.critic_warmup > 0:
            student.requires_grad_(True)
            logstd.requires_grad_(True)
        HS, A, LP, R, D, V, PR = [], [], [], [], [], [], []
        h_init = h_state.clone() if args.stateful else None
        ep_succ = ep_coll = 0
        for t in range(args.steps):
            h = obs["student"]
            with torch.no_grad():
                if args.stateful:
                    frame = h[:, -1:, :]
                    mean_seq, h_next = student.forward_seq(frame, h_state)
                    mean = mean_seq[:, 0]
                else:
                    mean = student(h)
                dist = torch.distributions.Normal(mean, logstd.exp())
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
                o = torch.cat([obs["pose"], obs["vel"]], dim=1)
                priv = torch.cat([o, env.coef.reshape(n, -1)], dim=1)
                v = critic(priv)
            obs2, r, done, info = env.step(a)
            HS.append((h[:, -1, :] if args.stateful else h).detach())
            PR.append(priv.detach())
            A.append(a.detach()); LP.append(lp.detach())
            R.append(r.detach()); D.append(done.float()); V.append(v.detach())
            ep_succ += int(info["success"].sum()); ep_coll += int(info["collision"].sum())
            idx = torch.where(done)[0]
            if args.stateful:
                h_state = h_next.detach()
                if len(idx):
                    h_state[:, idx] = 0.0
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
        if args.ent_anneal:
            ent = 0.01 * (1 - 0.9 * it / max(args.iters - 1, 1))
        else:
            ent = args.ent
        mb = max(1, n // args.minibatches)
        pl = vl = 0.0
        for e in range(args.epochs):
            idx_all = torch.randperm(n, device=dev)
            for b in range(0, n, mb):
                sl = idx_all[b:b + mb]
                if args.stateful:
                    hc = h_init[:, sl]
                    means = []
                    hs = []
                    for st in range(args.steps):
                        out_t, hc = student.gru(HSs[st, sl].unsqueeze(1), hc)
                        means.append(student.head(out_t.squeeze(1)))
                        hs.append(hc.squeeze(0))
                        hc = hc * (1.0 - D[st:st + 1, sl, None])
                    mean = torch.stack(means).reshape(-1, args.act_dim)
                    h_last = torch.stack(hs).reshape(-1, args.hidden)
                else:
                    seq = HSs[:, sl].reshape(args.steps * len(sl), HIST, -1)
                    _, h_last = student.gru(seq)
                    h_last = h_last.squeeze(0)
                    mean = student.head(h_last)
                # LeWM/RMA aux supervision: hidden state predicts privileged
                # [pose, vel] (PR cols 0..5) + variance reg against collapse.
                aux_pred = student.aux_head(h_last)
                aux_tgt = PRs[:, sl].reshape(-1, PRs.shape[-1])[:, :6]
                loss_aux = ((aux_pred - aux_tgt).pow(2).mean()
                            + 0.01 * torch.relu(1.0 - h_last.std(dim=0)).mean())
                dist = torch.distributions.Normal(mean, logstd.exp())
                nlp = dist.log_prob(A[:, sl].reshape(-1, args.act_dim)).sum(-1)
                ratio = torch.exp(nlp - LP[:, sl].reshape(-1))
                pg1 = ratio * adv[:, sl].reshape(-1)
                pg2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv[:, sl].reshape(-1)
                loss_p = -torch.min(pg1, pg2).mean()
                loss_v = 0.5 * (critic(PRs[:, sl].reshape(-1, PRs.shape[-1]))
                                - ret[:, sl].reshape(-1)).pow(2).mean()
                loss_e = -dist.entropy().sum(-1).mean()
                if it < args.critic_warmup:
                    loss = loss_v
                else:
                    loss = loss_p + loss_v + ent * loss_e + args.aux_weight * loss_aux
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step(); opt.zero_grad()
                pl, vl, al = float(loss_p.detach()), float(loss_v.detach()), float(loss_aux.detach())
        sched.step()
        print(f"iter {it} pol {pl:.4f} val {vl:.4f} aux {al:.4f} std {float(logstd.exp().mean().detach()):.3f} succ {ep_succ} coll {ep_coll} {time.time()-t0:.1f}s", flush=True)
        if args.ckpt_every > 0 and (it + 1) % args.ckpt_every == 0:
            save_ckpt()
    save_ckpt()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
