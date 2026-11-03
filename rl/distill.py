"""DAgger distill: student on noisy history, labelled by the privileged teacher.

Measured L0 result: plateaus at val MSE ~0.07 and 0% success (docs/RL_PARKING.md,
clonability floor). Kept as a warm-start source for train_student.py --init.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

from sim import MecanumVecEnv, EnvConfig
from models import RUNS, Student, load_teacher, teacher_obs


@torch.no_grad()
def teacher_act(teacher, obs):
    return teacher(teacher_obs(obs))


def collect(teacher, student, env, episodes, noise=0.05, beta=0.0):
    # Per-EPISODE mixing: a whole episode rides the teacher (w.p. beta) or the
    # student. Per-step mixing from a random init injects a yank every ~5 steps,
    # so no episode ever completes the 1 s hold and the student never sees
    # success behavior. All visited states are labelled by the teacher either way.
    hists, labels = [], []
    left = episodes
    n = env.n
    while left > 0:
        m = min(left, n)
        obs = env.reset(torch.arange(m))
        use_teacher = torch.rand(m, device=env.device) < beta
        done = torch.zeros(m, dtype=torch.bool, device=env.device)
        while not bool(done.all()):
            h = obs["student"][:m]
            with torch.no_grad():
                a_s = student(h)
                a_t = teacher_act(teacher, {"pose": obs["pose"][:m], "vel": obs["vel"][:m]})
            hists.append(h.detach().cpu()); labels.append(a_t.detach().cpu())
            a_exec = torch.where(use_teacher[:, None].expand_as(a_s), a_t,
                                 (a_s + noise * torch.randn_like(a_s)).clamp(-1, 1))
            full = torch.zeros(n, 3, device=env.device)
            full[:m] = a_exec
            obs_full, _, d, _ = env.step(full)
            obs = {"pose": obs_full["pose"][:m], "vel": obs_full["vel"][:m],
                   "student": obs_full["student"][:m]}
            done = done | d[:m].to(env.device)
        left -= m
    return torch.cat(hists), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--level", default="L3")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-decay", type=float, default=0.05)
    ap.add_argument("--bc-iters", type=int, default=6,
                    help="initial pure-teacher behavior-cloning iters (beta=1)")
    ap.add_argument("--buffer", type=int, default=120000,
                    help="rolling buffer; large enough to retain teacher demos")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, _ = load_teacher(args.teacher, dev)
    student = Student().to(dev)
    print(f"student params: {student.n_params()}")
    env = MecanumVecEnv(args.envs, EnvConfig(level=args.level, device=dev,
                                         seed=args.seed))
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    buf_h, buf_y = None, None
    for it in range(args.iters):
        # BC warm-start (beta=1): fill the buffer with clean teacher demos so
        # later student rounds can't evict good behavior (rolling buffer would
        # otherwise end up holding only 1 s failure prefixes).
        if it < args.bc_iters:
            beta = 1.0
        else:
            beta = max(0.8 - (it - args.bc_iters) * args.beta_decay, 0.0)
        h, y = collect(teacher, student, env, args.episodes, beta=beta)
        buf_h = h if buf_h is None else torch.cat([buf_h, h])[-args.buffer:]
        buf_y = y if buf_y is None else torch.cat([buf_y, y])[-args.buffer:]
        perm = torch.randperm(len(buf_h))[:50000]
        loss_sum, loss_n = 0.0, 0
        student.train()
        for e in range(args.epochs):
            for b in range(0, len(perm), 2048):
                sl = perm[b:b + 2048]
                pred = student(buf_h[sl].to(dev))
                loss = (pred - buf_y[sl].to(dev)).pow(2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 0.5); opt.step()
                loss_sum += float(loss); loss_n += 1
        # validation error on fresh batch
        student.eval()
        with torch.no_grad():
            vh, vy = collect(teacher, student, env, 8, noise=0.0)
            verr = (student(vh.to(dev)) - vy.to(dev)).pow(2).mean()
        print(f"dagger {it} mse {loss_sum/max(loss_n,1):.4f} val {float(verr):.4f} buf {len(buf_h)}", flush=True)
    RUNS.mkdir(exist_ok=True)
    out = args.out or str(RUNS / f"student_dagger_{args.level}.pt")
    torch.save({"student": student.state_dict(), "level": args.level}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
