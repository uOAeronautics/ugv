"""DAgger distill: student on noisy history, labelled by the privileged teacher.

Options for the L1+ blind-spawn regime (see scout notes 2026-11-13): --fov-search
wraps executed student actions with the deploy blind-search spin reflex so
student rounds visit post-acquisition states, and --aux-weight adds LeWM/RMA
aux supervision (aux_head(h) -> priv [pose,vel]). Produces warm-start
checkpoints for train_student.py --init.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from models import RUNS, Student, load_teacher, teacher_obs
from sim import EnvConfig, MecanumVecEnv


@torch.no_grad()
def teacher_act(teacher, obs):
    return teacher(teacher_obs(obs))


def collect(teacher, student, env, episodes, noise=0.05, beta=0.0, fov_search=False):
    # Per-EPISODE mixing: a whole episode rides the teacher (w.p. beta) or the
    # student. Per-step mixing from a random init injects a yank every ~5 steps,
    # so no episode ever completes the 1 s hold and the student never sees
    # success behavior. All visited states are labelled by the teacher either way.
    # fov_search wraps the executed STUDENT actions with the deploy blind-search
    # reflex (spin toward imu_yaw = 0 after 1 s without a tag, zero translation;
    # mirrors StudentPolicy.step): on L1+ ~45% of spawns start with the tag
    # outside the camera FOV, and an unassisted student translates blindly into
    # the skids before ever visiting states near the goal.
    hists, labels, privs = [], [], []
    left = episodes
    n = env.n
    while left > 0:
        m = min(left, n)
        obs = env.reset(torch.arange(m))
        use_teacher = torch.rand(m, device=env.device) < beta
        done = torch.zeros(m, dtype=torch.bool, device=env.device)
        while not bool(done.all()):
            active = ~done
            h = obs["student"][:m]
            with torch.no_grad():
                a_s = student(h)
                a_t = teacher_act(teacher, {"pose": obs["pose"][:m], "vel": obs["vel"][:m]})
            hists.append(h[active].detach().cpu())
            labels.append(a_t[active].detach().cpu())
            privs.append(torch.cat([obs["pose"][:m], obs["vel"][:m]], dim=1)[active].detach().cpu())
            a_exec = torch.where(
                use_teacher[:, None].expand_as(a_s),
                a_t,
                (a_s + noise * torch.randn_like(a_s)).clamp(-1, 1),
            )
            if fov_search:
                age = h[:, -1, 3]
                blind = (~use_teacher) & (age >= 1.0)
                if bool(blind.any()):
                    imuw = h[:, -1, 5]
                    imuyaw = h[:, -1, 6]
                    u_turn = (-1.8 * imuyaw - 0.3 * imuw).clamp(-0.6, 0.6)
                    a_exec[blind] = torch.stack(
                        [
                            torch.zeros_like(u_turn[blind]),
                            torch.zeros_like(u_turn[blind]),
                            u_turn[blind],
                        ],
                        dim=1,
                    )
            full = torch.zeros(n, 3, device=env.device)
            full[:m] = a_exec
            obs_full, _, d, _ = env.step(full)
            obs = {
                "pose": obs_full["pose"][:m],
                "vel": obs_full["vel"][:m],
                "student": obs_full["student"][:m],
            }
            done = done | d[:m].to(env.device)
        left -= m
    return torch.cat(hists), torch.cat(labels), torch.cat(privs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--level", default="L3")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-decay", type=float, default=0.05)
    ap.add_argument(
        "--bc-iters",
        type=int,
        default=6,
        help="initial pure-teacher behavior-cloning iters (beta=1)",
    )
    ap.add_argument(
        "--buffer",
        type=int,
        default=120000,
        help="rolling buffer; large enough to retain teacher demos",
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--head-w", type=int, default=64)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--aux-weight",
        type=float,
        default=0.1,
        help="LeWM/RMA aux supervision weight: MSE(aux_head(h), priv [pose,vel])",
    )
    ap.add_argument(
        "--fov-search",
        action="store_true",
        help="wrap executed student actions with the blind-search spin reflex",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, _ = load_teacher(args.teacher, dev)
    student = Student(h=args.hidden, head_w=args.head_w).to(dev)
    print(f"student params: {student.n_params()}")
    env = MecanumVecEnv(args.envs, EnvConfig(level=args.level, device=dev, seed=args.seed))
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    buf_h, buf_y, buf_p = None, None, None
    for it in range(args.iters):
        # BC warm-start (beta=1): fill the buffer with clean teacher demos so
        # later student rounds can't evict good behavior (rolling buffer would
        # otherwise end up holding only 1 s failure prefixes).
        if it < args.bc_iters:
            beta = 1.0
        else:
            beta = max(0.8 - (it - args.bc_iters) * args.beta_decay, 0.0)
        h, y, p = collect(
            teacher, student, env, args.episodes, beta=beta, fov_search=args.fov_search
        )
        buf_h = h if buf_h is None else torch.cat([buf_h, h])[-args.buffer :]
        buf_y = y if buf_y is None else torch.cat([buf_y, y])[-args.buffer :]
        buf_p = p if buf_p is None else torch.cat([buf_p, p])[-args.buffer :]
        perm = torch.randperm(len(buf_h))[:50000]
        loss_sum, loss_n = 0.0, 0
        student.train()
        for _e in range(args.epochs):
            for b in range(0, len(perm), 2048):
                sl = perm[b : b + 2048]
                seq = buf_h[sl].to(dev)
                pred = student(seq)
                loss = (pred - buf_y[sl].to(dev)).pow(2).mean()
                if args.aux_weight > 0:
                    _, h_last = student.gru(seq)
                    aux = student.aux_head(h_last.squeeze(0))
                    loss = loss + args.aux_weight * (aux - buf_p[sl].to(dev)).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                opt.step()
                loss_sum += float(loss)
                loss_n += 1
        # validation error on fresh batch
        student.eval()
        with torch.no_grad():
            vh, vy, _ = collect(teacher, student, env, 8, noise=0.0)
            verr = (student(vh.to(dev)) - vy.to(dev)).pow(2).mean()
        print(
            f"dagger {it} mse {loss_sum / max(loss_n, 1):.4f} val {float(verr):.4f} buf {len(buf_h)}",
            flush=True,
        )
    RUNS.mkdir(exist_ok=True)
    out = args.out or str(RUNS / f"student_dagger_{args.level}.pt")
    torch.save(
        {
            "student": student.state_dict(),
            "level": args.level,
            "hidden": args.hidden,
            "head_w": args.head_w,
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
