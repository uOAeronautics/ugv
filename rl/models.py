"""Networks shared by training, evaluation and export.

Student is mirrored in NumPy by src/ugv/policy.py; export.py checks parity
against that deployed implementation.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn

from sim import STUDENT_DIM

RUNS = Path(__file__).resolve().parent / "runs"  # gitignored checkpoints/logs
TEACHER_OBS = 6    # true pose(3) + true body velocity(3)
PRIV_EXTRA = 48    # posterior-draw coefficients [16,3], critic only


class Actor(nn.Module):
    """Privileged teacher actor (train only, never ships)."""

    def __init__(self, obs_dim=TEACHER_OBS, act_dim=3, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                                 nn.Linear(w, w), nn.Tanh(), nn.Linear(w, act_dim), nn.Tanh())
        self.logstd = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, o):
        return self.net(o)


class Critic(nn.Module):
    """Asymmetric value: privileged state + dynamics draw."""

    def __init__(self, in_dim=TEACHER_OBS + PRIV_EXTRA, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, w), nn.Tanh(), nn.Linear(w, w),
                                 nn.Tanh(), nn.Linear(w, 1))

    def forward(self, o):
        return self.net(o).squeeze(-1)


class Student(nn.Module):
    """GRU history encoder + MLP head. Deploy architecture (default h=64).

    aux_head is training-only LeWM/RMA latent supervision: predicts privileged
    [pose(3), vel(3)] from the GRU hidden state. export.py ships gru+head only,
    so the .npz and src/ugv/policy.py forward are unchanged (0.0 ms Pi delta).
    """

    def __init__(self, obs_dim=STUDENT_DIM, act_dim=3, h=64, head_w=64):
        super().__init__()
        self.gru = nn.GRU(obs_dim, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, head_w), nn.Tanh(),
                                  nn.Linear(head_w, head_w),
                                  nn.Tanh(), nn.Linear(head_w, act_dim), nn.Tanh())
        self.aux_head = nn.Linear(h, 6)

    def forward(self, hist, h0=None):
        _, h = self.gru(hist, h0)
        return self.head(h.squeeze(0))

    def forward_seq(self, seq, h0=None):
        out, h = self.gru(seq, h0)
        return self.head(out), h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def teacher_obs(obs):
    return torch.cat([obs["pose"], obs["vel"]], dim=1)


def priv_input(obs, env):
    """(actor obs, critic obs) for the privileged teacher."""
    o = teacher_obs(obs)
    return o, torch.cat([o, env.coef.reshape(env.n, -1)], dim=1)


def load_teacher(path, device):
    ckpt = torch.load(path, map_location=device)
    m = Actor(obs_dim=ckpt.get("obs_dim", TEACHER_OBS),
              act_dim=ckpt.get("act_dim", 3),
              w=ckpt.get("hidden", 256)).to(device)
    m.load_state_dict(ckpt["actor"])
    m.eval()
    return m, ckpt


def load_student(path, device):
    # strict=False so pre-aux checkpoints (no aux_head keys) and post-aux
    # checkpoints both load cleanly.
    ckpt = torch.load(path, map_location=device)
    m = Student(act_dim=ckpt.get("act_dim", 3),
                h=ckpt.get("hidden", 64),
                head_w=ckpt.get("head_w", 64)).to(device)
    m.load_state_dict(ckpt["student"], strict=False)
    m.eval()
    return m, ckpt


def gae_advantages(rewards, values, dones, last_v, gamma, lam):
    """Generalized advantage estimation. `dones[t]` ends transition t, so both
    the bootstrap value and the carried advantage are zeroed with `1 - D[t]`
    (carrying with `D[t+1]` leaks post-terminal returns backwards)."""
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nxt_v = last_v if t == rewards.shape[0] - 1 else values[t + 1]
        nt = 1 - dones[t]
        delta = rewards[t] + gamma * nxt_v * nt - values[t]
        gae = delta + gamma * lam * nt * gae
        adv[t] = gae
    return adv, adv + values
