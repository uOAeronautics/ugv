"""Potential-based parking reward. Keeps shaping separate from sim.

Phase-0 baseline (post user side-study): progress + success bonus + near-box
still pay. Yaw weight raised 0.3 -> 1.0: failures cluster at ~5.8 deg yaw while
5 deg is worth ~0.03 in the old potential, so yaw never mattered. Rate weight
is a parameter (winning L0-gate run used 0.0). Removed extras (zone grading,
first-slow seed, terminal proximity, mouth creep/rush) showed no effect in
single-seed runs and are gone; re-propose with 3-seed evidence if needed.
"""

from __future__ import annotations

import torch


def progress_potential(pose: torch.Tensor, lat_w: float = 2.0, yaw_w: float = 1.0) -> torch.Tensor:
    # Bounded dense shaping: metres + heading. Goal = origin, skid axis = yaw 0.
    # Lateral error priced extra (default 3x total): the slot allows +-47 mm/side
    # and policies otherwise tolerate the coupling-driven +x drift until the
    # mouth clips it (s1 failures: +52 mm mean offset by step 20).
    # Pre-mouth corridor funnel (active only when y > 0.15 m and |x| > 0.032 m):
    # while laterally misaligned with the skid slot (|x| >= 0.047 m clips at
    # y = 0.2624 m), forward progress below y = 0.42 m is gated in the
    # potential and approaching the skid tips without centering x is penalized.
    x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    in_front = torch.clamp((y - 0.15) / 0.10, 0.0, 1.0)
    outside = torch.clamp((x.abs() - 0.032) / 0.015, 0.0, 1.0) * in_front
    y_eff = y + outside * torch.relu(0.42 - y)
    dist = torch.sqrt(x.pow(2) + y_eff.pow(2) + 1e-6)
    mouth_pen = (
        3.0 * torch.relu(x.abs() - 0.032) * torch.clamp((0.65 - y) / 0.35, 0.0, 1.0) * in_front
    )
    return -(dist + yaw_w * yaw.abs() + lat_w * x.abs() + mouth_pen)


def step_reward(
    prev_pose,
    pose,
    act,
    prev_act,
    vel,
    success,
    near,
    collision,
    false_park,
    dt=0.02,
    rate_w=0.01,
    lat_w=2.0,
    yaw_w=1.0,
):
    prog = (
        progress_potential(pose, lat_w, yaw_w) - progress_potential(prev_pose, lat_w, yaw_w)
    ).clamp(-0.1, 0.1) * 5.0
    bonus = success.float() * 10.0
    # settle shaping: dense gradient toward braking at the goal. The sparse 1 s
    # hold bonus alone is never discovered by chance (50 near-zero actions in
    # a row), so stillness itself must pay per step.
    spd = (vel[:, :2].norm(dim=1) / 0.02 + vel[:, 2].abs() / 0.035) / 2
    still_pay = near.float() * 0.1 * (1 - spd.clamp_max(1.0))
    act_pen = -0.02 * near.float() * act.pow(2).mean(dim=1)
    rate = -rate_w * ((act - prev_act).pow(2).sum(dim=1))
    time = -0.005
    col = -5.0 * collision.float()
    fp = -5.0 * false_park.float()
    return prog + bonus + still_pay + act_pen + rate + time + col + fp
