"""Potential-based parking reward. Keeps shaping separate from sim."""
from __future__ import annotations
import torch


def progress_potential(pose: torch.Tensor) -> torch.Tensor:
    # Bounded dense shaping: metres + heading. Goal = origin, skid axis = yaw 0.
    dist = torch.sqrt(pose[:, 0].pow(2) + pose[:, 1].pow(2) + 1e-6)
    return -(dist + 0.3 * pose[:, 2].abs())


def step_reward(prev_pose, pose, act, prev_act, vel, success, first_slow,
                timeout, collision, false_park, dt=0.02):
    prog = (progress_potential(pose) - progress_potential(prev_pose)).clamp(-0.1, 0.1) * 5.0
    bonus = success.float() * 10.0
    # first-slow bonus: 50 consecutive slow steps are undiscoverable under
    # sampling noise, so the first slow moment pays +2 to seed the braking
    # gradient that still_pay then extends into the full hold.
    seed = first_slow.float() * 2.0
    # settle shaping: dense gradient toward braking at the goal. The sparse 1 s
    # hold bonus alone is never discovered by chance (50 near-zero actions in
    # a row), so stillness itself must pay per step. The pay grades outward to
    # 25 cm: without it the slot mouth offers only collision risk for ~zero
    # gain and the policy rationally hovers outside (observed twice: 0/1000).
    dist = torch.sqrt(pose[:, 0].pow(2) + pose[:, 1].pow(2) + 1e-6)
    prox = 1 - (dist / 0.25).clamp_max(1.0)
    # terminal proximity: episodes that survive 45 s near the goal (without
    # crashing) earn closeness pay. Early collisions make the value of the goal
    # region deeply negative and the policy learns to flee it (observed: full-
    # throttle escape from a perfect start); without an antidote nothing ever
    # visits the hold. Crashes are excluded: only survival near goal pays.
    spd = (vel[:, :2].norm(dim=1) / 0.02 + vel[:, 2].abs() / 0.035) / 2
    still_pay = 0.15 * prox * (1 - spd.clamp_max(1.0))
    end = timeout.float() * (1 - collision.float()) * (1 - success.float())
    prox_end = end * 5.0 * (1 - (dist / 0.5).clamp_max(1.0))
    # mouth caution: 89% of clips happen at the slot mouth (|y|<0.35) above
    # 0.3 m/s. Creeping there pays; rushing doesn't. (still_pay only covers
    # inside 25 cm, so the mouth itself had no caution gradient.)
    mouth = ((pose[:, 1].abs() < 0.35) & (pose[:, 1].abs() > 0.05)).float()
    band_speed = vel[:, :2].norm(dim=1) / 0.5
    creep = mouth * 0.05 * (1 - (vel[:, :2].norm(dim=1) / 0.15).clamp_max(1.0))
    rush = -mouth * 0.15 * band_speed.clamp_max(1.0)
    act_pen = -0.02 * prox * act.pow(2).mean(dim=1)
    rate = -0.01 * ((act - prev_act).pow(2).sum(dim=1))
    time = -0.005
    col = -5.0 * collision.float()
    fp = -5.0 * false_park.float()
    return prog + bonus + seed + still_pay + prox_end + creep + rush + act_pen + rate + time + col + fp
