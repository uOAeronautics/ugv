---
id: 2026-11-17-h-0008-evaluation-validity
date: 2026-11-17
author: fer-claude
question: The L1 baseline spread (cuda 989 with 9 collisions, CPU 959 with 19, a summary quote of 973) is RNG-ensemble noise plus an unrecorded dynamics flag, and a repeat with use_mean_dynamics and device recorded gives one number within 10 episodes.
parent: 2026-11-17-track-explore-loop
outcome: abandoned
failure_mode: none
refs: [H-0008]
tags: [rl, explore, evaluation-validity]
---

## Tried

The L1 baseline spread (cuda 989 with 9 collisions, CPU 959 with 19, a summary quote of 973) is
RNG-ensemble noise plus an unrecorded dynamics flag, and a repeat with use_mean_dynamics and
device recorded gives one number within 10 episodes.
Prediction: Repeats with the same seeds, the dynamics flag and the device recorded, on CPU and
cuda, agree to within 10 episodes of 1000.
Falsifier: The CPU and cuda scores differ by more than 10 episodes after the flags match, which
means the device or the code path changes the outcome.
Cost: One CPU and one cuda repeat of baseline_ekf_v2_L1. The cuda run is a GPU job: schedule it
with the GPU owner. Safety S0 (sim).

## Result

Closed inconclusive; evidence explore/results/H-0008-cpu.json at tag pre-pose-20261117.

## Lesson

Retrospective note for H-0008: CPU repeat 986/1000 lands within 10 of cuda 989 but 27 away from
the board's 959; the 959 figure has no file behind it.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
