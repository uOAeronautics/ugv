---
id: 2026-11-17-h-0004-evaluation-validity
date: 2026-11-17
author: fer-claude
question: The L_real failure is an action-to-duty mapping or frame-sign mismatch between the sim policy frame and the Pi path, not a learning limit - the policy's -dy action gives +0.25 m progress in sensitivity_L1ft_Lreal, which points to a sign or frame error that a training change cannot fix.
parent: 2026-11-17-track-explore-loop
outcome: negative
failure_mode: eval-mismatch
refs: [H-0004]
tags: [rl, explore, evaluation-validity]
---

## Tried

The L_real failure is an action-to-duty mapping or frame-sign mismatch between the sim policy
frame and the Pi path, not a learning limit: the policy's -dy action gives +0.25 m progress in
sensitivity_L1ft_Lreal, which points to a sign or frame error that a training change cannot fix.
Prediction: A frame and sign audit of the NumPy export, the Pi action-to-duty path and the
policy frame finds at least one sign or axis convention that differs, and fixing it lifts L_real
p0 success above 0.
Falsifier: The audit finds all conventions match (export parity 8.17e-08 already holds), and the
L_real p0 score stays at 0 after the frame check, which confirms a learning limit.
Cost: Offline audit of the policy frame against the Pi path and the export, CPU only, no car, no
motor. Safety S0.

## Result

Closed refuted; evidence explore/results/H-0004-audit.md at tag pre-pose-20261117.

## Lesson

Premise not re-measured before use; see the parent record's retrospective.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
