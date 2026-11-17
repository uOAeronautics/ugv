---
id: 2026-11-17-rl-student-l1-policy-on-car
date: 2026-11-17
author: fer-claude
question: Real car - Student L1 policy on car
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: plant-mismatch
refs: []
tags: [rl, real]
---

## Tried

`student_L1_best.npz`: 0/3 parked. Runs `20261113-182150-policy-ep1-front` (timeout, oscillation
~1.3 m), `20261113-182645-policy-ep2-mean` (lost tag after strafe-induced yaw),
`20261113-183857-policy-ep3-p0` (46 s oscillation, 0.57->1.10 m). Turn saturation |a2|>0.8 on
74% of ep3 ticks; 0% when gyro is zeroed in replay (docs/L7_FIELD_NOTES.md). Retry condition:
sim gyro noise and identified real plant in the model, the joint contract passing, and explicit
operator authorisation (RL_PARKING gate, PLAN Phase 5).

## Result

As recorded in explore/HISTORY.md section 2 (Real car) at tag pre-pose-20261117.

## Lesson

Student L1 policy on car: see Tried; no separate cause was recorded.

## Re-run only if

Retry condition: sim gyro noise and identified real plant in the model, the joint contract
passing, and explicit operator authorisation (RL_PARKING gate, PLAN Phase 5).
