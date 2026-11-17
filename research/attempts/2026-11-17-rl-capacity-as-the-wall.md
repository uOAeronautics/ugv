---
id: 2026-11-17-rl-capacity-as-the-wall
date: 2026-11-17
author: fer-claude
question: Teacher track (L0) - Capacity as the wall
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, teacher]
---

## Tried

`student_L0_H` (GRU-128, gamma 0.997, lam 0.99): 976/1000, 24 coll, versus GRU-64 at 991
(rl/PLAN.md State). Contradicted; a larger GRU is not a fix.

## Result

As recorded in explore/HISTORY.md section 2 (Teacher track (L0)) at tag pre-pose-20261117.

## Lesson

Capacity as the wall: see Tried; no separate cause was recorded.

## Re-run only if

The 2D RL stack is restored and the original retry condition holds.
