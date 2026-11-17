---
id: 2026-11-17-rl-warm-start-from-l0-gate-ckpt-no-curriculum
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - Warm-start from L0 gate ckpt, no curriculum
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, student]
---

## Tried

`student_L1_s0`: 0/1000, 866 coll, rl/runs/student_L1_s0.json. Board: iter 0 already dead;
L0->L1 spawn, yaw, 30 Hz, 80 ms and noise all hit at once. Retry: only with a curriculum
(below).

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

Warm-start from L0 gate ckpt, no curriculum: see Tried; no separate cause was recorded.

## Re-run only if

Retry: only with a curriculum (below).
