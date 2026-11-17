---
id: 2026-11-17-rl-aux-jepa-rma-head-warm-start-curric-100
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - Aux JEPA/RMA head, warm start, curric 100
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, student]
---

## Tried

`student_L1_jepa_s0`: 0/1000, 56 coll, 944 timeouts, rl/runs/student_L1_jepa_s0.json. Collisions
fell from 866 to 56 but success stayed 0. Retry: the aux head is kept (0 ms deploy cost); retry
only with a success-side change.

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

Aux JEPA/RMA head, warm start, curric 100: see Tried; no separate cause was recorded.

## Re-run only if

Retry: the aux head is kept (0 ms deploy cost); retry only with a success-side change.
