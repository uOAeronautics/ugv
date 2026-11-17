---
id: 2026-11-17-rl-direct-l2-warm-start-from-best-l1
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - Direct L2 warm start from best L1
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, student]
---

## Tried

`student_L2_s0`: 187/1000, 338 coll, 475 timeouts, rl/runs/student_L2_s0.json. A dynamics shock
is not overcome in 250 iterations on the full 60-particle posterior. Retry: only through the
curriculum/ADR route, not a direct warm start.

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

Direct L2 warm start from best L1: see Tried; no separate cause was recorded.

## Re-run only if

Retry: only through the curriculum/ADR route, not a direct warm start.
