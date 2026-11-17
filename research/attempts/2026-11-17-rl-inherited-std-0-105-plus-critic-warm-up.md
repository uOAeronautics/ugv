---
id: 2026-11-17-rl-inherited-std-0-105-plus-critic-warm-up
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - Inherited std 0.105 plus critic warm-up
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, student]
---

## Tried

`student_L1_unlocked_v1`: 0/1000, 743 coll, rl/runs/student_L1_unlocked_v1.json. The full-L1
phase destroyed the actor. Retry: do not inherit the L0 std on a warm start.

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

Inherited std 0.105 plus critic warm-up: see Tried; no separate cause was recorded.

## Re-run only if

Retry: do not inherit the L0 std on a warm start.
