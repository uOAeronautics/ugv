---
id: 2026-11-17-rl-teacher-lat-w-4-0-both-seeds
date: 2026-11-17
author: fer-claude
question: Teacher track (L0) - Teacher lat_w 4.0, both seeds
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: seed-bifurcation
refs: []
tags: [rl, teacher]
---

## Tried

Runs `teacher_L0_latw4_s0` (131/1000, 869 coll) and `teacher_L0_latw4_s1` (125/1000, 875 coll),
rl/runs/teacher_L0_latw4_s0.json and _s1.json. Both GATE FAIL. Failure mode is the slot-mouth
clip. Why: the board records the idea as dead after two seeds. Retry only with a different
mechanism, not lat_w alone.

## Result

As recorded in explore/HISTORY.md section 2 (Teacher track (L0)) at tag pre-pose-20261117.

## Lesson

the board records the idea as dead after two seeds.

## Re-run only if

Retry only with a different mechanism, not lat_w alone.
