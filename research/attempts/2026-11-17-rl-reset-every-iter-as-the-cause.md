---
id: 2026-11-17-rl-reset-every-iter-as-the-cause
date: 2026-11-17
author: fer-claude
question: Teacher track (L0) - Reset-every-iter as the cause
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: plant-mismatch
refs: []
tags: [rl, teacher]
---

## Tried

Run `teacher_L0_reset_s1` (55/1000, 945 coll), rl/runs/teacher_L0_reset_s1.json. Hypothesis
(persistent rollouts cause the bifurcation) rejected; the plant is the cause. Retry: not as a
fix for the teacher.

## Result

As recorded in explore/HISTORY.md section 2 (Teacher track (L0)) at tag pre-pose-20261117.

## Lesson

Reset-every-iter as the cause: see Tried; no separate cause was recorded.

## Re-run only if

Retry: not as a fix for the teacher.
