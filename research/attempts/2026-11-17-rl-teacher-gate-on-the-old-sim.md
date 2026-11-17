---
id: 2026-11-17-rl-teacher-gate-on-the-old-sim
date: 2026-11-17
author: fer-claude
question: Teacher track (L0) - Teacher gate on the old sim
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, teacher]
---

## Tried

`teacher_L0_gate` shows 1000/1000 in rl/runs/archive.jsonl, but that was on the pre-fix sim. On
the fixed sim the same recipe gave `teacher_L0_refixed` 848/1000 (152 coll). The "teacher
1000/1000" figure must not be cited as current.

## Result

As recorded in explore/HISTORY.md section 2 (Teacher track (L0)) at tag pre-pose-20261117.

## Lesson

Teacher gate on the old sim: see Tried; no separate cause was recorded.

## Re-run only if

The 2D RL stack is restored and the original retry condition holds.
