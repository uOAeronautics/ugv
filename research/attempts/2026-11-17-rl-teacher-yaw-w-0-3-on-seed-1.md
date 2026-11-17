---
id: 2026-11-17-rl-teacher-yaw-w-0-3-on-seed-1
date: 2026-11-17
author: fer-claude
question: Teacher track (L0) - Teacher yaw_w 0.3 on seed 1
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: seed-bifurcation
refs: []
tags: [rl, teacher]
---

## Tried

Run `teacher_L0_yaw03_s1` (103/1000, 897 coll), rl/runs/teacher_L0_yaw03_s1.json. Seed 0
(`teacher_L0_yaw03`) gives 989/1000 on the same code. Why: seed bifurcation between freeze and
clip modes; unresolved. Retry needs a change that passes on two seeds.

## Result

As recorded in explore/HISTORY.md section 2 (Teacher track (L0)) at tag pre-pose-20261117.

## Lesson

seed bifurcation between freeze and clip modes; unresolved.

## Re-run only if

Retry needs a change that passes on two seeds.
