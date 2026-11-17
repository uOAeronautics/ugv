---
id: 2026-11-17-rl-forward-model-loss-and-l-field
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - Forward-model loss and L_field
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: seed-bifurcation
refs: []
tags: [rl, student]
---

## Tried

Three runs, all 0/200 on L_real p0, seed 11 (all timeouts, 0 coll):
`student_field_base_s0_eval200`, `student_Lreal_C_fwd_s0_eval200` (rl/runs/archive.jsonl,
eval_metrics), and `student_fieldB_base_s0` (board only; 0 successes even at dr_scale 0). Why: a
300-iteration warm start does not cross the real deadzone/yaw-coupling barrier. Retry condition
(stated by the board): first reproduce a nonzero L_real baseline, then A/B `--fwd-weight`
against it. The forward model is neither supported nor refuted.

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

a 300-iteration warm start does not cross the real deadzone/yaw-coupling barrier.

## Re-run only if

Retry condition (stated by the board): first reproduce a nonzero L_real baseline, then A/B
`--fwd-weight` against it. The forward model is neither supported nor refuted.
