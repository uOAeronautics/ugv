---
id: 2026-11-17-rl-l-real-at-full-dr-scale-1-0
date: 2026-11-17
author: fer-claude
question: Student track (L1 and up) - L_real at full dr_scale 1.0
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: blind-zone
refs: []
tags: [rl, student]
---

## Tried

Board [EXP] student_Lreal_adr_v4_s0 row: 0/1000, 45 coll, 955 timeouts (the source gives no run
id for this result). The 0.40 m blind approach zone where `valid=0` zeros the tag is the open
cause.

## Result

As recorded in explore/HISTORY.md section 2 (Student track (L1 and up)) at tag pre-
pose-20261117.

## Lesson

L_real at full dr_scale 1.0: see Tried; no separate cause was recorded.

## Re-run only if

The 2D RL stack is restored and the original retry condition holds.
