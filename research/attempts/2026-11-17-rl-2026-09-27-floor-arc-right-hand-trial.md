---
id: 2026-11-17-rl-2026-09-27-floor-arc-right-hand-trial
date: 2026-11-17
author: fer-claude
question: Real car - 2026-09-27 floor arc, right-hand trial
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: safety-abort
refs: []
tags: [rl, real]
---

## Tried

Aborted by a trial-only accel check; no retry after a safety abort
(docs/archive/STATE_HISTORY.md, "2026-09-27 short floor arc tuning"). Retry only after the
operator reports the right-hand motion.

## Result

As recorded in explore/HISTORY.md section 2 (Real car) at tag pre-pose-20261117.

## Lesson

2026-09-27 floor arc, right-hand trial: see Tried; no separate cause was recorded.

## Re-run only if

Retry only after the operator reports the right-hand motion.
