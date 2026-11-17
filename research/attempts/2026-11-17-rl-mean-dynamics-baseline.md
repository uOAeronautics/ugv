---
id: 2026-11-17-rl-mean-dynamics-baseline
date: 2026-11-17
author: fer-claude
question: Baseline (EKF + staged) - Mean-dynamics baseline
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: plant-mismatch
refs: []
tags: [rl, baseline]
---

## Tried

`baseline_ekf_staged_L0_refixed`: 533/1000, 461 timeouts, rl/runs/archive.jsonl. Plant mismatch.
Retry: not on mean dynamics (see fact above).

## Result

As recorded in explore/HISTORY.md section 2 (Baseline (EKF + staged)) at tag pre-pose-20261117.

## Lesson

Mean-dynamics baseline: see Tried; no separate cause was recorded.

## Re-run only if

Retry: not on mean dynamics (see fact above).
