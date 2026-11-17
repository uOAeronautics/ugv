---
id: 2026-11-17-rl-baseline-l3-by-tuning
date: 2026-11-17
author: fer-claude
question: Baseline (EKF + staged) - Baseline L3 by tuning
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: blind-zone
refs: []
tags: [rl, baseline]
---

## Tried

`baseline_ekf_v2_L3`: 257/1000, 351 coll, rl/runs/baseline_ekf_v2_L3.json. Cap is the under-
drone blind finale (dist < 0.3 m tags lost). Board: needs estimator/model-ID work, not more
tuning. Retry: only with that work.

## Result

As recorded in explore/HISTORY.md section 2 (Baseline (EKF + staged)) at tag pre-pose-20261117.

## Lesson

Baseline L3 by tuning: see Tried; no separate cause was recorded.

## Re-run only if

Retry: only with that work.
