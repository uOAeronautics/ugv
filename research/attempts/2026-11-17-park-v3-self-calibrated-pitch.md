---
id: 2026-11-17-park-v3-self-calibrated-pitch
date: 2026-11-17
author: fer-claude
question: Does solving camera pitch as a 4th unknown (held across frames by the estimator) restore detection?
parent: 2026-11-17-park-v2-lag-and-two-point-solve
outcome: kept
failure_mode: none
refs: [src/ugv/park/perception.py]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`).

## Result

Detection at 1-3 m went from 22-37/60 to 60/60 per distance; full 42/96.

## Lesson

The camera mount is a constant: estimate it from the data instead of trusting the nominal value.

## Re-run only if

The camera mount or solver changes.
