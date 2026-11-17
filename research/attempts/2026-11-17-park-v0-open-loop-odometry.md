---
id: 2026-11-17-park-v0-open-loop-odometry
date: 2026-11-17
author: fer-claude
question: Does the state machine park when odometry integrates the commanded velocity through the nominal plant?
parent: 2026-11-17-track-3d-parking
outcome: negative
failure_mode: plant-mismatch
refs: [src/ugv/park]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`). Controller as in `controller.py`, estimator without any plant adaptation, perception with 6+ points only.

## Result

Ideal 96/96, detector-only 96/96, calibration-only 96/96, plant spread 0/96, full 0/96 (mostly collisions in APPROACH, off-target stops).

## Lesson

Perception was never the problem; open-loop odometry over a car whose gains and deadzones differ by tens of percent is. The blind finale needs measurements deeper in the bay and a calibrated motion model.

## Re-run only if

Never: superseded by v1-v6.
