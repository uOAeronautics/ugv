---
id: 2026-11-17-park-v2-lag-and-two-point-solve
date: 2026-11-17
author: fer-claude
question: With the nominal motor lag in odometry and a 2-point x,y solve, does the plant gap close?
parent: 2026-11-17-park-v1-planar-solve-and-speed-scale
outcome: negative
failure_mode: plant-mismatch
refs: [src/ugv/park]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`).

## Result

Ideal 96/96, plant 25/96, full 4/96: timeouts in SEARCH (no detection) and APPROACH (stuck).

## Lesson

Under full noise detection failed half the time: camera pitch error (1.5 deg moves points about 25 px) breaks a solver that assumes a level camera.

## Re-run only if

Never.
