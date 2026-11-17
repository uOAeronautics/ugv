---
id: 2026-11-17-park-v1-planar-solve-and-speed-scale
date: 2026-11-17
author: fer-claude
question: Do a floor-constrained solve (3+ points with a prior) and a per-axis odometry speed scale learned from fixes fix the plant gap?
parent: 2026-11-17-park-v0-open-loop-odometry
outcome: negative
failure_mode: no-op-step
refs: [src/ugv/park/perception.py]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`). Planar solve added; scale ratio of measured to odometry displacement between fixes.

## Result

Ideal fell to 50/96 (46 parked short by about 2 cm): the scale learned >1 because odometry ignored the motor lag. Fixes still stopped at x=-0.34 m: only nose and nose wheel are visible in the bay.

## Lesson

A learned correction absorbs whatever the model leaves out (here, lag) and then misapplies it; model the lag explicitly. Two keypoints on the centreline fix x and y but not yaw.

## Re-run only if

Never.
