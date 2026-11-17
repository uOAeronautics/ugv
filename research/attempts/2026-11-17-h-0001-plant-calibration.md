---
id: 2026-11-17-h-0001-plant-calibration
date: 2026-11-17
author: fer-claude
question: Fitting the sim plant to the measured real-car deadzones (forward 0.05, strafe 0.13, yaw 0.10) and gyro noise (0.63 rad/s) reduces the sim-to-real gap enough that the L1 p0 failure modes in the sim match the car's (oscillation, turn saturation).
parent: 2026-11-17-track-explore-loop
outcome: kept
failure_mode: none
refs: [H-0001]
tags: [rl, explore, plant-calibration]
---

## Tried

Fitting the sim plant to the measured real-car deadzones (forward 0.05, strafe 0.13, yaw 0.10)
and gyro noise (0.63 rad/s) reduces the sim-to-real gap enough that the L1 p0 failure modes in
the sim match the car's (oscillation, turn saturation).
Prediction: After the sim plant takes the measured deadzones (forward 0.05, strafe 0.13, yaw
0.10) and gyro std 0.63 rad/s, the L1 p0 failures in sim show the car's pattern: turn saturation
|a2|>0.8 on at least 50% of ticks, and the joint observation overlap moves above 0.5 from 0.0.
Falsifier: Turn saturation stays under 10% of ticks, or the L1 p0 score moves less than the
1000-episode noise band (about 1.5 points), or the fit residual on the probe-axes runs is no
lower than with the current sim plant.
Cost: Sim and CPU only, no car. Safety S0 (no motor command; the car needs a separate S4 gate
later).

## Result

Closed confirmed; evidence explore/results/H-0001-plant.json at tag pre-pose-20261117.

## Lesson

Retrospective note for H-0001: forward/yaw residuals identical between fitted and nominal
plants; strafe fit is degenerate; the saturation failure reproduced anyway.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
