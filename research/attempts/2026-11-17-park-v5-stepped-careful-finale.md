---
id: 2026-11-17-park-v5-stepped-careful-finale
date: 2026-11-17
author: fer-claude
question: Does a stepped finale (stop, measure still, one small move, calibrate the next move from the last) park robustly?
parent: 2026-11-17-park-v4-online-motor-identification
outcome: kept
failure_mode: none
refs: [src/ugv/park/controller.py]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`).

## Result

Plant 70/96, full 48/96; remaining collisions came from heading errors of 5-10 deg in the bay.

## Lesson

Measure-then-move turns an unknown plant into a repeatable one: each move is calibrated at the same speed just before the blind one.

## Re-run only if

The finale routine changes.
