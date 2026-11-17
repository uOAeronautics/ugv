---
id: 2026-11-17-h-0009-plant-calibration
date: 2026-11-17
author: fer-claude
question: The sim observation contract fails (joint overlap 0.0) because the sim loop is a constant 20 ms while the Pi loop is about 0.9 ms; a sim loop that draws its lag from the measured Pi distribution lifts the joint overlap above 0.5.
parent: 2026-11-17-track-explore-loop
outcome: negative
failure_mode: none
refs: [H-0009]
tags: [rl, explore, plant-calibration]
---

## Tried

The sim observation contract fails (joint overlap 0.0) because the sim loop is a constant 20 ms
while the Pi loop is about 0.9 ms; a sim loop that draws its lag from the measured Pi
distribution lifts the joint overlap above 0.5.
Prediction: A sim loop with the measured Pi lag distribution gives a joint whitened-KDE overlap
above 0.5 on the stage-park-0.45 run, against 0.0 at constant 20 ms.
Falsifier: Overlap stays below 0.1 with the measured lag, so loop lag is not the contract
failure.
Cost: Sim and CPU only, contract check on existing run data. Safety S0. The measured Pi lag is
read from existing logs, not from a new run.

## Result

Closed refuted; evidence explore/results/H-0009-contract.json at tag pre-pose-20261117.

## Lesson

Retrospective note for H-0009: 0.742 with today's tool; Pi lag lowers overlap.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
