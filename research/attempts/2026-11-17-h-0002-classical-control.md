---
id: 2026-11-17-h-0002-classical-control
date: 2026-11-17
author: fer-claude
question: A classical controller (the stage P go-to that parked 4/4 on the car, plus a dead-reckoned finale from the last tag fix) tuned to the measured real plant parks in sim at least as often as the best RL student on L1 p0, where the RL student sits at 966/1000 and the EKF baseline at 989/1000 on cuda.
parent: 2026-11-17-track-explore-loop
outcome: negative
failure_mode: none
refs: [H-0002]
tags: [rl, explore, classical-control]
---

## Tried

A classical controller (the stage P go-to that parked 4/4 on the car, plus a dead-reckoned
finale from the last tag fix) tuned to the measured real plant parks in sim at least as often as
the best RL student on L1 p0, where the RL student sits at 966/1000 and the EKF baseline at
989/1000 on cuda.
Prediction: The classical controller with the real-plant deadzones scores at least 956/1000 on
L1 p0 seed 11 with 0 collisions, the RL-on-L_real adoption threshold, without any learned
weights.
Falsifier: Score below 956, or any collision on L1 p0, or the finale (last 0.40 m, valid=0)
fails on more than 1% of episodes.
Cost: CPU sim evaluation only, seed 11, 1000 episodes. Safety S0 (sim; any car run needs its own
S4 gate).

## Result

Closed refuted; evidence explore/results/H-0002-classical.json at tag pre-pose-20261117.

## Lesson

Retrospective note for H-0002: 552/1000, 375 coll.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
