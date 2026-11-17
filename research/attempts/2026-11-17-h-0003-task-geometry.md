---
id: 2026-11-17-h-0003-task-geometry
date: 2026-11-17
author: fer-claude
question: A hybrid policy (RL student above 0.40 m, classical finale below it) beats RL alone on L_real p0, because the blind zone where valid=0 zeros the tag is the open cause of the L_real failures.
parent: 2026-11-17-track-explore-loop
outcome: abandoned
failure_mode: none
refs: [H-0003]
tags: [rl, explore, task-geometry]
---

## Tried

A hybrid policy (RL student above 0.40 m, classical finale below it) beats RL alone on L_real
p0, because the blind zone where valid=0 zeros the tag is the open cause of the L_real failures.
Prediction: The hybrid reaches at least 10 points more success than the RL-alone L_real p0 run
(0/200 to 318/1000 range) with collisions at the handover under 0.5%.
Falsifier: Hybrid gains under 10 points, or handover collisions reach 0.5% or more, or the
blind-zone entry rate is unchanged.
Cost: CPU sim evaluation on L_real, seed 11. Safety S0 (sim).

## Result

Never run; the 2D RL track it served was retired on 2026-11-17.

## Lesson

Not tested; nothing to conclude.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
