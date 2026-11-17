---
id: 2026-11-17-h-0007-evaluation-validity
date: 2026-11-17
author: fer-claude
question: The Gazebo versus fake-bridge gap (0/10 against 3/10) was an artefact of FakeBridge ignoring --seed, so spawn-paired seeds remove most of it.
parent: 2026-11-17-track-explore-loop
outcome: abandoned
failure_mode: none
refs: [H-0007]
tags: [rl, explore, evaluation-validity]
---

## Tried

The Gazebo versus fake-bridge gap (0/10 against 3/10) was an artefact of FakeBridge ignoring
--seed, so spawn-paired seeds remove most of it.
Prediction: With spawn-paired seeds the fake-bridge and Gazebo arms differ by at most 1 success
in 20 episodes for student_L1_best.npz.
Falsifier: The gap stays at 2 or more successes in 20 episodes with the seed fixed, so the plant
gap is real and not an artefact of seeding.
Cost: Gazebo CPU runs, spawn-paired seeds, 20 episodes per arm. Safety S0 (sim); no GPU job.

## Result

Never run; the 2D RL track it served was retired on 2026-11-17.

## Lesson

Not tested; nothing to conclude.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
