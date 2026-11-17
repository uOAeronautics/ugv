---
id: 2026-11-17-h-0006-evaluation-validity
date: 2026-11-17
author: fer-claude
question: The Gazebo decode-rate conflict (5.2% in the 10-episode camdr batch against 98.3% in the ep0 probe) is a lighting-DR flag wiring bug in the batch runner, not a vision failure.
parent: 2026-11-17-track-explore-loop
outcome: negative
failure_mode: eval-mismatch
refs: [H-0006]
tags: [rl, explore, evaluation-validity]
---

## Tried

The Gazebo decode-rate conflict (5.2% in the 10-episode camdr batch against 98.3% in the ep0
probe) is a lighting-DR flag wiring bug in the batch runner, not a vision failure.
Prediction: After the batch and probe set the lighting-DR flag identically, the 10-episode
decode rate is at least 95%, matching the ep0 probe.
Falsifier: Decode stays below 20% after the flags match, which means the batch runner has no
flag bug and the vision gap is real.
Cost: Gazebo CPU re-run of the 10-episode camdr batch with the probe flag set the same way as
ep0. Safety S0 (sim).

## Result

Closed refuted; evidence explore/results/H-0006-decode.json at tag pre-pose-20261117.

## Lesson

Premise not re-measured before use; see the parent record's retrospective.

## Re-run only if

The 2D RL stack is restored from the tag and this premise still matters.
