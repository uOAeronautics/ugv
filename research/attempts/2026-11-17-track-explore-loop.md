---
id: 2026-11-17-track-explore-loop
date: 2026-11-17
author: fer-claude
question: Ranked offline hypotheses (explore/ loop) to localise the sim-to-real gap of the 2D RL stack.
parent: none
outcome: abandoned
failure_mode: none
refs: []
tags: [rl, explore]
---

## Tried

Root of a track; children are the individual runs.

## Result

Ten hypotheses H-0001..H-0010 proposed, six closed on 2026-11-17 (archive: research/explore-
archive.jsonl; results under explore/results at tag pre-pose-20261117).
Retrospective (verbatim): We assumed pre-fix reference numbers still hold and never re-measured
them: the contract-overlap 0.0 premise (H-0009: 0.742 with today's tool; Pi lag lowers overlap),
the CPU-vs-cuda baseline spread (H-0008: CPU repeat 986/1000 lands within 10 of cuda 989 but 27
away from the board's 959; the 959 figure has no file behind it), and the Gazebo camdr probe
figure 98.3% (unverifiable in the tree; both re-runs decode ~46-48%). We also assumed transfer:
the field 4/4 stage go-to was expected to park in sim (H-0002: 552/1000, 375 coll) and the plant
fit was expected to discriminate (H-0001: forward/yaw residuals identical between fitted and
nominal plants; strafe fit is degenerate; the saturation failure reproduced anyway). Re-measure
the reference on the current tree before building on it; treat field success as a hypothesis,
not a fact.

## Lesson

Re-measure reference numbers before building on them: three premises (contract overlap 0.0, the
959 baseline, the 98.3% decode) had no file behind them.

## Re-run only if

A decision in research/decisions reopens the track.
