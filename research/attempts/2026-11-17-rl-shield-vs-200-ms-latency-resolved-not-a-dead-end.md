---
id: 2026-11-17-rl-shield-vs-200-ms-latency-resolved-not-a-dead-end
date: 2026-11-17
author: fer-claude
question: Real car - Shield vs 200 ms latency (resolved, not a dead end)
parent: 2026-11-17-track-2d-rl
outcome: kept
failure_mode: loop-latency
refs: []
tags: [rl, real]
---

## Tried

Horizon 0.18 s < latency 0.20 s, fires never (board [ADVERSARY] pi-inference-and-shield); fixed
by STOP_HORIZON 0.35 and dt_stale-aware sweep (board [SCOUT] fast-shield-and-latency).

## Result

As recorded in explore/HISTORY.md section 2 (Real car) at tag pre-pose-20261117.

## Lesson

Shield vs 200 ms latency (resolved, not a dead end): see Tried; no separate cause was recorded.

## Re-run only if

The 2D RL stack is restored and the original retry condition holds.
