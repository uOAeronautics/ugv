---
id: 2026-11-17-rl-oracle-hot-entry
date: 2026-11-17
author: fer-claude
question: Retired - Oracle hot entry
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: none
refs: []
tags: [rl, retired]
---

## Tried

`oracle_pd_L1`: 391/1000, 596 coll, med 4.9 s (rl/runs/archive.jsonl). Fast entries overshoot
into skids (docs/archive/FAILURE_REPLAY.md, C6). Modifier only; no separate reset bin.

## Result

As recorded in explore/HISTORY.md section 2 (Retired) at tag pre-pose-20261117.

## Lesson

Oracle hot entry: see Tried; no separate cause was recorded.

## Re-run only if

The 2D RL stack is restored and the original retry condition holds.
