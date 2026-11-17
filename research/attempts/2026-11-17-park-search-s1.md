---
id: 2026-11-17-park-search-s1
date: 2026-11-17
author: fer-claude
question: Can a failure-guided search over the 24 controller parameters improve on v6?
parent: 2026-11-17-park-v6-stepped-finale
outcome: kept
failure_mode: none
refs: [runs/park_search/s1/tree.jsonl, src/ugv/park/search.py]
tags: [park, search]
---

## Tried

`tools/park/search.py --out runs/park_search/s1`: 6 generations x 10 children, 128 common episodes (seed 7) each; each child changes 3 parameters weighted towards the levers of the parent's failure modes; PASS = beats the parent by 2+ episodes.

## Result

Root 111/128 on the search episodes; gen 1 best n5 118/128; gen 5 best n41 121/128; 60
candidates, 2 accepted (Search tree page). The search ran on the v6 code.

Held-out validation (1000 episodes, seed 11, current v7 code), paired on the same episodes:

| params | parked | 95% CI | collisions |
|---|---|---|---|
| default (v7) | 882 | 86.1-90.1% | 7 |
| s1 best (`config/park_params.json`) | 899 | 87.9-91.6% | 13 |

The tuned set parks 17 more episodes and wins every sweep condition (worse detector 261 vs 232
of 300, wider cars 247 vs 202, v6-default sweep), but collides 13 times against 7. The search
objective counts a collision like any other failure.

## Lesson

Parameter tuning helps a little and saturates early; and an objective that treats a collision
like a timeout will trade safety for success. The next search should score collisions as
worse than any other failure (or gate on them) before any parameters go near the car.

## Re-run only if

The controller structure changes, then search from the new default.
