---
id: 2026-11-17-park-search-s2
date: 2026-11-17
author: fer-claude
question: Does a search that may not trade collisions for success find parameters better than both the defaults and s1?
parent: 2026-11-17-park-search-s1
outcome: kept
failure_mode: none
refs: [runs/park_search/s2/tree.jsonl, config/park_params.json, src/ugv/park/search.py]
tags: [park, search]
---

## Tried

`tools/park/search.py --out runs/park_search/s2 --generations 5 --children 10 --n 160 --seed 21`
on the v7 code. PASS now also requires no more collisions than the parent.

## Result

51 candidates, 8 PASS; 6 more beat their parent on success but were rejected for colliding more.
Best n39 changed orbit_r 0.99, orbit_v 0.20, orbit_tol_deg 4.9, settle_s 0.23, measure_s 0.45,
step_frac 0.62, stop_tol_m 0.0054, max_retries 4. Held-out validation, the same 1000 episodes
(seed 11) as the other rows:

| params | parked | 95% CI | collisions |
|---|---|---|---|
| default (v7) | 882 | 86.1-90.1% | 7 |
| s1 best | 899 | 87.9-91.6% | 13 |
| **s2 best** (`config/park_params.json`) | **909** | **89.0-92.5%** | **7** |

Sweep (300 each): ideal 300, detector-only 296, plant-only 290, worse detector 259, worse
calibration 236, wider cars 227.

## Lesson

A collision gate costs little success (6 rejected candidates) and keeps the safety the
defaults had. Slower orbiting with a tighter line-up and shorter, more frequent measurement
pauses are what the search found.

## Re-run only if

The controller, plant or detector model changes; start from `config/park_params.json`.
