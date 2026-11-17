---
id: 2026-11-17-park-search-s1
date: 2026-11-17
author: fer-claude
question: Can a failure-guided search over the 24 controller parameters improve on v6?
parent: 2026-11-17-park-v6-stepped-finale
outcome: open
failure_mode: none
refs: [runs/park_search/s1/tree.jsonl, src/ugv/park/search.py]
tags: [park, search]
---

## Tried

`tools/park/search.py --out runs/park_search/s1`: 6 generations x 10 children, 128 common episodes (seed 7) each; each child changes 3 parameters weighted towards the levers of the parent's failure modes; PASS = beats the parent by 2+ episodes.

## Result

Root 111/128; gen 1 best n5 (k_radial 1.30, step_v 0.079, measure_s 0.51) 118/128; no further PASS through gen 5 (see the Search tree page). Validation on held-out seeds pending.

## Lesson

Parameter tuning saturates early; the remaining failures need structural changes (heading at the end), which is where the next attempt should go.

## Re-run only if

The controller structure changes, then search from the new default.
