---
id: 2026-11-18-park-search-s8
date: 2026-11-18
author: fernando
question: After the blind gate reached the config, can the parameter search recover parked episodes on faulty and slippery conditions without more collisions?
parent: 2026-11-18-park-blind-config
outcome: negative
failure_mode: underpowered
refs: [runs/park_search/s8/tree.jsonl, runs/park_search/s8/best.json, runs/park_eval/traction-2026-11-18.json, runs/park_eval/s2-blind.json]
tags: [park, search, faults, floor]
---

## Tried

`s8`: 2 generations x 6 children, 100 episodes of each condition per candidate (clean, faulty
`faults=1`, slippery `floor=1`, faulty+slippery; seed 7), starting from the current
`config/park_params.json` after the 4 cm blind gate was added. PASS required +8 parked episodes
over the parent and no more collisions.

## Result

Root 219/400 `[97, 17, 90, 15]`, 34 collisions. The best gen-1 candidate was 216 (37 col);
the best gen-2 candidate was 218 (32 col); no candidate reached the +8 gate, so `best.json` is
the root. This is a plateau: 12 mutations moved success by at most 6 and none improved both
conditions together.

The current config on held-out 300 episodes (seed 11) is clean 297/0 collisions, slippery
285/5, faulty 60/55, faulty+slippery 55/51; the 1000-episode clean run is 979/1000 with 1
collision (`runs/park_eval/s2-blind.json`).

Simulation only; not a hardware result.

## Lesson

Once the blind gate was actually in the config, the parameter search had nothing left to win.
The faulty+slippery failures are structural: in the bay the estimator often has 2-4 keypoints,
and asymmetric wheel faults bias the odometry until a fix lands. A gain change cannot make
those keypoints more observable. The next work is a pre-entry drive check or a slip-aware
estimator, not another parameter search; the 4 cm blind gate should stay because it converts
contacts into aborts.

## Re-run only if

The estimator or the bay observability changes (near-field keypoints, a drive check, a slip
bias), or a search is seeded with a candidate that already beats 219/400 on this batch.
