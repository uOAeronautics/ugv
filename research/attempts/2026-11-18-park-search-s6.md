---
id: 2026-11-18-park-search-s6
date: 2026-11-18
author: fernando
question: Does the s6 failure-guided candidate, alone or with the new health checks, beat the s2 parameters on held-out cars and on a slippery floor?
parent: 2026-11-17-park-search-s2
outcome: negative
failure_mode: underpowered
refs: [runs/park_search/s6/tree.jsonl, runs/park_search/s6/best.json, runs/park_search/final.json, config/park_params.json]
tags: [park, search, faults, floor]
---

## Tried

`s6`: 5 generations x 10 children, 200 episodes healthy and 200 faulty per candidate (seed 7),
starting from `config/park_params.json`; PASS required +2 episodes per condition and no more
collisions than the parent. Root 261/400 `[197, 64]`, 25 collisions; n6 PASS 271/400
`[196, 75]`, 20 collisions. I also combined n6 with the health checks from the broken-sensor
work (`gyro_check`, `camera_silent_s=3`, `health_innov_m=0.05`) in
`runs/park_search/final.json`.

Held-out: 300 episodes, seed 11, with `evaluate` on clean, slippery (`floor=1`), faulty
(`faults=1`) and faulty+slippery conditions; common random numbers across parameter sets.

## Result

| params | clean | slippery | faulty | faulty+slippery |
|---|---|---|---|---|
| config (s2, before blind gate) | 297, 0 col | 285, 5 col | **72**, 93 col | **63**, 87 col |
| s6 n6 | **297**, 2 col | 278, 9 col | 60, **60** col | 58, **60** col |
| n6 + health 0.05 | 294, 2 col | 257, 3 col | 25, 5 col | 15, 1 col |

The n6 collision reduction under faults is real (60 vs 93) but it costs 12 parked episodes,
and it is not free on a slippery floor (9 vs 5 collisions). The health combination is much
worse: 229 of 300 faulty runs and 243 of 300 faulty+slippery runs end in
"odometry disagrees with the camera" — a 5 cm innovation gate is tighter than ordinary fault
noise. A dark camera is still handled correctly by `camera_silent_s`.

Simulation only; these are not hardware results.

## Lesson

The s6 search was underpowered for the faulty condition: +11 episodes on 200 is inside the
paired noise, and the held-out set reverses the sign. Do not combine a search winner with
health thresholds that were validated on a different fault family. The next search must score
clean, faulty, slippery and faulty+slippery together and gate on collisions; that is task
`2026-11-18-park-slip-robustness`.

## Re-run only if

A candidate from a held-out-validated search improves success on all four conditions without
more collisions than `config/park_params.json`. The health gate needs its own threshold sweep
on fault severities below 1 before it is enabled by default.
