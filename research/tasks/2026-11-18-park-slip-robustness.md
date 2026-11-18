---
id: 2026-11-18-park-slip-robustness
date: 2026-11-18
author: fernando
owner: fernando
state: done
updated: 2026-11-18
due: none
after: []
check: none
refs: []
tags: []
result: [2026-11-18-park-search-s8]
---

## Goal

The controller still parks on a slippery floor but loses the car to collisions when the floor is
slippery *and* sensors are bad: 60/300 parked with 60 collisions (`s6/best.json`, seed 11,
faults+slip). Search the controller parameters against clean, faulty, slippery and
slippery+faulty conditions together, starting from the current config, and keep a candidate only
if it gains success without more collisions.

## Done when

A search under `runs/park_search/s7` has a held-out 300-episode result for all four conditions,
the candidate is compared with `config/park_params.json` on the same episodes, and the attempt
record says whether it is kept.

## Log
- 2026-11-18 fernando: claim
- 2026-11-18 fernando: s6 held-out was negative; see `2026-11-18-park-search-s6`. Launched s7
  from `config/park_params.json` with clean, faults, slip and faults+slip (4 generations, 8
  children, n=100 per condition).
- 2026-11-18 fernando: done
