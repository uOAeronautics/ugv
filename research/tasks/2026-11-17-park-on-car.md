---
id: 2026-11-17-park-on-car
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: [2026-11-17-pose-on-car, 2026-11-17-parking-state-machine, 2026-11-17-measure-gear-geometry]
check: none
refs: []
tags: [park, milestone, hardware]
---

## Goal

Milestone: the car parks under the real airframe with the state machine and model-based pose,
confirmed by the end frame. Needs the S4 gate in `docs/SAFETY.md` (human present, clear zone,
timeout) and a port of `ugv.park.controller` to the L7 harness as a `park` mode.

## Done when

N supervised runs with the end frame recorded; success rate with its interval in an attempt.
