---
id: 2026-11-18-park-blind-config
date: 2026-11-18
author: fernando
question: Did the kept blind gate make it into the scored config, and what does it cost on the current code?
parent: 2026-11-18-park-blind-gate
outcome: kept
failure_mode: blind-zone
refs: [config/park_params.json, src/ugv/park/controller.py]
tags: [park, faults, floor]
---

## Tried

`2026-11-18-park-blind-gate` recorded `blind_max_m=0.04` and `blind_waits=3` as kept, but
`config/park_params.json` never carried them, so every scored run used the defaults (1.0 m, 3
windows): ENTER would drive the whole remaining distance blind. I added the two values to the
config and re-scored 300 episodes at seed 11, clean, slippery (`floor=1`), faulty
(`faults=1`) and faulty+slippery.

I also tried a mid-move guard (`blind_move_s=0.5`: stop an ENTER move when the last fix is
older than 0.5 s). It is not kept.

## Result

| condition | before (default 1.0) | after (0.04 m) | mid-move guard |
|---|---|---|---|
| clean | 297, 0 col | 297, 0 col | – |
| slippery | 285, 5 col | 285, 5 col | – |
| faulty | 72, 93 col | 60, 55 col | 46, 65 col |
| faulty+slippery | 63, 87 col | 55, 51 col | 48, 54 col |

The 4 cm gate turns 38 faulty collisions into aborts and costs 12 parked episodes; that is the
intended trade (an abort costs a retry, a collision costs a person walking over). The mid-move
guard looked reasonable on one collision trace but lost on the batch (more collisions and fewer
parks), so it was removed rather than kept behind a flag.

## Lesson

A mechanism is not kept until the scored config carries it. The blind gate was written down as
kept but stayed at its unsafe default for several later runs; the config is part of the
artifact, not a note. The remaining faulty+slippery failures are mostly timeouts and
misalignment aborts, not contacts.

## Re-run only if

A held-out search recovers parked episodes on the faulty and faulty+slippery conditions without
more collisions than this config. That search is task `2026-11-18-park-slip-robustness`.
