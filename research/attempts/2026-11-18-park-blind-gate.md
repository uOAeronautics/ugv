---
id: 2026-11-18-park-blind-gate
date: 2026-11-18
author: fer-claude
question: Does refusing to move unseen far from the target stop exposure spells and lost frames from turning into collisions?
parent: 2026-11-18-park-near-field
outcome: kept
failure_mode: blind-zone
refs: [src/ugv/park/controller.py]
tags: [park, faults]
---

## Tried

With near-field points, the collision rate rose most with bad-exposure spells (+13 points).
ENTER's rule was: if the measure window saw nothing, drive the whole remaining distance blind,
even from 30 cm out. Now (`blind_max_m`, `blind_waits`): farther than blind_max_m, an empty
window means measure again; after blind_waits empty windows, back off. A/B seeds 21.

## Result

| | all faults (300) | healthy (200) |
|---|---|---|
| near-field | 106, 72 coll, 30 off | 200 |
| + blind gate 4 cm | 107, 33 coll, 4 off | 200 |
| + blind gate 2 cm | 101, 34 coll, 2 off | 200 |

Collisions halve; the remaining failures are aborts (safe). Kept at 4 cm; both are in the
search space (s5).

## Lesson

A collision costs a person walking over; an abort costs a retry. Prefer rules that turn the
first into the second, then win the successes back.

## Re-run only if

ENTER's structure changes.
