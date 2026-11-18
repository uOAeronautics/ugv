---
id: 2026-11-18-park-pitch-refit
date: 2026-11-18
author: fer-claude
question: When the camera mount is knocked mid-run, can the car re-fit its camera pitch from the 4-5 keypoints it sees near the airframe?
parent: 2026-11-18-park-heading-hold
outcome: kept
failure_mode: plant-mismatch
refs: [src/ugv/park/perception.py, src/ugv/park/estimator.py]
tags: [park, faults, perception]
---

## Tried

Knocked cars parked 4-6 cm short: the knock tilts the camera, the floor-constrained solve turns
a pitch error into a range error, and pitch was only re-fitted from fixes with 6+ keypoints,
which the car rarely has near the airframe (trace: true pitch 3.0 -> 0.7 deg, estimate frozen).
Now (`DetectorModel.refit_pitch`): with 4-5 points whose fit misses the gate, re-solve with pitch
free; accept if every residual passes the gate and pitch moved < 7 deg; the estimator then
moves its pitch 30% towards it. A/B seeds 21, k_hold 3.

## Result

| | all faults (200) | healthy (150) | knock (150) | lens (150) |
|---|---|---|---|---|
| hold | 64, 58 coll | 135 | 103 | 112 |
| hold + refit | 60, 57 coll | 136 | 115 | 124 |

Knock +12, lens +12, healthy and all-faults within noise. Made the default.

## Lesson

A shared nuisance parameter (pitch) needs a way back in when it changes; gate the re-fit on a
residual failure so noise alone never moves it.

## Re-run only if

The solver or the pitch model changes.
