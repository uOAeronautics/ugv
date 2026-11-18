---
id: 2026-11-18-park-fault-suite
date: 2026-11-18
author: fer-claude
question: How does the s2 controller hold up when the car and its camera are faulty, and when it starts close to the airframe on any side?
parent: 2026-11-17-park-search-s2
outcome: kept
failure_mode: plant-mismatch
refs: [src/ugv/park/faults.py, tools/park/diagnose.py, runs/park_eval/s2-faults.json, runs/park_eval/s2-close.json, runs/park_eval/diag-s2-faults.json]
tags: [park, faults, robustness]
---

## Tried

A fault model on top of the body-level plant, drawn per episode from its own RNG stream (a
fault-free run is bit-identical to before, checked on the 24-episode fingerprint): weak or
strong wheels through mecanum kinematics, battery level and sag (before the deadzone), wheel
stalls, slip, extra command delay, lost camera frames, false airframe detections, gyro scale
and bias walk, and camera faults expressed as detector error: exposure spells, motion blur,
sensor noise, a blocked image region, mount shake, a mid-run knock, lens (focal, principal
point, radial distortion) and fixed focus (defocus blur with range). Severity 1: each family on
with probability 0.5. Also `start="close"`: 0.4-1.3 m from the airframe on any side.
Params `config/park_params.json` (s2 n39), seed 11, `tools/park/eval.py --faults 1 --families`.

## Result

- Healthy cars, ring starts: 909/1000 (s2, unchanged).
- Close starts, healthy cars: 540/600 parked, 2 collisions (`s2-close`). Named awkward starts
  (40 seeds each, healthy car): nose-on 32, nose-oblique 36, beside the nose wheel 36, under a
  wingtip 38, abeam 37, tail 36. The orbit goes round the near side every time (nose-on splits
  20/20); no side-choice bug.
- All faults: 412/1000 parked, 241 collisions, 280 parked-off-target (`s2-faults`, run with the
  13 families before lens/focus were added).
- One family at a time (300 each): wheel 198, delay 215, stall 247, battery 253, occlusion 255,
  phantom 264, gyro 265, blur 266, slip 275, drops 275, exposure 275, shake 281 (healthy ~273).
- The first run of this suite had the gyro bias walk 10x too large (~3 deg/s after 30 s): its
  gyro row (197/300, 43 collisions) is invalid and was re-run.
- Diagnosis (400 faulty episodes, `diag-s2-faults.json`): collisions happen in ENTER (80/114)
  and BACKOFF (31), with the estimate 3.0 cm off the truth (median) against 3.5 cm clearance.
  Collision rate rises most with wheel faults (+24 points), lens (+15) and shake (+15).

## Lesson

Faults combine: the families one at a time cost 0-25 points each, together they cost 50 and
mostly as collisions. The finale's moves are open loop in heading, so a weak wheel curves a
straight creep into the gear; camera geometry faults bias the estimate by the clearance.

## Re-run only if

The fault model or its magnitudes change (they are guesses until measured on the car).
