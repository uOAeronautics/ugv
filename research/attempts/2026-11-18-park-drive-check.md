---
id: 2026-11-18-park-drive-check
date: 2026-11-18
author: fernando
question: Can a short deliberate strafe before the bay catch a dead wheel, without costing the clean run?
parent: 2026-11-18-park-sensor-health
outcome: negative
failure_mode: plant-mismatch
refs: [src/ugv/park/controller.py, src/ugv/park/estimator.py, runs/park_eval/s2-floor.json, runs/park_eval/traction-2026-11-18-floor.json]
tags: [park, faults, safety, enter]
---

## Tried

A drive probe: the car strafes 10 cm across the line at 0.14 m/s, stops, and the estimator
compares the raw camera fixes across the leg. A dead mecanum wheel cannot strafe — the other
three turn part of it into forward motion (a quarter, in the ideal mixing) — so the measured
forward/strafe slope exposes it. The slope is a Theil-Sen median over the fixes, so one bad fix
cannot decide it. Two placements, both at the config's parameters, paired seeds:

- after ORBIT, 1 m out, diagnosis before APPROACH;
- at the staging point (APPROACH → CHECK → ENTER), 0.35 m out, out-and-back so the car ends on
  the line, diagnosis after the approach.

The verdict was to abort when the slope exceeds 0.28. That threshold came from the measured
distributions at 0.4-1.3 m: healthy p50 0.027, p99 0.105, max 0.116 (n=118); dead wheel p50
0.370, 100% above 0.2.

## Result

| batch | off | check after orbit | check at staging |
|---|---|---|---|
| clean 1000 | 980 parked, 0 col | 980, 3 col | – |
| clean 300 | 297, 0 col | 298-299, 1-2 col | 295, 2 col |
| faulty 300 (seed 21) | 61, 61 col | 56, 41 col | 57, 50 col |
| wheel-only 200 | 184, 4 col | 180, 5-6 col | 181, 4 col |
| dead FL, forced, 100 | 0, 51 col | 0, 3 col | 0, 50 col |

The check detects the fault: a permanently dead front-left wheel goes from 51 collisions to 3
after the orbit. But it is not free. Early in the investigation a healthy car's few-point fix
(prior yaw, position re-solved) put the healthy tail at 0.34, one false abort per 300. Worse, the
strafe itself perturbs the bay entry: replaying the three clean collisions with the check off
parks all three. One was the check leaving a 5 mm lateral error, which made ENTER issue a 5 mm
strafe — shorter than the distance the car covers while the motors spin up — so the
achieved/commanded ratio collapsed to 0.45 and the next strafe doubled into the main gear.

At the staging point the check is too late: a dead wheel usually collides during the approach,
before the car ever reaches the staging point (51 → 50 collisions).

## Kept by-product

The 5 mm move above exposed a real bug in ENTER: it calibrated `_k` from moves of 1 cm or more,
which is less than `step_v` times the motor lag (0.08 m/s × 0.2 s ≈ 1.6 cm), so tiny moves
taught the controller a wrong gain. The floor is now 2 cm (`controller.py`, `_enter`). On the
current config that alone gives:

| condition (300, seed 11) | before | after |
|---|---|---|
| clean | 297, 0 col | 297, 0 col |
| slippery | 285, 5 col | 287, 3 col |
| faulty | 60, 55 col | 61, 49 col |
| faulty + slippery | 55, 51 col | 62, 49 col |

and the 1000-episode clean run is 980/1000 with 0 collisions (`runs/park_eval/s2-floor.json`),
against 979/1000 with 1 before. The drive check is not in `config/park_params.json`.

## Lesson

A deliberate probe is not a free measurement in a controller that calibrates itself from the
moves it makes: the probe changes the pose the bay entry starts from, and ENTER's boundary
behaviour is sensitive to millimetres, so the safety net creates the contacts it exists to
prevent. The coupling has to be identified from motion the car already makes — the orbit and the
approach command known mixed velocities with the airframe in view — which adds no geometry
error and no time. That regression is the next attempt.

## Re-run only if

The coupling is identified from existing motion (no deliberate strafe), or ENTER is made
insensitive to a few millimetres of entry offset (a confirmed alignment band wider than the
probe's residual). A check placed before the bay that moves the car is closed.
