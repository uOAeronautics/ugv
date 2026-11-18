---
id: 2026-11-18-park-sensor-health
date: 2026-11-18
author: fer-claude
question: When a sensor or the drive breaks mid-run (camera dark, gyro stuck or zero, a wheel dead), does the car notice and stop safely instead of hitting the airframe?
parent: 2026-11-18-park-blind-gate
outcome: open
failure_mode: safety-abort
refs: [src/ugv/park/faults.py, src/ugv/park/estimator.py, src/ugv/park/controller.py]
tags: [park, faults, safety]
---

## Tried

Fault family `broken`: at a random time 2-40 s the camera stops delivering frames, the gyro
sticks at its last value or reads zero, or the front-left wheel dies. 200 episodes per row
(seeds 21), params cand2 (near-field, blind gate).
Health checks (`ParkParams`, all off by default): `gyro_check` (25 identical gyro readings =
dead or stuck, since a live MEMS gyro always shows noise; heading then comes from the
identified motor model), `camera_silent_s` (no frame at all for that long: ABORT "camera
silent"), `health_innov_m` (EMA of how far camera fixes land from odometry; above it in
APPROACH/ENTER/BACKOFF: ABORT "odometry disagrees with the camera").

## Result

| broken | before | with gyro_check, camera_silent 3 s, innov 0.05 m |
|---|---|---|
| gyro stuck | 19 parked, 21 collisions | 54 parked, 0 collisions |
| gyro zero | 8 parked, 19 collisions | 41 parked, 0 collisions |
| camera dark | 23 parked, 1 collision, 33 timeouts | 22 parked, 34 "camera silent", 0 collisions |
| wheel dead | 20 parked, 19 collisions | 20 parked, 16 collisions |

Gyro failures are now handled (the car parks on its motor model and vision); a dark camera
ends in a prompt abort. A dead wheel still collides: the car reaches the staging point ~5 cm
off-centre beside the main gear and its first strafe comes out as a turn plus a lurch.
Two dead-wheel detectors were tried and removed: a sideways-drift ratio of measured forward
moves in ENTER (never triggered before the collision) and an "unexplained yaw while
translating" monitor (healthy cars reach 0.29 rad/s at the 95th percentile from roller slip,
so it cannot separate them).

## Lesson

Flat-lined signals are cheap to detect and the fallback (model instead of sensor) works. A
dead wheel needs a drive check that runs before the bay, e.g. a deliberate short strafe and
turn test while far from the airframe, compared against the identified model.

## Re-run only if

A pre-entry drive test exists, or the wheel fault model is replaced by a measured one.
