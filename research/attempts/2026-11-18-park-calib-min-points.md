---
id: 2026-11-18-park-calib-min-points
date: 2026-11-18
author: fer-claude
question: Does restricting online calibration (motor gains, gyro bias) to well-observed fixes stop the heading drift under the belly?
parent: 2026-11-18-park-pitch-refit
outcome: open
failure_mode: plant-mismatch
refs: [src/ugv/park/estimator.py]
tags: [park, faults, estimator]
---

## Tried

Delay-faulted cars parked 5-10 deg off: trace (delay +40 ms) shows the gyro-bias estimate drift
to -0.006 rad/s while fixes had 3-5 keypoints (yaw weakly observed), then 10 s of blind ENTER
turned it into -5 deg of heading error with the car standing still. `ParkParams.calib_min_points`
skips calibration on fixes with fewer keypoints. A/B seeds 21, k_hold 3 + pitch refit.

## Result

| | all faults (200) | healthy (150) | delay (150) | gyro (150) |
|---|---|---|---|---|
| 0 (before) | 60, 57 coll, 62 off | 136 | 105 | 130 |
| 6 | 53, 67 coll | 126 | 94 | 121 |
| 4 | 74, 70 coll, 45 off | 136 | 106 | 133 |

6 starves the calibration; 4 cuts off-target but collisions rise within noise. Left to the
search (`calib_min_points` in SPACE, s4 starts at 4).

## Lesson

Calibration needs observability gates, but too strict a gate starves it; tune the threshold
with the collision gate rather than by hand.

## Re-run only if

The calibration law changes.
