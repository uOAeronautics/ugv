---
id: 2026-11-18-park-near-field
date: 2026-11-18
author: fer-claude
question: The finale is driven blind (1.2 keypoints per frame in ENTER); would keypoints the camera can see from the bay change that?
parent: 2026-11-18-park-calib-min-points
outcome: kept
failure_mode: blind-zone
refs: [src/ugv/park/perception.py, runs/park_eval/diag-cand.json, runs/park_eval/diag-cand-near.json]
tags: [park, perception, skeleton]
---

## Tried

Diagnosis of faulty-car collisions (400 episodes, s3 n4 params + calib_min_points 4): 111/125
in ENTER, during a forward move with no keypoint seen for ~0.9 s, the car 3 cm past the target
(nose wheel) with the estimate 3-6 cm behind. The 12-point skeleton is out of view from the
bay: ENTER averages 1.2 keypoints per frame. From the parked camera the belly is at 37-63 deg
elevation (half-FOV 20.4 deg) and only the nose wheel's upper half is in view. Added three
near-field points from the mesh: the nose tyre's left, right and top extremes
(`DetectorModel.near_field`, `perception._near_points`). A/B seeds 21.

## Result

| | all faults (300) | healthy (200) |
|---|---|---|
| 12 points | 109 parked, 99 coll, 71 off-target | 175, 21 off-target |
| + 3 tyre points | 106 parked, 72 coll, 30 off-target | 200, 0 off-target |

Made the simulator's default. It is a proposal for the detector: the real one must be trained
to output the three points (task `2026-11-18-pose-near-field-keypoints`); until then the
simulated numbers assume a detector we do not have.

## Lesson

Choose keypoints for where the precision is needed, not only for a global pose: the skeleton
was designed from far away, and the 3.5 cm finale happens where it cannot be seen.

## Re-run only if

The camera height or field of view changes, or the real detector cannot find the tyre points.
