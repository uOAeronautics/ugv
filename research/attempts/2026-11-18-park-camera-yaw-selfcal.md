---
id: 2026-11-18-park-camera-yaw-selfcal
date: 2026-11-18
author: fer-claude
question: Can the estimator self-calibrate the camera's yaw offset (mount knock, lens centre) from how well airframe samples taken at different places agree?
parent: 2026-11-18-park-heading-hold
outcome: negative
failure_mode: underpowered
refs: []
tags: [park, faults, estimator]
---

## Tried

Every fix with 4+ keypoints stored (odometry pose, raw measurement) for the last 4 s. Once the
car had moved 0.2 m, a grid over the yaw offset (current +-3 deg, 0.25 deg steps) chose the
offset whose corrected samples of the airframe position (odom frame) had the least variance;
the estimate moved 30% towards it per fix. Measurements were rotated back about the lens
before fusion. Behind a flag; A/B on seeds 21 with k_hold 3.

## Result

| | all faults (200) | healthy (150) | knock (200) | lens (200) |
|---|---|---|---|---|
| hold | 64, 58 coll | 135 | 141 | 150 |
| hold + self-cal | 34, 78 coll | 97 | 96 | 103 |

On healthy cars with a true offset of -0.2..-0.8 deg the estimate wandered between -7 and +8
deg: the sample spread is dominated by odometry drift and detector noise, not by the offset.
Code removed.

## Lesson

Sample agreement in the odometry frame cannot see a 1-3 deg camera offset under this noise.
A usable observation needs a reference the camera offset does not touch, measured while the
odometry is trustworthy: e.g. the direction of a short gyro-held straight drive against the
track of measured positions (focus of expansion), or a known straight edge at rest.

## Re-run only if

Odometry drift between fixes drops several-fold (measured wheel odometry), or a direct
reference for the camera's heading exists (calibration target, focus-of-expansion estimate).
