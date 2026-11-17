---
id: 2026-11-17-pose-smoke-keypoint-pnp
date: 2026-11-17
author: fer-claude
question: Does PnP on the rigid keypoint skeleton recover the airframe pose from synthetic views, and how fast does it degrade with keypoint pixel noise?
parent: none
outcome: kept
failure_mode: synthetic-data
refs: [tools/pose/smoke.py, src/ugv/pose]
tags: [pose, smoke]
---

## Tried

`uv run python tools/pose/smoke.py [--views car]` (seed 11): 40 synthetic 640x360 views
(fx = 484 px, the car camera at half resolution), noise backgrounds, random roll and off-centre
aim. `generic`: distance 0.6-4 m, elevation -5..35 deg. `car`: distance 0.4-3.5 m, camera
0.06-0.12 m below the airframe origin. Ground-truth 2D keypoints of the 12-point skeleton (nose, tail, wingtips, 3 rotor hubs, 2 V-tail tips, nose wheel, 2 main
wheels)
(visible ones only) plus Gaussian pixel noise, then RANSAC EPnP + LM.

## Result

| views | keypoint noise | solved | within 5 deg and 10 cm | median rotation | median position |
|---|---|---|---|---|---|
| generic | 0 px | 35/40 | 35/40 | 0.00 deg | 0.000 m |
| generic | 1 px | 35/40 | 35/40 | 0.51 deg | 0.018 m |
| generic | 3 px | 32/40 | 20/40 | 1.87 deg | 0.056 m |
| car | 0 px | 33/40 | 33/40 | 0.00 deg | 0.000 m |
| car | 1 px | 33/40 | 33/40 | 0.49 deg | 0.014 m |
| car | 3 px | 27/40 | 18/40 | 2.52 deg | 0.075 m |

Unsolved views at 0 px have fewer than 6 visible keypoints (close or cropped views). Synthetic
only: a smoke test of the geometry and solver, not evidence about real images.

## Lesson

The skeleton is well conditioned when the keypoints are right; the whole problem is a detector
that places them within about 1 px at 640x360, plus a fallback for close views where fewer
than 6 keypoints are visible (the careful-parking range), so its error budget is set by this table.

## Re-run only if

The keypoint set, camera model or pose ranges change.
