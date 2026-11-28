---
id: 2026-11-24-pose-network-in-challenge-loop
date: 2026-11-24
author: fer-claude
question: How well does the trained keypoint network read the challenge's rendered car-camera frames once it replaces the fabricated detector, why does it fail, and does retraining on the missing cases fix it?
parent: 2026-11-17-pose-kp-crop-v1
outcome: kept
failure_mode: synthetic-data
refs: [src/ugv/pose/detector.py, src/ugv/lab/record.py, src/ugv/pose/frame_eval.py, tools/pose/eval_frames.py, 28c46f7, src/ugv/pose/render.py]
tags: [pose, detector, challenge, evaluation]
---

## Tried

The challenge used to build its "detections" from the true pose plus noise (deleted, see the
no-stand-ins rule in `AGENTS.md`). The browser now posts the rendered 1280x720 mono frame; the
server runs the two-stage ONNX detector (`ugv.pose.detector`) and the estimator fuses its points.

Measurement: `ugv-lab --record runs/challenge_frames/diag1` saved 428 frames (healthy sensors,
the six named starts plus eight random ones, autopilot spinning in SEARCH) with true keypoints
projected from the exact pose and mount the browser rendered (overlay checked by eye). Scored
with `tools/pose/eval_frames.py` (hit = within 20 px of its own keypoint). These frames were
never trained on.

Diagnosis, each checked on these frames or in the code:

1. No negatives: `make_dataset.py` put the airframe in every training image, and visibility is a
   global pooled logit. Stage 1 alone fired on 100% of the 263 frames with no airframe in view,
   6.0 invented points each.
2. Wrong viewpoint distribution: training cameras always aimed at the airframe (at most 0.35
   half-widths off centre); the car's camera is level, 6.4 cm up, facing anywhere.
3. Brightness is not the cause: challenge frames average 148 vs 82 in training, but gamma 1.6 /
   2.2 and CLAHE at inference left frames-with-6-correct at 10-12%.

Fix tried (v2, GPU free after the `strata` engine was stopped at the owner's request):
`make_dataset.py` with 55% level car-camera views 0.25-3.5 m around the airframe facing anywhere
(20% of those facing away: empty pictures), a clean bright room background for 35% of them, and
crop-stage crops cut the way the deployed stage 1 proposes them (plus random empty crops). 24k
images each, 30 epochs, same network. Runs `runs/pose_train/kp-v2`, `runs/pose_train/kp-crop-v2`.

## Result

Two-stage detector on `runs/challenge_frames/diag1` (165 frames with the airframe, 263 without):

| | v1 (kp-v1 + kp-crop-v1) | v2 (kp-v2 + kp-crop-v2) |
|---|---|---|
| invented points per empty frame | 1.13 | 0.14 |
| empty frames with any detection | 31.9% | 14.1% |
| precision | 0.483 | 0.816 |
| recall | 0.289 | 0.356 |
| frames with 6+ correct points | 10.9% | 13.3% |
| same, under 1 m (107 frames) | 2.8% | 6.5% |
| median error of correct points | 4.7 px | 5.6 px |

Stage 1 alone: v2 fires on 27.8% of empty frames (v1: 100%). Mixed pairs: the crop stage
carries most of the precision gain (kp-v1 + kp-crop-v2: precision 0.691, 0.15 false points).

The dominant failure is truncation. 149 of the 165 frames with the airframe show it cut by the
frame edge (the challenge is mostly close range). Whole airframe in view (16 frames): 6+ correct
points on 75% of frames for both v1 and v2. Cut by the edge (149): v1 4.0%, v2 6.7%; recall
0.197 -> 0.293.

Inference 6-8 ms per frame (both stages, desktop CPU, 2 threads). In headless Chromium with
software WebGL the 1280x720 camera reached the server at only ~1.6 Hz, so headless autopilot runs
(0 parked in 3 + 1 rounds, car stuck in SEARCH) do not measure the controller; the HUD now shows
the network's frame rate.

v2 synthetic validation (its own harder generator, not comparable to v1's): kp-v2 PnP within
5 deg / 10 cm 21%, kp-crop-v2 42%.

## Lesson

Retraining on the cases that were missing (empty pictures, level unaimed views) fixed the false
alarms and doubled precision without hurting the whole-airframe case. It did not fix close
range: a heatmap + soft-argmax head cannot place a keypoint outside the image and loses
truncated views. The published answer to truncation is dense per-pixel keypoint voting (PVNet,
CVPR 2019, arXiv 1812.11788, tested on Truncation LINEMOD); this project's own answer for the
bay is the near-field tyre keypoints (task `2026-11-18-pose-near-field-keypoints`) with the
floor-constrained 2-3 point solve. More images of the old kind would not have helped: coverage,
not count, was the gap. Next: task `2026-11-24-pose-detector-v3-close-range`.

## Re-run only if

The generator, the network head or the challenge renderer changes. Re-score on
`runs/challenge_frames/diag1` (local, not in Git; re-record with `ugv-lab --record` if missing).
