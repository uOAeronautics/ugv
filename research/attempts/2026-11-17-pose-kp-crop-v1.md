---
id: 2026-11-17-pose-kp-crop-v1
date: 2026-11-17
author: fer-claude
question: Does a second stage on a zoomed crop (256x256 of the full 1280x720 camera) close the precision gap of kp-v1?
parent: 2026-11-17-pose-kp-v1-synthetic
outcome: kept
failure_mode: synthetic-data
refs: [train/pose_kp.py, tools/pose/make_dataset.py]
tags: [pose, learning]
---

## Tried

`tools/pose/make_dataset.py --crop --n 24000`: each sample is the square crop a first stage
would propose (margin 1.1-1.6x, jittered), rendered with the crop's own intrinsics (a crop of
a pinhole image is a pinhole image), over the same crop of a real frame. Same network and
training as kp-v1 (30 epochs, 1.18 M parameters). Run `runs/pose_train/kp-crop-v1`.

## Result

Synthetic validation: keypoint error median 3.4 px at 1280x720 (kp-v1: 8.0), p90 14.5 px
(34); visibility 0.93 (0.90); PnP within 5 deg and 10 cm 69% (39%). ONNX on one idle desktop
CPU thread: 5.7 ms at 256x256.

Parking simulator with `config/park_params.json` and a matched detector model (2.9 px per
axis, 7.5% misses, 6% outliers), 300 episodes, seed 11: 249 parked (95% CI 78-87%), 9
collisions (3%). The kp-v1-matched model on the same episodes: 236 parked, 22 collisions (7.3%).
The 2 px model: 909/1000, 0.7% collisions.

## Lesson

Resolution was the limit: zooming halves the keypoint error, and in the parking simulator
detector precision shows up first as collisions, not as lower success. The crop assumes a good
first-stage proposal; the two stages have not been run end to end yet.

## Re-run only if

The crop policy, network or data generator changes, or the two stages are chained.
