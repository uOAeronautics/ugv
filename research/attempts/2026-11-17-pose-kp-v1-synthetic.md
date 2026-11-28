---
id: 2026-11-17-pose-kp-v1-synthetic
date: 2026-11-17
author: fer-claude
question: How precise is a Pi-sized keypoint detector trained only on domain-randomised renders, on held-out renders?
parent: 2026-11-17-pose-smoke-keypoint-pnp
outcome: negative
failure_mode: synthetic-data
refs: [28c46f7, 28c46f7]
tags: [pose, learning]
---

## Tried

`tools/pose/make_dataset.py --n 24000`: the airframe rendered at random car-height (60%) and
sphere (40%) poses, 320x180 grayscale, composited over 9,045 real car-camera frames (70%) or
noise, random/plain/fixed textures, blur, noise, gamma, occluders. `train/pose_kp.py`:
MobileNetV3-small (ImageNet, mono first conv) + stride-4 heatmap head with soft-argmax and a
visibility head, 1.18 M parameters, 30 epochs, 22k train / 2k validation, RTX 3090 (about 8 min).

## Result

Validation (synthetic, same generator): keypoint error median 2.0 px at 320x180 (8.0 px at the
camera's 1280x720), p90 8.4 px (34 px at 1280); visibility accuracy 0.90; PnP solved on 79% of
views, within 5 deg and 10 cm on 38.6%. ONNX: 10.5 ms per frame on one desktop CPU thread
(idle; 23 ms measured while the CPU was loaded by a search). Run `runs/pose_train/kp-v1`.

What that precision buys in the parking simulator (detector model matched to it: 6.8 px per
axis at 1280, 10% misses, 10% outliers; 300 episodes, seed 11): 218/300 parked (95% CI
67-77%), 20 collisions (6.7%), against 882/1000 and 0.7% with the 2 px model. Not good enough
for the car.

## Lesson

At 320x180 the detector is about 4x short of the 1-2 px budget the smoke test set
(`2026-11-17-pose-smoke-keypoint-pnp`): input resolution, not training length, is the limit (the
error curve was flat over the last 10 epochs). Next: two-stage detect-then-crop at full
resolution, and stride-2 heatmaps. Says nothing about real frames yet.

## Re-run only if

The input resolution, crop stage or data generator changes.
