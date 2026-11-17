---
id: 2026-11-17-keypoint-detector
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: [2026-11-17-real-pose-eval-set]
check: none
refs: []
tags: [pose, learning]
---

## Goal

Train a heatmap keypoint detector for the 11-point skeleton on synthetic views
(`src/ugv/pose/render.py`), with augmentation: random and real-photo backgrounds, camera
intrinsics jitter, blur, exposure, lens distortion, partial crops and occlusion. Pose = PnP on
detected keypoints. Baseline encoder: a small ImageNet CNN; the eval set decides.

## Done when

An attempt reporting pose error on synthetic held-out views and on the real eval set, beside
the SIFT baseline; the synthetic-to-real gap is the headline number.
