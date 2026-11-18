---
id: 2026-11-18-pose-camera-faults-train
date: 2026-11-18
author: fer-claude
owner: fernando
state: open
updated: 2026-11-18
due: none
after: []
check: none
refs: []
tags: [pose, faults, gpu]
---

## Goal

Retrain the keypoint detectors (kp-v1 whole frame, kp-crop-v1 crop) with the camera faults the
car meets, so a bad frame costs a little accuracy instead of a wrong pose. `train/pose_kp.py`
now applies `camera_faults` with per-image probabilities (`AUG`): brightness/contrast always,
over/under-exposure 15%, defocus blur 25% (fixed-focus lens: near or far is soft), motion blur
20% (turning and mount shake), sensor noise 50%, 1-3 blocked rectangles 30%. Geometry faults
(mount shake/knock, lens distortion, focal and principal point) are left to the solver and the
parking simulator (`faults.py`), not the detector. One GPU job at a time.

## Done when

Both detectors retrained (same data, same epochs) and scored on the clean synthetic validation
set and on a faulted copy of it; an attempt records both against kp-v1 / kp-crop-v1, and the
parking simulator's `DetectorModel` is updated from the faulted numbers. Real-frame numbers
come only with `2026-11-17-real-pose-eval-set`.
