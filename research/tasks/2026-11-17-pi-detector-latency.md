---
id: 2026-11-17-pi-detector-latency
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: []
check: none
refs: []
tags: [pose, hardware]
---

## Goal

Measure the keypoint detector (`runs/pose_train/<run>/model.onnx`) on the Pi 4 with ONNX
Runtime: median and p95 per frame at 320x180, 1 and 4 threads, fp32 and int8. S0 (no motor),
but it needs the Pi, which `docs/STATE.md` says is paused: a person must free it first.

## Done when

An attempt with the numbers and the run id.
