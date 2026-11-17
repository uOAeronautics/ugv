---
id: 2026-11-17-rl-gazebo-l6-latw5-s0-at-l1
date: 2026-11-17
author: fer-claude
question: Gazebo and transfer - Gazebo L6, latw5_s0 at L1
parent: 2026-11-17-track-2d-rl
outcome: negative
failure_mode: seed-bifurcation
refs: []
tags: [rl, gazebo]
---

## Tried

`student_L1_latw5_s0_gz_L1_emu`: 0/20, 20 timeouts,
rl/runs/gazebo_l6/student_L1_latw5_s0_L1_20ep_s11.json (npz sha fe44ba93cc739f6a). L2 0/20
(`..._L2_20ep_s11.json`). Vision-in-the-loop with photometric DR: 0/10, decode 5.2%
(rl/runs/gazebo_l6/student_L1_latw5_s0_L1_camdr_10ep_s11.json). Neither decode figure is usable
(§4; H-0006 refuted). Retry: after the fake-bridge plant is aligned to `real_plant` and the
paired diagnostic is re-run with spawn-paired seeds.

## Result

As recorded in explore/HISTORY.md section 2 (Gazebo and transfer) at tag pre-pose-20261117.

## Lesson

Gazebo L6, latw5_s0 at L1: see Tried; no separate cause was recorded.

## Re-run only if

Retry: after the fake-bridge plant is aligned to `real_plant` and the paired diagnostic is re-
run with spawn-paired seeds.
