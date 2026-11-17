---
id: 2026-11-17-park-heading-tail
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: []
check: none
refs: []
tags: [park]
---

## Goal

Find which part of the car spread drives the final-heading failures (76 of 91 off-target
episodes in `default`; a nominal car ends within 0.5 deg at the 90th percentile). Ablate one
plant factor at a time (gains, deadzones, lags, strafe slip, speed noise) on 300 episodes, then
fix the mechanism (e.g. a final gyro-closed heading trim after the last forward move).

## Done when

An attempt under `2026-11-17-park-v7-fixed-pitch-few-points` with the factor named and the
fix scored on the same seeds.
