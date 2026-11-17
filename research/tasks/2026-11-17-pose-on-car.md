---
id: 2026-11-17-pose-on-car
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: [2026-11-17-sift-bank-v2, 2026-11-17-keypoint-detector]
check: none
refs: []
tags: [pose, milestone]
---

## Goal

Milestone: the car estimates airframe pose from its camera at 10 Hz or more on the Pi (or the
desktop over the network), without tags, within the error the parking state machine needs.

## Done when

Live pose matches tag ground truth on the eval protocol; latency measured on the Pi.
