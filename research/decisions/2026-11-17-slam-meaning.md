---
id: 2026-11-17-slam-meaning
date: 2026-11-17
author: fer-claude
status: active
supersedes: []
refs: [docs/POSE.md]
tags: [slam, pose, question]
---

## Decision

The supervisor's "SLAM" is simultaneous localization and mapping in the usual sense
(https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping), confirmed by Fernando on
2026-11-17. For this project it means: the car builds a map of the room while localising in
it, and the airframe is one landmark in that map whose geometry is known exactly. Model-based
airframe pose is the measurement that ties the map to the airframe; SLAM keeps the car
localised when the airframe leaves the camera (behind the car, under the belly, occluded) and
gives a drift-free frame for the orbit and approach.

Order: airframe pose first (it is needed either way), then a SLAM layer. The feature-matching
work (SIFT or learned features on the airframe) doubles as the front end of a feature-based
SLAM (ORB-SLAM style), so it is not wasted. On a Pi 4 a full visual SLAM is heavy; the cheap
first version is pose-graph odometry: wheel + gyro dead reckoning, corrected by every airframe
pose fix, with room features (already tracked in `src/ugv/parking.py` as the tag-loss bridge)
as extra landmarks.

## Why

The parking failures on the car were losses of the target (tag out of view during strafe and
under the drone). Mapping the room solves exactly that: the target can be out of view while the
car still knows where it is relative to it.

## Revisit if

The airframe stays in view for the whole manoeuvre in practice (then odometry between fixes is
enough and SLAM is not needed).
