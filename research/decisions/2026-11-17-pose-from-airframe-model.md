---
id: 2026-11-17-pose-from-airframe-model
date: 2026-11-17
author: fer-claude
status: active
supersedes: []
refs: [simulation/gazebo/strl_001]
tags: [pose, perception]
---

## Decision

The next direction is 6-DoF pose of the STRL-001 airframe from one car-camera image, using the
final 3D model (`simulation/gazebo/strl_001/meshes/`) and synthetic training data. The
airframe is a rigid body (control surfaces held at zero), so pose is a rigid skeleton of named
keypoints (wingtips, nose, tail, rotor hubs, landing gear) solved with PnP, plus model-based
feature matching. ArUco tags are kept only as ground truth for evaluation, not as the
long-term localiser. Plan and experiment list: `docs/POSE.md`.

## Why

Supervisor, 2026-11-17: tags are unreliable in the long run (loss under the drone, glare,
range); the airframe never changes, so its own geometry and texture can be matched (SIFT or
learned features). A model gives pose at any distance the airframe is visible, which is what
the parking approach needs (orbit while facing the airframe, then align with the bay).

## Revisit if

Real frames show the airframe too small or occluded at the parking distances for any
model-based method; then a fixed marker field on the floor is the fallback.
