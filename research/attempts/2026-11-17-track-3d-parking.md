---
id: 2026-11-17-track-3d-parking
date: 2026-11-17
author: fer-claude
question: Can a mechanistic controller, using model-based airframe pose, park the car under STRL-001 in a simulator built from the mesh and the measured plant?
parent: none
outcome: open
failure_mode: none
refs: [docs/PARK_SIM.md, src/ugv/park]
tags: [park, root]
---

## Tried

Root of the 3D parking track (decision 2026-11-17-pose-from-airframe-model). Children are the successive controller and estimator versions, each scored on the same conditions.

## Result

In progress. Best so far 872/1000 under full noise (`2026-11-17-park-v6-stepped-finale`).

## Lesson

Each version below fixed the failure mode the previous one exposed; the order matters more than the final numbers.

## Re-run only if

A child changes the plant model, the target or the success definition.
