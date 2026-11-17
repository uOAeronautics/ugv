---
id: 2026-11-17-park-v7-fixed-pitch-few-points
date: 2026-11-17
author: fer-claude
question: Does holding camera pitch fixed for 3-5 point fixes (so they can update yaw) reduce the heading failures?
parent: 2026-11-17-park-v6-stepped-finale
outcome: kept
failure_mode: none
refs: [src/ugv/park/perception.py]
tags: [park]
---

## Tried

Same 400 episodes (seed 11) as the first 400 of `default`, paired.

## Result

359/400 parked vs 349/400 for v6; collisions 2 vs 4. Heading 90th percentile unchanged at about 4.7 deg.

## Lesson

A small real gain from better-conditioned yaw at close range; the heading tail is elsewhere (car spread), see the open child task.

## Re-run only if

The solver changes.
