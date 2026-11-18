---
id: 2026-11-18-pose-near-field-keypoints
date: 2026-11-18
author: fer-claude
owner: fernando
state: open
updated: 2026-11-18
due: none
after: []
check: none
refs: []
tags: [pose, skeleton, detector]
---

## Goal

Extend the detector's skeleton from 12 to 15 points: the nose tyre's left, right and top
extremes (`ugv.park.perception._near_points`), which the parking simulator now assumes
(attempt `2026-11-18-park-near-field`: healthy cars 175 -> 200/200 parked, faulty-car
collisions 99 -> 72 of 300). Regenerate the dataset with the 15 names (including close views
from inside the bay, 0.1-0.4 m from the nose wheel, which the current generator rarely makes),
retrain both stages, and report the tyre points' error separately.

## Done when

An attempt with per-point error for the three tyre points on synthetic close views, and on the
real evaluation set once it exists. If they cannot be found reliably, set `near_field=False`
and re-score the controller.
