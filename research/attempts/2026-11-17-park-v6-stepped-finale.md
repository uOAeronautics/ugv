---
id: 2026-11-17-park-v6-stepped-finale
date: 2026-11-17
author: fer-claude
question: With gyro noise tied to motor vibration, few-point fixes kept off the yaw, a scaled RLS prior and steady-duty sampling, how well does it park?
parent: 2026-11-17-park-v5-stepped-careful-finale
outcome: kept
failure_mode: none
refs: [runs/park_eval/default.json, docs/PARK_SIM.md]
tags: [park, eval]
---

## Tried

Ablation batch as above, then `tools/park/eval.py --name default --sweep` (1000 episodes, seed 11, plus a 300-episode robustness sweep).

## Result

Full 872/1000 (95% CI 85.0-89.1%), 15 collisions (1.5%), median 29 s to park. Sweep: ideal 100%, detector-only 99%, plant-only 95%, worse detector 77%, worse calibration 74%, 1.5x wider cars 67%. Off-target failures (91) are mostly heading just over 5 deg (76 of 91).

## Lesson

The controller is robust to the measured plant spread; what is left is heading at the end, and it comes from the car spread (nominal car: 90th-percentile heading error 0.5 deg).

## Re-run only if

The plant model, detector model or tolerance changes.
