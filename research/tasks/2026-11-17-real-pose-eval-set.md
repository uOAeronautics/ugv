---
id: 2026-11-17-real-pose-eval-set
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: []
check: none
refs: []
tags: [pose, data]
---

## Goal

Collect the evaluation set every pose method is judged on: 50-100 car-camera frames of the
real airframe at the parking distances (0.3-3 m), each with ground-truth pose from the ArUco
tags (14/6/3) or a measured placement. Tags are allowed here: they are the ruler, not the method.
S0 for the car (camera only; no motor).

## Done when

Frames and poses are on disk (not in Git), with a manifest and loader in `src/ugv/pose/`, and a
`ledger.py` attempt records how ground truth was obtained and its own error.
