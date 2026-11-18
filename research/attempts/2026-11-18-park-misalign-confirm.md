---
id: 2026-11-18-park-misalign-confirm
date: 2026-11-18
author: fer-claude
question: Most remaining failures are "misaligned in the bay" aborts; are they real, and can confirmation or averaging win them back without collisions?
parent: 2026-11-18-park-blind-gate
outcome: negative
failure_mode: underpowered
refs: [src/ugv/park/controller.py, src/ugv/park/estimator.py]
tags: [park, faults, estimator]
---

## Tried

Search s5 (from near-field + blind gate, healthy + faulty + close, 150 each): root 258/450 with
25 collisions; 127 failures were "misaligned in the bay; out of retries"; two generations found
nothing that passed the collision gate. Traced 269 misaligned back-offs (120 faulty episodes):
207 were false (estimated |y| > 3 cm, true |y| < 3 cm), with the estimate's lateral and yaw
errors coupled (few-point fixes at short range). Then: `misalign_confirm` (back off only after
N consecutive misaligned windows, measuring again in between) and `still_gain` (average
harder while the car stands still). A/B seeds 21, near-field on.

## Result

| variant | all faults (300) | healthy (200) |
|---|---|---|
| blind gate (cand2) | 107, 33 coll, 4 off | 200 |
| confirm 2 | 129, 42 coll | 200 |
| confirm 3 | 133, 43 coll | 199 |
| confirm 3 + max_step 0.156 | 139, 55 coll | 197 |
| still_gain 0.1 | 122, 38 coll, 23 off | 197 |
| still_gain 0.1 + confirm 2 | 136, 49 coll | 198 |

Every variant buys successes with collisions at about the same ratio: on cars carrying every
fault family at once, lateral precision in the bay (~3 cm) is the limit against 3.5 cm of
clearance. Knobs kept (default off) and left to the collision-gated search (s6).

## Lesson

When the ratio of successes gained to collisions added is the same for every knob, the
bottleneck is the sensor, not the logic: the next step is perception (near-field keypoints,
a detector trained on camera faults, measured calibration), not more controller tuning.

## Re-run only if

Perception in the bay improves (real detector with tyre points, measured lens model).
