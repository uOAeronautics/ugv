---
id: 2026-11-18-park-heading-hold
date: 2026-11-18
author: fer-claude
question: Does holding heading by gyro during the finale's timed moves (and shielding the reverse) cut fault-driven collisions?
parent: 2026-11-18-park-fault-suite
outcome: kept
failure_mode: plant-mismatch
refs: [src/ugv/park/controller.py]
tags: [park, faults]
---

## Tried

`ParkParams.k_hold`: during an ENTER move, w = k_hold * (heading at move start - odometry
heading), odometry heading from the gyro. `shield_backoff`: run the collision shield in
BACKOFF. A/B on the same seeds (21), s2 params, n: all-faults 400, healthy 300, single family 200.

## Result

| variant | all faults | healthy | wheel | lens | delay |
|---|---|---|---|---|---|
| s2 | 150/400, 114 coll | 277/300 | 128/200 | 164/200 | 144/200 |
| + k_hold 3 | 159/400, 93 coll | 269/300 | 162/200 | 150/200 | 141/200 |
| + shield_backoff | identical | identical | identical | identical | identical |

Heading hold fixes the weak-wheel curve (off-target 53 -> 22) and cuts collisions by a fifth;
it costs a little on healthy cars and lens faults at this gain. The backoff shield never fires.

## Lesson

Close the loop on what the car can measure (gyro heading) inside every open-loop move. The
gain trades against lens faults; left to the search (`k_hold` in SPACE, s3).

## Re-run only if

The ENTER move structure changes.
