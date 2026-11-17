---
id: 2026-11-17-parking-state-machine
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: [2026-11-17-pose-on-car]
check: none
refs: []
tags: [parking, fsm, milestone]
---

## Goal

Design the parking behaviour as a declared state machine (`src/ugv/fsm.py`):
SEARCH (rotate in place until the airframe is detected) -> ACQUIRE (pose converges) ->
ORBIT (strafe on a circle while yawing to keep the airframe centred, until the car is on the
bay axis behind the tail) -> APPROACH (drive in along the axis) -> CAREFUL_PARK (slow, last
0.4 m dead-reckoned from the last good pose, end-frame check) -> PARKED. Each state lists its
failure modes and exits: pose lost, pose jump, too close, timeout, collision risk (shield),
operator stop. Simulate it against synthetic pose noise first.

## Done when

Transition table and failure-mode table in `docs/PARKING.md`; the machine runs in simulation
from random car and airframe start poses with a measured success rate; no motor change without
the S4 gate.
