---
id: 2026-11-24-pose-detector-v3-close-range
date: 2026-11-24
author: fer-claude
owner: fernando
state: open
updated: 2026-11-24
due: none
after: []
check: none
refs: [research/attempts/2026-11-24-pose-network-in-challenge-loop.md, research/tasks/2026-11-18-pose-near-field-keypoints.md]
tags: [pose, detector, challenge, learning]
---

## Goal

The trained detector reads the challenge camera well enough for the autopilot to park from the
named starts, measured on recorded frames it never trained on, with no fabricated component
anywhere in the loop (`AGENTS.md`, no-stand-ins rule).

## Where it stands (2026-11-24)

Attempt `2026-11-24-pose-network-in-challenge-loop`: the network runs in the loop (v2 default,
6-8 ms per frame). False alarms are fixed (0.14 invented points per empty frame, precision 0.82).
The bottleneck is truncation: 149 of 165 frames with the airframe show it cut by the frame edge;
on those only 6.7% have 6+ correct points (75% when the whole airframe is in view). The 15-point
skeleton the controller assumes in the bay (near-field tyre points) does not exist in the network.

## Plan, in order (each step: one attempt record with numbers on the held-out frames)

1. **Held-out sets first.** Record a second set with `uv run ugv-lab --record
   runs/challenge_frames/<name>` in a real browser (GPU, ~10 Hz; check the HUD rate), including
   ENTER/BACKOFF close-range frames from manual driving into the bay. Never train on recorded
   frames that are used for scoring; split by session.
2. **15 keypoints.** Add the three nose-tyre points (`ugv.park.perception._near_points`) to the
   network: `train/pose_kp.py` `NK`, `make_dataset.py` labels, `ugv.pose.detector` (indices
   12-14 are already what the server expects). Task `2026-11-18-pose-near-field-keypoints`.
3. **Truncation.** Either add truncated training views at bay range (camera inside the bay
   looking at the nose gear, airframe mostly outside the picture), or replace the soft-argmax
   head with per-pixel keypoint voting (PVNet, arXiv 1812.11788). Decide by the cut-off-frames
   number in `tools/pose/eval_frames.py`, not by synthetic validation.
4. **Streamed data.** The CPU renderer makes ~30 images/s on 16 cores, the GPU trains at ~1,800/s:
   replace the fixed 24k set with a rolling pool (generator workers refresh ~20% of the pool per
   epoch) so training never repeats a set. Keep validation fixed.
5. **Autopilot gate.** Only after 1-4: autopilot rounds in a real browser from each named start,
   healthy then bad sensors, parked count reported with the frame rate. The batch simulator
   (`ugv.park.perception.DetectorModel`) stays out of tuning until it runs the network.

## Not to do

Do not restore any detector built from the true pose; do not tune controller gains to hide
detector misses; do not report synthetic-validation numbers as evidence; do not train on the
frames used for scoring. Real car frames (`2026-11-17-real-pose-eval-set`) remain the only
evidence about the method; the challenge is a rendered simulator.

## Done when

On held-out challenge frames: 6+ correct points on most frames under 1 m (with the tyre points),
false points per empty frame at or below v2's 0.14, and the autopilot parking from the named
starts in a real browser, each with an attempt record and run ids.
