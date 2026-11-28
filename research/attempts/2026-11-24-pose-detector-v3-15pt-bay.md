---
id: 2026-11-24-pose-detector-v3-15pt-bay
date: 2026-11-24
author: fer-claude
question: Do the 3 nose-tyre keypoints (15-pt skeleton) and bay-range truncated training views fix close-range and cut-off challenge frames?
parent: 2026-11-24-pose-network-in-challenge-loop
outcome: kept
failure_mode: none
refs: [src/ugv/pose/mesh.py, src/ugv/pose/render.py, src/ugv/pose/dataset.py, src/ugv/lab/record.py, src/ugv/pose/frame_eval.py, 28c46f7, 28c46f7, tools/pose/eval_frames.py]
tags: [pose, detector, challenge, close-range, truncation]
---

## Tried

Worked steps 1-3 of `research/tasks/2026-11-24-pose-detector-v3-close-range.md`:

1. **Held-out 15-point frame sets** (never trained on, recorded in a real browser with `ugv-lab --record`):
   - `runs/challenge_frames/v3a`: 2,683 frames (1,557 empty, 1,126 with airframe, 783 under 1 m, 438 under 1 m with 6+ points in view) under **Bad sensors** (default page setting: occluders, defocus, motion blur).
   - `runs/challenge_frames/v3h`: 1,261 frames (620 empty, 641 with airframe, 401 under 1 m, 328 under 1 m with 6+ points in view) under **Healthy sensors** across all 6 named starts + random (both manual close-range bay entries and autopilot rounds, 5-8 Hz in the HUD during concurrent GPU load).
2. **15 keypoints (`v3a`)**: appended `tyre_left`, `tyre_right`, `tyre_top` (`ugv.pose.mesh.keypoints(..., near=True)`, indices 12-14) to the renderer, dataset generator, recorder, and `train/pose_kp.py`. Trained fixed 24k sets (`runs/pose_data/v3a`, `runs/pose_data/crop-v3a`) for 30 epochs: runs `runs/pose_train/kp-v3a` and `runs/pose_train/kp-crop-v3a`.
3. **Bay-range truncated views (`v3b`)**: added `render.sample_bay_pose` (camera on the bay run-in looking at the nose gear and tyre points with the rest of the airframe cut off) to `ugv.pose.dataset` (`--bay`). Trained fixed 24k sets (`runs/pose_data/v3b`, `runs/pose_data/crop-v3b`) for 30 epochs: runs `runs/pose_train/kp-v3b` and `runs/pose_train/kp-crop-v3b`.

## Result

Scored with `tools/pose/eval_frames.py` (hit <= 20 px, threshold 0.5).

**Healthy sensors, 15-point (`runs/challenge_frames/v3h`, 1,261 frames):**

| Metric | v2 (`kp-v2,kp-crop-v2`, 12-pt) | v3a (`kp-v3a,kp-crop-v3a`, 15-pt) | v3b (`kp-v3b,kp-crop-v3b`, 15-pt + bay) |
|---|---|---|---|
| false points / empty frame | 0.07 | 0.01 | 0.01 |
| false alarm frames | 7.3% | 0.6% | 1.5% |
| precision | 0.750 | 0.836 | 0.879 |
| recall | 0.289 | 0.438 | 0.585 |
| swap rate | 1.0% | 0.6% | 0.4% |
| median error | 6.4 px | 6.1 px | 5.5 px |
| whole airframe in view (70 fr) | 44.3% | 100.0% | 97.1% |
| cut-off frames (571 fr) | 3.9% | 37.0% | 54.8% |
| under 1 m (401 fr) | 4.0% | 32.7% | 53.4% |
| under 1 m, 6+ in view (328 fr) | 4.9% | 39.9% | 65.2% |
| 1-2 m (240 fr) | 15.4% | 62.5% | 69.6% |

**Bad sensors, 15-point (`runs/challenge_frames/v3a`, 2,683 frames):**

| Metric | v2 | v3a | v3b |
|---|---|---|---|
| false points / empty frame | 0.10 | 0.02 | 0.03 |
| precision / recall | 0.626 / 0.236 | 0.667 / 0.294 | 0.660 / 0.323 |
| cut-off frames (1,069 fr) | 1.2% | 17.9% | 19.7% |
| under 1 m (783 fr) | 1.9% | 19.7% | 21.7% |
| under 1 m, 6+ in view (438 fr) | 4.8% | 35.2% | 38.8% |

**Baseline set, healthy 12-point (`runs/challenge_frames/diag1`, 428 frames, `--n-kp 12`):**
v2: 0.14 false pts/empty, prec 0.816, cut-off 6.7%, under 1 m 6.5%.
v3a: 0.02 false pts/empty, prec 0.823, whole 87.5%, cut-off 6.7%, under 1 m 4.7%.
v3b: 0.05 false pts/empty, prec 0.818, whole 87.5%, cut-off 4.7%, under 1 m 2.8% (12-pt only; `diag1` does not score the 3 tyre points).

Decision on step 3: bay-range truncated views already lifted healthy cut-off frames from 3.9% (v2) to 54.8% (v3b) and under-1m frames from 4.0% to 53.4% (65.2% when 6+ points are in view) while keeping CPU latency at 5.6 + 6.0 ms (1-thread) / 2.8 + 2.6 ms (4-thread). Keep the heatmap + soft-argmax architecture rather than switching to PVNet voting, and address the two remaining bottlenecks in step 4 (rolling pool): global-pooled visibility (`18.9%` of visible close-range points on `diag1` were already within 20 px in stage 1 yet rejected with `prob <= 0.5`), and uniform distance sampling (`sample_car_level_pose` puts only 23% of general-azimuth views under 1 m).

## Lesson

Adding the 3 nose-tyre keypoints plus bay-range truncated views raises healthy under-1m 6+-correct rate from 4.0% (v2) to 53.4% (65.2% with 6+ in view) and cuts false points per empty frame to 0.01 without adding voting-head latency; next train on the rolling pool with spatial visibility and log-uniform close-range sampling.

## Re-run only if

The airframe mesh, keypoint definition or challenge renderer changes.
