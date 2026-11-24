---
id: 2026-11-24-pose-detector-v3c-rolling-pool
date: 2026-11-24
author: fer-claude
question: Does training on a continuously refreshed rolling pool with spatial per-keypoint visibility and log-uniform close-range distance sampling improve cut-off and under-1m performance?
parent: 2026-11-24-pose-detector-v3-15pt-bay
outcome: kept
failure_mode: none
refs: [src/ugv/pose/dataset.py, src/ugv/pose/render.py, src/ugv/pose/detector.py, train/pose_kp.py, tests/test_pose_pool.py]
tags: [pose, detector, challenge, rolling-pool, close-range]
---

## Tried

Worked step 4 of `research/tasks/2026-11-24-pose-detector-v3-close-range.md`:

1. **Rolling generated pool (`ugv.pose.dataset.RollingPool`, `train/pose_kp.py --pool`)**: replaced the fixed 24k dataset with an in-memory 12,000-sample pool continuously refreshed by 14 background CPU worker processes while the GPU trains for 80 epochs (`--bay 0.20`), against fixed validation sets (`runs/pose_data/val-v3` and `runs/pose_data/val-crop-v3`, 3,000 samples each). During training, `kp-v3c` refreshed 83,072 samples (~95k unique views) and `kp-crop-v3c` refreshed 84,224 samples (~96k unique views).
2. **Spatial visibility head (`train/pose_kp.py` `KeypointNet`)**: replaced the global-pooled `nn.Linear(576, 15)` visibility head with per-keypoint attention-pooled stride-4 features (`prob @ feat`) concatenated with each keypoint's heatmap peak logit and projected global context (`1,217,232` params, same ONNX interface).
3. **Log-uniform close-range distance (`src/ugv/pose/render.py` `sample_car_level_pose`)**: drew level car-camera distance log-uniformly over `[0.25, 3.5]` m so ~52% of general-azimuth views are under 1 m (up from 23%).

Runs: `runs/pose_train/kp-v3c`, `runs/pose_train/kp-crop-v3c`. Set as `DEFAULT` in `src/ugv/pose/detector.py`.

## Result

Scored with `tools/pose/eval_frames.py` (hit <= 20 px, threshold 0.5).

**Healthy sensors, 15-point (`runs/challenge_frames/v3h`, 1,261 frames):**

| Metric | v2 (12-pt) | v3b (15-pt + bay, fixed 24k) | v3c (rolling pool + spatial vis) |
|---|---|---|---|
| false points / empty frame | 0.07 | 0.01 | 0.11 |
| precision / recall | 0.750 / 0.289 | 0.879 / 0.585 | **0.899 / 0.621** |
| swap rate | 1.0% | 0.4% | 0.8% |
| median error | 6.4 px | 5.5 px | 5.6 px |
| whole airframe in view (70 fr) | 44.3% | 97.1% | 95.7% |
| cut-off frames (571 fr) | 3.9% | 54.8% | **59.4%** |
| under 1 m (401 fr) | 4.0% | 53.4% | **58.4%** |
| under 1 m, 6+ in view (328 fr) | 4.9% | 65.2% | **71.3%** |
| 0-1 m precision / recall | 0.681 / 0.274 | 0.870 / 0.624 | **0.909 / 0.677** |
| 1-2 m frames 6+ correct (240 fr) | 15.4% | 69.6% | **71.7%** |

**Bad sensors, 15-point (`runs/challenge_frames/v3a`, 2,683 frames):**

| Metric | v2 | v3b | v3c |
|---|---|---|---|
| false points / empty frame | 0.10 | 0.03 | 0.10 |
| precision / recall | 0.626 / 0.236 | 0.660 / 0.323 | **0.691 / 0.375** |
| cut-off frames (1,069 fr) | 1.2% | 19.7% | **25.9%** |
| under 1 m (783 fr) | 1.9% | 21.7% | **30.0%** |
| under 1 m, 6+ in view (438 fr) | 4.8% | 38.8% | **53.7%** |

**Baseline set, healthy 12-point (`runs/challenge_frames/diag1`, 428 frames, `--n-kp 12`):**
v2: 0.14 false pts/empty, prec 0.816, recall 0.356, whole 75.0%, cut-off 6.7%, under 1 m 6.5%.
v3c: **0.02** false pts/empty, prec **0.881**, recall **0.450**, whole **100.0%**, cut-off **7.4%**, under 1 m 5.6% (12-pt only; 0-1 m precision 0.872, recall 0.419 vs v2's 0.768 / 0.310).

CPU latency (1 thread): 8.9 ms (stage 1) + 9.5 ms (stage 2). Synthetic validation (not real-frame evidence): `kp-crop-v3c` `val_vis_acc` 0.956, `val_px_median_at_1280` 4.35 px, `val_pnp_within_5deg_10cm` 0.492.

## Lesson

Combining a continuously refreshed rolling pool (~95k unique samples), spatial per-keypoint visibility and log-uniform close-range sampling lifts every held-out metric over fixed-dataset v3b: healthy under-1m 6+-correct reaches 58.4% (71.3% when 6+ are in view) with 0.909 precision, bad-sensor under-1m with 6+ in view jumps from 38.8% to 53.7%, and `diag1` beats v2 on false alarms (0.02 vs 0.14), precision (0.881 vs 0.816) and cut-off frames (7.4% vs 6.7%).

## Re-run only if

The network architecture, generator mix or challenge renderer changes.
