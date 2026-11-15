# 12 · Plane-belly keypoint and CNN experiment: results

Run id: `20261114-112058-belly12` · script: `docs/council_v2/exp_belly_vision.py` · seeds train/test/clutter 0/1000/2000.

## Setup (synthetic, reproducible)

- Frames: 160x120 grayscale. Camera intrinsics fx=fy=121.0 px (car camera 968 px at 1280x720, scaled), principal point at the image centre, no lens distortion.
- Scene: wing checkerboard (z=+0.20 m), twin props, main-gear wheels with hubs (x=±0.45 m), inverted-V fins with DICT_4X4_50 tags 14 (left) and 6 (right) at 50 mm, belly tag 3. Distractors: chair legs, boxes, shoes, dark discs, decoy binary grids. Gain 0.45-1.5x, glare blobs in half the frames, noise, PSF blur.
- Train: 16000 frames, 70% plane views with a part in view and the rest distractor-only. Off-frame plane views are excluded from training and calibration: their label is ambiguous (wing visible, parts just outside the frame), and the test set does not contain them. Test: 500 target-visible frames (rejection-sampled, so every frame has at least one part in view) + 100 distractor-only frames.
- Test range: camera distance 0.20-2.0 m; 32 frames <0.45 m (ArUco-failure regime), 157 frames with off-axis angle >=40 deg.
- Pose: camera (x, y) in the plane frame and yaw = azimuth of the camera image-up vector. Units mm and deg.

## Results (held-out test set, same frames for every method)

| Method | Recall % | Close <0.45 m recall % | Steep >=40 deg recall % | Pose coverage % | Pos MAE mm | Yaw MAE deg | Close pos MAE mm | Distractor FPR % | CPU ms / frame (1 thread) |
|---|---|---|---|---|---|---|---|---|---|
| ArUco-only | 0.8 | 0.0 | 1.9 | 0.8 | 259.0 | 5.04 | nan | 0.0 | 0.73 |
| HOG + linear-SVM windows | 64.8 | 40.6 | 68.2 | 10.2 | 2630.7 | 98.35 | 7931.7 | 50.0 | 19.12 |
| Keypoint CNN + LeJEPA aux | 98.6 | 90.6 | 98.1 | 98.6 | 546.1 | 27.43 | 234.4 | 10.0 | 1.34 |
| CNN keypoints -> PnP | 98.6 | 90.6 | 98.1 | 98.6 | 1374.3 | 56.69 | 2648.7 | 10.0 | 1.41 |
| ArUco-only @320x240 (sensitivity) | 13.0 | 0.0 | 14.0 | 13.0 | 425.3 | 4.89 | nan | 0.0 | 0.39 |

Presence thresholds were chosen on a separate 200-frame calibration set (seed 3000, not the test set) by balanced accuracy: classical tau=-0.42 (62.9%), CNN p>0.99 (96.7%). ArUco has no threshold.

Tag in view: 80.2% of target frames have a full tag facing the camera and in frame. ArUco recall on those frames: 1.0%. ArUco cannot see the wheels, so its headline recall on all target frames is capped by tag visibility.

CNN keypoint error on visible parts: 18.60 px (160x120 frame). Row `CNN keypoints -> PnP` solves pose geometrically from the CNN's four keypoints; it is a diagnostic variant, not a separate network.

CNN parameters: 118,514. CNN CPU latency (batch 1, 1 thread): median 1.34 ms, mean 1.37 ms. Training time: CNN 33.0 s on cuda; classical SVM 7.1 s; data generation 7.7 s. Total wall time 63.3 s.

Recall is computed on the 500 target-visible frames; FPR on the 100 distractor-only frames. Pose MAE is over frames where the method both detects and outputs a pose.

![comparison](belly_vision_comparison.png)

Composite rows (top to bottom): close-range target, steep-angle target, mid-range target, distractor-only. Columns: ground truth keypoints (green), ArUco detections (cyan), best HOG+SVM windows (magenta), CNN keypoints when p>0.99 (yellow crosses).

## ArUco at car-like resolution (opt-in pass, same protocol)

Run with `--aruco-fullres` (300 target + 50 distractor frames, 1280x960 render, fx=968 = the car camera's intrinsics):

| Method | Recall % | Close <0.45 m recall % | Tag-in-view recall % | Pos MAE mm | Yaw MAE deg | Distractor FPR % | CPU ms / frame |
|---|---|---|---|---|---|---|---|
| ArUco-only @1280x960 | 58.3 | 46.7 | 71.1 (tags in view: 82.0%) | 181.4 | 1.93 | 0.0 | 3.4 |

Raw numbers: `12_belly_vision_aruco_fullres.json`.

## Reading the results (authored after run `20261114-112058-belly12`)

**What this says about the question (ArUco vs. HOG vs. learned keypoints):**

- **ArUco is the best pose source when a tag is in view, but only at car-like resolution.** At 160x120 it recalls 1% of
  frames where a full tag is in view: the tags are a few pixels across. A head-on tag decodes at 0.3 to 0.8 m with under 3 mm
  pose error at both scales, so the renderer and the pose conventions are correct; the failure is resolution. At
  1280x960 with car intrinsics, ArUco recalls 71% of in-view frames with 1.9° yaw error and 0 false positives. Its
  limit is the 29% of frames it misses, and it cannot see wheels at all.
- **The learned keypoint CNN is the only method here that finds the target across the full range.** It recalls 98.6%
  overall and 90.6% at <0.45 m, where ArUco drops to 0%. Its distractor FPR is 10% in this run. Its pose is not usable
  yet: 546 mm and 27° MAE. Its keypoints are off by 18.6 px on average at 160x120, so keypoint localisation is the
  bottleneck, not detection. Feeding its keypoints through PnP is worse (1374 mm), so the bottleneck is in keypoints.
- **HOG + linear SVM is not worth pursuing as built.** It recalls 64.8% with 50% distractor FPR. It fires on glare,
  decoy grids and dark discs, and its pose is unusable.

**Recommendation for the council (not yet a decision):**

1. Keep ArUco as the primary pose source when a tag is in view, at car resolution. It has zero false positives on this
   data, and its pose path is verified.
2. Treat the CNN as a presence gate and a fallback for wheels and fins when tags drop out at close range. Do not use its
   pose for control yet.
3. Before any LeJEPA claim, run an ablation with the auxiliary loss weight set to 0. This experiment did not isolate the
   LeJEPA contribution, so the CNN numbers above cannot be attributed to it.
4. Next: retrain the CNN at 1280x960 (the car's regime, not 160x120), train longer to close the keypoint gap, and check
   against real frames before any hardware use.

**Caveats (read before quoting any number):**

- **The renderer is a simplified proxy.** The parts are flat-shaded primitives with no real textures, and the lighting
  model is crude. The sim-to-real gap is the largest unknown, and these numbers measure the proxy, not the car.
- **Single run per method, single seed.** Repeated runs of the final protocol gave CNN distractor FPR 0 to 10%, recall
  98.2 to 99.8%, position MAE 546 to 625 mm and yaw MAE 25 to 27°. ArUco and HOG numbers did not change between runs.
  FPR is quantised at 1% per frame (100 distractor frames).
- **Training was budget-limited.** The CNN ran 20 epochs on 16k frames to fit the 90 s limit, and its loss was still
  falling. Treat the CNN numbers as a lower bound on what this architecture can do.
- **Calibration is on a separate set, but the CNN threshold sits at the top of its grid** (p>0.99 here, 0.54 to 1.00
  across runs). Its presence scores are close to binary on this data, so the FPR is sensitive to that choice.
- **Optimistic for every method:** exact intrinsics, no lens distortion, no motion blur, no label noise.
- **Yaw convention:** the experiment's yaw is the azimuth of the camera image-up vector. Before any use on the car it must
  go through `pose2_from_ccw` / `pose2_to_ccw` in `src/ugv/parking.py` and be checked against the car frame.
- **Nothing ran on the Pi.** The car is paused for charging per `HANDOFF.md`, and this experiment is desktop-only.

## How to reproduce

```sh
cd rl && uv run --with opencv-contrib-python-headless python ../docs/council_v2/exp_belly_vision.py          # main, ~65 s on RTX 3090
cd rl && uv run --with opencv-contrib-python-headless python ../docs/council_v2/exp_belly_vision.py --aruco-fullres  # opt-in, ~12 s
```

`rl/pyproject.toml` has no OpenCV, and the experiment does not change it. `--with` adds OpenCV for that run only. No
weights, frames or videos are saved; only the PNG composite, the JSON results and this report are written.
