---
id: 2026-11-24-pose-detector-v4-reliability
date: 2026-11-24
author: fer-claude
owner: unassigned
state: open
updated: 2026-11-24
due: none
after: []
check: uv run ruff check . && uv run ruff format --check . && uv run pytest -q
refs: [research/decisions/2026-11-24-offline-pose-eval-and-reliability-gate.md, research/attempts/2026-11-24-pose-detector-v3c-autopilot-gate.md, research/attempts/2026-11-24-pose-detector-v3c-rolling-pool.md, research/attempts/2026-11-24-pose-detector-v3-15pt-bay.md]
tags: [pose, detector, reliability, handoff]
---

## Goal

Take the trained 15-point keypoint detector from "can park" (`v3c`: 4/6 healthy, 1/6 bad sensors, 3 collisions in `n = 12` live browser rounds) to **reliably parking** (`>= 90%` healthy, `0` collisions, strong bad-sensor rate) using fast offline evaluation and failure-frame decomposition, without falling into local minima or running live Chromium loops.

### Handoff state (2026-11-24, training paused)

- **Code**: commits `a1b1378` (15-pt skeleton, bay-range generator, `RollingPool`) and `749138e` (v3c default, spatial per-keypoint visibility, log-uniform close-range distance, tight nose-wheel refinement crop, span-conditioned `solve_planar`).
- **Active best model on disk** (older `v1`/`v2`/`v3a`/`v3b` checkpoints and fixed 24k datasets removed; their numbers live in `research/attempts/`):
  - `runs/pose_train/kp-v3c` + `runs/pose_train/kp-crop-v3c` (`DEFAULT` in `src/ugv/pose/detector.py`).
- **Fixed validation & frozen held-out sets on disk** (never train on `challenge_frames/`):
  - Synthetic val: `runs/pose_data/{val-v3,val-crop-v3}` (3,000 each).
  - Held-out Challenge frames: `runs/challenge_frames/diag1` (428 healthy 12-pt), `v3h` (1,261 healthy 15-pt), `v3a` (2,683 bad-sensor 15-pt).
- **Current `v3c` baseline (with nose-wheel crop + span-conditioned `solve_planar`)**:
  - `v3h` (healthy 15-pt): false pts/empty `0.11`, prec `0.888`, recall `0.614`, med `4.5 px`, cut-off `57.8%`, under 1 m `56.1%` (`68.6%` when 6+ in view); bay-entrance `p90 |dy| = 1.7 cm`, `p90 |dyaw| = 1.2 deg`, `abort_rate = 2.4%`.
  - `v3a` (bad sensors 15-pt): prec `0.691`, recall `0.375`, cut-off `25.9%`, under 1 m `30.0%` (`53.7%` when 6+ in view).
  - `diag1` (healthy 12-pt): false pts/empty `0.02`, prec `0.881`, cut-off `7.4%`, whole `100%`.
- **Known open failure modes (decompose before retraining)**:
  1. `+3.8 cm` x-overshoot collision into the nose wheel during `ENTER` (`beside-nose-gear`, healthy).
  2. `-11 to -13.5 px` systematic vertical bias on `hub_left`/`hub_right` in Challenge bay frames (`0 px` on `val-v3`) — likely pitch-black rotors/propellers vs uniform/sine albedo in `src/ugv/pose/render.py`.
  3. Bad-sensor recall drop (`0.375` on `v3a`) causing dropouts and collisions during `ENTER`/`BACKOFF`.
  4. Regression test for the `0.35 m` horizontal-span yaw guard in `src/ugv/park/perception.py` is now in `tests/test_park.py` (`test_solve_planar_holds_yaw_on_tight_nose_cluster`).

### Plan, in order (decision `2026-11-24-offline-pose-eval-and-reliability-gate`)

1. **Fast offline closed-loop evaluator**: build a Python runner (`n >= 50` matched seeds, `ugv.pose.render` + real ONNX `KeypointDetector`, zero fabricated stand-ins, zero live Chromium) that reports parked/collision/abort/timeout rates with 95% binomial CI and renders PNG keypoint overlays of failed/colliding frames.
2. **Decompose failures**: inspect the rendered failure frames from (1) plus the `-13 px` hub bias on `v3h` and bad-sensor misses on `v3a`.
3. **Resume training (`v4`)**: add per-part albedo randomization (dark rotors/tyres vs light body) and stronger occluder/defocus/exposure augmentation in `ugv.pose.dataset` / `train/pose_kp.py`; A/B test PVNet per-pixel keypoint voting vs the heatmap head on `v3h` + `v3a` + the `n >= 50` offline closed-loop gate before promoting.
4. **Final milestone check**: confirm on real frames (`2026-11-17-real-pose-eval-set`) once the Pi is off charge (`docs/STATE.md`), and run a final Challenge check only after offline gates pass.

## Done when

On the `n >= 50` offline closed-loop gate (no stand-ins): `>= 90%` healthy parked with `0` collisions, bad-sensor parked rate and `v3a` under-1m recall clearly above `v3c` (outside the 95% binomial interval), `solve_planar` regression test in `tests/`, and every experiment logged as a `parent`-linked attempt in `research/attempts/`.
