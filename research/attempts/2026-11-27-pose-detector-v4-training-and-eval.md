---
id: 2026-11-27-pose-detector-v4-training-and-eval
date: 2026-11-27
author: agy
question: Does adding per-part albedo randomization (dark rotors/nacelles vs light body), stronger camera-fault augmentation reusing apply_image_faults, and seed-11 DAgger mixing improve held-out per-frame scores and closed-loop parking reliability over v3c on both seed 11 and held-out seed 12?
parent: 2026-11-27-offline-park-network-eval
outcome: kept
failure_mode: none
refs: [src/ugv/pose/render.py, src/ugv/pose/dataset.py, src/ugv/park/camera.py, tests/test_pose.py, tests/test_park_camera.py, research/tasks/2026-11-24-pose-detector-v4-reliability.md]
tags: [pose, park, detector, v4, dagger, evaluation]
---

## Tried

1. **Generator fixes (`src/ugv/pose/render.py`, `src/ugv/pose/dataset.py`, `src/ugv/park/camera.py`):**
   - Added `SDF_PART_RGB`, `part_albedo`, and `random_part_albedo` in `src/ugv/pose/render.py` (light body `[0.55, 0.98]`, dark rotors/nacelles `[0.03, 0.30]` with per-part jitter, medium/accent control surfaces); delegated `ugv.park.camera.sdf_albedo` to `part_albedo` without duplication. Used `random_part_albedo` on 50% of rolling-pool samples (20% plain paint, 30% procedural texture).
   - Reused `ugv.park.camera.apply_image_faults` in `ugv.pose.dataset.sample` (generalizing occluder slice bounds to `gray.shape[:2]`) so 45% of rolling-pool samples receive camera faults matching `ugv.park.faults`: log-uniform defocus/smear blur up to `25 px` at 1280x720 scaled by `cam.K[0,0] / FULL.K[0,0]`, linear HDR soft-knee exposure (`+0.7..+3.5` or `-2.5..-0.6` stops), sensor noise (`noise_px` `0.5..2.5`), black occluders (`15–40% W × 15–50% H`), plus 25% directional motion blur (`5–11 px` kernels).
   - Fixed validation sets `runs/pose_data/{val-v3,val-crop-v3}` and held-out Challenge sets `runs/challenge_frames/{v3h,v3a,diag1}` were left untouched.
2. **GPU training (NVIDIA GeForce RTX 3090, sequential, no other compute jobs active):**
   - Stage 1 (`runs/pose_train/kp-v4`, `664 s` / `11.1 min`, `84,480` pool replacements): `cd train && uv run python pose_kp.py --data ../runs/pose_data/val-v3 --out ../runs/pose_train/kp-v4 --epochs 80 --pool 12000 --bay 0.20 --workers 14 --dagger ../runs/pose_data/dagger-v3c-healthy ../runs/pose_data/dagger-v3c-bad --dagger-frac 0.25`. Exported `model.onnx` (`8.94 ms` 1-thread CPU).
   - Stage 2 (`runs/pose_train/kp-crop-v4`, `741 s` / `12.4 min`, `91,392` pool replacements; total `1,405 s` / `23.4 min`): same flags plus `--crop --data ../runs/pose_data/val-crop-v3 --out ../runs/pose_train/kp-crop-v4`. Exported `model.onnx` (`9.70 ms` 1-thread CPU).
   - **Leakage note:** `dagger-v3c-healthy` and `dagger-v3c-bad` (`4,000 + 4,000` frames) came from seed-11 `v3c` closed-loop runs, so seed-11 evaluations of `v4` have training-set leakage from those seed-11 failure trajectories; seed-12 evaluations (`n = 200` healthy for both `v3c` and `v4`) have zero leakage.
3. **Evaluations (synthetic/Challenge renders of the real mesh, not real-camera evidence):**
   - Per-frame `tools/pose/eval_frames.py` on `runs/challenge_frames/{v3h,v3a,diag1}` (`v4` vs `v3c`, threshold `0.5`, hit `<= 20 px`).
   - Closed-loop `tools/park/eval.py --perception network --params config/park_params.json --start mixed --workers 14`: `network-v4-healthy` (seed 11, `n=200`), `network-v4-bad` (seed 11, `n=120`, `--faults 1`), `network-v3c-healthy-s12` (seed 12, `n=200`), `network-v4-healthy-s12` (seed 12, `n=200`).

## Result

### Per-frame held-out Challenge sets (`tools/pose/eval_frames.py`, threshold `0.5`)

| Set (frames) | Model | False pts / empty | Precision | Recall | Swap rate | Med px | Whole 6+ | Cut-off 6+ | Under 1 m 6+ | Under 1 m (6+ in view) | 0–1 m rec / prec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3h` (1,261) | `v3c` | 0.11 | 0.888 | 0.614 | 0.017 | **4.5** | 95.7% | 57.8% | 56.1% | 68.6% | 0.662 / 0.889 |
| `v3h` (1,261) | `v4` | **0.08** | **0.907** | **0.659** | **0.014** | 5.1 | **100.0%** | **63.6%** | **65.1%** | **79.6%** | **0.714 / 0.923** |
| `v3a` (2,683) | `v3c` | **0.10** | 0.667 | 0.362 | 0.060 | 7.0 | 40.4% | 24.3% | 28.0% | 50.0% | 0.428 / 0.767 |
| `v3a` (2,683) | `v4` | 0.22 | **0.677** | **0.423** | **0.043** | 7.0 | **59.6%** | **25.8%** | **28.6%** | **51.1%** | **0.488 / 0.775** |
| `diag1` (428, 12-pt) | `v3c` | **0.02** | **0.881** | 0.450 | 0.024 | 4.9 | **100.0%** | 7.4% | 5.6% | 15.0% | 0.419 / 0.872 |
| `diag1` (428, 12-pt) | `v4` | 0.13 | 0.860 | **0.466** | **0.018** | **4.6** | 93.8% | **12.8%** | **10.3%** | **27.5%** | 0.417 / 0.864 |

Synthetic validation (unchanged `val-v3` / `val-crop-v3`, not real-frame evidence): `kp-v4` `val_vis_acc = 0.921`, `val_px_median_at_1280 = 10.31 px`, `val_pnp_within_5deg_10cm = 0.285`; `kp-crop-v4` `val_vis_acc = 0.948`, `val_px_median_at_1280 = 5.25 px`, `val_pnp_within_5deg_10cm = 0.435`.

### Closed-loop image-in-the-loop parking (`--perception network`, `--start mixed`, `config/park_params.json`)

| Run ID | Detector | Seed | Condition | n | Parked | Wilson 95% CI | Collisions | Wall s (14w) | Missed views | `hub_left` bias (du, dv) / med | `hub_right` bias (du, dv) / med |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---|---|
| `network-v3c-healthy` | `v3c` | 11 | healthy | 200 | 124/200 (62.0%) | `55.1–68.4%` | 21 | 798.2 | 1.83% | `(-3.62, -8.79)` / `10.87 px` | `(+3.50, -4.49)` / `8.23 px` |
| `network-v4-healthy` *(seed-11 DAgger leak)* | `v4` | 11 | healthy | 200 | 131/200 (65.5%) | `58.7–71.7%` | **11** | 632.1 | 3.79% | **`(+0.93, +0.61)` / `4.87 px`** | **`(-0.90, -0.09)` / `4.44 px`** |
| `network-v3c-bad` | `v3c` | 11 | `--faults 1` | 120 | 12/120 (10.0%) | `5.8–16.7%` | **17** | 1222.7 | 25.51% | `(-4.91, -9.45)` / `13.11 px` | `(+4.83, -5.81)` / `10.72 px` |
| `network-v4-bad` *(seed-11 DAgger leak)* | `v4` | 11 | `--faults 1` | 120 | 17/120 (14.2%) | `9.0–21.5%` | 30 | 702.1 | **13.94%** | **`(-0.13, +0.69)` / `5.08 px`** | **`(-1.43, +0.69)` / `5.09 px`** |
| `network-v3c-healthy-s12` | `v3c` | 12 | healthy (held-out) | 200 | 118/200 (59.0%) | `52.1–65.6%` | 26 | 714.9 | 2.15% | `(-3.70, -8.77)` / `11.26 px` | `(+3.59, -4.29)` / `8.02 px` |
| `network-v4-healthy-s12` | `v4` | 12 | healthy (held-out) | 200 | 126/200 (63.0%) | `56.1–69.4%` | **18** | 621.6 | 3.84% | **`(+0.54, +0.30)` / `4.82 px`** | **`(-0.89, -0.36)` / `4.46 px`** |

**Failure breakdown (FSM last active state & starts):**
- `network-v4-healthy` (seed 11): `parked` `131`, `parked-off-target` `49` (`49` `ENTER` -> `PARKED`; `0/49` `|dx| > 2 cm` med `0.34 cm`, `17/49` `|dy| > 2 cm`, `48/49` `|dyaw| > 5 deg` med `7.02 deg` max `14.19 deg`), `collision` `11` (`10` `ENTER`, `1` `BACKOFF`), `abort:timeout` `5` (`2` `APPROACH`, `1` `SEARCH`, `1` `ORBIT`, `1` `ENTER`), `abort:misaligned in the bay; out of retries` `4` (`ENTER`). By start: `tail` `7/10` (`1` col), `nose-on` `5/10` (`0` col), `nose-oblique` `8/10` (`1` col, up from `3/10`), `beside-nose-gear` `5/10` (`2` col), `under-wingtip` `8/10` (`1` col), `side` `6/10` (`0` col), `random:ring` `92/140` (`6` col, down from `17`).
- `network-v4-healthy-s12` (seed 12, held-out): `parked` `126`, `parked-off-target` `45` (`45` `ENTER` -> `PARKED`; `0/45` `|dx| > 2 cm` med `0.32 cm`, `16/45` `|dy| > 2 cm`, `45/45` `|dyaw| > 5 deg` med `7.00 deg` max `11.45 deg`), `collision` `18` (`17` `ENTER`, `1` `BACKOFF`; vs `v3c` `26` in `ENTER`), `abort:timeout` `6` (`3` `APPROACH`, `2` `ENTER`, `1` `SEARCH`), `abort:misaligned in the bay; out of retries` `5` (`ENTER`; vs `v3c` `14`). By start (`v4` vs `v3c`): `tail` `7/10` vs `4/10`, `nose-on` `8/10` vs `6/10`, `nose-oblique` `6/10` vs `8/10`, `beside-nose-gear` `4/10` vs `5/10`, `under-wingtip` `8/10` vs `7/10`, `side` `7/10` vs `7/10`, `random:ring` `86/140` (`15` col) vs `81/140` (`18` col).
- `network-v4-bad` (seed 11, `--faults 1`): `parked` `17`, `abort:timeout` `40` (`19` `SEARCH`, `8` `ACQUIRE`, `7` `ENTER`, `4` `APPROACH`, `2` `ORBIT`; down from `52`), `collision` `30` (`15` `ENTER`, `6` `BACKOFF`, `4` `SEARCH`, `3` `APPROACH`, `2` `ORBIT`; up from `17`), `abort:lost in the bay` `15` (`ENTER`), `abort:misaligned in the bay` `12` (`ENTER`), `parked-off-target` `6` (`ENTER`). Fault-family parked rates: `focus` still `0/55` (`0.0%` vs `17/65 = 26.2%` off), `blur` `11/61` (`18.0%`, up from `9.8%`), `exposure` `8/56` (`14.3%`, up from `8.9%`), `lens` `6/62` (`9.7%`, up from `4.8%`), `occlusion` `5/59` (`8.5%`), `broken` `4/63` (`6.3%`, up from `1.6%`), `wheel` `5/61` (`8.2%`, up from `4.9%`). Under `--faults 1`, `gear_nose`/`tyre_*` picked up a `+3.7..+5.4 px` vertical bias (`p90` `18.9–27.0 px`), converting former `SEARCH` timeouts into bay collisions.

**Default & task gate decision:**
- Seed-12 healthy Wilson 95% CIs overlap (`v4` `56.1–69.4%` vs `v3c` `52.1–65.6%`), so `DEFAULT` in `src/ugv/pose/detector.py` stays `("kp-v3c", "kp-crop-v3c")`.
- `research/tasks/2026-11-24-pose-detector-v4-reliability.md` acceptance gate (`>= 90%` healthy parked, `0` collisions) is NOT met; task remains `claimed`.

## Lesson

Per-part albedo randomization eliminated the dark-rotor vertical hub bias (`-8.8 / -4.5 px` -> `< 1 px`) and cut healthy collisions by 31–48% while lifting `v3h` under-1m 6+-in-view recall from `68.6%` to `79.6%`, yet ~23% of healthy episodes still end `parked-off-target` exclusively on `|dyaw| > 5 deg` (median `7.0 deg`, `|dx| ~ 0.3 cm`), showing that closing the gap to `>= 90%` requires investigating yaw error accumulation in `ENTER` (after wide-span points leave view) and heavy-defocus/bad-sensor bay robustness (`focus` `0/55`, `30` bad-sensor collisions) rather than albedo alone.

## Re-run only if

The `ENTER` yaw-drift / pre-entry alignment error is decomposed, or detector architecture/head (e.g., PVNet voting), cleaner/less-noisy DAgger split, or fixed-focus defocus handling changes.
