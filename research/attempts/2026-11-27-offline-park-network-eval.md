---
id: 2026-11-27-offline-park-network-eval
date: 2026-11-27
author: agy
question: What is the true offline closed-loop parking reliability of kp-v3c + kp-crop-v3c when the batch simulator renders 1280x720 images and applies image-level faults directly to pixels instead of sampling DetectorModel?
parent: 2026-11-24-pose-detector-v3c-autopilot-gate
outcome: kept
failure_mode: none
refs: [src/ugv/park/camera.py, src/ugv/park/dagger.py, src/ugv/park/episode.py, src/ugv/park/evaluate.py, tools/park/eval.py, 28c46f7, docs/PARK_SIM.md, research/tasks/2026-11-24-pose-detector-v4-reliability.md]
tags: [pose, park, detector, evaluation, dagger]
---

## Tried

Built the offline image-in-the-loop evaluator (`src/ugv/park/camera.py`, `src/ugv/park/dagger.py`, `src/ugv/park/evaluate.py`, `tools/park/eval.py --perception network`) and DAgger dataset dump/mixing hook (`--dump-frames`, `train/pose_kp.py --dagger`):

1. **Image-in-the-loop perception (`NetworkPerception`)**: on every 10 Hz camera tick (`Episode._camera`), renders the 1280x720 frame from the true plant pose using `ugv.pose.render` (`WIDTH=1280, HEIGHT=720, F=968.0`, `CAMERA_X_M`, `CAMERA_Z_M`, `bay().floor_z`) with SDF per-part materials (`body` 0.85, control surfaces `(0.95, 0.45, 0.1)`, `nacelle`/`prop` 0.15) and `room_background`, applies camera faults to pixels (OpenCV radial distortion `k1/k2`, isotropic Gaussian blur from motion smear + defocus, `exp2(stops)` + soft-knee exposure, Gaussian pixel noise, black occluder box) and frame drops via `FaultState.frame_lost()` (with `phantom_rate=0.0`), runs two-stage CPU ONNX `KeypointDetector` (`kp-v3c` + `kp-crop-v3c`, 1 thread/worker), and feeds detected points into `measurement_from_detections`.
2. **Evaluations run (CPU multiprocessing, 14 workers, `config/park_params.json`, seed 11, `--start mixed`, synthetic renders of the real mesh, not real-camera evidence)**:
   - Healthy (`runs/park_eval/network-v3c-healthy`): `n=200` (`10` repeats x `6` Challenge presets = `60` + `140` `random:ring`), `--dump-frames runs/pose_data/dagger-v3c-healthy --dump-max 4000`.
   - Bad sensors (`runs/park_eval/network-v3c-bad`, `--faults 1`): `n=120` (`6` repeats x `6` Challenge presets = `36` + `84` `random:ring`), `--dump-frames runs/pose_data/dagger-v3c-bad --dump-max 4000`.
3. **Training hook**: `train/pose_kp.py --dagger` (`nargs="+"`, `--dagger-frac 0.25`) mixes stage-1 (`320x180`) and stage-2 (`crop/`, `256x256`) DAgger datasets into training batches alongside `RollingPool` (not executed on GPU here).

## Result

| Condition | Run ID | n | Parked | Wilson 95% CI | Collisions | Wall time (14 workers) | Mean tick (render + detect) |
|---|---|---|---|---:|---:|---:|---:|
| Healthy | `network-v3c-healthy` | 200 | **124/200 (62.0%)** | `55.1-68.4%` | 21 | 798.2 s | 147.1 ms (`115.1 + 32.0 ms`) |
| Bad sensors (`--faults 1`) | `network-v3c-bad` | 120 | **12/120 (10.0%)** | `5.8-16.7%` | 17 | 1222.7 s | 205.7 ms (`172.2 + 33.6 ms`) |

**Outcomes & FSM state breakdown:**
- Healthy (`network-v3c-healthy`): `parked` `124`, `parked-off-target` `43` (`43` `ENTER` -> `PARKED`), `collision` `21` (`21` in `ENTER`), `abort:misaligned in the bay; out of retries` `12` (`12` `ENTER` -> `ABORT`). **All 76 healthy failures occurred in `ENTER`.** `43/43` off-target episodes exceeded `5 deg` yaw (`|dyaw|` med `6.63 deg`, max `11.93 deg`), while `0/43` exceeded `2 cm` in `x` (`|dx|` med `0.26 cm`). Collisions had `|dx|` med `12.67 cm`, `|dy|` med `4.05 cm`, `|dyaw|` med `9.18 deg`.
- Bad sensors (`network-v3c-bad`): `parked` `12`, `abort:timeout` `52` (last active: `33` `SEARCH`, `7` `ENTER`, `6` `ACQUIRE`, `5` `APPROACH`, `1` `BACKOFF`), `abort:misaligned in the bay; out of retries` `20` (`ENTER`), `collision` `17` (`8` `ENTER`, `4` `BACKOFF`, `2` `SEARCH`, `2` `APPROACH`, `1` `ORBIT`), `abort:lost in the bay; out of retries` `15` (`ENTER`), `parked-off-target` `4` (`ENTER`).

**By start (`healthy` / `bad`):**
- `tail`: `6/10` (`1` col) / `1/6` (`1` col)
- `nose-on`: `6/10` (`1` col) / `0/6` (`1` col)
- `nose-oblique`: `3/10` (`0` col, `6` off-target) / `0/6` (`1` col)
- `beside-nose-gear`: `5/10` (`1` col, `4` off-target) / `0/6` (`1` col)
- `under-wingtip`: `8/10` (`1` col) / `0/6` (`1` col)
- `side`: `6/10` (`0` col, `4` off-target) / `0/6` (`3` col)
- `random:ring`: `90/140` (`64.3%`, `17` col) / `11/84` (`13.1%`, `9` col)

**Keypoint error & missed-view decomposition:**
- Healthy (`70,133` rendered frames, `1.83%` missed airframe views, `0.001` false pts/empty): `hub_left` recall `0.8593`, mean bias `(-3.62, -8.79) px`, med `10.87 px`, p90 `32.73 px`; `hub_right` recall `0.8843`, bias `(+3.50, -4.49) px`, med `8.23 px`, p90 `25.93 px`; `wingtip_left/right` med `13.32 / 14.80 px` (p90 `37.84 / 41.67 px`); `vtail_tip_right` recall `0.5486`; `nose` recall `0.7278`; `gear_nose` recall `0.9884` (med `2.86 px`); `tyre_left/right/top` recall `0.9965/0.9950/0.9982` (med `2.45/2.03/1.97 px`).
- Bad sensors (`75,547` rendered frames, `25.51%` missed airframe views, `0.010` false pts/empty): `hub_left` recall `0.7565`, bias `(-4.91, -9.45) px`, med `13.11 px`; `hub_right` recall `0.7680`, bias `(+4.83, -5.81) px`, med `10.72 px`; `gear_nose` recall `0.8313` (med `5.16 px`); `tyre_*` recall `0.829-0.836` (med `5.42-5.99 px`).
- Bad-sensor fault-family lifts: `focus` `0/55` (`0.0%` vs `18.5%`, `-18.5 pp`), `broken` `1/63` (`1.6%` vs `19.3%`, `-17.7 pp`: `camera goes dark` `0/15`, `gyro sticks` `0/12`, `a wheel dies` `0/19`, `gyro reads zero` `1/17`), `lens` `4.8%` vs `15.5%` (`-10.7 pp`), `wheel` `4.9%` vs `15.3%` (`-10.3 pp`).

**DAgger dumps written:** `runs/pose_data/dagger-v3c-healthy` (`4,000` frames at `320x180` + `4,000` crops at `256x256`) and `runs/pose_data/dagger-v3c-bad` (`4,000` + `4,000`).

## Lesson

The sampled `DetectorModel` (`980/1000`, `0` collisions) overstates `kp-v3c + kp-crop-v3c` closed-loop reliability (`124/200` healthy with `21` collisions, `12/120` bad-sensor with `17` collisions): all `76` healthy failures happen in `ENTER` from `hub_left`/`hub_right` vertical bias (`-8.8 / -4.5 px` on dark rotors vs training procedural textures) biasing yaw just prior to the bay, while bad sensors lose `25.5%` of airframe views (especially fixed-focus defocus `0/55` and broken-sensor runs without `gyro_check`/`camera_silent_s` enabled in `config/park_params.json`).

## Re-run only if

Retraining with per-part albedo randomization, stronger blur/defocus/occlusion, and the `dagger-v3c-{healthy,bad}` pools (e.g. `cd train && uv run python pose_kp.py --data ../runs/pose_data/val-v3 --out ../runs/pose_train/kp-v4 --epochs 80 --pool 12000 --bay 0.20 --workers 14 --dagger ../runs/pose_data/dagger-v3c-healthy ../runs/pose_data/dagger-v3c-bad --dagger-frac 0.25`) or controller/health-check changes produce a new candidate.
