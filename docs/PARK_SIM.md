# Parking simulator

`src/ugv/park/` simulates the whole task, find the airframe and park under it, with every
piece that decides success on the real car taken from measurement or from the 3D model. CPU
only, S0. Results: `uv run ugv-lab` (Park page: results, replay, tuning); history: the attempt
tree under `research/attempts/2026-11-17-track-3d-parking.md`.

## The strict camera protocol (challenge = pipeline)

The batch simulator (`run()`, `tools/park/*`) samples the detector's statistics; the challenge
page does not. One rendered frame feeds both the human's eyes and the estimator, so what the
challenge shows cannot flatter the car:

1. The browser renders the car camera through a real sensor chain: linear HDR scene with the
   true mount (fixed mount error, then per-frame shake and the one-time knock) → the true
   lens (radial distortion k1/k2, principal-point offset, monotone full-well curve with
   saturation) → mono luma gained in stops → 8-bit quantization. Lens defocus and motion blur
   are applied to the pixels (isotropic kernel, as a simplification of directional streak),
   never as a CSS filter on the display.
2. That exact 1280x720 frame is drawn for the human and sent, as mono bytes, to the server,
   where the trained YOLO-pose network reads it (`ugv.pose.detector.YoloPoseDetector`:
   `yolo11n-pose-v3-416-int8`, single-stage ONNX on the CPU). Nothing on this path knows the true
   pose. Without `runs/pose_train/yolo11n-pose-v3-416-int8/model.onnx` the page says so and the car
   gets no camera. How well the earlier two-stage detector read these frames (retired):
   `research/attempts/2026-11-24-pose-detector-v3c-rolling-pool.md` and
   `research/attempts/2026-11-24-pose-detector-v3c-autopilot-gate.md`.
3. The network's 15 keypoints (12 skeleton + 3 nose-tyre points) go to `perception.solve` with
   the nominal camera, exactly as the car would, and the estimator fuses them (`solve_planar`
   requires a horizontal 3D keypoint span >= 0.35 m before freeing yaw, holding yaw from the
   gyro/prior on tight nose-wheel or single-hub clusters). A frame that arrives while the previous one is still
   being read is dropped, as on the car. `Keypoints` and `Network ms` in the HUD come from this
   path.

Still approximated, on purpose: frame drops are drawn browser-side from the round seed; motion
blur is isotropic, not a directional streak; exposure spells layer on top of the fixed setting;
the render adds no sensor noise. The phantom family (a drone-looking impostor in view) is not
fired in the challenge — the picture never contains an impostor. The batch simulator below
still samples a detector model instead of running the network; under the no-stand-ins rule
(`AGENTS.md`) its numbers are not reported or tuned against until it does.

## What is modelled, and from where

| Piece | Module | Source |
|---|---|---|
| Bay, obstacles, target | `geometry.py` | the STRL-001 mesh: every surface point lower than the car (body 0.044 m, camera stand 0.084 m) is an obstacle; the target is the deepest pose between the main-gear axle and the nose wheel whose straight tail entry clears by the most (3.5 cm per side) |
| Car | `plant.py` | measured gains, deadzones and onsets (`docs/L7_FIELD_NOTES.md`), duty cap 0.44; each episode draws a different car: gains ±25%, deadzones ±40%, lags ±50%, strafe-to-yaw slip, gyro vibration noise |
| Floor | `plant.Floor` | traction: strafe and yaw authority lost to the surface, ground velocity lagging the body through turns, and grip that changes from patch to patch; a model, not a measurement |
| Camera and detector | `perception.py` | the 15 keypoints (12 skeleton + 3 near-field tyre points) projected through the car camera (1280x720, fx 968, 6.4 cm off the floor); 2 px noise, 10% misses, 3% gross outliers; per-episode focal (2%), mount yaw (1 deg) and pitch (1.5 deg) errors; 10 Hz, 0.1 s late. The batch path; the challenge renders and analyses the actual frame instead (see The strict camera protocol) |
| Pose solver | `perception.solve` | 6-DoF PnP when 6+ points are seen, then a floor-constrained solve of x, y, yaw (and camera pitch with 6+ points); with 2-5 points, from the estimator's prior |
| Estimator | `estimator.py` | the airframe as a static landmark in an odometry frame; gated fusion; odometry through the identified motor model and lag |
| Motor model | `actuator.py` | gain and deadzone per axis identified online: yaw from the gyro, forward and strafe from fix-to-fix motion |
| Behaviour | `controller.py` | the state machine in `docs/POSE.md`; the careful ENTER routine calibrates each move from the last, but a move shorter than `step_v` times the motor lag (~1.6 cm) reflects the lag, not the gain, and is not used |

Success is judged on the true pose, never the estimate: PARKED within ±2 cm, ±2 cm, ±5 deg of
the target, and no contact at any time.

## Faults and awkward starts

The table above is a healthy car with calibration errors. `faults.py` adds what goes wrong on
the day, per episode, from its own random stream (a fault-free run is bit-identical). At
severity 1 each family is present with probability 0.5:

| Family | What | Magnitude at severity 1 |
|---|---|---|
| wheel | one or two wheels weaker or stronger, through mecanum kinematics | ±10-40% |
| battery | supply vs calibration day, before the deadzone; sags while driving | 80-108%, 0-10%/min |
| stall | one wheel cuts out for 50-300 ms | 0.1-1 per s |
| slip | per-wheel traction loss each tick | 5-20% |
| delay | extra command latency | 20-60 ms |
| drops | camera frames lost, sometimes in runs | 5-30% |
| phantom | a false airframe detection at a plausible range | 1-5% of frames |
| gyro | scale error and wandering bias | ±5%, 0-0.001 rad/s/√s |
| exposure | over/under-exposed spells that hide most keypoints | 0.05-0.3 per s, 0.3-2 s |
| exposure, fixed | auto-exposure off, set permanently high: `exposure_stops`, never clears | +1..+4 stops in the challenge selector |
| blur, noise | motion blur with turn rate; sensor noise | 2-10 px per rad/s; 0.5-2 px |
| occlusion | a fixed region of the image blocked | 2-20% of the image |
| shake | mount jitter per frame (also blurs) | 0.2-1 deg |
| bump | the mount knocked once, at a random time | 1-4 deg |
| lens | focal, principal point, radial distortion the solver does not know | ±6%, ±25 px, k1 -0.2..0.1 |
| focus | fixed focus: defocus blur grows with distance from the focus | focus 0.3-3 m, 2-8 px/dioptre |

`--start close` places the car 0.4-1.3 m from the airframe on any side (nose and wingtips
included). The challenge page drives the same faulty cars live, and named awkward starts
(nose-on, beside the nose wheel, under a wingtip) are in `ugv.lab.sim.PRESETS`. Its exposure
selector pins the car's exposure permanently high (0 = auto) on top of the round's other
faults; it applies live to the running round through the browser's sensor chain, and the
detector reads the same pixels. Attempts: `research/attempts/2026-11-18-park-*`. Magnitudes are
guesses until measured on the car; the per-family sweep (`eval.py --families`) says which ones matter.

## Floor traction

`plant.Floor` is the surface, separate from the car's own faults. Severity `f` (0 clean, 1
polished concrete or a wet bay) draws a grip loss of 5-40% on strafing and yaw, a ground
velocity that lags the body by up to 0.5 s at walking speed (the car keeps sliding when it
stops commanding), and a smooth grip field that varies by up to ±50% over about a metre. A
clean floor integrates the commanded velocity exactly, so `floor=0` leaves every earlier
result unchanged. The controller sees only the gyro and the camera; it has to learn the
surface through its own motion.

`eval.py --floor 1` scores the main run on the wet-floor draw; `--sweep` includes
`slippery` and `slippery+faults`. The challenge exposes it through the environment selector:
the lab is almost clean, the polished hangar is 0.55, the wet loading bay is 1.0, and a real
Poly Haven panorama gets 0.25. With `config/park_params.json`, 300 episodes, seed 11:

| Condition | Parked | Collisions |
|---|---|---|
| clean | 297/300 | 0 |
| slippery | 287/300 | 3 |
| faulty | 61/300 | 49 |
| faulty + slippery | 62/300 | 49 |

(`runs/park_eval/traction-2026-11-18-floor.json`.) Traction is not the main fault cost: a bad
car already collides often, and the floor adds little to that. Severity 1 is a stress test that
can draw several fault families at once; it is not a typical car. Both numbers are simulation;
the real floor is unmeasured.

## Camera, faults and panoramas in the challenge

The car camera is 1280×720 at fx 968, and the page draws exactly that sensor frame: the
full-screen ride and the inset both render the same 16:9 rectangle with the same letterbox,
the same exposure and motion blur at the same visual scale, and the same blocked region. A
dropped frame paints the frame black and says NO SIGNAL instead of leaving the previous image
up. The orbit view is an observer, not the sensor: it never receives the camera's faults.

The environment selector has six procedural sets plus the merged CC0 panorama set indexed in
`src/ugv/lab/static/hdri.json`: Poly Haven's original 998 (preserved in `static/hdri-polyhaven.json`)
and 213 pre-fetched captures from OpenHDRI (43 outdoor, 17 indoor, 60 total, CC0) and ambientCG
(Indoor/Day/Evening/Night Environment HDRIs, ~153 captures, CC0). The extended panoramas are
pre-downloaded at 2K by `tools/hdri_horizons/fetch_panos.py` into `runs/cache/hdri_v2/`; the lab
server serves those files directly (ids `ohl_*`, `acg_*`), falling back to Poly Haven's CDN for the
plain ids. The repository therefore carries no panorama data. The panorama is both the scene and the
image-based light. It renders on a grounded dome whose equator is the y=0 floor plane (a
directional equirect sample, not the infinite `scene.background`), so the photograph's horizon sits
at floor level at every azimuth, and an opaque ground disc at y=0 deprojects the panorama's ground
half onto the floor plane: each floor point samples the panorama pixel that photographed that
distance from the capture point (camera height ≈ 1.55 m, `uEyeH` in `scene3d.js`), so the ground
stays at true scale from orbit height down to the car camera's 6 cm. Shadows still land on a
transparent catcher riding just above the disc. A
panorama whose camera was not level can ship a per-id horizon offset in `static/hdri-horizons.json`
(absent ids default to 0), shifting the dome's sampled `v` so the photograph's floor/wall junction
lands on the y=0 plane. `tools/hdri_horizons/predict_horizons.py` prefills the offsets from the
cached `.hdr` files; `tools/hdri_horizons/dashboard.py` is the review UI (pitch/roll scrubbing,
reviewed flags in `static/hdri-horizons-reviewed.json`).

## Results (seed 11, 1000 episodes, full noise)

| Version | Parked | 95% CI | Collisions | Record |
|---|---|---|---|---|
| v6, default parameters | 872/1000 | 85.0-89.1% | 15 | `runs/park_eval/default.json` |
| v7 (3-5 point fixes update yaw), default parameters | 882/1000 | 86.1-90.1% | 7 | `runs/park_eval/v7-default.json` |
| v7, parameters from search s1 | 899/1000 | 87.9-91.6% | 13 | `runs/park_eval/s1-best.json` |
| v7, search s2 parameters before the blind gate | 909/1000 | 89.0-92.5% | 7 | `runs/park_eval/s2-best.json` |
| v7, s2 + blind gate | 979/1000 | 96.8-98.6% | 1 | `runs/park_eval/s2-blind.json` |
| **v7, current config (s2 + blind gate, ENTER calibrates only moves over the motor lag)** | **980/1000** | **96.9-98.7%** | **0** | `runs/park_eval/s2-floor.json` |

The first four are the same 1000 episodes. Search s1 scored a collision like any other failure
and bought success with collisions; s2 rejects any candidate that collides more than its parent
and beats both. The current config adds the blind gate that a later attempt recorded as kept
(`research/attempts/2026-11-18-park-blind-config.md`). The ENTER calibration floor
(`2026-11-18-park-drive-check`: a move shorter than `step_v` times the motor lag measures the
lag, not the gain) is the only difference from the 979 row; the 980 row is a fresh run, so the
pair is not a controlled comparison. Use `--params config/park_params.json`.

Robustness sweep of v6 `default` (300 episodes each): ideal world 100%, detector noise only 99%,
car spread only 95%, worse detector (4 px, 20% misses, 8% outliers) 77%, worse calibration
(2.5 deg mount yaw, 3 deg pitch) 74%, 1.5x wider spread of cars 67%. The residual failures are
mostly final heading just over 5 deg; with a nominal car the 90th-percentile heading error is
0.5 deg, so it comes from the car spread, not the camera.

## Closed-loop image evaluation (`--perception network`)

`tools/park/eval.py --perception network` closes the loop in Python without a browser: on each
10 Hz camera tick it renders the 1280x720 frame from the true plant pose using `ugv.pose.render`
with the SDF per-part airframe materials (`body` 0.85, control surfaces orange, `nacelle`/`prop`
0.15) and `room_background`, applies image-level camera faults (radial lens distortion, Gaussian
blur, exposure stops/spells, sensor noise, occluder) directly to the pixels (`src/ugv/park/camera.py`),
runs `ugv.pose.detector.YoloPoseDetector` (`yolo11n-pose-v3-416-int8`) on CPU, and feeds the
detections into `perception.measurement_from_detections`. These numbers are **synthetic renders of
the real mesh, not real-camera evidence**.

With `config/park_params.json`, seed 11, `--start mixed` (named Challenge starts plus random `ring`
starts):

| Condition | n | Parked | 95% CI | Collisions | Record |
|---|---|---|---|---|---|
| healthy (60 presets + 140 ring) | 200 | 124/200 (62.0%) | 55.1-68.4% | 21 | `runs/park_eval/network-v3c-healthy.json` |
| bad sensors (`--faults 1`, 36 presets + 84 ring) | 120 | 12/120 (10.0%) | 5.8-16.7% | 17 | `runs/park_eval/network-v3c-bad.json` |

All 76 healthy failures occurred in `ENTER` (`43` `parked-off-target` all exceeding the 5 deg yaw
bound, `21` `collision`, `12` `abort:misaligned in the bay; out of retries`), driven by vertical
bias on the dark-rotor hubs (`hub_left` `-8.79 px`, `hub_right` `-4.49 px`). Under `--faults 1`,
missed airframe views rise from `1.83%` to `25.51%` (`52` timeouts, `33` last active in `SEARCH`,
and `0/55` parked when fixed-focus defocus is active); see
`research/attempts/2026-11-27-offline-park-network-eval.md`.

## Commands

```sh
uv run python tools/park/eval.py --name <name> [--params <json>] --sweep --traces 16
uv run python tools/park/eval.py --name <name> --perception network --params config/park_params.json --start mixed --n 200 --dump-frames runs/pose_data/<name>
uv run python tools/park/eval.py --name wet --params config/park_params.json --floor 1
uv run python tools/park/search.py --out runs/park_search/<name>     # failure-guided tuning
uv run ugv-lab                                                       # dashboard, port 8768
```

## What it does not show

The simulator cannot say whether the real detector reaches 2 px on real frames, whether the
mesh's gear geometry matches the airframe (its scale is provisional), or how the floor and
battery change the plant. Those are the real evaluation set, a tape-measure check of the gear
and the S4-gated car runs in `research/tasks/`.
