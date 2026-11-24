# Pose of the airframe from the car camera

Direction set on 2026-11-17 (`research/decisions/2026-11-17-pose-from-airframe-model.md`):
estimate the 6-DoF pose of the STRL-001 airframe from one car-camera image, using its final 3D
model, then park with a declared state machine. Tags stay only as ground truth for evaluation.
Live work and its order: `uv run python tools/research/ledger.py roadmap`.

## The problem, stated precisely

- **Object**: one rigid airframe (`simulation/gazebo/strl_001/meshes/`, model frame x nose,
  y left, z up, metres; 1.22 m span). Control surfaces are held at zero, so the skeleton is
  rigid and the pose is `(R, t)` in SE(3); there is no articulated pose to estimate.
- **Camera**: the car camera, 0.064 m off the floor, 1280x720, fx = 968 px (66.9 deg
  horizontal FOV). It looks slightly up at the belly from 0.2-3.5 m.
- **Output the parking needs**: the airframe's position and heading in the car frame (x, y,
  yaw on the floor) at 10 Hz or more, with an error budget set by the bay: about 2 cm and 2 deg
  in the last 0.4 m (`docs/PARKING.md`), looser further out.
- **Skeleton**: 15 named keypoints from mesh extremes (`src/ugv/pose/mesh.py`, `keypoints(..., near=True)`):
  12 airframe points (nose, tail, wingtips, three rotor hubs, two V-tail tips, nose wheel, two main
  wheels) plus 3 near-field nose-tyre points (`tyre_left`, `tyre_right`, `tyre_top`, indices 12-14)
  that stay in view inside the bay when the rest of the airframe is cut off. The wheels sit on the
  floor, which adds the ground-plane constraint (z = 0 at the contacts) for free.

## How model-based pose estimation works, and what we use

The task is *instance-level* 6-DoF pose: one known, rigid object with an exact CAD model.
That is the easiest version of pose estimation; the hard parts are the domain gap (synthetic
training, real camera) and small, distant or partly visible views. The standard families:

| Family | Idea | Fit here |
|---|---|---|
| Keypoints + PnP (BB8, PVNet, HRNet/YOLO-pose style heatmaps) | a CNN finds a few named 3D points in the image; PnP turns them into pose | **first choice**: small, fast, exact geometry, runs on a Pi |
| Dense correspondences (DPOD, GDR-Net, ZebraPose) | every pixel predicts its 3D model coordinate; PnP-RANSAC | more robust to occlusion; heavier |
| Direct regression | the network outputs R, t | least accurate; no geometry |
| Render-and-compare (DeepIM, CosyPose, MegaPose, FoundationPose) | render the model at a guess, compare, refine | most accurate; desktop GPU, not the Pi; a good offline ground-truth tool |
| Classic features (SIFT/ORB vs reference renders) | match image features to features of known 3D points | no training; needs texture; the supervisor's suggestion, kept as a baseline |

Two facts of this task make it easier still, and the code uses both
(`src/ugv/park/perception.py`):

1. **Both bodies stand on the floor.** The relative pose has 3 unknowns (x, y, yaw), not 6.
   Two keypoints already fix x and y; three or more spanning at least 0.65 m horizontally fix yaw
   (on a tight cluster such as the 5 cm nose wheel plus at most one hub, yaw stays at the
   gyro/prior). That is why the car keeps measuring deep into the bay where only the nose wheel
   and tyre points are in view.
2. **The camera mount is constant.** Its pitch is solved as a 4th unknown whenever 6+ well-spanned
   points are visible and then held, so the car calibrates its own mount as it drives.

### Robustness: size, lighting, background, "fake but realistic", angles

| Threat | Training-side answer | Run-time answer |
|---|---|---|
| Size (distance) | log-uniform 0.25-3.5 m plus 20% bay-range views; heatmaps at stride 4 with soft-argmax and spatial per-keypoint visibility | two-stage crop-and-zoom plus a tight 256x256 nose-wheel refinement crop when the main crop exceeds 400 px |
| Lighting | random light direction and gain, gamma, blur, sensor noise | auto-exposure off, fixed shutter (4 ms worked on the car) |
| Background | composite the render over **real car-camera frames** of the lab (9,045 in `calibration/runs`), noise fields and flat-shaded rooms | the residual gate in the solver rejects stray points |
| Texture of the real airframe | random procedural textures, plain paint and the fixed texture (per-part rotor/tyre contrast planned for v4) | the detector learns shape, not paint |
| Synthetic-to-real gap | domain randomisation (`ugv.pose.dataset.RollingPool`), then fine-tune on real frames labelled by the tags | evaluate only on held-out / real sets (`research/decisions/2026-11-24-offline-pose-eval-and-reliability-gate.md`) |
| Viewing angle | 55% level car-camera views (including empty away-facing views), 20% bay run-in, 25% sphere/low aimed | the orbit keeps the airframe in view from the side and the tail |

Training uses a rolling generated pool (`train/pose_kp.py --pool`, `ugv.pose.dataset.RollingPool`,
~95k unique generated views per run against fixed `val-v3` / `val-crop-v3`) rather than a fixed
24k dataset on disk. The decisive data remains small and real: the 50-100 frame evaluation set
(`2026-11-17-real-pose-eval-set`) and, later, real frames labelled by the tags during drives.

### Current detector state (2026-11-24, training paused)

- **Active default**: `runs/pose_train/{kp-v3c,kp-crop-v3c}` (`DEFAULT` in `src/ugv/pose/detector.py`);
  older checkpoints (`v1`, `v2`, `v3a`, `v3b`) and fixed 24k sets are removed from `runs/` with
  their results and failure modes preserved in `research/attempts/`.
- **Measured on frozen held-out Challenge sets** (`2026-11-24-pose-detector-v3c-rolling-pool`,
  `2026-11-24-pose-detector-v3c-autopilot-gate`): healthy `v3h` (1,261 frames) precision 0.888,
  median 4.5 px, 6+ correct on 56.1% of under-1m frames (68.6% with 6+ in view), 0.11 false
  pts/empty; bad-sensor `v3a` (2,683 frames) under-1m with 6+ in view 53.7%; `diag1` (12-pt)
  0.02 false pts/empty, precision 0.881, cut-off 7.4%.
- **Next handoff** (training paused until offline gates exist):
  `research/tasks/2026-11-24-pose-detector-v4-reliability.md` and
  `research/decisions/2026-11-24-offline-pose-eval-and-reliability-gate.md`.

### Execution on a Raspberry Pi 4

The trained detector (`train/pose_kp.py`) is MobileNetV3-small with a stride-4 heatmap head and
per-keypoint attention-pooled visibility: 1.22 M parameters, 320x180 mono stage 1 + 256x256
stage 2, **~8.9 ms + ~9.5 ms on one desktop CPU thread** with ONNX Runtime.
A Pi 4's Cortex-A72 is several times slower: expect roughly 40-80 ms per frame (int8
quantisation roughly halves it). Measuring this on the Pi is a task. That is fast enough
because the estimator fuses fixes at whatever rate arrives (the simulator assumes 10 Hz with
100 ms latency) and odometry carries the pose in between. Heavy models (render-and-compare,
FoundationPose) belong on the desktop GPU, as ground truth or over the network.

### World model + action head, or a mechanistic controller?

The parking controller in `src/ugv/park/` is itself a small **mechanistic world model**: a
motor model identified online (gain and deadzone per axis), a lag model, an estimator, and a
declared state machine that plans against them. It parks 87% of 1000 randomised simulated
episodes (Lab dashboard, Parking page), every failure is labelled, and nothing needs a GPU.

A learned alternative is a LeJEPA encoder (fine-tuned, not frozen) feeding an
action-conditioned latent predictor (the world model) and a small action head trained by
imitating the state machine in simulation. On a Pi 4 that is affordable only if the encoder is
the same small backbone as the detector (an MLP head costs nothing). Recommendation: keep the
mechanistic controller for motion (it is interpretable, testable and already robust to the
measured plant spread) and spend learning on perception, where the open risk is. Revisit the
learned policy if real runs show failures the mechanistic model cannot express.

## What the smoke test already says (synthetic)

`tools/pose/smoke.py`, records `research/attempts/2026-11-17-pose-smoke-*`:

- PnP on the true keypoints with 1 px noise recovers pose within 5 deg and 10 cm on all solved
  views (median 0.5 deg, 1.5-2 cm). At 3 px about half the views fail that bar. The detector's
  budget is therefore about 1 px at 640x360.
- Close views (the careful-parking range) often show fewer than 6 keypoints: the finale needs
  another cue (gear contacts on the floor plane, the belly texture, or dead reckoning).
- SIFT matching against 48 random reference renders fails completely with the current smooth
  procedural texture. This says the texture and bank design matter, not that SIFT fails on the
  real airframe.

## Experiments, in dependency order

Each one is a task in `research/tasks/`; its result is an attempt record.

1. **Real evaluation set** (`real-pose-eval-set`). 50-100 car-camera frames of the real
   airframe at parking distances, ground truth from the tags (14/6/3) or measured placement.
   Everything below is judged on these frames; synthetic numbers are only a debugging aid.
2. **Airframe texture** (`airframe-texture`). The airframe's appearance is a design variable.
   Candidates: photo-texture the mesh from real photos; print a corner-rich seeded noise
   pattern on stickers at known model coordinates; or optimise a procedural texture for
   detectability in simulation. Score each by SIFT repeatability across synthetic views.
   PySR (symbolic regression) only fits the last option: a search over small closed-form
   texture functions scored by that repeatability. It is a search tool, not a detector.
3. **Model-based feature matching** (`sift-bank-v2`). Supervisor's suggestion: references on a
   viewing sphere, rendered with the chosen texture, each with a triangle-id map so every
   matched feature has a 3D point; match per reference, keep the best, PnP + RANSAC. No
   training. Also try learned matchers (SuperPoint + LightGlue) if SIFT is weak at range.
4. **Keypoint detector** (`keypoint-detector`). Heatmap CNN for the 12 keypoints, trained only
   on synthetic views from `src/ugv/pose/render.py`, with domain randomisation: random and
   real-photo backgrounds, intrinsics and distortion jitter, exposure, blur, noise, partial
   crops, occluders. Pose = PnP; a reprojection gate and a left/right chirality test reject
   wrong solutions (council report 06 found 4-point solutions accept about 10% of swapped
   labels; 6 or more points rejected them all).
5. **LeJEPA encoder** (`lejepa-pose-finetune`). Does a LeJEPA-pretrained encoder, fine-tuned
   rather than frozen, or trained jointly with the keypoint loss, beat the ImageNet baseline on
   the real set? Three arms with equal data and budget, several seeds.
6. **Pose on the car** (`pose-on-car`, milestone). The best method runs live; latency is
   measured on the Pi (or the desktop over the network).
7. **Parking state machine** (`parking-state-machine`, milestone). See below.

## Parking as a declared state machine

Implemented and scored in simulation: `src/ugv/park/` (`docs/PARK_SIM.md`). Each state has an
entry condition, an exit on success and an exit per failure mode; the table is
`ugv.park.controller.TABLE` on top of `src/ugv/fsm.py`, so an undeclared transition fails.

| State | Does | Leaves on success | Failure modes and exits |
|---|---|---|---|
| SEARCH | rotate in place, scan for the airframe | detection held for N frames | timeout: stop and report |
| ACQUIRE | hold still, let the pose estimate settle | pose variance below a bound | pose jumps (left/right flip): back to SEARCH |
| ORBIT | strafe on a circle around the airframe while yawing to keep it centred | car on the fuselage line behind the tail | pose lost: SEARCH; pose jump: ACQUIRE |
| APPROACH | drive along the line to the staging point, correcting lateral offset and heading | aligned at the staging point | lost far out: SEARCH; shield: BACKOFF |
| ENTER | careful routine: stop, measure still, one small move (gyro turn, strafe, or a fraction of the remaining distance), repeat; each move's result calibrates the next | at the target | misaligned or shield: BACKOFF (bounded retries) |
| BACKOFF | reverse out along the line | back at the staging point | lost: SEARCH; out of retries: ABORT |
| PARKED | stop, confirm with the end frame | | |

The orbit-and-face strategy keeps the airframe in view for as long as possible, which is when a
model-based pose is reliable; the blind finale is short and starts from a converged pose.
Simulate the machine against pose noise drawn from the measured detector error before any
motor run (S4 gate, `docs/SAFETY.md`).

## Open questions

- SLAM: resolved as standard SLAM; plan in `research/decisions/2026-11-17-slam-meaning.md`.
- Which airframe points touch the floor, measured on the real airframe (the mesh says nose
  wheel plus two main wheels).

## Prior analysis

The 2026-11-14 council reports (`docs/archive/council_2026-11-14/`) studied this before the
final model existed: 03 airframe geometry and bay corridor, 06 keypoint CNN, gates and Pi
runtime, 07 LeJEPA encoders on the Pi, 08 ground-plane homography from gear contacts and
fiducial alternatives. Their numbers predate the final mesh; re-derive before citing.
