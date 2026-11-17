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
- **Skeleton**: 12 named keypoints from mesh extremes (`src/ugv/pose/mesh.py`): nose, tail,
  wingtips, three rotor hubs, two V-tail tips, nose wheel, two main wheels. The wheels sit on
  the floor, which adds the ground-plane constraint (z = 0 at the contacts) for free.

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

Each state has an entry condition, an exit on success and an exit per failure mode; the table
lives in `src/ugv/fsm.py` so an undeclared transition fails in tests.

| State | Does | Leaves on success | Failure modes and exits |
|---|---|---|---|
| SEARCH | rotate in place, scan for the airframe | detection held for N frames | timeout: stop and report |
| ACQUIRE | hold still, let the pose estimate settle | pose variance below a bound | pose jumps (left/right flip): back to SEARCH |
| ORBIT | strafe on a circle around the airframe while yawing to keep it centred | car on the bay axis behind the tail | pose lost: SEARCH; obstacle (shield): stop |
| APPROACH | drive in along the bay axis, correcting lateral error | distance under 0.4 m | drift off axis: back out to ORBIT |
| CAREFUL_PARK | slow; dead-reckon from the last good pose plus gear contacts | end-frame check passes | collision risk or timeout: stop, back out |
| PARKED | stop, confirm with the end frame | | |

The orbit-and-face strategy keeps the airframe in view for as long as possible, which is when a
model-based pose is reliable; the blind finale is short and starts from a converged pose.
Simulate the machine against pose noise drawn from the measured detector error before any
motor run (S4 gate, `docs/SAFETY.md`).

## Open questions

- Supervisor: "SLAM for communication" (`ask-supervisor-slam`). Readings: a shared map so
  either vehicle can localise the other, or visual SLAM on the car with the airframe as a
  landmark. Feature placement on the airframe is part of the same question.
- Which airframe points touch the floor, measured on the real airframe (the mesh says nose
  wheel plus two main wheels).

## Prior analysis

The 2026-11-14 council reports (`docs/archive/council_2026-11-14/`) studied this before the
final model existed: 03 airframe geometry and bay corridor, 06 keypoint CNN, gates and Pi
runtime, 07 LeJEPA encoders on the Pi, 08 ground-plane homography from gear contacts and
fiducial alternatives. Their numbers predate the final mesh; re-derive before citing.
