# Parking simulator

`src/ugv/park/` simulates the whole task, find the airframe and park under it, with every
piece that decides success on the real car taken from measurement or from the 3D model. CPU
only, S0. Results: `uv run ugv-lab` (Park page: results, replay, tuning); history: the attempt
tree under `research/attempts/2026-11-17-track-3d-parking.md`.

## What is modelled, and from where

| Piece | Module | Source |
|---|---|---|
| Bay, obstacles, target | `geometry.py` | the STRL-001 mesh: every surface point lower than the car (body 0.044 m, camera stand 0.084 m) is an obstacle; the target is the deepest pose between the main-gear axle and the nose wheel whose straight tail entry clears by the most (3.5 cm per side) |
| Car | `plant.py` | measured gains, deadzones and onsets (`docs/L7_FIELD_NOTES.md`), duty cap 0.44; each episode draws a different car: gains ±25%, deadzones ±40%, lags ±50%, strafe-to-yaw slip, gyro vibration noise |
| Camera and detector | `perception.py` | the 12 skeleton keypoints projected through the car camera (1280x720, fx 968, 6.4 cm off the floor); 2 px noise, 10% misses, 3% gross outliers; per-episode focal (2%), mount yaw (1 deg) and pitch (1.5 deg) errors; 10 Hz, 0.1 s late |
| Pose solver | `perception.solve` | 6-DoF PnP when 6+ points are seen, then a floor-constrained solve of x, y, yaw (and camera pitch with 6+ points); with 2-5 points, from the estimator's prior |
| Estimator | `estimator.py` | the airframe as a static landmark in an odometry frame; gated fusion; odometry through the identified motor model and lag |
| Motor model | `actuator.py` | gain and deadzone per axis identified online: yaw from the gyro, forward and strafe from fix-to-fix motion |
| Behaviour | `controller.py` | the state machine in `docs/POSE.md`; the careful ENTER routine calibrates each move from the last |

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
| blur, noise | motion blur with turn rate; sensor noise | 2-10 px per rad/s; 0.5-2 px |
| occlusion | a fixed region of the image blocked | 2-20% of the image |
| shake | mount jitter per frame (also blurs) | 0.2-1 deg |
| bump | the mount knocked once, at a random time | 1-4 deg |
| lens | focal, principal point, radial distortion the solver does not know | ±6%, ±25 px, k1 -0.2..0.1 |
| focus | fixed focus: defocus blur grows with distance from the focus | focus 0.3-3 m, 2-8 px/dioptre |

`--start close` places the car 0.4-1.3 m from the airframe on any side (nose and wingtips
included). The challenge page drives the same faulty cars live, and named awkward starts
(nose-on, beside the nose wheel, under a wingtip) are in `ugv.lab.sim.PRESETS`.
Attempts: `research/attempts/2026-11-18-park-*`. Magnitudes are guesses until measured on
the car; the per-family sweep (`eval.py --families`) says which ones matter.

## Results (seed 11, 1000 episodes, full noise)

| Version | Parked | 95% CI | Collisions | Record |
|---|---|---|---|---|
| v6, default parameters | 872/1000 | 85.0-89.1% | 15 | `runs/park_eval/default.json` |
| v7 (3-5 point fixes update yaw), default parameters | 882/1000 | 86.1-90.1% | 7 | `runs/park_eval/v7-default.json` |
| v7, parameters from search s1 | 899/1000 | 87.9-91.6% | 13 | `runs/park_eval/s1-best.json` |
| **v7, parameters from search s2 (`config/park_params.json`)** | **909/1000** | **89.0-92.5%** | **7** | `runs/park_eval/s2-best.json` |

All four are the same 1000 episodes. Search s1 scored a collision like any other failure and
bought success with collisions; s2 rejects any candidate that collides more than its parent and
beats both. Use `--params config/park_params.json`.

Robustness sweep of v6 `default` (300 episodes each): ideal world 100%, detector noise only 99%,
car spread only 95%, worse detector (4 px, 20% misses, 8% outliers) 77%, worse calibration
(2.5 deg mount yaw, 3 deg pitch) 74%, 1.5x wider spread of cars 67%. The residual failures are
mostly final heading just over 5 deg; with a nominal car the 90th-percentile heading error is
0.5 deg, so it comes from the car spread, not the camera.

## Commands

```sh
uv run python tools/park/eval.py --name <name> [--params <json>] --sweep --traces 16
uv run python tools/park/search.py --out runs/park_search/<name>     # failure-guided tuning
uv run ugv-lab                                                       # dashboard, port 8768
```

## What it does not show

The simulator cannot say whether the real detector reaches 2 px on real frames, whether the
mesh's gear geometry matches the airframe (its scale is provisional), or how the floor and
battery change the plant. Those are the real evaluation set, a tape-measure check of the gear
and the S4-gated car runs in `research/tasks/`.
