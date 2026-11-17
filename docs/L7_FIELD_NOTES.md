# L7 Field Notes (2026-11-13)

Historical record of the 2026-11-13 runs. Paths under `rl/` refer to the retired 2D RL stack at tag `pre-pose-20261117`.

Terse record of real-car L7 tests on 2026-11-13. Evidence lives under `calibration/runs/l7/`.

## Harness (`tools/l7/`)

- `tools/l7/l7_run.py` (+ `tools/l7/deploy.sh`, `tools/l7/run.sh`, `tools/l7/fit_probe.py`, `tools/l7/pull_frames.sh`) runs on the Pi over the dashboard websocket while preserving the `600 ms` watchdog and `1800/4095` PWM cap.
- Modes: `probe`, `stage`, `policy`, `watch`. Logs `ticks.jsonl`, `frames.jsonl`, `imu.jsonl`, `meta.json`, `summary.json`, and sampled JPEGs under `calibration/runs/l7/`.
- Frame convention: `rl/sim.py` axes are rotated 90 deg from the camera; handled by the `rl_sim` remap, removed with the policy mode.

## Student Policy on Real Car (`rl/runs/gazebo_l6/student_L1_best.npz`)

- Real car: `3` runs, `0` parked:
  - ep1 (`calibration/runs/l7/20261113-182150-policy-ep1-front/summary.json`): timeout (`42.96 s`), oscillating around `~1.3 m` (`end_est y=1.39 m`).
  - ep2 (`calibration/runs/l7/20261113-182645-policy-ep2-mean/summary.json`): `lost` (`11.31 s`), lost tag after strafe-induced yaw.
  - ep3 (`calibration/runs/l7/20261113-183857-policy-ep3-p0/summary.json`): timeout (`45.98 s`), oscillating while drifting from `0.57 m` (`acquired_pose y=0.568 m`) to `1.10 m` (`end_pose_tag y=1.103 m`).
- Sim comparison: same student scores `954-966/1000` on L1 in the retired 2D sim and `0/20` in Gazebo L6 (timeouts); records under `research/attempts/` (root `2026-11-17-track-2d-rl`).

## Cause Analysis

- Turn action saturates: `|a2| > 0.8` on `74%` of ep3 ticks (`calibration/runs/l7/20261113-183857-policy-ep3-p0/ticks.jsonl`).
- Offline replay of ep3 with the gyro-rate input zeroed yields `0%` saturation (`|a2| > 0.8`).
- Real gyro-rate std is `0.63 rad/s` vs `0.002` in sim (`rl/sim.py`).
- Student forward command also scales with valid-pose rate.
- `[inferred]` Gyro-rate noise/magnitude mismatch drives `a2` saturation, and lower valid-pose rate suppresses forward progress.

## Plant Identification (`calibration/runs/l7/20261113-181040-probe-axes/fit.json`)

- Forward: `2.3 m/s` per duty (`2.319`), deadzone `0.05` (`0.051`), onset `0.18 s`.
- Strafe: `~1.6 m/s` per duty, deadzone `~0.13`, onset `0.25-0.37 s`.
- Yaw: `15.7 rad/s` per duty (`897.2 deg/s`), deadzone `0.10` (`0.101`), onset `0.07 s`.
- Sim profile (`simulation/gazebo/mecanum_bayesian_profile.json`): `deadzone = 0.0`; particle 0 gives `2.49 rad/s` at full turn command with `~1 m/s` translation coupling.

## Vision, Camera & Drone Tag Constellation

- Real drone tags (`DICT_4X4_50`, `50 mm`, loose hanging paper cards with almost no white quiet zone on one edge, tilted/rolled `~10-25 deg`; `calibration/runs/l7/tagmap.json`, `simulation/gazebo/drone_model.py`, `simulation/gazebo/vision_in_loop.py`):
  - Tag `14`: `+y` entry side, `x=-0.028 m`, `y≈0`, `z≈0.14 m`, pitched `~25 deg`.
  - Tag `6`: `+x` side, online co-observation estimate relative to tag 14 `x +0.14, y -0.09` (low confidence, `26` obs; `calibration/runs/l7/20261113-181622-stage-front/summary.json`).
  - Tag `3`: `-x` side, estimate `x -0.13, y 0.0` (low confidence, `93` obs; `calibration/runs/l7/20261113-183626-watch-spawn/summary.json`).
  - Fourth `-y` tag: ID unreadable due to wires across it (omitted from sim constellation).
- Stock detector found tag `6` in `~25%` of frames (cards lack a quiet zone on one edge).
- Pi camera shutter was `16 ms` at gain `3`, blurring tags during yaw (`calibration/runs/l7/vision-camera.conf.bak-20261113`); changed on Pi to `4 ms` / gain `8` in `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf` (backup at `~/l7_runs/vision-camera.conf.bak-20261113` on Pi).
- Half-resolution detection + full-res corner refinement + KLT fallback yields tags in `97%` of frames in motion on one staging run, `~13 poses/s`, `~65 ms` latency.

## Classical Go-To Baseline & Geometry / Process Limits

- Go-to P controller (`stage` mode) reached target pose in `7-11 s` on `4` runs (`4 cm` / `5 deg` tolerance):
  - `calibration/runs/l7/20261113-181438-stage-toD/summary.json` (`10.84 s`)
  - `calibration/runs/l7/20261113-181622-stage-front/summary.json` (`10.0 s`)
  - `calibration/runs/l7/20261113-182536-stage-front3/summary.json` (`7.38 s`)
  - `calibration/runs/l7/20261113-183026-stage-front4/summary.json` (`6.49 s`)
- Geometry limit: tag `14` leaves the vertical field of view about `0.4 m` out; tag centre `~0.14 m` above floor.
- Process gap: the Gazebo `0/20` result was not a release gate, and no automatic comparison of real policy inputs against the sim distribution existed.

## Proposals (Not Results)

- Add real gyro-rate noise + identified real plant parameters to sim.
- Add an observation-contract audit comparing real inputs to the sim distribution as a pre-drive gate.
- Add a runtime monitor on action sign-flip and saturation.
- Replace loose paper tags with rigid tag cards carrying a full white border.
- Add an inner yaw-rate control loop.
- Use the classical go-to P controller as the L7 baseline.
