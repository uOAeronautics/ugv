# L7 Preflight Checklist (real robot, no actuation until all green)

Scope: Pi dashboard (`src/ugv/web.py`) deploy readiness. No S3/S4 motion until
every box is checked. Confirmation flags live in `config/parking.json`.

## 1. Camera intrinsics
- [ ] Calibrate camera matrix + distortion at deployment resolution (1280x720).
- [ ] Set `camera_matrix`, `distortion` in `config/parking.json`.
- [ ] Set `camera_intrinsics_confirmed: true`.
- [ ] Verify ArUco range/pose against tape measure at 0.5/1.0/2.0 m.

## 2. Camera mount
- [ ] Fix pan/tilt servos; record pulses in `camera_mount_servo`.
- [ ] Measure `camera_to_car` transform (translation + rotation).
- [ ] Set `camera_mount_confirmed: true`.
- [ ] Confirm dashboard pan/tilt matches calibrated mount (blocker otherwise).

## 3. Gear clearance + vehicle envelope
- [ ] Measure vehicle width/length/height; set `vehicle.*`.
- [ ] Measure gear `height_clearance_m`, `inward_slope`; set `gear.confirmed: true`.
- [ ] Set `parking_goal` pose; verify `clearance_margin_m + sigma` inflation.
- [ ] Confirm `/api/parking/plan` returns a route with no blockers.

## 4. Rotation-trial dynamics refit
- [ ] Run bounded rotation trials (wheels-off-ground first, then clear-zone S4).
- [ ] Refit `motion_model` (forward/reverse mps, left/right dps, stops).
- [ ] Set `motion_model.confirmed: true`.
- [ ] Current values are nominal medians with filter/perspective/slip error (see config notes).

## 5. Confirmation flags (all must read true / non-null)
- [ ] `tag_size_confirmed`, `camera_intrinsics_confirmed`, `camera_mount_confirmed`
- [ ] `gear.confirmed`, `motion_model.confirmed`, `localization_error_confirmed`
- [ ] `position_error_bound_m`, `parking_goal`, `camera_to_car`, `drone_to_anchor`
- [ ] `readiness()` blockers empty on live `/api/parking` snapshot.

## 6. Safety gates (docs/SAFETY.md)
- [ ] S3/S4 only with wheels-off-ground + clear-zone + human-present + timeout.
- [ ] All motion finite-duration + explicit stop/finally; watchdog 600 ms live.
- [ ] `aruco-lost | frame-stale>450ms | owner-disconnect => brake` verified.
- [ ] Credentials runtime-only; none in logs/docs/git.

## 7. Dashboard observability (required tiles before L7)
- [ ] ArUco age, EKF sigma, IMU yaw/bias, stage, swept-clear margin, ms/tick, watchdog visible.

## 8. 30-second SysID + turn-column refit (src/ugv/calibration.py)
- [ ] Emit the excitation CSV (50 Hz, exactly 30.0 s, 1500 steps):
  `python3 -m ugv.calibration --emit-csv /tmp/sysid_cmds.csv`.
  Bounded and safe by construction: every pulse <= 0.80 s, |cmd| <= 0.75.
  Covers (a) breakaway deadzone ramps 0.05..0.30 x 0.25 s on
  forward/strafe/turn, (b) forward + strafe step pulses at 3 magnitudes
  (0.35/0.55/0.75, 0.8 s on + 0.4 s brake) for `drive_tau` / `brake_tau`
  and translational `coef` columns, (c) CW/CCW yaw probes (+/-0.35,
  +/-0.55, 0.6 s on + 0.4 s off) for the uncalibrated turn column.
- [ ] Drive the car (or sim) through the CSV at 50 Hz, logging per step
  `cmd_vx,vy,w`, `dt_s`, body velocity `v_vx,vy,w` (ArUco pose deltas),
  and IMU `yaw_dps` to `/tmp/sysid_log.csv`.
- [ ] Refit:
  `python3 -m ugv.calibration --fit-csv /tmp/sysid_log.csv --prior simulation/gazebo/mecanum_bayesian_profile.json --out simulation/gazebo/mecanum_bayesian_profile.json`.
  Fits deadzone (grid 0..0.25), `drive_tau` / `brake_tau` (run-bounded
  step onsets/offsets, free-asymptote grid), and the 16x3 `coef` with
  Bayesian prior regularization: translational columns by weighted ridge,
  turn column as `C[:,2] = yaw_scale*C0[:,2] + cross_couple*C1[:,1] + D`
  (weak ridge on `D` only, so unexcited terms stay bounded).
  Synthetic-plant gate (`tests/test_calibration_sysid.py`, perturbed
  dz 0.14 / yaw_scale 0.7 / coupling 0.25): all params within 8%,
  and `JoystickMapper` (smooth dz pre-comp + grid `inv` decoupling +
  bisection turn inversion + gyro yaw-hold when turn == 0) zeroes yaw
  drift during pure strafe.
- [ ] Copy the refit `deadzone` / `coef` / taus into `MeanModel`
  (`linearize` / `inv`) and the yaw channel (`AdaptiveYaw.turn`); confirm
  `JoystickMapper`-equivalent stick mapping in `web.py` `_apply_vector`
  (body-centric + field-centric drive with dz compensation).
- [ ] Pi 4B shield verification (`rl/runs/pi_inference_bench.json`, retired stack, tag `pre-pose-20261117`):
  stateful GRU-64 + vectorized swept_clear p50 **0.134 ms** single-core
  host (taskset -c0, 10k iters) -> **0.47–0.54 ms** Pi 4B projected =
  **PASS vs 5.0 ms budget** (~9–10x headroom); export parity 4.8e-7
  (< 1e-5 gate); shield PASS under 200 ms stale pose
  (`tests/test_shield.py` keeps the swept-clear checks; the policy shield tests went with the retired policy, STOP_HORIZON 0.18 -> 0.35 =
  0.4445 m sweep > 0.254 m latency travel at vmax, dt_stale
  forward-prediction of sweep start).
