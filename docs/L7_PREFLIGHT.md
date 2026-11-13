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
- [ ] ArUco age, EKF sigma, IMU yaw/bias, stage/policy, swept-clear margin, ms/tick, watchdog visible.
