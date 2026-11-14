# UGV Agent Knowledge Base — Verified Facts

## Environment & Repo Rules
- Workspace Cwd: `/home/fer/Documents/ugv`.
- Pi host: `ssh pi` (`advanced@192.168.8.100`, see `docs/NETWORK.md`).
- Pi service: `ugv-dashboard.service` on port `:8765`, runtime `.venv/bin/python`.
- Local runner: `uv run` from repo root (and `cd rl && uv run` for CUDA/torch RL).
- Agent KB tool: `python3 .agent_kb/kb.py {index|query|slice|facts}`.

## Frame & Hardware Constants
- World yaw convention: `yaw=0` faces `-y`, CCW+ (`tools/l7/l7_run.py`, `src/ugv/policy.py`, `src/ugv/autopark.py`, `rl/sim.py`).
- Sign hazard: `src/ugv/parking.py` `Pose2` uses CW+ yaw (opposite of `l7_run.py` / `policy.py` / `autopark.py`). Use `pose2_from_ccw` / `pose2_to_ccw`.
- Tag `14` anchor: `[-0.028, 0.0, pi/2]`, `tag_y=-0.08`.
- Tag & camera constants: `TAG_SIZE_M=0.050`, `CAM_FWD_M=0.071`.
- Camera intrinsics: `1280x720`, `K` nominal `fx=968`.
- Camera blind zone: tag `14` exits FOV at `y ~ 0.40 m`.
- Actuator scale: `PWM_LIMIT=1800`, `KDUTY=0.4396`.
- Chassis clearance: skid clear width `0.254-0.31 m`.

## Verified Plant (fit.json)
- Forward channel: `2.32 m/s/duty`, deadzone `0.051`, onset `0.18s`.
- Strafe channel: `~1.6 m/s/duty`, deadzone `~0.13`, onset `0.25-0.37s`.
- Strafe cross-coupling: induced yaw drift `5-15 deg/s`.
- Yaw channel: `15.7 rad/s/duty`, deadzone `0.101`, onset `0.07s`.
- IMU noise floor: real stationary gyro floor `0.00077 rad/s`.

## Key Line Map & Open P0/P1s — tools/l7/l7_run.py
- `tag_solutions` (L151): AprilTag candidate pose enumeration.
- `TagBits` (L187): tag bit/geometry state.
- `vision_worker` (L207): asynchronous camera/AprilTag capture loop.
- `Localizer` (L362): P0 IPPE planar flip & outlier leak (L408-473).
- `Localizer` timestamp skew: `pose_t=cap` vs `dr_t=now` mismatch (L460-493).
- `Drive` (L499): low-level motion interface.
- `GoTo` (L569): P0 missing `swept_clear` & blind `<0.4m` DR guards (L870-892).
- `GoTo` rest & acquire bugs: unvalidated `rest_pose` (L955-956), `acquire` liveness (L786-793).

## Key Line Map & Open P0/P1s — src/ugv/parking.py
- `Pose2` (L69): CW+ yaw convention vs CCW+ in `l7_run.py` & `src/ugv/policy.py`.
- `swept_clear` (L114-174): swept-volume collision check against blockers.
- `readiness()` (L223): controller/sensor readiness gating.
- `ParkingEstimator` (L345): parking state estimation.

## Key Line Map & Open P0/P1s — src/ugv/web.py
- `RobotController`: live robot control & telemetry owner.
- `_guard_check` (L255-295): safety guard check before motion commands.
- `select_controller` (L384): `staged_ekf` and `rl_student` remain stubbed.
- `_parking_loop` (L587): main web-driven parking control loop.

## Key Line Map & Open P0/P1s — Static UI & Config
- `src/ugv/static/` CSS: broken `.control-bar` CSS grid layout.
- `src/ugv/static/tactical.js`: tactical overlay collides with `.key-overlay`.
- `config/parking.json`: 13 unconfirmed blockers in workspace config.
- `config/parking.json`: wrong `drone_tag_ids [9,12,13,14]`.

## Current Sync & Pause Status (2026-11-13)
- `ferpc` `main` is at `1b2590d`+ (`HANDOFF.md` added); Pi (`adv034`, `~/ugv`) is paused for charging at **`2fefd95`** (behind `1b2590d`).
- Before Auto Park or wheel tests: sync latest `main` to `pi:~/ugv`, preserve `~/ugv/runs/student.npz` and `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf` (`UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8`), and restart `ugv-dashboard.service`.
- Last stationary pose (`20261113-193354-watch-diag`): `x=+0.741 m, y=-0.052 m, yaw=-85.3 deg`, facing Tag `6` (`0.62 m`, `obs=0` in `tagmap.json`).
- `tests/`: 155 passed, 0 failed (`1b2590d`).
## Orbit Routing, UI Cleanup & Gazebo/CV Notes (87af776, d1bf093)
- Browser UI cleanup (`87af776`): removed unused `vision.js`, `ort.wasm.min.js`, `models/yolo11n-320.onnx`, and `vendor/` ONNX WASM assets.
- Autopark orbit routing (`d1bf093`): `_staging_orbit_waypoints` routes side/rear poses along a `0.72-1.15 m` perimeter around `(0, 0)` so `swept_clear` avoids drone skids; verified `156 passed` (`tests/`) and 0 `"blocked"` ticks across closed-loop rollouts (`right` corridor@step 88, facing@step 138 (8.28s) -> `(+0.01, +0.54, -15.8 deg)`; `left` corridor@step 91, facing@step 142 (8.52s) -> `(-0.01, +0.54, +16.2 deg)`; `rear` corridor@step 158, facing@step 238 (14.28s) -> `(-0.02, +0.55, +15.6 deg)`).
- Yaw/turn signs verified: `probe-axes` `+ccw -> +w`; `GoTo` reduces `|yaw_err|`; `autopark.py` negates CCW+ yaw when calling `parking.py` `Pose2` (CW+ `+z`).
- Gazebo vs real gap: `freenove_world.sdf` camera matches `1280x720 L8 fx=fy=968` + `key_light`/`fill_light`, while `VisionSensors` renders tags-only without speckled/specular concrete floor, foam skids + grey end-caps, or lab clutter (pulled Pi frames in `calibration/runs/l7/20261113-193354-watch-diag/frames/`).
