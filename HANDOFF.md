# UGV Project Handoff — Pre-Charging Tactical Pause (2026-11-13)

## 1. Tactical Pause Reason & Physical Robot State
- **Pause reason**: Raspberry Pi (`adv034`, `advanced@192.168.8.100` / `ssh pi`) is paused for battery charging. **Do NOT touch SSH / Pi until charging is complete.**
- **Last known physical pose** (read-only `tools/l7/l7_run.py watch` run `calibration/runs/l7/20261113-193354-watch-diag`):
  - Motion state: stationary (`armed=false`, `stopped=true`, all wheels `0`).
  - Pose (drone/policy frame: skids along `y`, `yaw=0` faces `-y`, CCW+): **`x = +0.741 m, y = -0.052 m, yaw = -85.3 deg`** (`summary.json` `start_pose`: `[0.7466, -0.0487, -1.4845 rad]`), on the **right side** of the drone.
  - Visual target: facing **Tag 6** (`0.62 m` away). Tag `14` and Tag `3` are **not in FOV** from this pose.
  - Camera/vision telemetry: `1280x720`, `vision_ms_p50 = 50.0 ms`, `K` nominal `fx = fy = 968.0`.

## 2. Git & Pi Sync State
- **`ferpc` `main` commits (pushed to `origin`)**:
  - `3bccaca` — `feat(l7): add real-car L7 harness, contract audit, and field logs`
  - `7f6035f` — `feat(rl,ui): add L_real sim plant and compass needle wrap fix`
  - `1be1015` — `feat(gazebo): update drone tag constellation to 14, 6, 3`
  - `1b0e7dc` — `fix(l7,policy): harden L7 safety, fit_probe dynamics, and swept_clear yaw sign`
  - `2fefd95` — `feat(ui,autopark): overhaul transparent HUD and add web Auto Park`
  - `1b2590d` — `fix(autopark,aruco): add IMU breakaway boost, safe Tag 6 bootstrap, and aruco race fix` (Auto Park IMU breakaway boost + `dz_yaw=0.16`, safe `obs==0` Tag 6 bootstrap for `acquiring`/`staging` while gating `entering` on calibrated fix, `ArucoFollower` shutdown race fix, Localizer `dr_pose` reset P1)
  - Handoff commit on `main` (this file, `calibration/pi_service_snapshot.txt`, `20261113-193354-watch-diag` logs, `runs/student.npz`).
- **CRITICAL SYNC WARNING**:
  - Pi (`~/ugv` on `adv034`) is currently at **`2fefd95`**, **BEHIND** `ferpc` commit `1b2590d`+!
  - Before running Auto Park or wheel tests, bundle/sync latest `main` to `pi:~/ugv` and run `sudo systemctl restart ugv-dashboard.service`.
  - **Preserve on Pi during sync**:
    - `~/ugv/runs/student.npz`
    - `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf` (`UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8`)

## 3. Verified Checks
- `tests/` full suite: **155 passed, 0 failed** (verified after `1b2590d` edits).
- `rl/test_rl.py`: run with `rl/.venv/bin/python` from `Cwd=/home/fer/Documents/ugv/rl`.
- Headless Chromium screenshot verification: **passed** (`0` console errors at `1440x900` and `960x720`, `backdropFilter === "none"`, `ort` loaded, live MJPEG `1280x720` verified).

## 4. Why Auto Park Stalled Pre-Pause (Fixed in `1b2590d`, Pending Pi Deploy)
1. **Yaw deadzone below scrub breakaway**: `StageGoto` `dz_yaw = 0.10` produced `±523 PWM`, below real turn scrub friction breakaway (`600–800 PWM`).
   - *Fix (`1b2590d`)*: raised `dz_yaw = 0.16` + added IMU closed-loop breakaway boost & rate damping in `src/ugv/autopark.py`.
2. **Uncalibrated Tag 6 filtered out**: car was facing Tag `6` (`obs = 0` in `calibration/runs/l7/tagmap.json`), which `autopark.py` dropped completely.
   - *Fix (`1b2590d`)*: safe bootstrap uses `obs == 0` priors **only** in `acquiring`/`staging` to orbit around skids to `(0.0, 0.55, 0.0)`, then requires calibrated Tag `14` or `3` (`obs != 0`) before `entering`.
3. **Stale overlay race on stop**: `ArucoFollower.run` race left a stale `TAG 6` overlay frozen after `asyncio.to_thread(track_all)`.
   - *Fix (`1b2590d`)*: fixed shutdown/clear race in `src/ugv/aruco.py` and `src/ugv/web.py`.

## 5. Data Preserved From Pi (`pi` -> `ferpc`)
- `calibration/` — `1.2G` total
- `calibration/runs/l7/` — `366M` across 25 run dirs (including all `frames/`; `*.jpg` images remain cleanly `.gitignore`d)
- `calibration/runs/l7/20261113-193354-watch-diag/` — latest read-only watch diagnostic (`meta.json`, `summary.json`, `frames.jsonl`, `imu.jsonl`, `ticks.jsonl`, `ws_state.jsonl`, plus local ignored `end.jpg` & `frames/`)
- `calibration/runs/l7/tagmap.json` — `187 B`, SHA-256 `f2c6582dcdd80a345cf0cd0f187de3d108dddfcc40d74bf90379bfcffefe9782` (`14: obs=-1`, `3: obs=129`, `6: obs=0`)
- `runs/student.npz` — `98,169 B`, SHA-256 `a895b0d5aacd81e03d2613f1a1d302405188734fc9defa9db14664bc1377f33e`
- `calibration/pi_service_snapshot.txt` — `15,522 B` (captures `ugv-dashboard.service`, `vision-camera.conf`, and Pi directory snapshot)

## 6. Next Steps When Pi Is Charged (Exact Runbook)
1. **Sync & Restart Service**:
   - Bundle/sync latest `main` to `pi:~/ugv` (preserving `~/ugv/runs/student.npz` and `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf`).
   - Run `sudo systemctl restart ugv-dashboard.service` on Pi and verify `GET /api/state`.
2. **Verify Auto Park Fix (`1b2590d`)**:
   - Click **Auto Park** in web UI (or send `{type: "auto-park", enabled: true}` over `/ws`).
   - Verify IMU breakaway orbits from Tag `6` around drone to `(0.0, 0.55, 0.0)`, locks calibrated Tag `14`/`3`, enters `(0.0, 0.02, 0.0)`, brakes cleanly, and verify `end.jpg` + `|imu_w| < 2 deg/s`.
3. **Extended Car-Feel & Tag-6 Calibration Capture Pack**:
   - Restage cleanly in front/around drone.
   - Run extended probe:
     `l7_run.py probe --baseline-s 10 --coast-s 0.8 --probe-duty 0.12,0.18,0.25,0.35,0.44 --probe-axes fwd,left,ccw,fwd+left,fwd+ccw --dz-yaw 0.16`
   - Run `fit_probe.py` & `audit_contract.py`.
   - Run a `watch` / staging sweep seeing Tags `6 + 14 + 3` simultaneously so Tag `6` gets `obs > 0` calibrated in `tagmap.json`.
   - *(Optional if human present)*: ChArUco/checkerboard camera intrinsics capture.
4. **Wrap & Distribute**:
   - Commit curated capture outputs, `git push`, and sync to Mac.

## 7. Local Agent Knowledge Base
- Tool: `python3 /home/fer/Documents/ugv/.agent_kb/kb.py` (`facts`, `query`, `slice`, `index`).
- Index: `/home/fer/Documents/ugv/.agent_kb/index.json` (`870` symbols, `25` L7 runs).
- Facts: `/home/fer/Documents/ugv/.agent_kb/FACTS.md`.
## 8. Post-Pause Offline Verification, Orbit Routing & Vision/Sim Notes (`87af776`, `d1bf093`)
- **Commits added on `main` (pushed to `origin`)**:
  - `87af776` — `refactor(ui): remove unused browser vision.js and ONNX WASM assets` (removed unused `src/ugv/static/vision.js`, `ort.wasm.min.js`, `models/yolo11n-320.onnx`, and `vendor/` WASM/ORT runtime assets; updated `index.html`, `styles.css`, and `tests/test_hud_payload.py`).
  - `d1bf093` — `fix(autopark): orbit perimeter around drone skids to prevent swept_clear blocks` (added perimeter orbit waypoint routing in `src/ugv/autopark.py` and skid-clearing coverage in `tests/test_autopark.py`).
- **Verification**:
  - `pytest tests/`: **156 passed, 0 failed** (full repo `pytest`: **190 passed, 4 subtests passed**).
  - **Closed-loop kinematic orbit rollout** (`dt = 0.06s`, `gear_obstacles` `clear_width_m=0.314, bar_length_m=0.305, bar_radius_m=0.03, inward_slope=0.0`, `AutoParkRunner` + `Localizer` holonomic velocity & yaw integration toward `(0.0, 0.55, 0.0)`): **0 `"blocked"` ticks**, all 3 poses reached `abs(x) <= 0.20, y >= 0.42` facing toward the drone (`right` corridor@step 88, facing@step 138 (8.28s) -> `(+0.01, +0.54, -15.8 deg)`; `left` corridor@step 91, facing@step 142 (8.52s) -> `(-0.01, +0.54, +16.2 deg)`; `rear` corridor@step 158, facing@step 238 (14.28s) -> `(-0.02, +0.55, +15.6 deg)`). Note: during combined fwd+strafe+turn (`duty_to_dashboard` wheel saturation scaling), yaw duty is scaled down while orbiting and finishes aligning once near `(0.0, 0.55)`.
- **`swept_clear` blocking & yaw/turn sign findings**:
  - Direct straight-chord staging from side/rear poses passed through `(0, 0)` (the drone skids), causing `swept_clear` to reject commands and block. `_staging_orbit_waypoints` now routes around a `0.72–1.15 m` perimeter arc toward `(0.0, 0.55, 0.0)` while facing the drone.
  - Yaw/turn signs: `probe-axes` `+ccw -> +w` verified; `GoTo` reduces `|yaw_err|` in CCW+ policy/world frame (`autopark.py`, `tools/l7/l7_run.py`, `rl/policy.py`). Legacy `src/ugv/parking.py` (`Pose2`) uses CW+ `+z` yaw convention separate from `autopark.py`, so `autopark.py` negates yaw when calling `swept_clear(Pose2(x, y, -yaw), ...)`.
- **Gazebo vs real screenshot gap & on-Pi skid-on-floor CV notes**:
  - `freenove_world.sdf` / `simulation/gazebo/environment.py` camera matches `1280x720` `L8` `fx = fy = 968` with directional `key_light` and `fill_light`, whereas `VisionSensors` renders tags-only frames without speckled/specular concrete floor texture, foam skids + grey end-caps, or lab clutter.
  - Pulled Pi frames (`calibration/runs/l7/20261113-193354-watch-diag/frames/`, 14 frames) highlight real-world floor specular glare, foam skids with grey end-caps, and background lab clutter as primary visual gaps for on-Pi skid-on-floor CV.

## 9. Interrupted Work Committed 2026-11-14 — Live Pi Profiler (Pending Pi Deploy)
- **What**: zero-overhead `PerfMonitor` (`src/ugv/perf.py`: `decode/detect/subpix/pnp/plan/drive/aruco/broadcast_ms`, rolling `p50/p95/Hz`, Pi CPU%/load/temp/RSS, `33.33 ms` budget) wired through `TagDetector.last_timings`, `ArucoFollower`, `RobotController` auto-park loop + `payload()` (`state.perf`, 5 Hz snapshot cache); dashboard `Perf` toggle + live stacked-bar canvas panel (`33.3 ms` + `50 ms` markers, `backdrop-filter: none` preserved).
- **Files**: `src/ugv/perf.py`, `tests/test_perf.py` (new); `src/ugv/aruco.py`, `src/ugv/autopark.py`, `src/ugv/web.py`, `src/ugv/static/{app.js,index.html,styles.css}`, `tests/test_hud_payload.py`; `.agent_kb/` tool snapshot.
- **Verification (ferpc)**: full `pytest` **199 passed, 4 subtests passed**; `test_perf.py` + `test_hud_payload.py` **12 passed**; `node --check app.js` OK; new files `ruff check` clean (repo-wide `ruff`/format failures pre-exist).
- **NOT done**: on-Pi screenshot check of `#perf-panel`; `pi:~/ugv` sync + `systemctl restart`; 180-degree auto-park flip forensic audit (replay + gyro/turn sign-chain) was still running when quota hit — no fix applied, no conclusion recorded.
- **Pi memory — measured live 2026-11-14 (`adv034`, up 39 min, streaming 30 fps + IMU, wheels idle)**: board `Pi4B` 2 GB — `MemTotal 1888972 kB` (`free -h`: total `1.8Gi`, used `540Mi`, `buff/cache 1.2Gi`, available `1.3Gi`, swap unused). `throttled=0x0`, `temp 60.3 C`, load `1.43–1.56`. Top procs: `uvicorn` dashboard `RSS 163768 kB` (`8.6%`, `VmHWM` = same = session peak incl. earlier auto-park per `auto_park_pose` in state); `rpicam-vid` `RSS 67016 kB` (`3.5%`, `VmHWM` same). Desktop stack (`labwc`/`wf-panel`/`pcmanfm`/`wayvnc`) adds ~`250 MB`. Headroom is comfortable; no swap pressure. True auto-park high-load trace still needs new `PerfMonitor` deployed + a run.
- **Vision high-pressure test 2026-11-14, no actuation**: 30 s tight loop over `GET /api/calibration/frame` + `TagDetector.detect_jpeg` on-Pi: **473 frames, 468 tag detections, fetch+detect mean `45.5 ms` / p50 `44.3 ms` / p95 `52.4 ms`** (detect path alone exceeds the `33.3 ms` frame budget — consistent with `vision_ms_p50 = 50 ms` in watch-diag). Loader proc peak RSS `64.5 MB`; dashboard `uvicorn` RSS flat at `160.4 MB` before and peak (no growth over 473 requests); system used `532 -> 539 MB`, available `1312 -> 1304 MB`. No leak, no swap, no pressure.
