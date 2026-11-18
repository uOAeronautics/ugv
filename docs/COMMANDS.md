# COMMANDS

Canonical commands verified against existing scripts and docs. For Pi SSH/URL details see [`docs/NETWORK.md`](NETWORK.md).

## Canonical command table

| Task | Canonical command | Notes |
|---|---|---|
| Unit tests (root) | `uv run pytest -q` | Runs `tests/` and `simulation/gazebo/test_*.py` (torch-free). |
| Lint & format check | `uv run ruff check . && uv run ruff format --check .` | Configured in `pyproject.toml`. |
| Gazebo run (GUI) | `./simulation/gazebo/launch_control.command` | Opens Gazebo + local console on `127.0.0.1:8767`. |
| Field harness (L7) | `tools/l7/run.sh <mode> [l7_run.py args...]` | Deploys L7 harness to Pi, runs `probe\|stage\|watch`, pulls logs. |
| Pi deploy / restart | `ssh pi 'cd ~/ugv && sudo systemctl restart ugv-dashboard.service'` | Preserve `vision-camera.conf`; see [`STATE.md`](STATE.md). |
| Live tasks / your inbox | `uv run python tools/research/ledger.py status [--as <you>]` | Blocked and stale first |
| Roadmap | `uv run python tools/research/ledger.py roadmap` | Milestone tasks and what each waits on |
| Prior attempts on a topic | `uv run python tools/research/ledger.py list --grep <topic>` | Read before re-running anything |
| New record / move a task | `uv run python tools/research/ledger.py new <decision\|attempt\|task> <slug> --author <you>`; `claim\|block\|release\|done\|drop <id> --as <you>` | `done` needs `--result` |
| Check records | `uv run python tools/research/ledger.py check` | Same rules as `tests/test_research.py` |
| Pose smoke test (synthetic, CPU) | `uv run python tools/pose/smoke.py [--views car] --out runs/pose_smoke` | Renders the airframe, PnP from keypoints, SIFT bank; see `docs/POSE.md` |
| Parking simulation: score | `uv run python tools/park/eval.py --name <name> [--params <json>] --sweep --traces 16 [--floor 1]` | 1000 episodes, seed 11; `--floor` sets traction severity; writes `runs/park_eval/<name>.json`; see `docs/PARK_SIM.md` |
| Parking simulation: tune | `uv run python tools/park/search.py --out runs/park_search/<name>` | Failure-guided search; `tree.jsonl` keeps every candidate |
| Lab dashboard | `uv run ugv-lab` | Port 8768, keys 1-4: Park (sim results, replay, tuning), Challenge (drive the sim car, or watch the tuned controller from the same start; six simulated sets and 998 Poly Haven CC0 panoramas), Pose (detector views with skeleton and mesh), Ledger (roadmap, tasks, attempts). Simulation only; never talks to the car |
| Pose training data | `uv run python tools/pose/make_dataset.py --out runs/pose_data/<name> --n 24000` | Airframe composited over real car-camera frames |
| Pose detector preview (CPU) | `cd train && uv run python preview.py --run ../runs/pose_train/<name>` | Runs the ONNX model on 48 held-out views for the Lab's Pose page (`preview/`) |
| Pose detector training (GPU) | `cd train && uv run python pose_kp.py --data ../runs/pose_data/<name> --out ../runs/pose_train/<name>` | Separate CUDA env (`train/pyproject.toml`); exports ONNX |

## Pi CLI & bounded hardware diagnostics

Run on the Pi (`cwd:=/home/advanced/ugv`) with safety gates from [`docs/SAFETY.md`](SAFETY.md):

```sh
uv run ugv --help
uv run ugv snapshot                         # OV5647 still; 500ms wait; rot180
uv run ugv snapshot --output <file.jpg>
uv run ugv servo-test <pan|tilt>            # ch8/ch9 1500->1400->1600->1500us; gate: clear linkage
uv run ugv wheel-test [wheel-name]          # PCA9685 ch0..7 900/4095 200ms; gate: chassis-lifted + human
uv run ugv led-test                         # 8xWS2812 SPI0; unsupported on PCB V3
sudo .venv/bin/ugv led-test-gpio18          # same via PWM/DMA GPIO18; requires root
uv run ugv-dashboard                        # local FastAPI/WS server on :8765
```

## Entry-point index (purpose → canonical script)

### Package & deployed CLIs (`src/ugv/`)

| Purpose | Script |
|---|---|
| Robot hardware CLI (`snapshot`, `servo-test`, `wheel-test`, `led-test`) | `uv run ugv` (`src/ugv/__init__.py`) |
| Dashboard web/WebSocket server | `uv run ugv-dashboard` (`src/ugv/web.py`) |
| 30 s SysID CSV generator & Bayesian profile refitter | `src/ugv/calibration.py` |

### L7 field harness (`tools/l7/`)

| Purpose | Script |
|---|---|
| Sync L7 harness + `src/ugv/` deploy modules to `pi:l7/` | `tools/l7/deploy.sh` |
| Deploy, run one L7 mode on Pi, and pull JSON/JSONL logs | `tools/l7/run.sh` |
| Pull raw camera JPEGs from `pi:l7_runs/` | `tools/l7/pull_frames.sh` |
| On-Pi L7 controller (`probe`, `stage`, `watch`) | `tools/l7/l7_run.py` |
| Analyze `probe` run dynamics (deadzone, onset, steady-state gains) | `tools/l7/fit_probe.py` |

### 3D model

One canonical airframe: `simulation/gazebo/strl_001/` (9,956 triangles, `tags.json` holds ArUco 14/6/3). `uv run python tools/build_strl_assets.py` regenerates the Lab dashboard assets from it; `uv run python tools/build_car_assets.py` does the same for the Freenove car from `simulation/gazebo/assembly.py`. Full-resolution originals and earlier experiments were moved out of the repo to `~/ugv_archive_2026-11-16/`.

### Mesh optimisation (`tools/mesh_opt/`)

Render meshes only; nothing here touches the car. Workflow and lessons: `.agents/skills/mesh-optimization/SKILL.md`.
All run with `uv run --no-project --python 3.12 --with trimesh --with numpy --with scipy --with embreex --with pillow` (add `--with pymeshlab` for `py_optimize.py`, `--with networkx` for `export_glb.py`).

| Purpose | Script |
|---|---|
| Gate a simplified mesh against its original (silhouette, depth, normals, surface distance); exit 0 = pass | `tools/mesh_opt/meshgate.py ORIG CAND --out DIR [--seed N]` |
| Simplify every STL with a quadric collapse, stepping down to the smallest count that passes the gate (dense organic parts: body, nacelles) | `tools/mesh_opt/py_optimize.py --out DIR` |
| Simplify per loose part to a deviation bound inside headless Blender (thin and flat parts: props, control surfaces) | `flatpak run org.blender.Blender -b --factory-startup --python $PWD/tools/mesh_opt/blender_optimize.py -- --out DIR` |
| Pack per-link STLs into one indexed GLB with crease-split normals and a node per link | `tools/mesh_opt/export_glb.py IN_DIR OUT.glb` |

### Gazebo simulation (`simulation/gazebo/`)

| Purpose | Script |
|---|---|
| Launch Gazebo + local browser drive console (`:8767`) | `simulation/gazebo/launch_control.command` |
| Launch drone docking Gazebo scene | `simulation/gazebo/launch_drone.command` |
| Launch reconstructed 24×12 m lab Gazebo scene | `simulation/gazebo/launch_lab.command` |
| Generate SDF world files (`freenove`, `drone`, `room`) | `simulation/gazebo/generate.py` |
| Build native `ugv::TTMotorSystem` C++ Gazebo plugin | `simulation/gazebo/build_motor_plugin.py` |
| Browser drive console backend (FastAPI + gz-transport) | `simulation/gazebo/control_panel.py` |
| Bounded simulation-only drive demo | `simulation/gazebo/demo_drive.py` |
| Run standalone EKF baseline episodes in Gazebo | `simulation/gazebo/run_baseline_gazebo.py` |
| Run parallel `GZ_PARTITION` Gazebo batch workers | `simulation/gazebo/parallel_worlds.py` |
| Regenerate `assets/test_pad.png` floor texture | `simulation/gazebo/make_test_pad.py` |
| Render Blender overview/approach/room images | `simulation/gazebo/render_drone.py`, `simulation/gazebo/render_blender.py` |
| Render & assemble room video comparison contact sheets | `simulation/gazebo/render_video_comparison.py`, `simulation/gazebo/make_room_comparison.py` |
| Live Gazebo physics/camera/IMU checks (`.control-venv`) | `simulation/gazebo/check_mecanum_car.py`, `check_mecanum_physics.py`, `check_motor_physics.py`, `check_reset_physics.py`, `check_lap_physics.py`, `check_camera.py`, `check_imu_pipeline.py` |
| Browser Playwright checks for Gazebo console | `simulation/gazebo/check_browser_controls.cjs`, `check_browser_camera.cjs`, `check_browser_imu.cjs` |
| Offline Gazebo model unit tests (run by root `pytest`) | `simulation/gazebo/test_*.py` |

### Calibration & system identification (`tools/calibration/`)

| Purpose | Script |
|---|---|
| Run bounded differential-drive floor trials on Pi | `tools/calibration/run_trials.py` |
| Run bounded mecanum identification trials on Pi | `tools/calibration/run_mecanum_trials.py` |
| Read-only telemetry capture during manual driving | `tools/calibration/record_observed.py` |
| Analyze floor/lap sessions (`analysis.json`) | `tools/calibration/analyze.py`, `tools/calibration/analyze_laps.py` |
| Fit empirical Gazebo drive profile / IMU noise profile | `tools/calibration/fit_profile.py`, `tools/calibration/fit_imu_noise.py` |
| Fit Bayesian mecanum model / native motor params / IMU noise | `tools/calibration/mecanum_fit.py`, `fit_mecanum_native.py`, `mecanum_imu_noise.py` |
| Visual–inertial floor homography alignment | `tools/calibration/mecanum_vision.py` |
| Replay recorded trials in Gazebo (differential / mecanum / guarded) | `tools/calibration/evaluate_gazebo.py`, `evaluate_mecanum_gazebo.py`, `evaluate_mecanum_guarded.py` |
| Plot Gazebo vs real-car calibration comparison | `tools/calibration/report.py` |
| Calibration & online guard unit tests | `tools/calibration/test_calibration.py`, `tools/calibration/test_mecanum_online_guard.py` |

### Developer and media tools (`tools/`, `tests/`)

| Command | Description |
|---|---|
| `uv run ugv-lab` | Lab dashboard: park results, parking challenge, pose, ledger (port 8768) |

| Purpose | Script |
|---|---|
| Read-only ArUco pipeline profiler on Pi | `tools/profile_aruco.py` |
| Single bounded floor arc trial | `tools/tune_arc.py` |
| Generate printable 50 mm ArUco PDF (`docs/aruco-tags-50mm.pdf`) | `tools/generate_aruco_tags.py` |
| Browser keyboard/follow contract check | `tests/test_keyboard.cjs` |
