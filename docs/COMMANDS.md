# COMMANDS

Canonical commands verified against existing scripts and docs. For Pi SSH/URL details see [`docs/NETWORK.md`](NETWORK.md).

## Canonical command table

| Task | Canonical command | Notes |
|---|---|---|
| Unit tests (root) | `uv run pytest -q` | Runs `tests/` and `simulation/gazebo/test_*.py` (torch-free). |
| Unit tests (RL) | `cd rl && uv run pytest -q test_rl.py` | Runs RL env/model/export parity tests in `rl/.venv`. |
| Lint & format check | `uv run ruff check . && uv run ruff format --check .` | Configured in `pyproject.toml`. |
| RL eval gate | `cd rl && uv run python evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics --policy runs/<ckpt>.pt --json runs/<ckpt>.json` | Sole success gate (`evaluate.py`); `--no-mean-dynamics` pins the p0 gate plant (the mean has zero differential yaw). |
| RL eval replay traces | `cd rl && uv run python evaluate.py --policy runs/<ckpt>.pt --save-traces runs/<name>.json` | `ugv_replay_v1` bundle for the Observatory arena; `--fov-search` mirrors the deploy blind-search reflex, `--dr-scale`/`--consistent-particles` override level DR. |
| RL training (teacher) | `cd rl && uv run python train_teacher.py --level L0 --envs 512 --iters 120 --seed 0 --out runs/teacher_L0_s0.pt` | Privileged PPO teacher on 3090. |
| RL training (student) | `cd rl && uv run python train_student.py --level L0 --iters 300 --seed 0 --out runs/student_L0_s0.pt` | Deployable GRU student PPO. |
| Gazebo run (GUI) | `./simulation/gazebo/launch_control.command` | Opens Gazebo + local console on `127.0.0.1:8767`. |
| Gazebo run (policy) | `uv run python simulation/gazebo/run_policy_episodes.py --weights <policy.npz>` | Headless L6 policy episode runner; `--bridge fake\|gazebo`, `--save-traces <path>` writes an Observatory replay bundle. |
| Field harness (L7) | `tools/l7/run.sh <mode> [l7_run.py args...]` | Deploys L7 harness to Pi, runs `probe\|stage\|policy\|watch`, pulls logs. |
| Pi deploy / restart | `ssh pi 'cd ~/ugv && sudo systemctl restart ugv-dashboard.service'` | Preserve `~/ugv/runs/student.npz` and `vision-camera.conf`; see [`HANDOFF.md`](../HANDOFF.md). |

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
| Standalone NumPy `StudentPolicy` benchmark / parity CLI | `src/ugv/policy.py` |
| Synthetic simulation / physics helper CLI | `src/ugv/simulation.py` |

### L7 field harness (`tools/l7/`)

| Purpose | Script |
|---|---|
| Sync L7 harness + `src/ugv/` deploy modules to `pi:l7/` | `tools/l7/deploy.sh` |
| Deploy, run one L7 mode on Pi, and pull JSON/JSONL logs | `tools/l7/run.sh` |
| Pull raw camera JPEGs from `pi:l7_runs/` | `tools/l7/pull_frames.sh` |
| On-Pi L7 controller (`probe`, `stage`, `policy`, `watch`) | `tools/l7/l7_run.py` |
| Analyze `probe` run dynamics (deadzone, onset, steady-state gains) | `tools/l7/fit_probe.py` |
| Dump L1 sim rollout channels to `.npz` for offline contract audit | `tools/l7/dump_sim_reference.py` |
| Offline sim-to-real contract audit on L7 logs | `tools/l7/audit_contract.py` |

### RL training & evaluation (`rl/`)

| Purpose | Script |
|---|---|
| Train privileged PPO teacher | `rl/train_teacher.py` |
| Train deployable GRU student via PPO | `rl/train_student.py` |
| DAgger distillation from teacher to student (optional) | `rl/distill.py` |
| Seeded 1000-episode evaluation gate (`evaluate.py`) | `rl/evaluate.py` |
| Benchmark `StagedBaseline` EKF controller in `rl/sim.py` | `rl/baselines.py` |
| Export student `.pt` to NumPy `.npz` and check parity | `rl/export.py` |
| Render top-down episode PNG/MP4 visualizations | `rl/visualize.py` |
| Query/seed experiment lineage archive (`rl/runs/archive.jsonl`) | `rl/experiment_tree.py` |
| Regenerate `tests/fixtures/policy_trace.npz` golden fixture | `rl/make_policy_fixture.py` |

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
| Run student policy or baseline episodes in Gazebo L6 | `simulation/gazebo/run_policy_episodes.py` |
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

### Developer, media & KB tools (`tools/`, `tests/`, `.agent_kb/`)

| Command | Description |
|---|---|
| `uv run ugv-observatory` | UGV Research Observatory (port 8768) |

| Purpose | Script |
|---|---|
| Read-only ArUco pipeline profiler on Pi | `tools/profile_aruco.py` |
| Single bounded floor arc trial | `tools/tune_arc.py` |
| Generate printable 50 mm ArUco PDF (`docs/aruco-tags-50mm.pdf`) | `tools/generate_aruco_tags.py` |
| Render uOttawa Aeronautics showcase MP4 reels | `tools/render_uottawa_reel.py` |
| Browser keyboard/follow contract check | `tests/test_keyboard.cjs` |
| Shared agent knowledge base indexer & query tool | `.agent_kb/kb.py` |
