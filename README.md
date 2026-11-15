<img src="docs/assets/banner.png" alt="ugv field chart: a mecanum car's traverse into the 0.31 m drone-skid bay, ArUco stations 14, 6 and 3, RL ladder L0-L7 stamped at L7" width="100%">

<p align="center"><i>A small mecanum car that parks itself under a drone, from simulation to the floor.</i></p>

<p align="center">
  <a href="https://github.com/uOAeronautics/ugv/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/uOAeronautics/ugv/actions/workflows/ci.yml/badge.svg?branch=main"/></a>
</p>

<p align="center">
  <a href="docs/COMMANDS.md"><b>Commands</b></a> ·
  <a href="docs/SAFETY.md"><b>Safety</b></a> ·
  <a href="docs/RL_PARKING.md"><b>RL parking</b></a> ·
  <a href="docs/L7_FIELD_NOTES.md"><b>Field notes</b></a> ·
  <a href="docs/INDEX.md"><b>All docs</b></a>
</p>

---

# UGV

Autonomous drone-skid parking stack for an `FNK0043` / Freenove 4WD mecanum car on Raspberry Pi 4B (`ugv-dashboard.service` on port `8765`). The repo pairs a torch-free Pi dashboard, ArUco + BMI270 IMU closed-loop control, and a `swept_clear` safety shield with PyTorch `MecanumVecEnv` RL teacher/student training, Gazebo 3D physics and vision verification, and an Arduino Nano 33 BLE Sense IMU companion.

**[Commands](docs/COMMANDS.md)** · [Safety gates](docs/SAFETY.md) · [Hardware](docs/HARDWARE.md) · [Network & SSH](docs/NETWORK.md) · [Field notes](docs/L7_FIELD_NOTES.md) · [All docs](docs/INDEX.md)

## Status

Simulation metrics come from `rl/evaluate.py` (1000 episodes, seed 11); hardware L7 metrics come from the [2026-11-13 field runs](docs/L7_FIELD_NOTES.md).

| Domain | Metric | Measured | Source |
|---|---|---:|---|
| Sim (`rl/sim.py`) | L0 teacher (`runs/teacher_L0_gate.pt`) | `1000/1000` (0 coll) | [RL parking](docs/RL_PARKING.md) |
| Sim (`rl/sim.py`) | L0 student (`student_L0_ppo_v2_s0.pt` / `student_L0_lat_s1.pt`) | `991/1000` (6 coll) / `1000/1000` (0 coll) | [RL parking](docs/RL_PARKING.md), [Plan](rl/PLAN.md) |
| Sim (`rl/sim.py`) | Best L1 student (`student_L1_best.npz`) | `966/1000` L1 (0 coll) · `0/20` Gazebo L6 | [State](docs/STATE.md), [Field notes](docs/L7_FIELD_NOTES.md) |
| Sim (`rl/sim.py`) | `StagedBaseline` EKF (`baseline_ekf_staged_L*`) | `975` L0 · `896` L1 · `448` L2 · `215` L3 · `148` L4 | [RL parking](docs/RL_PARKING.md) |
| Sim throughput | `MecanumVecEnv` CPU (`512`–`2048` envs) / Gazebo lockstep | `290k–302.6k` steps/s / `50.0 D-steps/s` (`1.0x` RT) | [Sim throughput](docs/SIM_THROUGHPUT.md) |
| Policy inference | NumPy `StudentPolicy` GRU-64 (desktop CPU, full vs stateful) | `1.47 ms/step` vs `0.04 ms/step` | [Sim throughput](docs/SIM_THROUGHPUT.md) |
| Hardware (Pi 4B / L7) | Classical go-to P (`stage` mode, `4 cm` / `5 deg` tol) | `4/4` reached target (`6.49–10.84 s`) | [Field notes](docs/L7_FIELD_NOTES.md) |
| Hardware (Pi 4B / L7) | L1 student (`student_L1_best.npz`) | `0/3` parked (timeouts / tag loss) | [Field notes](docs/L7_FIELD_NOTES.md) |
| Hardware (Pi 4B / L7) | ArUco + KLT (`1280x720`, `4 ms` shutter / gain `8`) | `97%` motion decode (`~13 poses/s`, `~65 ms`) | [Field notes](docs/L7_FIELD_NOTES.md) |

- Real-car plant ID (`probe`) measured `2.3 m/s` forward (deadzone `0.05`, onset `0.18 s`), `~1.6 m/s` strafe (deadzone `~0.13`, onset `0.25–0.37 s`), and `15.7 rad/s` yaw per duty (deadzone `0.10`, onset `0.07 s`), vs `0.0` deadzone in `simulation/gazebo/mecanum_bayesian_profile.json`.
- Real gyro-rate std (`0.63 rad/s` vs `0.002 rad/s` in `rl/sim.py`) drove `|a2| > 0.8` turn saturation on `74%` of real ep3 ticks (`0%` when gyro-rate is zeroed in offline replay).

## Layout

| Path | Contents |
|---|---|
| `src/ugv/` | Torch-free Pi runtime: FastAPI/WS dashboard (`web.py`), ArUco + KLT vision, BMI270 IMU heading, `swept_clear` shield, `StagedBaseline`, and NumPy `StudentPolicy` |
| `rl/` | CUDA RL subproject (`rl/.venv`): `MecanumVecEnv` (`sim.py`), PPO teacher/student trainers, `evaluate.py` (1000-episode gate), and `.npz` export |
| `simulation/gazebo/` | Gazebo 3D car/drone/lab worlds, native `ugv::TTMotorSystem` plugin, camera/IMU models, and browser drive console |
| `arduino/` | Standalone Nano 33 BLE Sense Rev2 firmware (`firmware/MotionStream`), BLE streaming, and VQF dashboard |
| `tools/` | L7 field harness (`tools/l7/`) and system identification / calibration scripts (`tools/calibration/`) |
| `docs/` | Topic documentation indexed in [docs/INDEX.md](docs/INDEX.md) (safety, network, commands, RL ladder, throughput, field notes) |
| `tests/` | Root torch-free unit and dashboard tests (`pytest` also covers `simulation/gazebo/test_*.py`) |

## Setup

Root Python `>= 3.13` stays torch-free for Pi deployment; `rl/` uses a separate CUDA environment (`rl/.venv`).

```bash
uv sync
uv run ugv --help
uv run ugv-dashboard                 # FastAPI/WS dashboard on port 8765
uv run pytest -q                     # root + simulation/gazebo tests
./simulation/gazebo/launch_control.command   # Gazebo + local console on :8767
```

- **RL training (`rl/`)**: `cd rl && uv sync && uv run pytest -q test_rl.py`; see [rl/README.md](rl/README.md) and [docs/RL_PARKING.md](docs/RL_PARKING.md).
- **Arduino IMU companion**: standalone BLE + VQF orientation dashboard on port `8766`; see [arduino/README.md](arduino/README.md).
- **Docs & safety**: check [docs/SAFETY.md](docs/SAFETY.md) and [HANDOFF.md](HANDOFF.md) before any Pi SSH or motor actuation. See [docs/INDEX.md](docs/INDEX.md) for all documents and [docs/COMMANDS.md](docs/COMMANDS.md) for the canonical command table.

References: [Freenove FNK0043](https://github.com/Freenove/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi) · [official docs](https://docs.freenove.com/projects/fnk0043/en/latest/)
