# Project scope

A Freenove 4WD mecanum car (Raspberry Pi 4B) that parks itself between a drone's skids, localising
from ArUco tags and a gyro. The repo holds the Pi runtime and dashboard, the RL training stack, the
Gazebo simulation, calibration tooling and the field harness. This is a team repository that drives
real hardware: a change that alters motion is a hardware change, whatever the commit calls it.

## Safety comes first

`docs/SAFETY.md` defines the levels `S0` read-only → `S4` motor. The default is `S0`.

- Never raise a level without the gate that document names. `S4` needs wheels off the ground or a
  clear zone, a human present and a timeout.
- Read `HANDOFF.md` before any contact with the Pi. If it says the car is charging or paused, do
  not ssh, deploy or move it.
- One agent owns the car at a time, and one job owns the GPU at a time.
- Do not change gains, caps, watchdog timing, `config/parking.json` or the systemd units as a side
  effect of other work. The dashboard owns the motors: 600 ms watchdog, PWM cap 1800/4095.
- A deploy must preserve `~/ugv/runs/student.npz` and the camera drop-in `vision-camera.conf`.
- An estimate-based "arrived" is not a parked car. Confirm with the end frame.

## Commands

`docs/COMMANDS.md` is the single command table and entry-point index. Add a row there when you add
a runner; do not add a second way to do the same thing.

Checks, required before every commit and run by CI:

```sh
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
cd rl && uv run pytest -q test_rl.py        # when rl/ changed; needs the CUDA environment
```

## Two environments

| Where | Python | Holds |
|---|---|---|
| repo root (`pyproject.toml`) | ≥ 3.13 on the desktop; the Pi runs the same package | `src/ugv`, tools, Gazebo tests. No torch: the deployed policy is NumPy. |
| `rl/` (`rl/pyproject.toml`, `rl/.venv`) | ≥ 3.11, CUDA torch | Training, evaluation, export. See `rl/AGENTS.md`. |

Code under `src/ugv` must import without torch.

## Where things live

| Path | Holds |
|---|---|
| `src/ugv/` | Pi runtime: dashboard (`web.py`), vision, heading, parking, the NumPy `StudentPolicy` |
| `rl/` | Simulator (`sim.py`), teacher and student training, `evaluate.py` (the success gate) |
| `simulation/gazebo/` | Worlds, models, bridge, policy episode runner |
| `tools/l7/` | Field harness: deploy, run one mode on the Pi, pull records |
| `tools/calibration/` | SysID trials, fits and reports |
| `calibration/` | Curated fits and manifests. Raw session logs stay local and on the Pi, ignored by Git |
| `config/` | Runtime configuration read on the car |
| `docs/` | One topic per file, indexed in `docs/INDEX.md`; dated history in `docs/archive/` |
| `tests/` | Root unit tests |

Source of truth per topic: status → `HANDOFF.md` and `docs/AGENT_BOARD.md`; safety →
`docs/SAFETY.md`; network and SSH → `docs/NETWORK.md`; RL plan and gates → `rl/PLAN.md` and
`docs/RL_PARKING.md`; field findings → `docs/L7_FIELD_NOTES.md`. State a fact in one place and link
to it from the others.

## Frames and tags

- Policy frame: drone at the origin, skids along y, the car enters from +y facing −y; pose is
  `(x, y, yaw)`.
- Yaw is counter-clockwise positive in `autopark.py`, `policy.py`, `tools/l7/l7_run.py` and
  `rl/sim.py`. `parking.Pose2` is clockwise positive. Cross the boundary only through
  `pose2_from_ccw` / `pose2_to_ccw` in `src/ugv/parking.py`.
- The drone carries tags 14, 6 and 3. The older layout 9/12/13/14 survives in some configs and
  simulation flags; check which one a file means before using it.

## Evidence rules

- `rl/evaluate.py`, 1000 episodes, seed 11, is the only success gate. Training loss is not success.
- Simulation success is not hardware success. Before a retrain, measure the sensor at rest, check
  loop lag and action-to-duty parity; a policy that oscillates is usually a loop problem.
- Record measured numbers with their run id. Never commit weights, raw logs, frames, videos,
  credentials or machine-specific paths.

## Code shape

- Reusable code goes in a package; a runner is a thin `main()`.
- Aim for files under about 300 lines so one can be read in a sitting. Split at a real seam
  (separate responsibilities), never mechanically. A longer file that does one thing is fine.
- Delete what is superseded. Git history is the archive.

## Skills

Project skills live in `.agents/skills/` (`.claude/skills` and `.opencode/skills` link there):

- `ugv-research`: the RL research loop (levels, evaluation gate, archive, board).
- `mastermind`: delegating to subagents and gating their output.
- `orchestrate-and-gate`: log-first diagnosis and sim-to-real release gates.
