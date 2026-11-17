# Project scope

A Freenove 4WD mecanum car (Raspberry Pi 4B) that should park itself under the STRL-001 airframe.
Direction since 2026-11-17: estimate the airframe's 6-DoF pose from the car camera using its 3D
model and synthetic data, then park with a declared state machine (`docs/POSE.md`). The 2D RL
stack is retired (tag `pre-pose-20261117`). The repo holds the Pi runtime and dashboard, the pose
code, the Gazebo simulation, calibration tooling and the field harness. This is a team repository that drives
real hardware: a change that alters motion is a hardware change, whatever the commit calls it.

## Safety comes first

`docs/SAFETY.md` defines the levels `S0` read-only → `S4` motor. The default is `S0`.

- Never raise a level without the gate that document names. `S4` needs wheels off the ground or a
  clear zone, a human present and a timeout.
- Read `docs/STATE.md` before any contact with the Pi. If it says the car is charging or paused, do
  not ssh, deploy or move it.
- One agent owns the car at a time, and one job owns the GPU at a time.
- Do not change gains, caps, watchdog timing, `config/parking.json` or the systemd units as a side
  effect of other work. The dashboard owns the motors: 600 ms watchdog, PWM cap 1800/4095.
- A deploy must preserve `~/ugv/runs/student.npz` and the camera drop-in `vision-camera.conf`.
- An estimate-based "arrived" is not a parked car. Confirm with the end frame.

## Research memory and coordination

`research/` is the only memory and work queue: decisions, attempts and tasks, one Markdown file
each (`research/README.md`, skill `research-ledger`). Attempts form a tree through `parent`
(Dream-RSI style), so what failed and why can be replayed instead of re-run.

1. Start: `git pull --ff-only`, then `uv run python tools/research/ledger.py status --as <you>`.
2. Before designing or running: `ledger.py list --grep <topic>`. A `negative`, `invalid` or
   `retracted` attempt is closed unless its "Re-run only if" holds.
3. Claim the task (`ledger.py claim`; refused until its `after` tasks are done), do the work.
4. Finish: an attempt record for any result, then `ledger.py done --result <attempt-id|commit>`.
5. The roadmap is `ledger.py roadmap`, computed from task `after` links; there is no roadmap file.

## Commands

`docs/COMMANDS.md` is the single command table and entry-point index. Add a row there when you add
a runner; do not add a second way to do the same thing.

Checks, required before every commit and run by CI:

```sh
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Environment

Root `pyproject.toml`, Python >= 3.13, same package on the desktop and the Pi. No torch: code
under `src/ugv` must import without it. Pose training, when it starts, gets its own CUDA
environment in a subdirectory, as the retired `rl/` had.

## Where things live

| Path | Holds |
|---|---|
| `src/ugv/` | Pi runtime: dashboard (`web.py`), vision, heading, parking, state machines (`fsm.py`) |
| `src/ugv/pose/` | Airframe mesh, keypoint skeleton, synthetic renderer, PnP and SIFT pose |
| `src/ugv/research/`, `research/` | Ledger schema and checks; the decision, attempt and task files |
| `tools/pose/`, `tools/research/` | Pose smoke test; `ledger.py` |
| `simulation/gazebo/` | Worlds, models, bridge, policy episode runner |
| `tools/l7/` | Field harness: deploy, run one mode on the Pi, pull records |
| `tools/calibration/` | SysID trials, fits and reports |
| `calibration/` | Curated fits and manifests. Raw session logs stay local and on the Pi, ignored by Git |
| `config/` | Runtime configuration read on the car |
| `docs/` | One topic per file, indexed in `docs/INDEX.md`; dated history in `docs/archive/` |
| `tests/` | Root unit tests |

Source of truth per topic: car status → `docs/STATE.md`; direction and experiments →
`docs/POSE.md`; work, decisions and past attempts → `research/`; safety → `docs/SAFETY.md`;
network and SSH → `docs/NETWORK.md`; field findings → `docs/L7_FIELD_NOTES.md`. State a fact in
one place and link to it from the others.

## Frames and tags

- Policy frame: drone at the origin, skids along y, the car enters from +y facing −y; pose is
  `(x, y, yaw)`.
- Yaw is counter-clockwise positive in `autopark.py`, `policy.py`, `tools/l7/l7_run.py` and
  the retired `rl/sim.py`. `parking.Pose2` is clockwise positive. Cross the boundary only through
  `pose2_from_ccw` / `pose2_to_ccw` in `src/ugv/parking.py`.
- The drone carries tags 14, 6 and 3. The older layout 9/12/13/14 survives in some configs and
  simulation flags; check which one a file means before using it.

## Evidence rules

- Synthetic pose numbers debug the pipeline; only the real evaluation set
  (`research/tasks/2026-11-17-real-pose-eval-set.md`) is evidence about the method.
- Simulation success is not hardware success. Before trusting a controller on the car, measure
  the sensor at rest, loop lag and action-to-duty parity.
- Record measured numbers in an attempt with their run id. Never commit weights, raw logs,
  frames, videos, credentials or machine-specific paths.

## Code shape

- Reusable code goes in a package; a runner is a thin `main()`.
- Aim for files under about 300 lines so one can be read in a sitting. Split at a real seam
  (separate responsibilities), never mechanically. A longer file that does one thing is fine.
- Delete what is superseded. Git history is the archive.

## Skills

Project skills live in `.agents/skills/` (`.claude/skills` and `.opencode/skills` link there):

- `research-ledger`: reading and writing `research/` (session start, claiming, finishing, asking).
- `mastermind`: delegating to subagents and gating their output.
- `orchestrate-and-gate`: log-first diagnosis and sim-to-real release gates.
- `mesh-optimization`: simplify heavy meshes into game-ready ones behind a camera-truth gate.

Skills follow the Agent Skills spec (agentskills.io): a directory with `SKILL.md`, `name`
matching the directory, and a `description` that says when to use it. Keep them under
~500 lines and push detail into `references/`. `agents.md` spec: plain Markdown, no required
fields; the nearest `AGENTS.md` to a file wins.
