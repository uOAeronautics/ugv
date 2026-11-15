# INDEX

`purpose:=minimal-token context; audience:=LLM; prose:=min; truth:=typed`

## Load order

```text
L0 ../README.md
L1 STATE.md + SAFETY.md + NETWORK.md
L2 HARDWARE.md + COMMANDS.md + DASHBOARD.md
L3 PROTOCOL.md
```

## Epistemics

```text
✓ observed/tested   ~ inferred   ? unknown/unverified   × absent/disabled/fail
! hazard/blocker    ∆ changed    @ locator              # reference
```

`conflict := newest-observation > inference > vendor-doc > assumption`
`mutation := declare intent -> safety gate -> smallest effect -> verify -> STATE∆`
`never-store := passwords|tokens|private-keys|router-secrets`

## Document map

- [`README.md`](../README.md) — project overview, verified status metrics, layout, and quickstart.
- [`HANDOFF.md`](../HANDOFF.md) — pre-charging Pi pause state, sync warnings, and next-step runbook.
- [`docs/STATE.md`](STATE.md) — current hardware/software status and pointers to field notes and archive.
- [`docs/NETWORK.md`](NETWORK.md) — single source of truth for Pi SSH (`ssh pi` via `glinet`), IP, and Wi-Fi hardening.
- [`docs/COMMANDS.md`](COMMANDS.md) — canonical command table, grouped script index, and bounded hardware diagnostics.
- [`docs/SAFETY.md`](SAFETY.md) — S0–S4 actuation gates, watchdog/stop invariants, and abort rules.
- [`docs/HARDWARE.md`](HARDWARE.md) — measured car envelope/mass, camera, IMU, power topology, and I2C bus layout.
- [`docs/DASHBOARD.md`](DASHBOARD.md) — WebSocket/MJPEG dashboard contract, manual controls, and Follow behavior.
- [`docs/ARUCO.md`](ARUCO.md) — 50 mm DICT_4X4_50 detection, curled/oblique rescue, follow control, and failsafes.
- [`docs/PARKING.md`](PARKING.md) — provisional tag/IMU/scene map, `swept_clear` planner, and entry gates.
- [`docs/RL_PARKING.md`](RL_PARKING.md) — RL goal, parked thresholds, L0–L7 ladder, contracts, and known gaps.
- [`docs/DYNAMICS.md`](DYNAMICS.md) — 100 Hz IMU/PWM capture, drift-bracketed identification, and Gazebo effort fit.
- [`docs/MECANUM_REAL2SIM.md`](MECANUM_REAL2SIM.md) — Bayesian mecanum movement model, visual–inertial scale, and heldout metrics.
- [`docs/L7_PREFLIGHT.md`](L7_PREFLIGHT.md) — L7 real-robot preflight checklist, confirmation flags, and 30 s SysID workflow.
- [`docs/L7_FIELD_NOTES.md`](L7_FIELD_NOTES.md) — 2026-11-13 L7 real-car field test results, plant ID, and camera/tag notes.
- [`docs/PIPELINE.md`](PIPELINE.md) — RL orchestrator/scout/adversary workflow, 1-GPU rule, and monotonic gating.
- [`docs/SIM_THROUGHPUT.md`](SIM_THROUGHPUT.md) — measured env/eval/Gazebo/NumPy throughput and worker settings.
- [`docs/AGENT_BOARD.md`](AGENT_BOARD.md) — RL experiment ledger, adversary verdicts, and gate decisions.
- [`docs/PROTOCOL.md`](PROTOCOL.md) — compact agent coordination grammar.
- [`docs/archive/STATE_HISTORY.md`](archive/STATE_HISTORY.md) — archived dated state snapshots (`2026-08-29`..`2026-10-17`).
- [`docs/archive/FAILURE_REPLAY.md`](archive/FAILURE_REPLAY.md) — archived Dream-RSI failure-replay curriculum proposal (`2026-11-13`).
- [`docs/archive/council_2026-11-14/`](archive/council_2026-11-14/) — archived council_v2 analysis reports (`2026-11-14`; scratch scripts/outputs removed).
- [`rl/README.md`](../rl/README.md) — RL subproject file map and L0 training/eval recipes (3090 CUDA env).
- [`rl/PLAN.md`](../rl/PLAN.md) — RL phase work queue, measured gate state, and guardrails.
- [`rl/AGENTS.md`](../rl/AGENTS.md) — RL agent rules, safety gates, and torch isolation constraints.
- [`simulation/gazebo/README.md`](../simulation/gazebo/README.md) — Gazebo Freenove/drone/lab models, Blender previews, and keyboard console.
- [`simulation/gazebo/MECANUM_MECHANICS.md`](../simulation/gazebo/MECANUM_MECHANICS.md) — passive mecanum roller geometry, native plugin params, and replay commands.
- [`simulation/gazebo/MECANUM_REAL2SIM.md`](../simulation/gazebo/MECANUM_REAL2SIM.md) — native mecanum Gazebo candidate `pilot02_v1` metrics and limitations.
- [`simulation/gazebo/CAMERA_MODEL.md`](../simulation/gazebo/CAMERA_MODEL.md) — simulated monochrome camera geometry, intrinsics, and browser checks.
- [`simulation/gazebo/IMU_MODEL.md`](../simulation/gazebo/IMU_MODEL.md) — simulated noisy BMI270 IMU profile, topics, and fitting limits.
- [`simulation/gazebo/MOTOR_MODEL.md`](../simulation/gazebo/MOTOR_MODEL.md) — `ugv::TTMotorSystem` DC motor/brake plugin build and validation.
- [`simulation/gazebo/DRONE_MODEL.md`](../simulation/gazebo/DRONE_MODEL.md) — quadcopter CAD/ArUco constellation assumptions and collision checks.
- [`simulation/gazebo/ROOM_MODEL.md`](../simulation/gazebo/ROOM_MODEL.md) — reconstructed 24×12 m robotics workshop layout and corridor assumptions.
- [`simulation/gazebo/ROOM_DETAILS.md`](../simulation/gazebo/ROOM_DETAILS.md) — workshop interior furniture/clutter pass and camera anchors.
- [`simulation/gazebo/ROOM_WEST_BAYS.md`](../simulation/gazebo/ROOM_WEST_BAYS.md) — west-wall glazing and bay detail pass.
- [`calibration/README.md`](../calibration/README.md) — calibration session directory layout and raw JPEG retention policy.
- [`arduino/README.md`](../arduino/README.md) — Nano 33 BLE Sense Rev2 standalone BLE firmware, VQF dashboard, and stress tools.
