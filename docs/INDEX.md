# INDEX

`purpose:=minimal-token context; audience:=LLM; prose:=min; truth:=typed`

## Load order

```text
L0 ../README.md
L1 STATE.md + SAFETY.md + NETWORK.md
L2 HARDWARE.md + COMMANDS.md + DASHBOARD.md
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
- [`docs/STATE.md`](STATE.md) — whether the car may be touched: Pi pause, deploy state, hardware.
- [`docs/PARK_SIM.md`](PARK_SIM.md) — parking simulator: what is modelled from where, results, limits.
- [`docs/POSE.md`](POSE.md) — current direction: airframe pose from the 3D model, experiments in order, parking state machine.
- [`research/README.md`](../research/README.md) — decisions, attempts (a tree of what was tried) and tasks; `tools/research/ledger.py`.
- [`docs/NETWORK.md`](NETWORK.md) — single source of truth for Pi SSH (`ssh pi` via `glinet`), IP, and Wi-Fi hardening.
- [`docs/COMMANDS.md`](COMMANDS.md) — canonical command table, grouped script index, and bounded hardware diagnostics.
- [`docs/SAFETY.md`](SAFETY.md) — S0–S4 actuation gates, watchdog/stop invariants, and abort rules.
- [`docs/HARDWARE.md`](HARDWARE.md) — measured car envelope/mass, camera, IMU, power topology, and I2C bus layout.
- [`docs/DASHBOARD.md`](DASHBOARD.md) — WebSocket/MJPEG dashboard contract, manual controls, and Follow behavior.
- [`docs/ARUCO.md`](ARUCO.md) — 50 mm DICT_4X4_50 detection, curled/oblique rescue, follow control, and failsafes.
- [`docs/PARKING.md`](PARKING.md) — provisional tag/IMU/scene map, `swept_clear` planner, and entry gates.
- [`docs/DYNAMICS.md`](DYNAMICS.md) — 100 Hz IMU/PWM capture, drift-bracketed identification, and Gazebo effort fit.
- [`docs/MECANUM_REAL2SIM.md`](MECANUM_REAL2SIM.md) — Bayesian mecanum movement model, visual–inertial scale, and heldout metrics.
- [`docs/L7_PREFLIGHT.md`](L7_PREFLIGHT.md) — L7 real-robot preflight checklist, confirmation flags, and 30 s SysID workflow.
- [`docs/L7_FIELD_NOTES.md`](L7_FIELD_NOTES.md) — 2026-11-13 L7 real-car field test results, plant ID, and camera/tag notes.
- [`docs/STATE_MACHINES.md`](STATE_MACHINES.md) — audit of implicit state machines, what is now an explicit `Machine`, ranked open items.
- [`docs/archive/STATE_HISTORY.md`](archive/STATE_HISTORY.md) — archived dated state snapshots (`2026-08-29`..`2026-10-17`).
- [`docs/archive/2026-11-27-freenove-car.md`](archive/2026-11-27-freenove-car.md) — archived Freenove 4WD mecanum car summary and preserved calibration pointers.
- [`docs/archive/council_2026-11-14/`](archive/council_2026-11-14/) — 2026-11-14 council reports; 03, 06, 07 and 08 are prior analysis for `POSE.md`.
- [`simulation/gazebo/README.md`](../simulation/gazebo/README.md) — Gazebo payload/Freenove, drone, and lab models, Blender previews, and keyboard console.
- [`simulation/gazebo/CAR_MODEL.md`](../simulation/gazebo/CAR_MODEL.md) — payload car CAD conversion, Gazebo frame, measured dimensions, and unmeasured hardware TODOs.
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
