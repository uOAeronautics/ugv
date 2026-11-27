---
id: 2026-11-27-replace-freenove-car
date: 2026-11-27
author: fer-agy
status: active
supersedes: []
refs: [simulation/gazebo/CAR_MODEL.md, simulation/gazebo/payload_dimensions.json, simulation/gazebo/assets/car/payload_assembly/SOURCE.md, docs/archive/2026-11-27-freenove-car.md]
tags: [hardware, simulation, scope]
---

## Decision

The Freenove FNK0043 4WD mecanum car is replaced by the new compact N20/BNO055/M12 payload car
(`Payload Assembly.usdz`, `118.85 × 149.00 × 60.74 mm`, `39.90 mm` 7-roller mecanum wheels) as the
default simulated vehicle in `simulation/gazebo/generate.py` (`--car payload`, default). The
Freenove procedural model remains selectable via `--car freenove`, its calibration/validation files
are preserved as evidence for earlier attempts, and its summary is archived in
`docs/archive/2026-11-27-freenove-car.md` (orchestrator tag `freenove-car-20261127`).

No Freenove physical or sensor calibration (mass, TT motor torque/PWM curves, mecanum slip fit,
Arducam UC-788 intrinsics, or Nano BMI270 noise profile) may be transferred to the payload car.
The Gazebo payload car model omits `<inertial>`, motor plugins, and calibrated sensor plugins until
the physical hardware is characterised (`2026-11-27-characterise-payload-car`). The Pi runtime
under `src/ugv/` is untouched until the new car's SBC, motor driver, camera, and BNO055 interfaces
are verified.

## Why

The owner is replacing the Freenove platform with the custom payload car CAD (`Payload Assembly.usdz`,
SHA-256 `2694510a1e074a20dbdaf9ca40801a286d04770c918ecb6e26b7208e518f8380`), which has completely
different geometry (`70.80 mm` wheelbase vs `139.7 mm`, `39.90 mm` wheels vs `63.5 mm`, `60.74 mm`
overall height), actuators (N20 gearmotors vs TT motors), IMU (BNO055 vs BMI270), and camera
(`M12_lens` on `40×40 mm` PCB). Treating Freenove fits as valid for the new vehicle would violate
the repository's no-fabricated-stand-ins evidence rule.

## Revisit if

A revised CAD revision supersedes `Payload Assembly.usdz` (re-run `convert_payload_usd.py` and
re-gate with `tools/mesh_opt/meshgate.py`), or once physical hardware characterisation populates
measured mass, inertia, motor, camera, and IMU parameters.
