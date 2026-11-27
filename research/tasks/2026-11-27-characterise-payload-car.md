---
id: 2026-11-27-characterise-payload-car
date: 2026-11-27
author: fer-agy
owner: fernando
state: open
updated: 2026-11-27
due: none
after: []
check: none
refs: [simulation/gazebo/CAR_MODEL.md, simulation/gazebo/payload_dimensions.json, simulation/gazebo/payload_car_model.py, docs/SAFETY.md]
tags: [hardware, calibration, simulation]
---

## Goal

Bench-characterise the physical payload car hardware before enabling calibrated dynamic simulation
or porting `src/ugv/` runtime drivers (all parameters are currently UNKNOWN; do not reuse Freenove
values). In the interim, `simulation/gazebo/payload_prior.json` defines an explicit Monte Carlo
prior (truncated at the owner's 2026-11-27 `"probably under 300g"` bound) for exploratory Gazebo
runs only; no controller tuned on those prior samples may be trusted on hardware.

1. **Mass & inertia:** weigh total assembled vehicle, individual 7-roller mecanum wheel, battery
   (`2S LiPo`), and chassis; record centre of mass and populate `UNMEASURED_CHASSIS_MASS_KG_TODO` /
   `UNMEASURED_WHEEL_MASS_KG_TODO` in `simulation/gazebo/payload_car_model.py`.
2. **SBC & motor driver board:** identify the `PCB_56x41` SBC, OS, bus/pinout of the motor driver,
   safe PWM/duty bounds, hardware watchdog/stop path, and battery voltage sense (S0→S4 safety gates).
3. **N20 motor & mecanum SysID:** identify N20 gear ratio, free-run speed, stall torque, deadzone,
   wheel mounting handedness (verify whether the physical car uses standard top-view X-pattern or
   matches the raw USD's O-on-top / X-on-floor `A`/`B` wheel placement), and floor
   translation/strafe/yaw response under S4 bench/floor gates.
4. **Camera intrinsics:** identify the sensor module behind `M12_lens__1_`, capture checkerboard
   calibration frames, and measure `fx`, `fy`, `cx`, `cy`, distortion, shutter type, and FOV.
5. **BNO055 IMU:** verify bus/address, operating mode, body-to-sensor mounting matrix, and resting
   noise/bias drift over a stationary capture.

## How each measurement collapses the Monte Carlo prior (`payload_prior.json`)

| Physical measurement | Prior parameters collapsed / replaced |
|---|---|
| **Kitchen-scale weighing (total assembled car)** | Replaces the bottom-up CAD volume + datasheet mass distribution and the `0.300 kg` owner upper bound (`total_mass_kg`). Resolves whether FDM infill + LiPo + SBC land under or above `300 g` (untruncated CAD prior spans `288–382 g`). |
| **Component weighing (1 wheel, 2S LiPo, chassis without battery) + balance-point check** | Collapses `pla_petg_fdm` / `tpu_roller_fdm` infill densities, `lipo_2s_72x35x17` and `sbc_56x41` envelope masses, and locks down chassis/wheel mass split, `assembly_com_m`, and `inertia_kg_m2`. |
| **Motor label / wheel-off-ground free-run RPM + supply voltage check (S4 bench)** | Collapses the discrete 8-variant `n20_motor_variants` prior (`gear_ratio`, `rated_voltage_v`, `no_load_speed_radps`) and `battery_pack_voltage_v` × `supply_efficiency`. |
| **Visual inspection of wheel roller orientation** | Resolves CAD `mecanum_hand_sign` discrepancy (raw USD has O-on-top / X-on-floor with `27.9 mm` yaw moment arm vs standard top-view X-pattern with `98.7 mm` moment arm). |
| **One floor `probe-axes` run (forward, strafe, yaw steps + coast, S4)** | Collapses `wheel_deadzone`, `wheel_duty_exponent`, `wheel_voltage_gain` per-wheel asymmetry, `wheel_drive_tau_s`, `wheel_brake_tau_s`, `brake_damping` / `brake_resistance`, `roller_mu`, and `roller_bearing_damping` (locking in real top speed, accel, strafe/forward ratio, yaw rate, and stop distance). |
| **Checkerboard capture + stationary IMU log (S0/S1)** | Supplies camera intrinsics and BNO055 noise/mounting matrix (omitted from the dynamics prior). |

## Done when

Measured mass/inertia, motor SysID, driver interface spec, calibrated camera intrinsics, and BNO055
resting noise profile are recorded in attempt/decision records, `simulation/gazebo/payload_dimensions.json`
and `payload_car_model.py` carry measured parameters without `TODO` placeholders, and `src/ugv/`
driver requirements are specified for the new board.
