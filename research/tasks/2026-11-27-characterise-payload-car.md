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

Bench-characterise the physical payload car hardware before enabling dynamic simulation or porting
`src/ugv/` runtime drivers (all parameters are currently UNKNOWN; do not reuse Freenove values):

1. **Mass & inertia:** weigh total assembled vehicle, individual 7-roller mecanum wheel, battery
   (`2S LiPo`), and chassis; record centre of mass and populate `UNMEASURED_CHASSIS_MASS_KG_TODO` /
   `UNMEASURED_WHEEL_MASS_KG_TODO` in `simulation/gazebo/payload_car_model.py`.
2. **SBC & motor driver board:** identify the `PCB_56x41` SBC, OS, bus/pinout of the motor driver,
   safe PWM/duty bounds, hardware watchdog/stop path, and battery voltage sense (S0→S4 safety gates).
3. **N20 motor & mecanum SysID:** identify N20 gear ratio, free-run speed, stall torque, deadzone,
   and floor translation/strafe/yaw response under S4 bench/floor gates.
4. **Camera intrinsics:** identify the sensor module behind `M12_lens__1_`, capture checkerboard
   calibration frames, and measure `fx`, `fy`, `cx`, `cy`, distortion, shutter type, and FOV.
5. **BNO055 IMU:** verify bus/address, operating mode, body-to-sensor mounting matrix, and resting
   noise/bias drift over a stationary capture.

## Done when

Measured mass/inertia, motor SysID, driver interface spec, calibrated camera intrinsics, and BNO055
resting noise profile are recorded in attempt/decision records, `simulation/gazebo/payload_dimensions.json`
and `payload_car_model.py` carry measured parameters without `TODO` placeholders, and `src/ugv/`
driver requirements are specified for the new board.
