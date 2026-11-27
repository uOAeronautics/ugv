# Freenove 4WD Mecanum Car (Archived 2026-11-27)

Historical record of the retired Freenove FNK0043 4WD mecanum car platform, replaced as the default
simulated vehicle by the payload car (`simulation/gazebo/CAR_MODEL.md`, decision
`research/decisions/2026-11-27-replace-freenove-car.md`). The orchestrator tags the last
Freenove-default commit as `freenove-car-20261127`.

Do not transfer any Freenove dimensions, mass, motor torque/PWM fits, mecanum slip models, camera
intrinsics, or IMU noise profiles to the new payload car.

## Hardware summary (verified on `adv034`)

- **Compute:** Raspberry Pi 4B (`adv034`), Debian 13 (`aarch64`), service `ugv-dashboard.service`
  on port `8765`.
- **Car board:** Freenove PCB V3.0 (`i2c-1`: `0x40` PCA9685 PWM/motor controller, `0x48` ADC,
  `0x70` all-call); dashboard PWM cap `1800/4095`, `600 ms` motor watchdog.
- **Drive:** 4 × yellow TT DC gearmotors with 63.5 mm × 32 mm (assumed) X-pattern mecanum wheels.
- **Camera:** Arducam UC-788 Rev.B / Pivariety mono global-shutter (`1280x720@30`), upright
  40 × 40 mm white stand (`40 mm` back from board front edge, lens `20 mm` above main board),
  nominal uncalibrated `fx = fy = 968`, deployed shutter `4000 us` / gain `8`.
- **IMU:** Arduino Nano 33 BLE Sense Rev2 (BMI270) over USB (`/dev/ttyACM0`, `~100 Hz` JSON).
- **Power:** 2 × 3.7 V cells in Freenove holder (`~8.85 V` charged).

## Measured & modelled dimensions

From `docs/HARDWARE.md` and `simulation/gazebo/dimensions.json` (2026-10-17):

| Quantity | Value | Source |
|---|---|---|
| Total mass | `577.4 g` (`445.4 g` rigid link + wheels/carriers) | Measured (`docs/HARDWARE.md`, `simulation/gazebo/mass_distribution.json`) |
| Overall length (tire-to-tire) | `8.0 in` (`203.2 mm`) | Measured |
| Overall width (tire outer) | `5.75 in` (`146.05 mm`) | Measured |
| Main PCB envelope | `7.5 × 3.75 in` (`190.5 × 95.25 mm`), `1.6 mm` assumed thick | Measured / assumed |
| Wheel diameter | `2.5 in` (`63.5 mm`) | Measured |
| Wheelbase | `5.5 in` (`139.7 mm`) | Derived from symmetric axles |
| Wheel-center track | `140.0 mm` (mecanum assumed; `120.65 mm` original tires) | Assumed (`dimensions.json`) |
| Motor bottom clearance | `1.0 in` (`25.4 mm`) | Measured |
| Main board top height | `1.75 in` (`44.45 mm`) | Measured |

## Preserved files & calibration evidence

All Freenove calibration, validation, and model files remain in the repository as evidence for
earlier attempts (`research/attempts/`) and remain selectable in the Gazebo generator via
`python3 simulation/gazebo/generate.py --car freenove`:

- **Gazebo procedural model:** `simulation/gazebo/assembly.py`, `simulation/gazebo/dimensions.json`,
  `simulation/gazebo/tt_geometry.py`, `simulation/gazebo/mecanum_geometry.py`,
  `simulation/gazebo/mass_distribution.json`, `simulation/gazebo/mass_properties.py`,
  `simulation/gazebo/freenove_world.sdf`.
- **Motor & mecanum fits:** `simulation/gazebo/motor_profile.json`,
  `simulation/gazebo/real_robot_profile.json`,
  `simulation/gazebo/mecanum_native_profile.json`,
  `simulation/gazebo/mecanum_native_profile_pilot02_v1.json`,
  `simulation/gazebo/mecanum_bayesian_profile.json`,
  `simulation/gazebo/brake_effort_corrections.json`,
  `simulation/gazebo/MOTOR_MODEL.md`, `simulation/gazebo/MECANUM_MECHANICS.md`,
  `simulation/gazebo/MECANUM_REAL2SIM.md`.
- **Camera & IMU fits:** `simulation/gazebo/camera_profile.json`, `simulation/gazebo/camera_model.py`,
  `simulation/gazebo/CAMERA_MODEL.md`, `simulation/gazebo/imu_noise_profile.json`,
  `simulation/gazebo/imu_model.py`, `simulation/gazebo/IMU_MODEL.md`.
- **Curated real-car calibration & field notes:** `calibration/`, `docs/HARDWARE.md`,
  `docs/DYNAMICS.md`, `docs/MECANUM_REAL2SIM.md`, `docs/L7_FIELD_NOTES.md`, `docs/STATE.md`.
- **Pi runtime (unchanged until payload car electronics are characterised):** `src/ugv/` (PCA9685
  motor driver, Arducam vision, Nano BMI270 heading).
