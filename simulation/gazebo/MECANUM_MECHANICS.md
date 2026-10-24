# Mecanum mechanics

The photo estimate uses 63.5 mm wheel diameter, 32 mm width, 140 mm center track,
and 139.7 mm wheelbase. The prior 577.4 g total mass is retained, so the new wheel
mass distribution remains an assumption until the converted robot is weighed.

Each wheel has a metal hub and eight independently rotating cylinders at 45°.
Roller contacts are isotropic rubber; sideways travel comes from the free roller
joints. The chassis receives contact forces and the four hub joints receive
bounded motor torques. No plugin overwrites body velocity. Cylindrical rollers
approximate the real barrel shape and produce small contact ripple. Bearing drag,
friction and dimensions remain estimates.

Body X is forward, Y left, Z up. Positive physical wheel speed about +Y drives
forward. In FL, FR, RL, RR order, with `k=(wheelbase+track)/2`, the wheel speeds are:

```
[vx-vy-k*w, vx+vy+k*w, vx+vy-k*w, vx-vy+k*w] / radius
```

The hardware has reversed command polarity. The replay evaluator reorders the
recorded FL, RL, FR, RR stream to FL, FR, RL, RR and negates it exactly once.

## Native motor parameters

The `ugv::TTMotorSystem` plugin accepts `mecanum=true`, `wheelbase` in meters,
and four normalized duties on `/model/freenove/wheel_duty`. Duties span -1 to +1.
`cmd_vel` supports `linear.x`, `linear.y` and `angular.z`. The keyboard panel uses
W/S forward/reverse, A/D strafe, Q/E rotation, and PWM 0 through 4095.

Optional SDF elements contain four space-separated values in FL, FR, RL, RR order:

| Element | Default | Meaning |
| --- | --- | --- |
| `wheel_voltage_gain` | `1 1 1 1` | Positive physical direction gain |
| `wheel_reverse_voltage_gain` | `1 1 1 1` | Negative physical direction gain |
| `wheel_duty_exponent` | `1 1 1 1` | Loaded PWM response exponent applied after deadzone removal |
| `wheel_deadzone` | `0 0 0 0` | Normalized electrical input deadzone, rescaled above it |
| `wheel_drive_tau` | `0 0 0 0` | Aggregate effort-response time constant for nonzero command changes, seconds |
| `wheel_brake_tau` | `0 0 0 0` | Brake engagement time constant, seconds |
| `wheel_gain_stddev` | `0 0 0 0` | Relative Gaussian gain spread sampled at configure/reset |

The scalar `no_load_speed` is in rad/s and `stall_torque` is in Nm. Replay updates wheel joint velocity limits to the requested no-load speed. `roller_bearing_damping` and `roller_bearing_friction` are evaluator overrides for all passive roller joints, in Nm·s/rad and Nm respectively. These loaded aggregate parameters are not independent motor or bearing measurements.

`random_seed` is an unsigned integer, default zero. Sampling is deterministic for
an identical seed and reset sequence. Gains are clamped positive. Randomization
is optional and does not constitute a fitted uncertainty model by itself.

Motor torque follows the finite DC back-EMF relation, bounded by configured stall
torque. Zero duty engages finite braking resistance. Missing heartbeat after
400 ms engages braking immediately, bypassing configured brake take-up lag.
Legacy differential-drive empirical PWM tables and voltage gains are ignored in
mecanum/four-duty mode. Full voltage remains limited by physical stall torque.

## Reproduce checks and recorded replay

From the repository root:

```
simulation/gazebo/.control-venv/bin/python simulation/gazebo/check_mecanum_car.py
simulation/gazebo/.control-venv/bin/python tools/calibration/evaluate_mecanum_gazebo.py calibration/runs/RUN --output /tmp/replay.json
```

The first command checks all six signed body motions and stable support in the
full generated car. The second replays actual wheel telemetry in an isolated
transport partition, derives simulated body velocity from ideal Gazebo poses,
and compares it with the run's declared visual/inertial reference. Entire-trial
and active-command metrics are reported separately. It preserves recorded split
labels, so probe results cannot become heldout validation.

To supply fitted native parameters, pass `--motor-parameters FILE.json`, with the
SDF field names above either at the root or under `gazebo_motor_parameters`.
The evaluator writes metrics, per-trial velocity traces, and a simulator log.
It explicitly labels IMU/vision scale limitations. Passing geometry and motion
checks alone does not establish a faithful real2sim or sim2real dynamics match.

## Training and heldout native calibration

`tools/calibration/fit_mecanum_native.py RUN --output PROFILE.json` fits a bounded
first-order kinematic initialization using training trajectories only. Its model
includes directional wheel effort, PWM nonlinearity, response lag, and a lateral
attenuation converted to equivalent passive bearing drag. SciPy and NumPy are
required. Validate the resulting profile through actual native Gazebo replay;
the surrogate optimizer's success is not acceptance. Native replay preserves all
unobserved command history and censors translation metrics across rejected
vision intervals. Full gyro yaw remains independently compared over each trial.
