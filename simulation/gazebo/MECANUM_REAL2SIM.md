# Native mecanum candidate: pilot02 v1

The default car world now loads `mecanum_native_profile_pilot02_v1.json` when
available. This is a **candidate with partial visual observations**, not a fully
accepted real2sim model. Training used completed continuous trajectories 0–5 of
`2026-10-23-mecanum-continuous-pilot-02`. Trajectories 6–7 were replayed after the
parameters and continuous effort-lag semantics were fixed.

The model retains four finite-torque motors and 32 independently rotating,
45-degree passive rollers. PWM nonlinearity, directional effort gains, aggregate
effort lag, and passive bearing damping were initialized from training data. The
native Gazebo replay includes the complete actual wheel-command history, including
commands during intervals where the camera reference was rejected.

## Quantitative comparison

Translation below uses only contiguous accepted visual segments. Yaw uses the
bias-corrected gyro over all completed trajectory samples. Units are m/s for
translation and rad/s for yaw.

| Split and metric | Nominal | Native candidate |
| --- | ---: | ---: |
| Training forward velocity RMSE | 0.1984 | 0.1001 |
| Training lateral velocity RMSE | 0.0646 | 0.0530 |
| Training gyro yaw-rate RMSE, all samples | 0.4568 | 0.4355 |
| Heldout forward velocity RMSE | 0.1553 | 0.0791 |
| Heldout lateral velocity RMSE | 0.0569 | 0.0391 |
| Heldout gyro yaw-rate RMSE, all samples | 0.1867 | 0.2019 |
| Heldout forward RMSE, active command and valid vision | 0.1741 | 0.0901 |
| Heldout lateral RMSE, active command and valid vision | 0.0650 | 0.0444 |

Translation improved, while heldout yaw error increased. The candidate therefore
has an unresolved rotation/generalization gap. Accepted visual observations cover
40.2% of training samples and 43.5% of heldout samples. Missing intervals are not
filled with IMU-derived translation for these metrics. The heldout PWM4095
trajectory did not complete; completed heldout data covers PWM1400 and PWM2600.

Local displacement RMSE within observed heldout segments is 0.0634 m forward,
0.0360 m lateral, and 0.0700 rad yaw. Both headings are anchored at each observed
segment start. These are censored local rollouts, not full eight-second trajectory
endpoint errors. Full-trajectory translation acceptance is unavailable.

The metric scale is visual/inertial and shares accelerometer and camera mounting
errors; there is no independent surveyed external position reference. Motor speed,
voltage gain, bearing load and tire slip cannot be separately identified from
these data. The prior total mass and wheel dimensions also remain assumptions
for the converted robot. Zero parameter randomization is the default; uncertainty
is not invented to cover a deterministic mismatch.

## Reproduce

From the repository root, the generator uses only the Python standard library:

```
python3 simulation/gazebo/generate.py
python3 simulation/gazebo/generate.py --nominal-motors --output /tmp/ugv_nominal.sdf
python3 simulation/gazebo/generate.py --motor-profile simulation/gazebo/mecanum_native_profile_pilot02_v1.json
```

The generated SDF embeds the candidate filename, SHA256 and acceptance status.
The common `native_motor_parameters.apply_native_parameters` helper updates the
motor plugin, hub velocity/effort constraints, and passive bearing parameters
consistently. Room/drone worlds are regenerated only when explicitly requested.

Use the Gazebo control environment for native replay:

```
simulation/gazebo/.control-venv/bin/python tools/calibration/evaluate_mecanum_gazebo.py calibration/runs/2026-10-23-mecanum-continuous-pilot-02 --nominal-motors --split heldout --output /tmp/nominal_heldout.json
simulation/gazebo/.control-venv/bin/python tools/calibration/evaluate_mecanum_gazebo.py calibration/runs/2026-10-23-mecanum-continuous-pilot-02 --motor-parameters simulation/gazebo/mecanum_native_profile_pilot02_v1.json --split heldout --output /tmp/candidate_heldout.json
```

Saved results are `mecanum_continuous_nominal_validation.json`,
`mecanum_continuous_native_train_validation.json`, and
`mecanum_continuous_native_heldout_validation.json`, each with a matching
`.traces.jsonl` file. The primary metric fields explicitly separate valid visual
translation from whole-trajectory gyro yaw. Mechanics smoke checks additionally
cover six signed body motions, stable support, full PWM, finite torque, and
watchdog braking. Passing those checks does not close the real2sim accuracy gaps.
