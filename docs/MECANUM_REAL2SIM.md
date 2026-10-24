# Mecanum real2sim calibration, 23 October 2026

The current result is a probabilistic movement model with **partial heldout validation on visually observed trajectory segments**. It does not establish faithful full-trajectory real2sim or sim2real performance. Dark, blurred camera frames leave substantial translation gaps, particularly at higher power. There is no surveyed external pose reference.

The versioned candidate is [`mecanum_bayesian_profile.json`](../simulation/gazebo/mecanum_bayesian_profile.json), artifact `2026-10-23.continuous-pilot02.v1`. It contains model coefficients, 60 joint posterior draws, priors, grid probabilities, metrics, exclusions, source recording hashes and camera-scale uncertainty. It is an offline empirical world model, **not** a native Gazebo motor-parameter calibration or an online controller. Raw recordings under `calibration/runs/` are ignored by Git.

## Recorded evidence

| Recording | Raw IMU samples | Duration | Camera frames | Purpose |
| --- | ---: | ---: | ---: | --- |
| `2026-10-23-mecanum-probe-01` | 6,615 | 66.17 s | 1,429 | Short direction/power probes; 29 complete trials, trial 29 incomplete |
| `2026-10-23-mecanum-imu-rest-01` | 6,062 | 60.59 s | 1,323 | Dedicated stationary bias, noise and current tilt |
| `2026-10-23-mecanum-continuous-pilot-02` | 9,404 | 94.21 s | 1,917 | Eight complete eight-second movements, rest boundaries and baseline; trial 8 incomplete |

Continuous commands change forward/strafe direction every 0.2 s, including reversals and coupled wheel loading. PWM limits are 1,400, 2,600 and 4,095. Whole trials 0–5 are training data: two repetitions at each limit. Whole trials 6 and 7 are held out at 1,400 and 2,600. The final 4,095 heldout trial was interrupted and is excluded. Thus **there is no completed heldout full-power trajectory**.

Short probes use PWM 700, 1,400, 2,200, 3,200 and 4,095 with 0.10–0.16 s drive periods. They establish sign and transient response, but cannot establish steady speed or long-duration lag. They have no heldout split.

### Coordinates and polarity

Model velocity is `[forward m/s, left m/s, CCW rad/s]`. Recorded wheel order is `front-left, rear-left, front-right, rear-right`; these are actual signed hardware PWM values divided by 4,095. Hardware PWM polarity is retained in the empirical features.

The API uses **positive strafe for right**, with A/left sending a negative value. Both IMU launch acceleration and floor-camera motion confirm that convention. No strafe-sign reversal is needed in the hardware mixer.

## Translation reference and its uncertainty

The camera pipeline tracks floor features with forward/backward KLT checks, RANSAC homography and floor-plane decomposition. Velocity is derived from camera displacement and frame timestamps. Reference rotation and lever-arm correction use the IMU. Assumed intrinsics are focal length 968 pixels at 1,280 × 720; initial floor clearance is 65 mm. Approximate body offsets are IMU `[-0.051302, 0.002974] m` and camera `[0.10, 0] m`.

Visual–inertial alignment compares covariant visual velocity increments with integrated, rest-bracketed acceleration. It estimates shared height/scale and camera/body yaw on **training trials only**, with robust residual weighting and a whole-training-trial bootstrap. The alignment includes rotating-frame transport and approximate IMU/camera lever arms. Mounting direction also excludes heldout launches.

| Alignment | Height / scale | Bootstrap 95% interval | Integrated acceleration consistency RMSE |
| --- | ---: | ---: | ---: |
| Short probes | 66.35 mm | 64.84–68.16 mm | 0.0435 m/s, training only |
| Continuous pilot 02 | 59.92 mm | 57.46–61.43 mm | 0.0561 m/s training; 0.0716 m/s heldout |

The roughly **10% shift between recordings** exceeds their bootstrap intervals. Those intervals capture trial sampling variation, not all mounting, pitch, intrinsics, accelerometer calibration or floor-model error. Metric scale shares acceleration calibration and is labeled `visual_inertial_train_scaled`, never surveyed external truth.

The plain homography pipeline accepts 960 of 1,916 continuous image pairs. A fixed horizontal-floor, gyro-assisted alternative accepts 757 pairs and has worse scale consistency; it is preserved as a diagnostic, not selected as the reference. A separate dynamic-plane guard improves tracking but does not yet yield complete eight-second observations. Camera exposure changes made after this capture are not evidence that these gaps have been resolved.

### Observability and exclusions

An active-motion reference gap over 120 ms is not silently interpolated. Evaluation uses contiguous visually observed segments of at least 250 ms, preserving the original whole-trial training/heldout split. Each segment starts from its observed initial velocity. Endpoint errors below therefore refer to **segment endpoints**, not complete eight-second movements.

| Whole trial | PWM | Split | Observed segment duration / full recorded trial |
| --- | ---: | --- | ---: |
| 0 | 1,400 | train | 62.4% |
| 1 | 2,600 | train | 25.9% |
| 2 | 4,095 | train | 11.7% |
| 3 | 1,400 | train | 93.5% |
| 4 | 2,600 | train | 36.2% |
| 5 | 4,095 | train | 8.9% |
| 6 | 1,400 | heldout | 76.6% |
| 7 | 2,600 | heldout | 9.6% |

The denominator includes post-drive rest. There are 26 training segments and eight heldout segments; those eight segments come from **only two independent heldout trajectories**.

Probe trial 29 is excluded: approximately −542°/s yaw and 724°/s tilt rate accompanied abrupt camera yaw and blur. The images do not distinguish chassis lurch from IMU/mount flex. Other high-speed pure-rotation probes lose visual translation during blur, so their accelerometer-integrated translation is not accepted as independent linear-motion evidence. Stationary endpoint-constrained IMU translation remains explicitly provisional.

## Bayesian movement model

The deterministic response combines asymmetric wheel drive, nonlinear saturation/loading terms and first-order relaxation:

- Normalize actual wheel PWM to `u` and apply candidate deadzone `d`: `q = sign(u) max(|u| − d, 0) / (1 − d)`.
- Sixteen features concatenate positive `q`, negative `q`, `q |q|`, and `q mean(|q|)` for all four wheels. There is no equilibrium intercept.
- Target body velocity is `feature(q) × B`. Drive and zero-PWM braking have separate time constants.
- An exact first-order discrete decay predicts each transition. For fitting, transitions are thinned to at least 80 ms and time-varying wheel features are integrated over each transition. Averaging input over a transition remains an approximation during rapid switching.

A matrix-normal inverse-Wishart posterior jointly represents the 16 × 3 response coefficients and the full three-axis residual covariance. The weak coefficient row precision is `0.01 I`; inverse-Wishart degrees of freedom are 5 with scale `diag(0.02², 0.02², 0.08²)` in velocity units. These are response-model regularizers, not claims about intrinsic sensor noise.

An equal categorical prior covers 175 combinations: deadzone `{0, .05, .10, .16, .23}`, drive lag `{.04, .08, .15, .25, .4, .65, 1} s`, and braking lag `{.025, .06, .12, .25, .5} s`. All posterior weights use training data. Shared scalar lag preserves the conjugate cross-axis fit; separate axis lags are not inferred by this Bayesian model.

Each posterior rollout selects one joint parameter draw for the whole trajectory and then samples correlated transition disturbances. Coefficient uncertainty and noise are therefore propagated together. The stored 60 draws are reproducible with seed 29. This is a Bayesian fit; earlier bootstrap-only fits are not described as Bayesian.

The dominant continuous-data grid state is drive lag **250 ms**, braking lag **120 ms**, deadzone zero, with posterior probability about 0.69. Most remaining mass favors a 400 ms drive lag or a 0.05 deadzone. Short probes favored about 80 ms drive lag. That disagreement shows why short bursts alone were inadequate.

A prediction bug was corrected during analysis: the fitted conditional covariance was being multiplied by PWM again during rollouts. Removing this duplicate scaling restored agreement between the fitted likelihood and its predictive simulation. No arbitrary covariance multiplier was chosen to meet a coverage target.

## Heldout results

These numbers use the observed segments described above. Velocity coverage is the fraction of observed samples inside a 95% posterior predictive interval; it is descriptive because samples and segments within a trajectory are correlated.

| Model / metric | Forward | Left | Yaw |
| --- | ---: | ---: | ---: |
| Continuous model heldout velocity RMSE | 0.0528 m/s | 0.0324 m/s | 0.1453 rad/s |
| Continuous model heldout 95% velocity coverage | 96.9% | 86.3% | 92.8% |
| Short-probe model transferred to the same heldout segments: velocity RMSE | 0.1176 m/s | 0.0797 m/s | 0.1766 rad/s |
| Continuous model heldout segment endpoint RMSE | 0.0303 m | 0.0165 m | 0.0647 rad / 3.7° |
| Continuous model heldout 95% endpoint coverage | 87.5% | 100% | 87.5% |

The continuous model improves transfer from short probes, but lateral predictive coverage remains below its nominal 95%. Its intervals condition on the fitted visual scale and model family; systematic scale uncertainty is reported separately and is not fully propagated into these numbers.

All continuous commands have zero requested turn, yet measured yaw coupling is substantial. First-repetition peak yaw rates at PWM 1,400 / 2,600 / 4,095 are approximately **49 / 181 / 239°/s**; the second repetition gives **35 / 72 / 86°/s**. Heldout peaks are **28 / 69°/s**. This variation is part of the physical/model evidence, not stationary IMU white noise. The observed continuous subset does not validate independently commanded pure yaw at all powers.

## IMU calibration and noise separation

[`imu_noise_profile.json`](../simulation/gazebo/imu_noise_profile.json) retains the simulator's compatible schema, with the new recording hashes and current mounting. The calibrated body-to-sensor columns are:

| Body axis | Sensor X | Sensor Y | Sensor Z |
| --- | ---: | ---: | ---: |
| Forward | 0.20237 | −0.96361 | −0.17466 |
| Left | 0.96012 | 0.23036 | −0.15847 |
| Up | 0.19294 | −0.13563 | 0.97179 |

The forward probe direction is reorthogonalized against the newer stationary gravity. Matrix orthogonality error is below `5e-16` and determinant is one. This mathematical check does not prove the fixture remained rigid during high-power movement. The later continuous training launches independently estimate a slightly different basis, which is preserved in the Bayesian profile.

Stationary gyro bias is `(0.04538, 0.83157, −0.04925)°/s`; per-sample standard deviations are `(0.03911, 0.04111, 0.04400)°/s`. Acceleration standard deviations are `(0.000628, 0.000698, 0.001040) g`. The profile includes full covariance, adjacent-difference covariance, half-second block statistics and overlapping Allan deviations from about 10 ms through 5 s. Allan implementation recovers synthetic white-noise behavior at 10 ms, 100 ms and 1 s.

The roughly 0.04°/s rest gyro noise is over 200 times smaller than heldout yaw model RMSE. Dynamic residuals therefore principally reflect movement/model error rather than stationary sensor white noise. Moving acceleration also contains real chassis acceleration, passive-roller ripple and possible mounting motion. Reference noise, physical process variation and model discrepancy are not separately identifiable from these measurements. The posterior residual covariance must not be called intrinsic motor noise, nor indiscriminately injected into native physics as physical movement.

This is **not** a six-pose accelerometer gain calibration. Sixty seconds in one pose do not identify independent accelerometer offsets/gains, long-term Allan bias instability, temperature drift or power-cycle repeatability. The gravity-norm offset retained for simulator compatibility is an assumption along gravity.

## Native Gazebo versus the empirical model

The Gazebo car uses physical passive mecanum rollers and finite motor torque. The Bayesian velocity response above is a separate empirical model; its coefficient matrix cannot be copied into native wheel voltage gains without identifying that mapping.

[`mecanum_continuous_nominal_validation.json`](../simulation/gazebo/mecanum_continuous_nominal_validation.json) records the nominal physical model's replay. On its visually observed heldout timestamps, RMSE is approximately **0.155 m/s forward, 0.0569 m/s left, 0.164 rad/s yaw**. Its report explicitly marks nominal motors and provisional reference status. That replay evolves the whole physical trajectory, whereas the empirical segment evaluation conditions on velocity at each segment boundary; their numerical errors are not a controlled direct ranking.

A calibrated native motor/contact model, complete heldout full-power movement, and an independently measured full-trajectory pose reference remain required before claiming faithful Gazebo transfer or a sim2real acceptance result. Current evidence supports a partially validated empirical candidate and identifies where the nominal simulation and sensing need work.

## Subsequent native-physics fit

[`mecanum_native_profile.json`](../simulation/gazebo/mecanum_native_profile.json) fits physical motor parameters using pilot-02 training trials 0–5 only. Heldout trials 6/7 were replayed without tuning. The fitted PWM exponent is approximately 0.596, nominal motor free speed 33.27 rad/s, and motor lag 0.218 s; the model retains physical passive-roller bearing drag.

[`mecanum_continuous_native_heldout_validation.json`](../simulation/gazebo/mecanum_continuous_native_heldout_validation.json) reports heldout observed-camera translation RMSE of **0.0791 m/s forward and 0.0391 m/s left**, improving the nominal model's 0.1553 / 0.0569 m/s. Whole-trajectory gyro yaw RMSE is **0.2019 rad/s**, worse than nominal 0.1867 rad/s. Camera observations cover only **43.5%** of heldout timestamps; censored segment endpoint RMSE is **0.0634 / 0.0360 m and 0.0700 rad**. This remains provisional native-physics validation, with a translation improvement and a yaw regression rather than complete acceptance.

## New complete guarded trajectory: frozen-model evaluation

After camera/processing changes, recording `2026-10-23-mecanum-continuous-loops-01` completed a new **8.077-second** commanded translation loop at **PWM 1,000**. Every active online guard pair was accepted and healthy, with no tracking reset or missing-motion interpolation. Camera provenance requests 3,000 μs exposure and gain 12, with 480-pixel guard processing; recorder metadata is a request, not independent hardware-exposure verification.

This trajectory is evaluation only. The recorder labels it `train`, but **that label is ignored here**: the versioned pilot-02 model, scale and posterior were frozen, with no refitting or evaluation-driven hyperparameter changes. PWM 1,000 lies below the model's 1,400 minimum training limit, and the new direction cycle is two seconds. It tests transfer to a different operating pattern.

[`mecanum_guarded_ood_validation.json`](../simulation/gazebo/mecanum_guarded_ood_validation.json) stores raw-recording hashes, the frozen model hash, guard status, reference and prediction endpoints, predictive intervals and measurement limitations. Evaluation uses actual per-sample wheel PWM, starts from stationary velocity, derives yaw from rest-bracketed raw IMU data, and takes full relative position from the accepted online guard. Guard height is the training-derived **59.92 mm**; it remains relative visual–inertial reference, not externally surveyed pose.

| Complete-trajectory metric | Forward | Left | Yaw |
| --- | ---: | ---: | ---: |
| Velocity RMSE | 0.03985 m/s | 0.05929 m/s | 0.15495 rad/s |
| 95% predictive velocity coverage | 98.9% | **71.3%** | 95.4% |
| Signed full endpoint error | +0.1308 m | +0.0748 m | +0.00389 rad / +0.22° |
| Reference endpoint inside predictive 95% interval | yes | yes | yes |

Full endpoint position error is **15.06 cm**. The endpoint falls inside the posterior intervals, but that does not cancel the large position error or lateral undercoverage. This recording closes the *observation continuity* gap for one lower-power eight-second trajectory. It **does not establish faithful movement transfer**, higher-power complete-trajectory performance, independent metric accuracy, or pure-yaw acceptance.

The evaluator is `tools/calibration/evaluate_mecanum_guarded.py`; it has no fitting or hardware commands. JPEG receipt timestamps and guard velocity differences contribute temporal uncertainty, IMU yaw is shared with visual rotation compensation, and predictive intervals do not fully propagate systematic scale/mounting uncertainty.

## Reproduction

Run from the repository root with Python, NumPy and OpenCV available. Using an isolated analysis environment avoids installing Raspberry Pi-only dependencies on a desktop:

```sh
PYTHONPATH=src:tools/calibration uv run --no-project --with numpy \
  --with opencv-contrib-python-headless python tools/calibration/mecanum_vision.py \
  calibration/runs/2026-10-23-mecanum-continuous-pilot-02 --align-imu

PYTHONPATH=src:tools/calibration uv run --no-project --with numpy \
  python tools/calibration/mecanum_fit.py \
  calibration/runs/2026-10-23-mecanum-continuous-pilot-02 \
  --bayesian --visual-segments --posterior-draws 60

PYTHONPATH=src:tools/calibration uv run --no-project --with numpy \
  python tools/calibration/mecanum_imu_noise.py \
  calibration/runs/2026-10-23-mecanum-imu-rest-01 \
  calibration/runs/2026-10-23-mecanum-probe-01

PYTHONPATH=src uv run --no-project --with numpy --with pytest \
  python -m pytest tests/test_mecanum_dynamics.py -q
```

Model invariants, synthetic unseen-input recovery, posterior uncertainty and probability normalization pass five tests. Those checks validate implementation behavior; they do not substitute for the missing physical acceptance evidence.
