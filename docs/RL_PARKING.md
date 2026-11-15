# RL drone parking

`goal := UGV finds drone anywhere in lab and parks centred under deck without touching gear, onboard sensing only, Pi 4B deploy`

## Parked (success)

Default thresholds, confirm against measured gear clearance (`config/parking.json` unverified):

```text
lateral   := |dy| <= 30mm vs drone-centre goal
long      := |dx| <= 50mm
yaw       := <= 5deg vs skid axis
corners   := all 4 footprint corners inside skid corridor
still     := |v| < 1cm/s and |w| < 2deg/s for 1.0s
contact   := zero over episode
false-park:= declared parked while outside above = failure, counted separately
```

## Metrics per level (>= 1000 seeded episodes, held-out seeds, Wilson 95%)

`success-lb, collision-rate, false-parked-rate, timeout-rate, median/p95 time-to-park, p95 final pos/yaw err, action jerk, Pi per-step latency`

## Ladder (each gate must pass before next counts)

```text
L0 aligned 0.75m ahead / posterior draw 0 (p0) / perfect state
   gate := 100-episode canary (seed 11, p0): 100/100, 0 collisions. ADR anchor + test fixture.
           The standalone L0 1000-episode gate is retired (3 policies already >=991).
L1 0.5-1.5m, +-30deg bearing, +-60deg yaw, tags visible / p0 / nominal noise 80ms
   gate := deploy floor >=90% success, collisions <=0.5% (research target >=98%, not the floor).
           Adoption guard: no regression vs incumbent 966 (candidate >=956). p0 = --no-mean-dynamics.
L2 as L1 / full 60-draw posterior + extra DR (yaw couple, deadzone, lag, gain, sag) / nominal
   INTERIM gate := 12-particle turn-consistent subset (--consistent-particles), 1000 ep seed 11:
           adopt iff success >=528 AND collisions <=216 AND L1-p0 >=956. Re-baselines L2+ history.
           Spec-L2 >=95%/<=1%/beats-EKF stays the research target; the real fix is turn-channel SysID.
L3 as L1 / as L2 / drops 25-45%, outliers, 3s blackout, tag loss under drone, 80-200ms
   gate := >=95%, <=1%, false-parked <=0.5%, beats EKF + staged baseline
L4 4x4m open, any heading, drone may start out of view, random drone yaw / as L2 / as L3
   gate := >=90%, <=1%, median <=45s, beats EKF + staged baseline
L5 full 24x12m lab + furniture (`simulation/gazebo/room_dimensions.json`) / as L2 / as L3 + scene-bridge only
   gate := >=85%, <=1%
L6 Gazebo sim-to-sim: native rollers, rendered camera, real ParkingEstimator + RelativeHeading
   gate := >=90% of L3/L4 rate over >=100 episodes
L7 real robot, operator-authorised, drone powered off
   gate := staged >=8/10 L1 starts, then >=18/20 L4 starts, 0 contacts
```

### Gate compaction (D1/D2, approved 2026-11-14)

- **p0 is the gate plant.** L0/L1 evaluate on posterior draw 0 (`--no-mean-dynamics`). The
  posterior *mean* has zero differential yaw authority (the turn column is unidentified across
  draws), so it is a broken plant and is never the gate. The mean is kept only as the L0 ADR
  anchor reference, not for scoring.
- **Deploy floor vs research target.** The written >=98% L1 target never passed; 966 deployed
  under a de-facto >=90% floor. The floor is >=90% success / <=0.5% collisions; >=98% remains the
  research target. The 34 L1 failures are timeouts with 0 collisions: safety lives in the
  collision bar, speed is the 98% path. Adoption requires candidate >=956 (no regression vs 966).
- **L0 canary.** 100 episodes, seed 11, p0, 100/100, 0 collisions replaces the standalone L0
  1000-episode gate. L0 stays the ADR anchor (`sim.py` dr_scale=0) and a test fixture; its spawns
  are not folded into L1.
- **L2 interim subset.** Only 12/60 posterior draws are turn-consistent; the same checkpoint
  scores 966 on L1 but 135 on spec-L2, so the plant distribution (not sensing) is the cause.
  The interim L2 gate scores the 12-particle consistent subset (`--consistent-particles`, label
  `L2_cons`). This invalidates prior L2+ numbers and requires a re-baseline; the subset-matched
  EKF + staged baseline must be re-measured before any "beats EKF" claim. The permanent fix is
  pure-rotation turn-channel SysID (hardware-gated), not a mean-plant simplification.

## Contracts

```text
obs student := history ~1s of [tag rel pose x,y,yaw + age + valid, imu yaw-rate + heading, prev action, smoothed tag velocity] (13-dim, noisy, delayed, valid-gated)
obs teacher := true rel pose + true dynamics draw + true obstacle geometry (privileged, train only)
act := (vx, vy, w) -> wheel duty via MECANUM_MECHANICS.md mixer -> swept_clear shield -> SAFETY.md gates
shield := watchdog + sensor-expiry brake + ownership/STOP + parking.swept_clear vs inflated gear; policy never bypasses
```

## Provenance

```text
dynamics := simulation/gazebo/mecanum_bayesian_profile.json (60 draws, schema v2) + src/ugv/mecanum_dynamics.py features/step
mixer    := hardware PWM polarity retained by fit: translation channels negated, turn channel as-is;
            verified per-channel (+vx->+1.27, +vy->+0.63, +w->+2.20 m/s,rad/s steady-state, posterior mean)
imu      := simulation/gazebo/imu_noise_profile.json (BMI270 rest, body_to_sensor, bias)
camera   := simulation/gazebo/camera_profile.json (1280x720, 67x41deg nominal, L8 30Hz) + ArUco DICT_4X4_50
geom     := src/ugv/parking.py gear_obstacles/footprint_corners/swept_clear + room_dimensions.json
baseline := EKF + staged (src/ugv/staged_baseline.py, NumPy; bench rl/baselines.py, same 1000 eps/seed 11).
            Same inputs as the student (tag, gyro, own commands; known tag latency), no privileged state.
            "beats" = higher success at no higher collision rate. Current sim (runs/baseline_ekf_staged_L*.json):
            L0 975, L1 896, L2 448, L3 215, L4 148 /1000. The same class drives the Gazebo L6 harness.
oracle   := rl/evaluate.py `oracle_pd` (PD crab on TRUE pose): diagnostic only, not a gate.
            L0 998, L1 391, L2 161, L3 160, L4 53 /1000.
deploy   := NumPy student 0.90ms/step desktop CPU (Pi pending), sha-pinned .npz; GRU width read from weights
```

## Known gaps

```text
tags      := real 9/12/13/14 vs sim drone 0/1/9/12; add 13/14 to sim before L6
clearance := config/parking.json gear clearance unverified; thresholds above provisional
intrinsic := fx=fy=968 nominal, no checkerboard calibration
heldout   := no complete heldout full-power (4095) trajectory; MECANUM_REAL2SIM.md
rule      := sim success != real success; L7 needs measured clearance + SAFETY flags
turn      := posterior turn column unidentified (yaw per +0.2 turn spans -1.1..+1.0 rad/s across draws;
             fit used translation trials only); ~17% of L2+ draws cannot yaw both ways. Needs pure-rotation trials
L0/L1 dyn := rl/sim.py gate plant is posterior draw 0 (p0, --no-mean-dynamics), pinned by the
             D1 compaction. The posterior mean has zero differential yaw and is never the gate.
```

## Compute

```text
train := RTX 3090 CUDA, rl/ subproject (torch), 4096-8192 vec envs, teacher 3x256 (never ships)
deploy:= Pi 4B CPU, NumPy .npz student (default GRU-64 + 2x64 MLP ~23k params; --hidden/--head-w configurable), no torch/onnxruntime; .onnx optional fallback
```

## Run

Status and work queue: `rl/PLAN.md`. Agent rules: `rl/AGENTS.md`. Checkpoints and logs go to `rl/runs/` (gitignored). `runs/teacher_L0_gate.pt` is the L0-gate teacher (1000/1000, 0 collisions); `runs/student_L0_ppo_v2_s0.pt` the best L0 student (991/1000, 6 collisions). One seed per experiment; on failure retry one more seed before dropping the idea. The 1000-episode eval is the judge.

```text
train teacher := cd rl && uv run python train_teacher.py --level L0 --envs 512 --iters 120 --seed <s>   # gate-teacher recipe (defaults: rate 0, gamma 0.99, lam 0.95; it used reward yaw weight 0.3, now 1.0)
student ppo   := cd rl && uv run python train_student.py --level L0 --iters 300 --seed <s>             # best-student recipe; --init <ckpt> to continue
eval          := cd rl && uv run python evaluate.py --level L0 --episodes 1000 --policy runs/<ckpt>.pt --seed 11 --no-mean-dynamics  # p0 gate plant; or oracle_pd|brake|random
baseline      := cd rl && uv run python baselines.py --level L0 --episodes 1000 --seed 11 --json runs/baseline_ekf_staged_L0.json
L6 baseline   := uv run python simulation/gazebo/run_policy_episodes.py --baseline --latency <s>  # fake bridge until the Gazebo bridge exists
export        := cd rl && uv run python export.py --ckpt runs/<student>.pt --out runs/policy_L0.npz
pi bench      := uv run python -m ugv.policy --weights <policy.npz> --benchmark  # no motors
L6 plumbing   := uv run python simulation/gazebo/run_policy_episodes.py --weights <policy.npz>  # fake bridge; live Gazebo bridge not built
visualize     := cd rl && uv run python visualize.py --policy runs/<ckpt>.pt --level L0 --episodes 6 --seed 11
rl tests      := cd rl && uv run pytest test_rl.py
```
