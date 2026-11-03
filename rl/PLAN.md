# RL parking: follow-up plan

Spec and gates: `docs/RL_PARKING.md`. This file is the work queue. Execute phases in order;
a phase is done only when its acceptance checks pass. Update the "State" section when a
measured result changes.

## State (2026-11-03, measured)

```text
teacher L0    := RETRAINED post-Phase-0 (all fixes): 0/1000 success, 0 coll, 1000 timeout,
                 seed 11, runs/teacher_L0.pt/json. Policy stalls ~20cm out (pos p95 215mm).
                 --ent-anneal retrain: 0/1000, 3 coll, 997 timeout (pos p95 286mm),
                 runs/teacher_L0_anneal.pt/json. Exploration is not the blocker; the
                 3cm still-pay box gives no entry gradient (mouth = risk, no gain).
                 Fix: graded 25cm slow-zone pay (reward.py). Retrain running
horizon result    := 911/1000 success (lb 0.892), 89 coll, 0 timeout, med 3.2s,
                 seed 11, runs/teacher_L0_horizon.pt/json. Horizon theory confirmed:
                 collision phase iters ~30-90, collapse by ~120.
horizon2        := 881/1000 (lb 0.859), 119 coll. More training hurt: tail is
                 structural, not a refinement deficit.
creep result      := 933/1000 (lb 0.916), 67 coll. Better, not gate.
                 Clip speed ROSE to 0.68 m/s: rushing past the pay beats it.
                 Fix: -0.15/step rush penalty in band (unharvestable), keep pay.
                 Retrain running (--init creep, runs/teacher_L0_rush.pt).
zone pay result := 0/1000, 13 coll, 987 timeout (pos p95 39mm, yaw 2.9deg).
                 Policy enters to 4cm but fidgets (mean acts +-0.1, vel never
                 <1cm/s): 50 slow steps undiscoverable under sampling noise.
                 Fix: +2 first-slow seeding bonus. Retrain running
                 (runs/teacher_L0_seed.pt, 150 iters).
seed result     := 0/1000, 7 coll (pos p95 90mm). Training never completed a
                 hold once in 4 runs. Mean probe: spawned AT goal, full-throttle
                 escape — learned fear of goal region.
fear result     := 0/1000, 2 coll (pos p95 128mm). Antidote insufficient.
                 Structural cause found: gamma*lam = 0.9405 at 50 Hz discounts the
                 +10 bonus 190 steps out by 1e-5; effective horizon (2 s) < task
                 (4 s). v3 only worked because leak+crash-bonus smuggled goal
                 value backward. Fix: gamma 0.997, lam 0.99, steps 256 (flags,
                 no code change). Retrain running (runs/teacher_L0_horizon.pt).classical     := 0/128 L0 and L1, ~95% skid contact (runs/classical_L*.json)
student DAgger:= 0% L0, val MSE plateau ~0.07
student PPO   := train_student.py crashes before iter 0 (build_hists, Phase 0.1)
deploy        := src/ugv/policy.py NumPy forward 0.90 ms/step desktop; parity 1.4e-7 vs torch
L6            := fake-bridge plumbing only; student 0/10 there; live Gazebo bridge not built
tests         := root 130 passed (incl. simulation/gazebo), rl 4 passed
```

Key observation: at L0 the student tag is invalid on 71% of live steps of a teacher rollout.
`sim.py` drops the tag whenever `dist < 0.30` (under-drone loss) at every level, so the
student crosses the entire 0.30 m slot and the 1 s hold blind. The spec defines L0 as
perfect state and lists under-drone tag loss only from L3. The earlier "POMDP clonability
floor" explanation is unverified; this sensor gating is the more likely cause.

## Guardrails (all phases)

- Root package stays torch-free. Torch lives only in `rl/` (its own venv).
- Checkpoints and logs go to `rl/runs/` (gitignored). Never leave results only in `/tmp`.
- Every reported number comes from `evaluate.py` with a fixed `--seed` and episode count
  stated. Compare policies on the same seeds.
- A change to `sim.py` or `reward.py` invalidates earlier results. Re-measure the
  teacher before comparing students.
- No real-robot motion (L7) without explicit operator authorisation, measured gear
  clearance in `config/parking.json`, and the SAFETY.md flags.

## Phase 0: correctness bugs (fix before any further training)

Each item is a confirmed code defect. Fix it, add a focused test in `rl/test_rl.py`, then
retrain the L0 teacher once at the end of the phase and record the new numbers.

0.1 `train_student.py` `build_hists`: `frames` are time-major `[T,N,13]` while `init_hist`
    is `[N,50,13]`, and `torch.cat(..., dim=1)` mixes the layouts. Histories that span a
    reset are also rebuilt from pre-reset frames. Fix: store `obs["student"]` per step
    (`[T,N,50,13]`, ~340 MB at N=1024, T=128; acceptable on the 3090) and delete
    `build_hists`. Accept: a 2-iter smoke run completes; the stored history equals
    `env.hist` at each step.

0.2 `sim.reset(idx)` resamples dynamics (`_sample_dynamics(self.n)`), latency, drop rate,
    blackout window and the latency buffer for all envs, not only `idx`. Any partial reset
    changes the physics of running episodes. Fix: index every per-env tensor by `idx`, and
    make the latency buffer per env (ring tensor `[N,K,3]`, per-env `steps_back`).
    Accept: test that `reset(idx=[0])` leaves `coef`, `latency` and `drop_p` of env 1 unchanged.

0.3 `sim.reset` reseeds its generator with `cfg.seed + seed_offset` on every call. Training
    calls `reset(idx)` with offset 0, so every reset draws the same start poses. Dynamics
    picks use the global unseeded RNG, which makes evaluation non-deterministic. Fix: one
    persistent `torch.Generator` per env object for all sampling; evaluation passes explicit
    seeds. Accept: two `evaluate.py` runs with the same seed give identical JSON.

0.4 `sim.step` sets `self.prev_act = cmd` before calling `step_reward(..., cmd, self.prev_act)`,
    so the action-rate penalty is always 0. Fix: pass the previous action. Accept: test
    with a step change in action gives a nonzero rate term.

0.5 `reward.step_reward` pays the +10 terminal bonus on `done & near`, which includes a
    collision or timeout while near the goal. Fix: pay it on `success` only (pass it in).

0.6 GAE in `train_teacher.py` and `train_student.py` carries `gae` through
    `1 - D[t+1]`. `D[t]` marks the end of transition t, so advantages leak across episode
    boundaries. Fix: use `1 - D[t]` for both bootstrap and carry. Factor one `gae()`
    helper into `models.py` (or a new `ppo.py`) used by both scripts. Accept: unit test on
    a hand-computed 4-step sequence with one terminal.

0.7 Teacher entropy annealing never ran (`ent` is constant 0.001; see `train_teacher.py`).
    Decide on anneal versus constant and measure. Do not change it silently.

0.8 Train/deploy frame mismatch: `sim.py` zeroes the tag fields when the tag is invalid;
    `ugv.policy.StudentPolicy.step` feeds the passed tag values with `valid=0`. Fix: one
    frame spec; deploy zeroes invalid tags. Accept: golden-trace test where a recorded sim
    sequence (tag, age, valid, imu, prev_act) replayed through `StudentPolicy.step`
    reproduces `env.hist` exactly (store the trace as an `.npz` fixture under `tests/fixtures`).

Phase 0 exit: rl tests pass; teacher L0 retrained; L0 gate (>=99%, 0 collisions on 1000
seeded episodes) reported pass or fail with numbers. If the teacher fails the gate, tune
reward or collision margin until it passes. A student cannot exceed its teacher's gate.

## Phase 1: make L0 match its spec, then get a working student

1.1 Make sensor degradations per-level parameters in `LEVELS`: `under_drone_loss`
    (False for L0-L2, True from L3), `fov` limit, and whether tag noise is on. L0 = perfect
    state per spec: tag always valid, no noise, no latency.

1.2 Experiment (cheap, decisive): DAgger at the new L0 with `runs/teacher_L0.pt`.
    Hypothesis: with the tag visible through the slot, the student clones the teacher.
    Report val MSE at the slot mouth versus the earlier 0.07, and success on 1000 eps.

1.3 Student PPO at L0 (`train_student.py`, optionally `--init` from 1.2). Learned
    `logstd` instead of fixed 0.3. Log success/collision per iteration to `runs/*.log`.
    Accept: L0 gate passes with the student.

1.4 Tag-rate realism before L1: the camera/estimator delivers pose at about 30 Hz while
    the policy runs at 50 Hz. Emulate the estimator rate in the sim (hold last tag, age
    grows, valid only on fresh samples) and compute tag velocity from fresh samples using
    their real time delta. `ugv.policy` must not hard-code `/0.02`.

1.5 Verify the FOV geometry. `sim.py` computes bearing as `atan2(x, y) - yaw` on the car pose
    in the drone frame. Check it against the frame conventions in `src/ugv/parking.py`
    (yaw from +z toward +x) and `camera_to_car`. A car facing the drone must see it. Add a
    test with one pose facing toward the drone and one facing away.

## Phase 2: climb the ladder in sim (L1 -> L4)

For each level: teacher (if needed), student, `evaluate.py --episodes 1000 --seed <fixed>`,
JSON into `runs/`, and a result row in the State section above. Promote only on gate pass.

- Training horizon: `train_teacher.py` resets all envs every iteration (256 steps = 5 s),
  so L1+ episodes (up to 45 s) never reach late-episode states. Switch to persistent envs
  across iterations with bootstrapped truncation.
- Curriculum: warm-start each level from the previous level's checkpoint.
- L2 must also beat the ContinuousDocking baseline (`src/ugv/continuous.py`). Add it as
  an `evaluate.py` baseline driven from the sim tag stream.
- L3 adds `false_parked`. The env currently has no "declare parked" output, so
  `evaluate.py` reports `null`. Define the declare signal (policy output or shield rule),
  then measure it.
- L4/L5 need search behavior (drone out of view). Expect to add a "drone seen" memory or
  bearing prior to the obs. Make the decision per spec, not per convenience.

## Phase 3: deploy path on the Pi

3.1 Shield horizon. `StudentPolicy.step` checks one 20 ms sweep with nominal speeds
    (0.6 m/s, 2.5 rad/s) and no body-to-drone rotation. That cannot stop in time (brake
    tau is ~100 ms). Check the sweep over the stopping horizon, rotate by yaw, and use the
    measured per-channel steady-state speeds (+1.27/+0.63/+2.20). Add tests near each skid.

3.2 Integration: drive `StudentPolicy` from the real `ParkingEstimator`/`RelativeHeading`
    outputs behind an explicit opt-in (config flag, default off), through the existing
    motor guard, watchdog, ownership and STOP paths. The policy never writes PWM directly.

3.3 Pi 4B benchmark: `python -m ugv.policy --weights <npz> --benchmark` on the Pi. Record
    ms/step p50/p99 under the dashboard load. Budget: <5 ms at 50 Hz.

3.4 Pin the deployed `.npz` by sha in config; refuse to load a mismatched sha.

## Phase 4: L6 sim-to-sim in Gazebo

- Implement the `gazebo` bridge in `simulation/gazebo/run_policy_episodes.py` using
  `gazebo_bridge.py` (pose, reset, velocity seam via `control_state.py`) and the real
  estimator on the rendered camera.
- Add tags 13/14 to the sim drone (real drone uses 9/12/13/14; sim has 0/1/9/12).
- Gate: >= 90% of the L3/L4 sim rate over >= 100 episodes.

## Phase 5: L7 real robot (operator only)

Preconditions: measured gear clearance in `config/parking.json` with confirmation flags,
checkerboard intrinsics (currently fx=fy=968 nominal), L6 gate passed, operator present,
drone powered off. Staged starts per the spec ladder. Agents prepare scripts and checklists;
they do not start motion.

## Not in scope

- Rewriting the existing classical parking stack (`parking.py`, `continuous.py`).
- New simulators or RL frameworks. The batched torch env is sufficient through L5.
