# RL parking: follow-up plan

Spec and gates: `docs/RL_PARKING.md`. This file is the work queue. Execute phases in order;
a phase is done only when its acceptance checks pass. Update the "State" section when a
measured result changes.

## State (2026-11-03, measured; eval = evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics)

Current truth only. Replace a line when its result changes; do not append history.
Superseded experiment notes live in `runs/*.log`.

```text
teacher L0   ✓ GATE PASS  runs/teacher_L0_gate.pt  1000/1000, 0 coll, med 3.5s, p95 1mm/4.7deg
             (Phase-0 reward with yaw weight 0.3 [now 1.0], rate_w 0, gamma 0.99/lam 0.95, 512 envs, 120 iters)
             teacher alone on harder starts (256 eps): L1 103/256, L4 16/256 -> needs L1+ training
student L0   ✓ GATE PASS  runs/student_L0_lat_s1.pt  1000/1000, 0 coll, med 6.0s,
             p95 10mm/0.3deg. GRU-64, from scratch, 300 iters, lat_w 2.0.
             Mechanism confirmed: pricing lateral error 3x fixed the tolerated
             +x coupling drift (928 -> 1000 same seed 1). Slower than teacher.
             runs/student_L0_H.pt (GRU-128, gamma 0.997/lam 0.99, steps 256): 976/1000,
             24 coll, med 6.7s. "Capacity was the wall" is contradicted: GRU-64 got 991.
baseline     ✓ EKF + staged (src/ugv/staged_baseline.py; rl/baselines.py), 1000 eps seed 11:
             L0 975 (5 coll), L1 896 (30), L2 448 (128), L3 215 (307), L4 148 (227).
             runs/baseline_ekf_staged_L*.json. RL gates L2-L4 must beat it.
             L2+ limits: turn channel unidentified in the posterior (see RL_PARKING.md gaps).
             L3+: blind last 0.30 m (under_drone_loss) -> 7-20 cm dead-reckoning stop error.
oracle_pd    ~ diagnostic (true pose): L0 998, L1 391, L2 161, L3 160, L4 53. runs/oracle_pd_L*.json
DAgger       × retired (rl/distill.py removed); direct student PPO works. Old 0/1000 runs predate the sim fix.
deploy       ~ ugv.policy NumPy forward 0.90 ms/step desktop; parity 1.4e-7. Pi not measured.
L6 Gazebo    × fake-bridge plumbing only; live bridge not built
tests        ✓ root 131, rl 13
```

Fixes that made the student work (keep them):

```text
sim      tag min-range (dist > 0.10) blinded the 1 s hold at every level; now part of
         under_drone_loss (L0-L2 visible < 0.30 m, L3+ lost). Test: test_under_drone_loss_only_from_l3
student  envs persist across iterations (per-iter reset made rollouts 2.56 s < approach+hold);
         learned logstd (fixed 0.3 made the hold unreachable); rate_w default 0.0
phase 0  all items 0.1-0.8 done (GAE, idx-scoped reset, seeding, rate term, success-only
         bonus, frame spec). Entropy: constant 0.001 (gate teacher); --ent-anneal never helped.
```

## Guardrails (all phases)

- Root package stays torch-free. Torch lives only in `rl/` (its own venv).
- Checkpoints and logs go to `rl/runs/` (gitignored). Never leave results only in `/tmp`.
- Every reported number comes from `evaluate.py` with a fixed `--seed` and episode count
  stated. Compare policies on the same seeds.
- One seed per experiment. Pass the gate -> keep it and move on. Fail -> one more seed
  before concluding the idea does not work (seed variance spans 0-256/256 on identical code).
- One GPU job at a time (student PPO at 1024 envs uses ~10 GB). Do not edit a file while a
  job that imports it is starting.
- A change to `sim.py` or `reward.py` invalidates earlier results. Re-measure the
  teacher before comparing students.
- Gate compaction (D1/D2, approved 2026-11-14; see `docs/RL_PARKING.md`): the gate plant is
  p0 (`--no-mean-dynamics`); L0 is a 100-episode canary; L1 deploy floor is >=90% / <=0.5%
  collisions with adoption guard candidate >=956 vs incumbent 966; L2 is scored on the
  12-particle turn-consistent subset (`--consistent-particles`, `L2_cons`), adopt iff
  success >=528 AND collisions <=216 AND L1-p0 >=956. Prior L2+ numbers are invalidated and
  must be re-baselined; the subset-matched EKF + staged baseline must be re-measured before
  any "beats EKF" claim.
- No real-robot motion (L7) without explicit operator authorisation, measured gear
  clearance in `config/parking.json`, and the SAFETY.md flags.

## Phase 0: correctness bugs ✓ done

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

Phase 0 exit: rl tests pass; teacher L0 retrained; L0 canary (100 episodes, seed 11, p0,
100/100, 0 collisions) reported pass or fail with numbers. If the teacher fails the gate, tune
reward or collision margin until it passes. A student cannot exceed its teacher's gate.

## Phase 1: make L0 match its spec, then get a working student (1.1 ✓, 1.2 skipped, 1.3 in progress, 1.4 open, 1.5 ✓)

1.1 Make sensor degradations per-level parameters in `LEVELS`: `under_drone_loss`
    (False for L0-L2, True from L3), `fov` limit, and whether tag noise is on. L0 = perfect
    state per spec: tag always valid, no noise, no latency.

1.2 Skipped/retired: DAgger (`rl/distill.py` removed); direct student PPO works (see 1.3).

1.3 Student PPO at L0 (`train_student.py`). Learned `logstd` ✓. Best 991/1000, 6 coll.
    Next: inspect the collision episodes (`visualize.py`), one targeted change, also aim
    for time near the teacher's 3.5 s. Accept: L0 gate passes with the student.

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

- Training horizon: `train_teacher.py` still resets all envs every iteration (256 steps
  = 5 s), so L1+ episodes (up to 45 s) never reach late-episode states. Port the
  persistent-env loop from `train_student.py` before L1 teacher training.
- Curriculum: warm-start each level from the previous level's checkpoint.
- L2-L4 must also beat the EKF + staged baseline (`rl/baselines.py`, numbers in State).
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
