# Decision and experiment history (mined for explore/)

Scope: the superseded agent board (a copy is in `.steward/agents/A/notes`),
docs/archive/**, rl/PLAN.md,
docs/RL_PARKING.md, docs/L7_FIELD_NOTES.md, docs/PIPELINE.md, rl/runs/archive.jsonl,
and git log (105 commits, HEAD 556ec80). Every number is from a named source. Where
sources disagree, both values are given. Nothing was re-run. Run ids are copied as
written in the sources; the frame is the policy frame in CLAUDE.md. Tags 14/6/3 are the
current layout; 9/12/13/14 is the old one.

Plant key used below: **p0** = posterior draw 0 (`--no-mean-dynamics`), the gate plant
since D1/D2 (2026-11-14). **mean** = posterior mean (`use_mean_dynamics=true`), which
has zero differential yaw authority and is not a valid gate.

## 1. Established facts

- **Student L0 passes on the fixed sim.** `student_L0_lat_s1_fixedsim`: 1000/1000,
  0 coll, med 6.0 s, p0, seed 11 (rl/runs/student_L0_lat_s1_fixedsim.json).
- **Student L1 deploy floor is met on p0 by one run.** `student_L1_latw4_ft`: 966/1000,
  0 coll, 34 timeouts (rl/runs/student_L1_latw4_ft.json). It is the incumbent for the
  adoption guard (966). Its NumPy export is the sha-pinned `student_L1_best.npz`.
- **The lateral-clip fix is the L1 breakthrough.** Only lat_w changed between
  `student_L1_unlocked` (585/1000, 393 coll) and `student_L1_latw5_s0` (954/1000,
  0 coll) (rl/runs/student_L1_latw5_s0.json). Source: board session summary 2026-11-13.
- **Mean dynamics is not a usable plant.** Its turn channel is command-symmetric
  (+0.3 and -0.3 both give -0.0242 rad/s), so it has zero differential yaw. Source:
  board [ADVERSARY] baseline-failure-audit; RL_PARKING.md "Gate compaction" D1.
- **L1 EKF-staged baseline is ~986-989/1000 (working band, 2026-11-17).** cuda record
  989/1000, 9 coll (rl/runs/baseline_ekf_v2_L1.json); CPU repeat on the current tree
  986/1000, 13 coll (`rl/runs/baseline_ekf_v2_L1_cpu_repro_20261117.json`). The board's
  CPU 959/19 and the 973/23 quote have no primary artifact and were not reproduced (+27
  from the CPU repeat). Do not cite them. See §4.
- **Pi inference is fixed.** Stateful+shield p50 0.136 ms host (was 11.869 ms); Pi
  projection 0.47-0.54 ms against a 5 ms budget. Stale-200ms brake verified at yaw
  0/30/60. Sources: board [SCOUT] fast-shield-and-latency; rl/runs/pi_inference_bench.json.
  Export parity 8.17e-08 (<1e-5).
- **Real plant differs from sim plant.** Measured on the car
  (`calibration/runs/l7/20261113-181040-probe-axes/fit.json`): forward 2.3 m/s per duty,
  deadzone 0.05; strafe ~1.6 m/s, deadzone ~0.13; yaw 15.7 rad/s per duty, deadzone
  0.10. Real gyro std 0.63 rad/s vs 0.002 in sim
  (docs/L7_FIELD_NOTES.md). The fitted plant (`L1_fitted`: measured dz 0.05/0.13/0.10,
  gyro 0.63) reproduces the car's failure: student 0/1000, 83.6% raw turn saturation
  (§4; H-0001 confirmed). The fit is NOT a strong discriminator (fwd/yaw residuals
  identical fitted vs nominal); the strafe values are nominal (`fit.json` degenerate, n=3).
- **Classical go-to works on the car.** P controller in `stage` mode reached the 4 cm /
  5 deg target on 4/4 runs in 6.5-10.8 s (docs/L7_FIELD_NOTES.md; the four
  `calibration/runs/l7/20261113-18*-stage-*/summary.json` files). Does NOT transfer to
  the sim: classical go-to + dead-reckoned finale scored 552/1000, 375 coll on L1 p0
  (§4; H-0002 refuted).
- **Sim-to-real gap is the plant, not sensing (one paired diagnostic).**
  `gazebo-paired-seed`: fake bridge 0/10 vs Gazebo 3/10 for `student_L1_best.npz`
  (rl/runs/gazebo_l6/diag_paired_analysis.json). Caveat: FakeBridge ignored `--seed`
  at that time, so arms were not spawn-paired. H-0007 (spawn-paired re-run) open; the
  cross-bridge spawn streams are not spawn-identical by construction. Conclusion unverified.
- **Observation contract passes the gate (2026-11-17, H-0009).** With the current contract
  tool the joint whitened-KDE overlap (channels inter_tag_dt_s, tag_age_s, imu_w_rad_s;
  min 0.35) is 0.742 at constant 20 ms and 0.5146 with the sim lag drawn from the run's Pi
  loop_ms distribution (0.486-0.525 over seeds 11-15); marginals pass. The recorded 0.0
  survives only as a 4-channel diagnostic that adds loop_ms_s, which is not the gate. Loop
  lag lowers, not lifts, the overlap; Pi-lag is not the contract failure. See §4.
- **Action-sensitivity finding.** At L_real the policy's mean |action| is 0.55 at the
  mouth yet progress is ~0 (+8 mm/s), so the action is misdirected, not frozen. The
  sensitivity entry -0.2533 is a contrast (mean[progress(+axis) - progress(-axis)]),
  not absolute progress: absolute -dy progress is +0.08 m (mouth) / +0.13 m (blind), with
  mouth collision rate 0.75 (rl/runs/sensitivity_L1ft_Lreal.json). -dy points toward the
  drone; failure is by clearance, not by any frame or sign error (§4; H-0004 refuted).
- **Gate status.** RL_PARKING.md: the standalone L0 1000-episode gate is retired for a
  100-episode canary (seed 11, p0, 100/100, 0 coll). L1 deploy floor >=90% success and
  <=0.5% collisions; research target >=98% remains open.

## 2. Dead ends

Each entry: what was tried, evidence, why it failed, and the condition for a legitimate
retry. Superseded attempts are marked; they are not dead.

### 2.1 Teacher track (L0)

- **Teacher lat_w 4.0, both seeds.** Runs `teacher_L0_latw4_s0` (131/1000, 869 coll) and
  `teacher_L0_latw4_s1` (125/1000, 875 coll), rl/runs/teacher_L0_latw4_s0.json and
  _s1.json. Both GATE FAIL. Failure mode is the slot-mouth clip. Why: the board records
  the idea as dead after two seeds. Retry only with a different mechanism, not lat_w alone.
- **Teacher yaw_w 0.3 on seed 1.** Run `teacher_L0_yaw03_s1` (103/1000, 897 coll),
  rl/runs/teacher_L0_yaw03_s1.json. Seed 0 (`teacher_L0_yaw03`) gives 989/1000 on the
  same code. Why: seed bifurcation between freeze and clip modes; unresolved. Retry needs a
  change that passes on two seeds.
- **Reset-every-iter as the cause.** Run `teacher_L0_reset_s1` (55/1000, 945 coll),
  rl/runs/teacher_L0_reset_s1.json. Hypothesis (persistent rollouts cause the bifurcation)
  rejected; the plant is the cause. Retry: not as a fix for the teacher.
- **Teacher gate on the old sim.** `teacher_L0_gate` shows 1000/1000 in
  rl/runs/archive.jsonl, but that was on the pre-fix sim. On the fixed sim the same recipe
  gave `teacher_L0_refixed` 848/1000 (152 coll). The "teacher 1000/1000" figure must not
  be cited as current.
- **Entropy annealing.** `--ent-anneal` never helped (rl/PLAN.md Phase 0.7). Retry only
  as a single-change experiment with two seeds.
- **Capacity as the wall.** `student_L0_H` (GRU-128, gamma 0.997, lam 0.99): 976/1000,
  24 coll, versus GRU-64 at 991 (rl/PLAN.md State). Contradicted; a larger GRU is not a fix.

### 2.2 Student track (L1 and up)

- **Warm-start from L0 gate ckpt, no curriculum.** `student_L1_s0`: 0/1000, 866 coll,
  rl/runs/student_L1_s0.json. Board: iter 0 already dead; L0->L1 spawn, yaw, 30 Hz,
  80 ms and noise all hit at once. Retry: only with a curriculum (below).
- **Curriculum 100 alone.** `student_L1_curric`: 0/1000, 761 coll,
  rl/runs/student_L1_curric.json. Retry: not alone; the successful run combined it with
  lat_w 5.
- **Aux JEPA/RMA head, warm start, curric 100.** `student_L1_jepa_s0`: 0/1000, 56 coll,
  944 timeouts, rl/runs/student_L1_jepa_s0.json. Collisions fell from 866 to 56 but
  success stayed 0. Retry: the aux head is kept (0 ms deploy cost); retry only with a
  success-side change.
- **Inherited std 0.105 plus critic warm-up.** `student_L1_unlocked_v1`: 0/1000, 743 coll,
  rl/runs/student_L1_unlocked_v1.json. The full-L1 phase destroyed the actor. Retry: do not
  inherit the L0 std on a warm start.
- **From-scratch seed 1 freezes.** `student_L1_unlocked_s1`: 0/1000, 0 coll, 1000
  timeouts, rl/runs/student_L1_unlocked_s1.json. Opposite bifurcation to seed 0. Retry
  needs settle/precision reward work first (board open thread).
- **Direct L2 warm start from best L1.** `student_L2_s0`: 187/1000, 338 coll, 475 timeouts,
  rl/runs/student_L2_s0.json. A dynamics shock is not overcome in 250 iterations on the
  full 60-particle posterior. Retry: only through the curriculum/ADR route, not a direct
  warm start.
- **L2 on the full posterior.** `student_L2_cons_adr_s0`: 464/1000, 325 coll
  (rl/runs/student_L2_cons_adr_s0_L2_1000ep_s11.json was the full 60-draw result, 145/1000).
  Superseded by v2 below, not dead.
- **Forward-model loss and L_field.** Three runs, all 0/200 on L_real p0, seed 11 (all
  timeouts, 0 coll): `student_field_base_s0_eval200`, `student_Lreal_C_fwd_s0_eval200`
  (rl/runs/archive.jsonl, eval_metrics), and `student_fieldB_base_s0` (board only; 0
  successes even at dr_scale 0). Why: a 300-iteration warm start does not cross the real
  deadzone/yaw-coupling barrier. Retry condition (stated by the board): first reproduce a
  nonzero L_real baseline, then A/B `--fwd-weight` against it. The forward model is
  neither supported nor refuted.
- **L_real evaluated on mean dynamics.** `student_Lreal_adr_v4_s0`: 318/1000 (494 coll),
  rl/runs/student_Lreal_adr_v4_s0_Lreal_dr025_1000ep_s11.json, use_mean_dynamics=true.
  Its p0 score is 0/200. Retry: do not cite the 318 as a gate result.
- **L_real at full dr_scale 1.0.** Board [EXP] student_Lreal_adr_v4_s0 row: 0/1000, 45 coll,
  955 timeouts (the source gives no run id for this result). The 0.40 m blind approach
  zone where `valid=0` zeros the tag is the open cause.

### 2.3 Baseline (EKF + staged)

- **Mean-dynamics baseline.** `baseline_ekf_staged_L0_refixed`: 533/1000, 461 timeouts,
  rl/runs/archive.jsonl. Plant mismatch. Retry: not on mean dynamics (see fact above).
- **Baseline L3 by tuning.** `baseline_ekf_v2_L3`: 257/1000, 351 coll,
  rl/runs/baseline_ekf_v2_L3.json. Cap is the under-drone blind finale (dist < 0.3 m tags
  lost). Board: needs estimator/model-ID work, not more tuning. Retry: only with that
  work.

### 2.4 Gazebo and transfer

- **Gazebo L6, latw5_s0 at L1.** `student_L1_latw5_s0_gz_L1_emu`: 0/20, 20 timeouts,
  rl/runs/gazebo_l6/student_L1_latw5_s0_L1_20ep_s11.json (npz sha fe44ba93cc739f6a).
  L2 0/20 (`..._L2_20ep_s11.json`). Vision-in-the-loop with photometric DR: 0/10, decode
  5.2% (rl/runs/gazebo_l6/student_L1_latw5_s0_L1_camdr_10ep_s11.json). Neither decode
  figure is usable (§4; H-0006 refuted). Retry: after the fake-bridge plant is aligned
  to `real_plant` and the paired diagnostic is re-run with spawn-paired seeds.
- **Gazebo L2 consensus candidate.** `student_L2_cons_adr_s0_gz_L1`: 0/3 Gazebo; 0/20
  fake camera_dr (2 coll) (board [EXP] row). Retry: same as above.

### 2.5 Real car

- **Student L1 policy on car.** `student_L1_best.npz`: 0/3 parked. Runs
  `20261113-182150-policy-ep1-front` (timeout, oscillation ~1.3 m),
  `20261113-182645-policy-ep2-mean` (lost tag after strafe-induced yaw),
  `20261113-183857-policy-ep3-p0` (46 s oscillation, 0.57->1.10 m). Turn saturation |a2|>0.8
  on 74% of ep3 ticks; 0% when gyro is zeroed in replay (docs/L7_FIELD_NOTES.md).
  Retry condition: sim gyro noise and identified real plant in the model, the joint
  contract passing, and explicit operator authorisation (RL_PARKING gate, PLAN Phase 5).
- **Shield vs 200 ms latency (resolved, not a dead end).** Horizon 0.18 s < latency
  0.20 s, fires never (board [ADVERSARY] pi-inference-and-shield); fixed by STOP_HORIZON
  0.35 and dt_stale-aware sweep (board [SCOUT] fast-shield-and-latency).
- **2026-09-27 floor arc, right-hand trial.** Aborted by a trial-only accel check; no
  retry after a safety abort (docs/archive/STATE_HISTORY.md, "2026-09-27 short floor arc
  tuning"). Retry only after the operator reports the right-hand motion.
- **Pre-RL motor probes.** Recorded in docs/archive/STATE_HISTORY.md (2026-10-13
  motor trials suspended on wired supply; Pi unreachable after LiPo depletion). No motor
  probe was issued after the outage. Retry: operator instruction.

### 2.6 Retired

- **DAgger (`rl/distill.py`).** Retired; direct student PPO works (rl/PLAN.md "1.2
  skipped/retired"; git history keeps the file). Retry: not stated; treat as closed.
- **Oracle hot entry.** `oracle_pd_L1`: 391/1000, 596 coll, med 4.9 s (rl/runs/archive.jsonl).
  Fast entries overshoot into skids (docs/archive/FAILURE_REPLAY.md, C6). Modifier only; no
  separate reset bin.

## 3. Open questions

1. **Adoption of L2_cons v2.** `student_L2_cons_adr_v2_s0` meets the L2 subset gate
   (528 succ, 216 coll, `student_L2_cons_adr_v2_s0_L2_cons_1000ep_s11.json`) but scores
   918/1000 on L1-p0 (`student_L2_cons_adr_v2_s0_L1_1000ep_s11.json`). The RL_PARKING rule
   requires L1-p0 >= 956. By that rule it is not adopted. Decision needed.
2. **Sweep-reflex gate plant.** `rl/runs/gate_sweep_L2_s11.json` records
   `use_mean_dynamics: true` (135 succ). The board's [GATE] sweep-reflex row claims no
   regression on 135->143. Confirm the sweep was measured on p0 before relying on it.
3. **Which L1 baseline number is true.** Resolved 2026-11-17: working band 986-989/1000
   (cuda 989/9; the CPU repeat measured 986/13, `rl/runs/baseline_ekf_v2_L1_cpu_repro_20261117.json`).
   The 959/19 and 973/23 figures have no artifact and may not be used. See §4.
4. **Seed bifurcation.** Freeze (0 coll, all timeouts) vs clip (many coll) under identical
   code on seeds 0 and 1 for teacher and student. Unresolved in every track.
5. **Gazebo decode 5.2% vs 98.3%.** Resolved 2026-11-17: not a lighting-DR flag-wiring bug.
   The 5.2% batch does not reproduce at HEAD (45.6-62.2% in matched configs); the 98.3%
   probe has no artifact and cannot be checked; runs are not bitwise reproducible at fixed
   seed. See §4.
6. **Paired-seed Gazebo gap.** 0/10 fake vs 3/10 Gazebo was measured before FakeBridge
   honoured `--seed`. Re-run required before the plant-not-sensing conclusion is final.
   H-0007 (spawn-paired re-run) was in flight at the time of writing; the two bridges
   draw spawns from different distributions, so arms are seed-paired, not spawn-identical.
7. **Real-car gyro and loop lag.** Loop lag: measured; not the contract failure
   (H-0009 refuted; lag LOWERS overlap). Gyro std 0.63: in the fitted plant and
   reproduces the saturation failure (H-0001 confirmed). Remaining: onset delays and
   camera rate unfitted; strafe fit degenerate.
8. **Belly vision.** Parked behind L2 >= 80%; the old experiment is in
   docs/archive/council_2026-11-14/. Not started.
9. **GPU run with `--blind-mouth-frac`.** Implemented in `7757879`; the board says it has
   not been run.
10. **Failure-replay bins R1-R5.** Defined in docs/archive/FAILURE_REPLAY.md. The sources
    do not say whether a mixed-bin run was trained; treat as unverified.
11. **Current PLAN State is stale.** rl/PLAN.md State is dated 2026-11-03 and still lists
    pre-fix numbers; the board is the newer source. Reconcile when PLAN is next edited.

## 4. Corrections (2026-11-17, six-closure roadmap run)

These supersede earlier entries. Ledger predictions are append-only and stay as written; this
section is the authority on where the evidence landed. All numbers trace to named artifacts.

- **Superseded: "Joint whitened-KDE overlap 0.0 (§1, observation contract)."** With the current
  contract tool (`tools/observatory/contract.py`, gate channels `inter_tag_dt_s`, `tag_age_s`,
  `imu_w_rad_s`, min overlap 0.35), the same run passes at 0.742 (20 ms) and at 0.5146 with the
  Pi lag drawn from the run's loop_ms distribution (seeds 11–15: 0.486–0.525). The 0.0 only
  reproduces as a 4-channel diagnostic with `loop_ms_s` added, which is not the gate. The lag
  LOWERS overlap; it does not lift it. Loop lag is not the contract failure.
  Source: `explore/results/H-0009-contract.json` (hypothesis H-0009, refuted). The "contract
  loop_ms fix" (`a7af35c`) likely post-dates the recorded 0.0; marginals were never in question.
- **Superseded: "L1 baseline figures unreconciled (§1; open question 3)."** CPU repeat on the
  current tree (2026-11-17): **986/1000 success, 13 collisions**, p0 seed 11, device cpu,
  `use_mean_dynamics=false` (`rl/runs/baseline_ekf_v2_L1_cpu_repro_20261117.json`). Within 10
  episodes of the cuda record (989/9, Δ−3) — consistent with the adversary's RNG-ensemble
  note. The board's CPU 959/19 was NOT reproduced (+27, no primary artifact); the 973/23 quote
  has no file. Working figure: EKF-staged L1 p0 seed 11 ≈ **986–989/1000**. Do not cite 959/973.
  Source: `explore/results/H-0008-cpu.json` (H-0008, inconclusive; the cuda half was never run).
- **Superseded: "Gazebo decode conflict 5.2% vs 98.3% (open question 5)."** The 5.2% batch does
  NOT reproduce at HEAD: matched-config re-runs decode at 45.6% (`camera_dr`), 48.2%
  (`camera --lighting-dr`), 62.2% (`--student-shield gazebo`), 30.3% (legacy 12/9/13/14 layout).
  Layout alone moves ~30 pp. The 98.3% probe has no artifact anywhere and cannot be checked;
  treat it as unverifiable, not as evidence. Runs are not bitwise reproducible at fixed seed
  (2803 vs 2616 frames, same effective config). `photometric_dr: false` in the batch JSON is a
  mislabel for `camera_dr` (the effective DR is on). Neither number may be used as a reference.
  Source: `explore/results/H-0006-decode.json` (H-0006, refuted).
- **"Action sensitivity −0.25 m for −dy (§1)" is a contrast, not progress.** Matrix entries are
  `mean[outcome(+axis) − outcome(−axis)]`; absolute −dy progress is +0.08 m (mouth) / +0.13 m
  (blind). The chain export → deployed NumPy → Pi duty path → sim is sign-consistent end to end
  (all conventions match; sim FOV bearing is cosmetically sign-inverted; a pre-existing CCW-yaw-
  in-CW-`Pose2` annotation remains at `src/ugv/policy.py:159` / `tools/l7/l7_run.py:1343` — a
  boundary-rule breach, not a numerics bug). No sign or frame mismatch exists in the chain.
  Source: `explore/results/H-0004-audit.md` (H-0004, refuted).
- **New established fact: the fitted plant reproduces the car's failure.** `L1_fitted` (L1 +
  real_plant, measured dz 0.05/0.13/0.10, gain fwd 2.3186 / yaw 15.66, gyro 0.63, behind new
  flags; L1/L_real/L3 rollouts bit-identical pre-edit): student_L1_best scores **0/1000**,
  17 collisions, 83.6% raw turn saturation (54.8% with deploy blind reflex) — matches the car's
  74.1% on ep3 (`20261113-183857-policy-ep3-p0`); nominal-plant control: 197/200, 0.14%. The
  joint overlap passed both plants (0.742 nominal / 0.618 fitted; gate 0.35). Caveats: the fit is
  NOT a strong discriminator (fwd/yaw residuals identical fitted vs nominal; strafe values are
  nominal because `fit.json` is degenerate, n=3; onsets and camera rate unfitted). Fusion of the
  measured deadzones+gains is what reproduces the failure, not any single channel.
  Source: `explore/results/H-0001-plant.json` (H-0001, confirmed). Use `L1_fitted` as the
  sim target for any fitted-plant training; do not treat the nominal posterior plant as
  representative of the car.
- **Refuted: the classical stage go-to + dead-reckoned finale transfers.** 552/1000, 375
  collisions vs RL student 966/1000; misses the 956 floor by 404; 41% finale failure rate below
  0.40 m; L1 p0 tag-loss only happens off-cone (dead-reckoning barely engaged). The field 4/4
  success does not transfer to the sim; tag geometry, latency and loss semantics differ. The
  hybrid H-0003 was therefore not built (cancelled). Source: `explore/results/H-0002-classical.json`.
- **Open questions re-filed (open question 6):** the spawn-paired re-run (H-0007) is in flight;
  cross-bridge spawn streams are NOT identical by construction (FakeBridge vs Gazebo LEVELS), so
  "spawn-paired" is approximate; 0/10 vs 3/10 conclusions from HISTORY §1 remain unusable until it
  reports. Question 4 (seed bifurcation) is untouched; question 2 (sweep-reflex gate plant still
  `use_mean_dynamics=true`) is untouched; question 9 (`--blind-mouth-frac` GPU run) is untouched.
- **Process prior:** re-measure any reference figure on the current tree before building on it
  (five of six closures this run traced to stale pre-fix numbers); record mouth-vs-epistemic
  scope separately (e.g. "field success transfers to sim" is a hypothesis, not a fact).
