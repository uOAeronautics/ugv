# AGENT BOARD

Protocol: one row per spawn/result/verdict/gate.
Format: `[TYPE] id | scope | status | evidence`.
`[SCOUT]` = task, `[EXP]` = measured eval, `[ADVERSARY]` = review,
`[GATE]` = gate decision. Full numbers live in `rl/runs/archive.jsonl`.

## State (2026-11-13, measured; eval = evaluate.py 1000 eps seed 11)

- StagedBaseline checkpoint `27c2b5c` (8 tests passed): RLS ID, Mahalanobis
  EKF, shield/retreat, L4 search. Baseline L0 975/1000 (5 coll).
- RL pipeline fixes in-progress (uncommitted tree: sim/reward/models,
  trainers, export, policy fixture; see `git status`).
- Gazebo lockstep in-progress (`simulation/gazebo/drone_model.py`,
  `run_policy_episodes.py` uncommitted; L6 = fake-bridge plumbing only).

## Pareto (eval JSONs, seed 11, n=1000 unless noted)

| policy | L | succ | coll | med s | pos p95 m | yaw p95 deg |
|---|---|---|---|---|---|---|
| teacher r0_fixed_s1 | L0 | 1000 | 0 | 3.5 | 0.001 | 4.7 |
| student lat_s1 GRU-64 | L0 | 1000 | 0 | 6.0 | 0.010 | 0.3 |
| ekf_staged | L0 | 975 | 5 | 8.3 | 0.025 | 4.5 |
| oracle_pd | L0 | 998 | 2 | 10.4 | 0.024 | 0.2 |
| ekf_staged | L1 | 896 | 30 | 12.3 | 0.327 | 8.2 |
| oracle_pd | L1 | 391 | 596 | 4.9 | 0.339 | 2.7 |
| ekf_staged | L2 | 448 | 128 | 14.1 | 1.711 | - |
| oracle_pd | L2 | 161 | - | - | - | - |
| ekf_staged | L3 | 215 | 307 | 13.0 | 1.854 | - |
| oracle_pd | L3 | 160 | - | - | - | - |
| ekf_staged | L4 | 148 | 227 | 40.3 | 4.444 | - |
| oracle_pd | L4 | 53 | 400 | 7.8 | 5.453 | 179.9 |

## [SCOUT]

| id | scope | status | evidence |
|---|---|---|---|
| [SCOUT] dashboard-selector | IMPL/dashboard-selector | done | web.py controller selector (manual/staged_ekf/rl_student, armed+IMU+blocker gating, brake+manual fallback, snapshot controller/stage/margin/ms/tick) + tests/test_dashboard_selector.py; py_compile OK; tests/ 115 passed |
| [SCOUT] ui-obs-01 | training-observatory | done | tools/training_ui.py summary output |
| [SCOUT] dash-audit-01 | pi-dashboard/deploy-readiness | done | web.py audit + docs/L7_PREFLIGHT.md; py_compile OK |
| [SCOUT] baseline-mean-flag | baselines --no-mean-dynamics (particle-0 plant) | done | rl/baselines.py:27,81-86; rl/test_rl.py flag-parsing test |
| [SCOUT] baseline-enter-gating | staged_baseline lateral ENTER margin gating + test | done | src/ugv/staged_baseline.py:611-619 (ENTER→APPROACH+retreat when \|x_est\|>=margin); tests/test_staged_baseline.py margin-gating test + centered-hold control; .venv/bin/pytest tests/test_staged_baseline.py -q → 10 passed |
| [SCOUT] reward-yaw-flag | reward yaw_w param + train_teacher --yaw-weight (default 1.0) | done | rl/reward.py:14,24; rl/train_teacher.py --yaw-weight; sim.py pass-through pending (EnvConfig.yaw_w + step_reward call) |
| [SCOUT] sim-yaw-wiring | EnvConfig.yaw_w (default 1.0) passed to step_reward | done | rl/sim.py EnvConfig.yaw_w + step_reward call; rl/test_rl.py yaw test; pytest test_rl.py -q |

## [EXP]

| id | result | status | evidence |
|---|---|---|---|
| [EXP] teacher_L0_refixed | L0 848/1000 succ, 152 coll, med 4.7s (GATE FAIL) | measured | rl/runs/teacher_L0_refixed.json, archive.jsonl parent teacher_L0_gate |
| [EXP] baseline_ekf_staged_L0_refixed | L0 533/1000 succ, 6 coll, 461 timeout, med 13.3s (mean-dyn default; no --no-mean-dynamics flag in baselines.py) | measured | rl/runs/baseline_ekf_staged_L0_refixed.json, archive.jsonl parent baseline_ekf_staged_L0 |
| [EXP] teacher_L0_yaw03 | L0 989/1000 succ, 10 coll, 1 timeout, med 4.4s, pos_p95 1mm, yaw_p95 0.3deg (yaw_w 0.3, p0, seed 0) | measured | rl/runs/teacher_L0_yaw03.json, archive.jsonl parent teacher_L0_refixed |
| [EXP] baseline_ekf_staged_L0_gated | L0 816/1000 succ, 158 coll, 26 timeout, med 9.8s, pos_p95 228mm (p0 + ENTER margin gating) | measured | rl/runs/baseline_ekf_staged_L0_gated.json, archive.jsonl parent baseline_ekf_staged_L0_refixed |
| [EXP] teacher_L0_yaw03_s1 | L0 103/1000 succ, 897 coll, 0 timeout, med 4.3s, pos_p95 79mm, yaw_p95 28.8deg (yaw_w 0.3, p0, seed 1; 2nd seed -> GATE FAIL, idea dead) | measured | rl/runs/teacher_L0_yaw03_s1.json, archive.jsonl parent teacher_L0_yaw03 |

## [ADVERSARY]

| id | verdict | citations |
|---|---|---|
| [ADVERSARY] review-d8650b2 | CONDITIONAL FAIL (gate-blocker: resim collapse; 8 pass / 3 conditional / 1 fail — see body) | rl/sim.py:336,349-358,371-385,399-416; src/ugv/policy.py:11-14,26,134-151; rl/runs/archive.jsonl:13; rl/PLAN.md:29 |
| [ADVERSARY] l0-refixed-failures | TEACHER: sim innocent (gate ckpt 200/200 on fixed sim p0) → retrain regression; mouth-clip x+0.05..0.11/y0.26/6deg/0.28m/s, pos_p95 270mm. BASELINE: refixed ran on mean-dyn (plant mismatch vs 975 p0 ref); matched-p0 rerun still regressed (160/200, coll x≈0.048=yaw-trim fail); 30Hz ruled out (L0 cam 50Hz). NEXT: teacher retrain yaw_w 0.3; baseline fix lateral ENTER gating then 1000ep --no-mean-dynamics | rl/sim.py:24; rl/reward.py:20; rl/evaluate.py:142; src/ugv/staged_baseline.py:609-645; probes 200/200,177/200,9/200,160/200 |

## [GATE]

| gate | decision | evidence |
|---|---|---|
| teacher L0 | PASS (prior) | runs/teacher_L0_gate.pt, archive.jsonl |
| student L0 | PASS (prior) | runs/student_L0_lat_s1.pt, 07f607b |
| baseline L0-L4 | MEASURED (prior) | runs/baseline_ekf_staged_L*.json, 27c2b5c |

## [EXP] gazebo-l6-smoke (2026-11-13, scout ses_e2a0bde0)
| id | result | status | evidence |
|---|---|---|---|
| [EXP] gz-L3-baseline-5ep | L3 5/5 succ, 0 coll, med 8.16s, wall 54.2s | pass | /tmp/opencode/gz_smoke/baseline_gazebo_L3.json |
| [EXP] gz-L3-policy-baseline-3ep | L3 3/3 succ | pass | run_policy_episodes.py --bridge gazebo --baseline |
| [EXP] gz-L3-policy-student-3ep | L3 2/3 succ, 0 coll, 1 timeout (after yaw-sign fix) | conditional | cmd_phys=(-act[1],act[0],-act[2]) |
| [ADVERSARY] teacher-10-coll | STOCHASTIC (seed-fixable): 14/15 mouth-clip same +50mm mode (cx+0.047-0.055/yaw14-21deg or cx-0.06-0.21), 1 runaway timeout; clip flips with particle/batching; yaw_w0.3>>1.0 (152->10). NEXT: seed-1 retry running; if FAIL then lat_w 2.0->4.0 only (keep yaw_w 0.3) | rl/reward.py:14-20; rl/sim.py:44-46,409-413; repro 985/14/1 vs JSON 989/10/1 |
