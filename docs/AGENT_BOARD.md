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
| [SCOUT] ui-obs-01 | training-observatory | done | tools/training_ui.py summary output |
| [SCOUT] dash-audit-01 | pi-dashboard/deploy-readiness | done | web.py audit + docs/L7_PREFLIGHT.md; py_compile OK |

## [EXP]

| id | result | status | evidence |
|---|---|---|---|
| _(empty)_ | | | |

## [ADVERSARY]

| id | verdict | citations |
|---|---|---|
| [ADVERSARY] review-d8650b2 | CONDITIONAL FAIL (gate-blocker: resim collapse; 8 pass / 3 conditional / 1 fail — see body) | rl/sim.py:336,349-358,371-385,399-416; src/ugv/policy.py:11-14,26,134-151; rl/runs/archive.jsonl:13; rl/PLAN.md:29 |

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
