# Failure-Replay Curriculum (Dream-RSI style, L1+ training)

Source: `rl/runs/*.json` (eval, seed 11 unless noted) + `rl/runs/archive.jsonl`
+ adversary probes on board. Reference: `rl/sim.py` `LEVELS` (L0 dist 0.75 /
yaw 0 / lat 0 / p0-or-mean; L1 dist 0.5-1.5 / bearing 30 / yaw 60 / lat 80 ms /
cam 30 Hz; L2 = L1 + full_posterior; L3 adds lat 0.08-0.20 / drops 0.25-0.45 /
outlier 0.02 / blackout 3 s / under_drone_loss <0.30 m; L4 dist 0.5-4.0 /
bearing 180 / yaw 180). Geometry: slot mouth clip zone x in [0.047, 0.11],
y ~ 0.26 (car half-width 0.08 vs SKID_X - GEAR_R = 0.127).

## Clusters (state at failure -> seed runs)

| # | cluster | failure state | seed runs (n_fail / N) |
|---|---|---|---|
| C1 | slot-mouth clip | pose x +0.05..+0.11 (or mirror -0.06..-0.21), y ~0.26, yaw 6-21 deg, speed ~0.28 m/s, collision at skid | teacher_L0_refixed (152/1000), teacher_L0_yaw03 (10/1000; repro probe 14/15 same mode + 1 runaway timeout) |
| C2 | yaw-trim bifurcation | yaw_p95 29-32 deg, pos_p95 80-111 mm, ~870-897/1000 collisions; seed 0 clean-ish -> seed 1 collapse under identical code (stochastic-train vs deterministic-eval mean-action mismatch on p0) | teacher_L0_yaw03_s1 (897), teacher_L0_latw4_s0 (869), teacher_L0_latw4_s1 (875) |
| C3 | plant-mismatch collapse | student ckpt trained on p0 resimmed on mean dynamics: 720 coll + 90 timeout, pos_p95 1.05 m, yaw_p95 151 deg, med 18.7 s. Baseline same cause: 461 timeout, yaw_p95 64 deg | student_L0_lat_s1_resim_mean (810/1000 fail), baseline_ekf_staged_L0_refixed (467/1000 fail) |
| C4 | stall / timeout drift | med 13-14 s, timeout 42-46%, pos_p95 0.55-1.71 m, yaw_p95 64-114 deg; slow EKF staged never reaches hold | baseline_ekf_staged_L0_refixed (461 timeout), baseline_ekf_staged_L2 (424 timeout + 128 coll) |
| C5 | sensor-degradation loss | L2+: latency 80-200 ms + drops 25-45% + 3 s blackout + under-drone blind <0.30 m + FOV loss (bearing 30-180 deg, yaw 60-180 deg, dist to 4 m). Baseline L3 307 coll/478 timeout; L4 227 coll/625 timeout, med 40.3 s; oracle (true pose!) still 40-60% coll -> dynamics, not just sensing | baseline_ekf_staged_L3, baseline_ekf_staged_L4, oracle_pd_L1 (596 coll), oracle_pd_L2/L3/L4 (~400 coll, yaw_p95 ~180 deg) |
| C6 | hot entry | oracle med 4.6-7.8 s but 40-60% collisions; fast entries overshoot into skids | oracle_pd_L1-L4, teacher_L0_rush (9 coll, med 2.8 s) |

C6 is a modifier (entry speed), not a separate reset bin: covered by spawning
C1/C2 states with nonzero initial velocity toward the slot.

## Reset distribution (per reset call, sums to 1.00)

Implement as a mixture over (level, spawn override, dynamics, latency).
`spawn_dist` / `full_posterior` overrides already exist in `EnvConfig`
(`rl/sim.py:68-80`); latency/drops/blackout stay per-level (no new flags
needed — pick the level that carries them).

| bin | ratio | level | spawn override | dynamics | sensor (from level) | covers |
|---|---|---|---|---|---|---|
| R1 mouth-clip | 0.30 | L1 | dist 0.30 (ring at y~0.26), bearing +/-10 deg, yaw +/-15 deg | p0 (`use_mean_dynamics=False`, `full_posterior=False`) | lat 80 ms, cam 30 Hz | C1 |
| R2 yaw-trim | 0.20 | L1 | dist (0.5, 1.0), bearing 30 deg, yaw +/-60 deg (edges) | p0 | lat 80 ms | C2 |
| R3 plant-mix | 0.20 | L1 | default L1 spawn | full_posterior=True (all 60 particles + wheel_gain 0.85-1.15, yaw_scale 0.5-1.5, sag 0.9-1.0); alternate 50/50 with mean (`full_posterior=False`, `use_mean_dynamics=True`) | lat 80 ms | C3 |
| R4 sensor-stress | 0.20 | L3 | default L3 spawn | full_posterior=True | lat U(0.08,0.20), drops U(0.25,0.45), blackout 3 s, under-drone loss | C5 (near/delay/drop/blackout/blind) |
| R5 search-far | 0.10 | L4 | default L4 spawn (dist 0.5-4.0, bearing 180, yaw 180) | full_posterior=True | same as L3 + far/out-of-view starts | C4 + C5-far |

Expected coverage: every reset lands within one recorded failure envelope
(R1 ~ mouth-clip pose; R2 ~ high-yaw entries that bifurcate; R3 ~ unseen-plant
episodes that collapsed the resim; R4 ~ delay/drop/blackout/blind combos that
produce L3 losses; R5 ~ far/out-of-view L4 timeouts). C6 hot entries arise
inside R1/R2 via approach speed, no extra bin.

## Use for L1+ training

- Warm-start from `runs/student_L0_lat_s1.pt` (gate ckpt, fixed-sim p0).
- Sample bins per env per reset with the ratios above; keep persistent-env
  rollouts (student loop) so late-episode hold states are reached.
- Do not retune reward weights to chase a bin; one change per experiment,
  judge by `evaluate.py --episodes 1000 --seed 11` per level.
- Re-seed check: any bin showing seed 0->1 bifurcation (C2 pattern) gets a
  second seed before a conclusion.
