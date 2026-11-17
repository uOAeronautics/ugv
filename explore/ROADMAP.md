# Roadmap: ranked hypotheses toward a parked car under the plane

Target: the car parks under the drone (STRL-001, tags 14/6/3), confirmed by the end frame, not
by an estimate-based "arrived". Ledger: `explore/HYPOTHESES.jsonl`, written only through
`tools/explore.py`. Facts and dead ends: `explore/HISTORY.md` (cited below by its sections).
Nothing here has been run. Every hypothesis is S0 (no motor command); any car step needs the
S4 gate in `docs/SAFETY.md` and a free car per `docs/STATE.md`.

Ranking is by expected information per unit cost: cheap offline checks that could invalidate
the current numbers come first. The rank is this list's order; IDs are the ledger's.

## Ranked

1. **H-0008** (evaluation-validity). The L1 baseline spread (cuda 989/9 coll, CPU 959/19 coll,
   quoted 973/23) must be one number before anything is compared against it. Cost: one CPU and
   one cuda repeat of `baseline_ekf_v2_L1`; the cuda run is a GPU job, schedule it with the GPU
   owner. Prediction: agreement within 10 episodes. Falsifier: over 10 episodes apart once the
   flags match.
2. **H-0004** (evaluation-validity, challenges RL-on-L_real). The `-dy` action giving +0.25 m
   progress (`rl/runs/sensitivity_L1ft_Lreal.json`) points to a frame or sign error, not a
   learning limit. Cost: offline audit of the policy frame, export and Pi duty path; no car.
   Falsifier: the audit finds every convention matching and L_real stays at 0.
3. **H-0002** (classical-control, challenges RL-on-L_real). A classical stage controller with a
   dead-reckoned finale, tuned to the measured plant, should match or beat the RL student on L1
   p0. Cost: CPU sim, 1000 episodes, seed 11. It rests on H-0008 for the baseline figure, so
   rank it after that check. Falsifier: below 956/1000 on L1 p0 or any collision.
4. **H-0006** (evaluation-validity). The Gazebo decode conflict (5.2% against 98.3%, HISTORY
   section 3, question 5) is likely a flag wiring error in the batch runner. Cost: a Gazebo CPU
   re-run. Falsifier: decode stays below 20% with the flags matched.
5. **H-0007** (evaluation-validity). The 0/10 against 3/10 gap predates the FakeBridge `--seed`
   fix (HISTORY section 3, question 6). Cost: Gazebo CPU, 20 spawn-paired episodes per arm.
   Falsifier: a gap of 2 or more successes persists with seeds fixed.
6. **H-0001** (plant-calibration). Fitting the sim to the measured deadzones and gyro std
   should reproduce the car's oscillation pattern (HISTORY section 1, real plant). Cost: sim
   and CPU. Falsifier: turn saturation stays under 10% of ticks after the fit.
7. **H-0009** (plant-calibration). The joint observation overlap of 0.0 may come from the
   constant 20 ms sim loop against the Pi's about 0.9 ms (HISTORY section 1, observation
   contract). Cost: contract check on existing logs. Falsifier: overlap stays below 0.1 with the
   measured lag.
8. **H-0010** (hardware). Whether the 0.63 rad/s real gyro std is vibration or sensor noise
   decides the gyro model. Cost: read-only logging with motors off; blocked until the car is
   free per `docs/STATE.md` (currently charging or paused, so not run). Falsifier: std above 0.3
   rad/s with motors off.
9. **H-0003** (task-geometry, challenges RL-on-L_real). A hybrid (RL above 0.40 m, classical
   below) may remove the blind-zone failure that HISTORY section 2.2 names as the open cause.
   Cost: CPU sim on L_real. Falsifier: under 10 points gained, or handover collisions at 0.5%
   or more. Ranked last because it depends on H-0002.
10. **H-0005** (perception). A belly camera could close the blind finale by vision. Cost: a rig
    and a static bench capture first; the car only later and free. Falsifier: detection under
    80% in any 0.10 m band. Ranked last: it needs hardware and is open question 8 in HISTORY.

## Open questions (not resolved here)

- **Adoption of `student_L2_cons_adr_v2_s0`.** It scores 918/1000 on L1 p0 against the 956
  threshold in `docs/RL_PARKING.md`. This roadmap does not decide it. H-0002 would give a
  classical reference for the same threshold, which may inform the decision; it does not
  settle it.
- **Which L1 baseline is true** (H-0008 addresses it).
- **Seed bifurcation** (freeze against clip under identical code on seeds 0 and 1). No
  hypothesis here targets it yet.
- **Sweep-reflex gate plant** (`gate_sweep_L2_s11.json` recorded with mean dynamics). Not
  covered; check before relying on it.
- **Failure-replay bins R1 to R5.** Unverified whether any mixed-bin run was trained.

## Caveats

- Figures are copied from `explore/HISTORY.md`; they are not re-measured. Several conflict
  across sources, and the conflicts are the subject of H-0006, H-0007 and H-0008.
- Family coverage: classical-control, task-geometry, evaluation-validity, plant-calibration,
  perception, hardware. The rl family has no hypothesis yet; H-0002 and H-0003 challenge it
  from the classical side.

---

## Status after the 2026-11-17 run (supersedes the figures woven through the items)

Closed: **H-0001** confirmed (fitted plant `L1_fitted` reproduces the car's 0/1000, 83.6%
turn saturation; source `explore/results/H-0001-plant.json`). **H-0002** refuted (classical
go-to + DR finale 552/1000, 375 coll; `explore/results/H-0002-classical.json`); H-0003 not
built (decision recorded in brainspace I25). **H-0004** refuted (conventions all match;
`explore/results/H-0004-audit.md`). **H-0006** refuted (wiring not causal; 98.3% unverifiable;
`explore/results/H-0006-decode.json`). **H-0008** inconclusive (cuda never run; CPU repeat 986).
**H-0009** refuted (lag lowers overlap; overlap 0.0 premise stale). In flight: **H-0007**
(ammunition pending). Unstarted: **H-0005** (hardware), **H-0010** (hardware).

Working reference figures now: EKF-staged L1 p0 seed 11 ≈ 986–989/1000 (cuda record + CPU repeat);
student incumbent 966; joint-overlap gate reads 0.742 (nominal) / 0.618 (L1_fitted) at min 0.35;
both Gazebo decode figures are unusable. The 5.2/98.3 decode pair, the 959/973 baseline pair and
the 0.0 overlap premise may not be cited.
