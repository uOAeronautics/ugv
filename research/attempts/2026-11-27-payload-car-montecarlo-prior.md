---
id: 2026-11-27-payload-car-montecarlo-prior
date: 2026-11-27
author: agy
question: Is the CAD-derived Monte Carlo dynamics prior for the unmeasured payload car physically sane across open-loop Gazebo step tests, and what physical discrepancies must bench characterisation resolve?
outcome: kept
failure_mode: none
refs: [8e2dabd, simulation/gazebo/payload_prior.json, simulation/gazebo/payload_montecarlo.json, simulation/gazebo/eval_payload_montecarlo.py, simulation/gazebo/build_payload_prior.py, simulation/gazebo/CAR_MODEL.md, research/decisions/2026-11-27-replace-freenove-car.md, research/tasks/2026-11-27-characterise-payload-car.md]
tags: [simulation, gazebo, payload-car, montecarlo, prior]
---

## Tried

Evaluated the committed Monte Carlo prior for the unmeasured `payload_car` (commit `8e2dabd`, `simulation/gazebo/payload_prior.json`, `simulation/gazebo/eval_payload_montecarlo.py`, status `PRIOR_SAMPLE_NOT_MEASURED` — NOT a calibrated vehicle model and NOT valid for trusting controllers on hardware):

1. **Bottom-up CAD + datasheet prior (`simulation/gazebo/build_payload_prior.py`, `payload_prior.json`):** extracted per-part volumes, centroids, and polyhedral inertia tensors from all `1,242` USD meshes in `payload_assembly.usdz` (`1,241` watertight + `1` tiny `0.0005 cm³` SMD pad via convex-hull fallback), paired with FDM/material density ranges and datasheet envelope mass ranges (`2S LiPo`, `4× N20` gearmotors across `8` common `6 V` / `12 V` variants, `BNO055`, `40×40 M12` camera, `56×41` SBC), truncated at the owner's `<= 0.300 kg` (`"probably under 300g"`) bound, plus drivetrain/roller friction and lag ranges.
2. **Open-loop Gazebo Monte Carlo batch (`mc-payload-20261127T193852Z`, `simulation/gazebo/payload_montecarlo.json`):** `N = 36` headless CPU Gazebo runs (`gz sim -s`, DART `pgs` solver, `1 ms` physics step, `dt = 0.02 s`, `Sensors` plugin stripped, seeds `0..35`), each running `0.5 s` settle + `1.5 s` full-duty (`1.0`) forward, strafe-left, and CCW-yaw drives with `1.0 s` / `0.6 s` braking (`total_wall_s = 78.36 s`).

## Result

**Batch `mc-payload-20261127T193852Z` (`simulation/gazebo/payload_montecarlo.json`):**

- **Sanity fraction:** **`36 / 36` (`100.0%`)** — zero tipping (`max |roll| = 0.405°`, `max |pitch| = 1.121°`), zero numerical blow-up (`z ∈ [-0.00073, +0.00053] m`), all axes signed correctly across all 8 N20 motor variants (`n20_100_1_6v`: 8, `n20_100_1_12v`: 7, `n20_50_1_12v`: 6, `n20_150_1_12v`: 5, `n20_210_1_6v`: 4, `n20_298_1_6v`: 3, `n20_50_1_6v`: 2, `n20_150_1_6v`: 1).

| Metric (`PRIOR_SAMPLE_NOT_MEASURED`, `N = 36`) | p05 | Median (p50) | p95 | Min – Max | Mean |
|---|---:|---:|---:|---:|---:|
| Sampled total mass (kg) | `0.27028` | `0.28794` | `0.29871` | `0.25855 – 0.29961` | `0.28703` |
| Forward top speed (m/s) | `0.17831` | `0.32336` | `0.87216` | `0.17760 – 1.23823` | `0.43745` |
| Forward acceleration (m/s²) | `0.71793` | `1.46166` | `3.10530` | `0.57803 – 4.36369` | `1.60876` |
| Strafe top speed (m/s) | `0.16949` | `0.29320` | `0.78041` | `0.16733 – 1.01343` | `0.38042` |
| Strafe / forward speed ratio | `0.75234` | `0.90995` | `0.96013` | `0.68492 – 0.97485` | `0.88795` |
| CCW yaw rate (rad/s) | `1.77061` | `3.21180` | `8.62019` | `1.74625 – 12.01404` | `4.31008` |
| Forward brake stop distance (m) | `0.00677` | `0.02343` | `0.08709` | `0.00534 – 0.17247` | `0.03468` |

**Two physical findings from the prior build (`payload_prior.json`, `CAR_MODEL.md`) that require hardware measurement (`2026-11-27-characterise-payload-car`):**

1. **Bottom-up mass vs. `300 g` owner bound:** the untruncated bottom-up CAD + datasheet mass distribution (50,000 draws) has median **`0.335 kg` (`334.8 g`)**, 5–95% `[287.8, 381.6] g`, with **only `12.2%` (`~12%`) of draws `<= 0.300 kg`** (truncated at `<= 0.300 kg`, the prior median shifts to `290.1 g`). Solid FDM prints plus a heavier LiPo/SBC would exceed `300 g`; a kitchen-scale measurement of the physical car is required to collapse this discrepancy.
2. **CAD wheel handedness discrepancy:** the raw USD assembly places `A` hubs on `FL`/`RR` and `B` hubs on `FR`/`RL` (`cad_raw_mecanum_hand_sign = -1`, O-pattern on top / X-pattern on floor), giving a yaw moment arm of only `|(track - wheelbase) / 2| = 27.9 mm`. The drivable Monte Carlo prior overrides this to standard top-view X-pattern (`mecanum_hand_sign = +1`, O-pattern on floor, yaw moment arm `(track + wheelbase) / 2 = 98.7 mm`, matching `mecanum_geometry.HANDEDNESS`); visual inspection of the physical wheels is required to confirm actual mounting.

## Lesson

The `payload_car` Monte Carlo prior simulates cleanly (`36/36` sane in `mc-payload-20261127T193852Z`), but the untruncated bottom-up mass median (`0.335 kg`, only `12.2% <= 300 g`) and raw USD wheel handedness (`cad_raw_mecanum_hand_sign = -1`, `27.9 mm` yaw arm vs. `98.7 mm` standard X-pattern) show that kitchen-scale weighing, visual wheel-handedness inspection, and an S4 floor `probe-axes` run must replace these priors before any controller is trusted on hardware.

## Re-run only if

Physical hardware measurements from `research/tasks/2026-11-27-characterise-payload-car.md` replace the prior distributions in `simulation/gazebo/payload_prior.json` or the CAD assembly changes.
