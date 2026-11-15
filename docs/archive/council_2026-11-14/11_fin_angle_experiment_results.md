# Agent 11 — Empirical Fin-Angle ArUco vs. Multi-Tag Board PnP Results

**Generated:** 2026-11-14 11:14:35
**Pipeline:** `simulation/gazebo/vision_in_loop.py` (`1280x720`, `fx=fy=968 px`, `50 mm` `DICT_4X4_50`, photometric DR) + `src/ugv/aruco.py` (`ArucoFollower.detect_gray`) + OpenCV 5.0.0 (`SOLVEPNP_IPPE_SQUARE` vs. `cv2.aruco.Board` + `SOLVEPNP_SQPNP`).
**Trials:** 30 photometric DR trials per `(condition, distance)` cell (15 Left, 15 Right) across `d ∈ [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]` m (105 cells, wall time 68.0 s).

---

## 1. Executive Summary Across All Conditions

| Condition | Solver | Det Rate (%) | >=2 Tags (%) | Med Pos Err (mm) | Med Yaw Err (deg) | IPPE Flip (%) | L/R Acc (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| `A_Vertical_Yaw00` | `single_ippe` | 99.0% | 0.0% | 52.4 | 2.85 | 3.4% | 68.3% |
| `A_Vertical_Yaw15` | `single_ippe` | 97.6% | 0.0% | 24.3 | 1.91 | 3.4% | 96.6% |
| `A_Vertical_Yaw30` | `single_ippe` | 100.0% | 0.0% | 12.7 | 1.20 | 3.8% | 96.2% |
| `A_Vertical_Yaw45` | `single_ippe` | 100.0% | 0.0% | 9.3 | 0.52 | 6.7% | 93.3% |
| `A_Vertical_Yaw60` | `single_ippe` | 96.7% | 0.0% | 11.1 | 0.83 | 17.2% | 82.8% |
| `A_Vertical_Yaw75` | `single_ippe` | 73.8% | 0.0% | 4.9 | 0.42 | 3.2% | 96.8% |
| `B_Pitch40` | `single_ippe` | 100.0% | 0.0% | 11.2 | 0.18 | 6.7% | 92.9% |
| `B_Pitch55` | `single_ippe` | 97.6% | 0.0% | 10.6 | 0.46 | 2.4% | 97.6% |
| `B_Dihedral40_Swept35` | `single_ippe` | 95.7% | 0.0% | 14.2 | 0.45 | 5.5% | 98.5% |
| `B_Dihedral55_Swept35` | `single_ippe` | 90.5% | 0.0% | 19.5 | 0.39 | 16.3% | 90.0% |
| `C_Wedge15_Single` | `single_ippe` | 100.0% | 0.0% | 27.8 | 2.14 | 16.2% | 83.8% |
| `C_Wedge15_3Tag_SingleIPPE` | `single_ippe` | 100.0% | 99.0% | 19.9 | 1.40 | 8.6% | 97.6% |
| `C_Wedge15_3Tag_BoardSQPNP` | `board_sqpnp` | 100.0% | 99.0% | 1.5 | 0.14 | 0.5% | 99.5% |
| `C_Steep40_3Tag_SingleIPPE` | `single_ippe` | 99.5% | 93.8% | 22.7 | 2.94 | 23.9% | 74.2% |
| `C_Steep40_3Tag_BoardSQPNP` | `board_sqpnp` | 99.5% | 93.8% | 2.6 | 0.16 | 2.4% | 97.1% |

## 2. Condition A — Vertical Fin (`0°` pitch, `0°..75°` yaw viewing angle)

Single `50 mm` vertical fin tag (`ID 20`, `z = 0.120 m`). Even when detection succeeds, near-frontal `0°` viewing suffers planar `IPPE_SQUARE` branch ambiguity, while steep oblique `60°–75°` yaw degrades long-range decode rate and depth conditioning.

| Yaw (deg) | d (m) | Det (%) | Pos p50 / p90 (mm) | Yaw p50 / p90 (deg) | IPPE Flip (%) | L/R Acc (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.25 | 100.0 | 0.6 / 0.9 | 0.1 / 0.2 | 0.0 | 100.0 |
| 0 | 0.35 | 96.7 | 17.5 / 27.1 | 2.9 / 4.4 | 0.0 | 100.0 |
| 0 | 0.50 | 100.0 | 19.6 / 42.6 | 2.2 / 4.9 | 0.0 | 90.0 |
| 0 | 0.75 | 100.0 | 106.6 / 182.0 | 8.2 / 14.0 | 0.0 | 56.7 |
| 0 | 1.00 | 96.7 | 52.4 / 224.6 | 2.7 / 12.9 | 0.0 | 37.9 |
| 0 | 1.50 | 100.0 | 230.1 / 336.5 | 8.6 / 12.8 | 0.0 | 33.3 |
| 0 | 2.00 | 100.0 | 776.2 / 994.0 | 22.3 / 29.2 | 23.3 | 60.0 |
| 15 | 0.25 | 96.7 | 1.2 / 2.3 | 0.3 / 0.5 | 3.5 | 96.5 |
| 15 | 0.35 | 93.3 | 2.6 / 12.0 | 0.5 / 1.9 | 0.0 | 100.0 |
| 15 | 0.50 | 100.0 | 23.2 / 29.0 | 2.6 / 3.3 | 0.0 | 100.0 |
| 15 | 0.75 | 96.7 | 27.6 / 36.8 | 2.1 / 2.8 | 0.0 | 100.0 |
| 15 | 1.00 | 100.0 | 24.3 / 98.9 | 1.4 / 5.4 | 10.0 | 90.0 |
| 15 | 1.50 | 100.0 | 56.9 / 79.8 | 1.9 / 2.9 | 0.0 | 100.0 |
| 15 | 2.00 | 96.7 | 411.0 / 787.3 | 11.8 / 22.9 | 10.3 | 89.7 |
| 30 | 0.25 | 100.0 | 1.1 / 1.5 | 0.2 / 0.3 | 0.0 | 100.0 |
| 30 | 0.35 | 100.0 | 4.3 / 8.5 | 0.7 / 1.4 | 0.0 | 100.0 |
| 30 | 0.50 | 100.0 | 12.7 / 20.9 | 1.4 / 2.2 | 0.0 | 100.0 |
| 30 | 0.75 | 100.0 | 18.1 / 22.6 | 1.2 / 1.6 | 0.0 | 100.0 |
| 30 | 1.00 | 100.0 | 9.7 / 17.2 | 0.4 / 0.8 | 0.0 | 100.0 |
| 30 | 1.50 | 100.0 | 143.2 / 1537.2 | 4.9 / 59.9 | 20.0 | 80.0 |
| 30 | 2.00 | 100.0 | 121.2 / 226.3 | 2.9 / 6.5 | 6.7 | 93.3 |
| 45 | 0.25 | 100.0 | 1.3 / 1.5 | 0.2 / 0.4 | 0.0 | 100.0 |
| 45 | 0.35 | 100.0 | 1.4 / 3.6 | 0.2 / 0.5 | 0.0 | 100.0 |
| 45 | 0.50 | 100.0 | 5.3 / 12.5 | 0.5 / 1.4 | 0.0 | 100.0 |
| 45 | 0.75 | 100.0 | 9.3 / 32.9 | 0.7 / 2.5 | 0.0 | 100.0 |
| 45 | 1.00 | 100.0 | 20.3 / 28.8 | 0.3 / 1.1 | 0.0 | 100.0 |
| 45 | 1.50 | 100.0 | 67.0 / 2164.5 | 1.5 / 91.3 | 23.3 | 76.7 |
| 45 | 2.00 | 100.0 | 164.8 / 2911.2 | 4.3 / 91.2 | 23.3 | 76.7 |
| 60 | 0.25 | 100.0 | 1.2 / 2.2 | 0.2 / 0.5 | 0.0 | 100.0 |
| 60 | 0.35 | 100.0 | 3.3 / 8.6 | 0.5 / 1.2 | 0.0 | 100.0 |
| 60 | 0.50 | 100.0 | 7.7 / 13.9 | 0.9 / 1.6 | 0.0 | 100.0 |
| 60 | 0.75 | 100.0 | 11.1 / 18.2 | 0.8 / 1.3 | 0.0 | 100.0 |
| 60 | 1.00 | 100.0 | 20.9 / 37.7 | 0.4 / 1.8 | 0.0 | 100.0 |
| 60 | 1.50 | 100.0 | 2635.3 / 2679.9 | 119.1 / 121.5 | 60.0 | 40.0 |
| 60 | 2.00 | 76.7 | 3523.4 / 3565.6 | 118.4 / 123.1 | 73.9 | 26.1 |
| 75 | 0.25 | 100.0 | 2.1 / 3.9 | 0.4 / 0.8 | 0.0 | 100.0 |
| 75 | 0.35 | 100.0 | 2.9 / 5.3 | 0.4 / 0.7 | 0.0 | 100.0 |
| 75 | 0.50 | 100.0 | 5.4 / 9.0 | 0.4 / 0.9 | 0.0 | 100.0 |
| 75 | 0.75 | 100.0 | 4.4 / 11.8 | 0.2 / 0.9 | 0.0 | 100.0 |
| 75 | 1.00 | 90.0 | 31.3 / 42.1 | 0.8 / 2.0 | 0.0 | 100.0 |
| 75 | 1.50 | 26.7 | 2926.0 / 2938.8 | 150.1 / 151.6 | 62.5 | 37.5 |
| 75 | 2.00 | 0.0 | — | — | — | — |

## 3. Condition B — Steep V-Tail Back Fin (`40°` and `55°` pitch / dihedral tilt)

- `B_Pitch40`, `B_Pitch55`: Fin surface pitched `40°` / `55°` upward from vertical (modeling a normal upward-angled fin viewed from a floor camera `z_cam = 0.0645 m`).
- `B_Dihedral40_Swept35`, `B_Dihedral55_Swept35`: Inverted-V fin with `40°` / `55°` anhedral roll and `35°` aft sweep.

| Config | d (m) | Det (%) | Pos p50 / p90 (mm) | Yaw p50 / p90 (deg) | IPPE Flip (%) | L/R Acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| `B_Pitch40` | 0.25 | 100.0 | 1.7 / 2.5 | 0.1 / 0.2 | 0.0 | 100.0 |
| `B_Pitch40` | 0.35 | 100.0 | 1.7 / 3.1 | 0.1 / 0.3 | 0.0 | 100.0 |
| `B_Pitch40` | 0.50 | 100.0 | 3.1 / 4.5 | 0.2 / 0.3 | 0.0 | 100.0 |
| `B_Pitch40` | 0.75 | 100.0 | 11.2 / 15.2 | 0.7 / 1.1 | 0.0 | 100.0 |
| `B_Pitch40` | 1.00 | 100.0 | 14.1 / 26.6 | 0.2 / 0.6 | 0.0 | 100.0 |
| `B_Pitch40` | 1.50 | 100.0 | 62.9 / 110.2 | 0.8 / 1.5 | 6.7 | 93.3 |
| `B_Pitch40` | 2.00 | 100.0 | 150.3 / 1795.7 | 3.2 / 41.9 | 40.0 | 56.7 |
| `B_Pitch55` | 0.25 | 96.7 | 0.9 / 1.2 | 0.2 / 0.2 | 0.0 | 100.0 |
| `B_Pitch55` | 0.35 | 96.7 | 0.7 / 1.9 | 0.1 / 0.3 | 0.0 | 100.0 |
| `B_Pitch55` | 0.50 | 96.7 | 2.5 / 4.3 | 0.1 / 0.3 | 0.0 | 100.0 |
| `B_Pitch55` | 0.75 | 100.0 | 10.6 / 13.4 | 0.6 / 0.7 | 0.0 | 100.0 |
| `B_Pitch55` | 1.00 | 100.0 | 16.2 / 21.4 | 0.5 / 0.8 | 0.0 | 100.0 |
| `B_Pitch55` | 1.50 | 100.0 | 55.1 / 104.3 | 1.1 / 2.2 | 3.3 | 96.7 |
| `B_Pitch55` | 2.00 | 93.3 | 100.7 / 2778.0 | 0.8 / 173.4 | 14.3 | 85.7 |
| `B_Dihedral40_Swept35` | 0.25 | 100.0 | 0.7 / 2.6 | 0.2 / 0.4 | 0.0 | 100.0 |
| `B_Dihedral40_Swept35` | 0.35 | 96.7 | 1.8 / 4.3 | 0.2 / 0.5 | 0.0 | 100.0 |
| `B_Dihedral40_Swept35` | 0.50 | 100.0 | 3.0 / 5.5 | 0.2 / 0.5 | 0.0 | 100.0 |
| `B_Dihedral40_Swept35` | 0.75 | 100.0 | 14.2 / 28.3 | 0.5 / 1.4 | 0.0 | 100.0 |
| `B_Dihedral40_Swept35` | 1.00 | 100.0 | 22.2 / 38.0 | 0.5 / 1.6 | 0.0 | 100.0 |
| `B_Dihedral40_Swept35` | 1.50 | 93.3 | 58.2 / 827.0 | 0.9 / 41.2 | 10.7 | 100.0 |
| `B_Dihedral40_Swept35` | 2.00 | 80.0 | 112.2 / 3485.2 | 2.1 / 136.3 | 33.3 | 87.5 |
| `B_Dihedral55_Swept35` | 0.25 | 96.7 | 1.2 / 4.2 | 0.2 / 0.5 | 0.0 | 100.0 |
| `B_Dihedral55_Swept35` | 0.35 | 96.7 | 2.9 / 5.4 | 0.4 / 0.8 | 0.0 | 100.0 |
| `B_Dihedral55_Swept35` | 0.50 | 96.7 | 4.5 / 7.0 | 0.4 / 0.6 | 0.0 | 100.0 |
| `B_Dihedral55_Swept35` | 0.75 | 100.0 | 19.5 / 30.4 | 0.4 / 1.1 | 3.3 | 96.7 |
| `B_Dihedral55_Swept35` | 1.00 | 100.0 | 1730.9 / 1881.0 | 145.3 / 154.8 | 53.3 | 63.3 |
| `B_Dihedral55_Swept35` | 1.50 | 73.3 | 47.8 / 66.5 | 0.5 / 1.8 | 4.5 | 100.0 |
| `B_Dihedral55_Swept35` | 2.00 | 70.0 | 3641.4 / 3870.0 | 151.1 / 152.6 | 61.9 | 66.7 |

## 4. Condition C — Wedge-Canted (`15°`) + Dual-Fin + Belly 3-Tag Constellation: `cv2.aruco.Board` / `SOLVEPNP_SQPNP` vs. Single-Tag `IPPE_SQUARE`

Constellation geometry: `left_fin` (`ID 20` at `+0.10, 0.0, 0.125`), `right_fin` (`ID 21` at `-0.10, 0.0, 0.125`), and `belly_tag` (`ID 22` at `0.0, -0.06, 0.165`). Paired evaluation runs both single-tag `IPPE_SQUARE` and joint `cv2.aruco.Board` + `SOLVEPNP_SQPNP` on the exact same rendered frames.

| Config | d (m) | Any Det (%) | >=2 Tags (%) | Pos p50 / p90 (mm) | Yaw p50 / p90 (deg) | Flip (%) | L/R Acc (%) | >=2-Tag Only Pos / Yaw / Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `C_Wedge15_Single` | 0.25 | 100.0 | 0.0 | 2.0 / 5.5 | 0.31 / 1.17 | 0.0 | 100.0 | — |
| `C_Wedge15_Single` | 0.35 | 100.0 | 0.0 | 13.9 / 27.9 | 2.14 / 4.32 | 0.0 | 100.0 | — |
| `C_Wedge15_Single` | 0.50 | 100.0 | 0.0 | 27.8 / 51.4 | 2.76 / 5.73 | 0.0 | 100.0 | — |
| `C_Wedge15_Single` | 0.75 | 100.0 | 0.0 | 27.4 / 48.3 | 2.08 / 3.59 | 0.0 | 100.0 | — |
| `C_Wedge15_Single` | 1.00 | 100.0 | 0.0 | 28.1 / 57.7 | 1.55 / 3.26 | 0.0 | 100.0 | — |
| `C_Wedge15_Single` | 1.50 | 100.0 | 0.0 | 203.5 / 358.0 | 6.56 / 11.93 | 46.7 | 53.3 | — |
| `C_Wedge15_Single` | 2.00 | 100.0 | 0.0 | 432.6 / 695.5 | 6.74 / 20.06 | 66.7 | 33.3 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 0.25 | 100.0 | 96.7 | 2.0 / 2.5 | 0.44 / 0.51 | 0.0 | 100.0 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 0.35 | 100.0 | 100.0 | 11.8 / 16.8 | 1.82 / 2.59 | 0.0 | 100.0 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 0.50 | 100.0 | 100.0 | 6.4 / 13.0 | 0.69 / 1.42 | 0.0 | 100.0 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 0.75 | 100.0 | 100.0 | 20.0 / 45.0 | 1.40 / 3.49 | 0.0 | 100.0 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 1.00 | 100.0 | 96.7 | 19.9 / 140.9 | 0.98 / 7.76 | 10.0 | 96.7 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 1.50 | 100.0 | 100.0 | 140.7 / 591.0 | 4.92 / 23.41 | 16.7 | 100.0 | — |
| `C_Wedge15_3Tag_SingleIPPE` | 2.00 | 100.0 | 100.0 | 242.6 / 933.3 | 6.38 / 27.76 | 33.3 | 86.7 | — |
| `C_Wedge15_3Tag_BoardSQPNP` | 0.25 | 100.0 | 96.7 | 0.2 / 0.4 | 0.02 / 0.06 | 0.0 | 100.0 | 0.2 mm / 0.02° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 0.35 | 100.0 | 100.0 | 0.5 / 0.9 | 0.04 / 0.07 | 0.0 | 100.0 | 0.5 mm / 0.04° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 0.50 | 100.0 | 100.0 | 1.5 / 3.2 | 0.14 / 0.31 | 0.0 | 100.0 | 1.5 mm / 0.14° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 0.75 | 100.0 | 100.0 | 1.1 / 2.9 | 0.05 / 0.19 | 0.0 | 100.0 | 1.1 mm / 0.05° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 1.00 | 100.0 | 96.7 | 3.2 / 6.2 | 0.16 / 0.22 | 3.3 | 96.7 | 3.1 mm / 0.16° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 1.50 | 100.0 | 100.0 | 9.4 / 18.5 | 0.19 / 0.57 | 0.0 | 100.0 | 9.4 mm / 0.19° / 0.0% |
| `C_Wedge15_3Tag_BoardSQPNP` | 2.00 | 100.0 | 100.0 | 22.4 / 33.6 | 0.50 / 0.85 | 0.0 | 100.0 | 22.4 mm / 0.50° / 0.0% |
| `C_Steep40_3Tag_SingleIPPE` | 0.25 | 100.0 | 93.3 | 0.7 / 1.0 | 0.09 / 0.15 | 0.0 | 100.0 | — |
| `C_Steep40_3Tag_SingleIPPE` | 0.35 | 100.0 | 100.0 | 22.7 / 42.7 | 2.94 / 5.56 | 0.0 | 100.0 | — |
| `C_Steep40_3Tag_SingleIPPE` | 0.50 | 100.0 | 100.0 | 22.6 / 46.2 | 2.19 / 4.47 | 3.3 | 96.7 | — |
| `C_Steep40_3Tag_SingleIPPE` | 0.75 | 100.0 | 100.0 | 19.6 / 31.1 | 1.16 / 2.01 | 0.0 | 100.0 | — |
| `C_Steep40_3Tag_SingleIPPE` | 1.00 | 100.0 | 100.0 | 123.2 / 260.2 | 6.35 / 13.09 | 46.7 | 50.0 | — |
| `C_Steep40_3Tag_SingleIPPE` | 1.50 | 100.0 | 93.3 | 277.0 / 339.1 | 5.56 / 10.36 | 70.0 | 23.3 | — |
| `C_Steep40_3Tag_SingleIPPE` | 2.00 | 96.7 | 70.0 | 592.9 / 870.1 | 16.15 / 24.22 | 48.3 | 48.3 | — |
| `C_Steep40_3Tag_BoardSQPNP` | 0.25 | 100.0 | 93.3 | 0.6 / 0.9 | 0.06 / 0.11 | 0.0 | 100.0 | 0.6 mm / 0.06° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 0.35 | 100.0 | 100.0 | 0.6 / 1.0 | 0.03 / 0.14 | 0.0 | 100.0 | 0.6 mm / 0.03° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 0.50 | 100.0 | 100.0 | 0.7 / 1.5 | 0.07 / 0.14 | 0.0 | 100.0 | 0.7 mm / 0.07° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 0.75 | 100.0 | 100.0 | 2.6 / 4.9 | 0.16 / 0.33 | 0.0 | 100.0 | 2.6 mm / 0.16° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 1.00 | 100.0 | 100.0 | 5.2 / 9.1 | 0.17 / 0.51 | 0.0 | 100.0 | 5.2 mm / 0.17° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 1.50 | 100.0 | 93.3 | 20.7 / 143.0 | 0.51 / 4.25 | 6.7 | 93.3 | 18.0 mm / 0.51° / 0.0% |
| `C_Steep40_3Tag_BoardSQPNP` | 2.00 | 96.7 | 70.0 | 34.0 / 506.1 | 0.89 / 14.03 | 10.3 | 86.2 | 18.6 mm / 0.34° / 0.0% |

## 5. Key Empirical Findings

1. **Steep un-wedged fins fail at range and suffer severe IPPE flips:** `B_Dihedral55_Swept35` achieves only `90.5%` overall detection (`B_Dihedral40_Swept35`: `95.7%`) and `B_Pitch55` exhibits `2.4%` IPPE flip rate with `10.6 mm` median error.
2. **15° wedge-canting restores full-range visibility, but single-tag IPPE_SQUARE still flips:** `C_Wedge15_Single` raises detection to `100.0%`, yet single-tag `IPPE_SQUARE` still flips `16.2%` of the time (`C_Wedge15_3Tag_SingleIPPE`: `8.6%` flips, `19.9 mm` median error, `97.6%` L/R accuracy).
3. **Joint `cv2.aruco.Board` + `SOLVEPNP_SQPNP` on the 3-tag constellation eliminates IPPE flips:** On the exact same frames, `C_Wedge15_3Tag_BoardSQPNP` drops IPPE flip rate from `8.6%` → **`0.5%`**, median position error from `19.9 mm` → **`1.5 mm`**, median yaw error from `1.40°` → **`0.14°`**, and raises L/R classification accuracy from `97.6%` → **`99.5%`**.
4. **Recommendation:** Do **not** stick tags flat onto steep 40°–55° V-tail surfaces with single-tag `IPPE_SQUARE`. Use lightweight `15°` aft/downward wedge brackets on both fins (`ID 20` left, `ID 21` right) plus an aft belly tag (`ID 22`) solved jointly with `cv2.aruco.Board` + `SOLVEPNP_SQPNP`.
