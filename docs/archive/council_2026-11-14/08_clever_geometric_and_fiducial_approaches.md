# 08 — Clever Geometric & Fiducial Perception: Ground-Plane Homography, V-Tail Dihedral Compensation, and Multi-Plane Hybrid

**Agent:** 08 (Alternative Clever Perception Explorer: Ground-Plane Homography, V-Tail Dihedral Ratio, & Fiducial/Keypoint Fusion)  
**Target:** `.` — Raspberry Pi 4B + UC-788 global-shutter monocular camera (`1280×720`, `fx = fy = 968 px`) + IMU parking under `STRL001` fixed-wing belly  
**Aircraft reference:** [`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png) (`STRL001`), [`03_strix_plane_geometry_model.md`](file://docs/council_v2/03_strix_plane_geometry_model.md), [`09_lr_disambiguation_and_belly_shield.md`](file://docs/council_v2/09_lr_disambiguation_and_belly_shield.md)  
**Verification script & data:** [`exp_homography_and_vtail.py`](file://docs/council_v2/exp_homography_and_vtail.py), [`exp_homography_and_vtail.json`](file://docs/council_v2/exp_homography_and_vtail.json)

---

## 1. Critical Audit of the Prior Draft & Existing Codebase

The previous draft of this document (written by `iceman-free`) contained generic textbook equations and **ignored the actual perception code already in `.`**. Auditing the codebase and running empirical simulations reveals five concrete gaps:

| Issue in Prior `08` Draft | Ground Truth in Codebase & Verification ([`exp_homography_and_vtail.py`](file://docs/council_v2/exp_homography_and_vtail.py)) |
|---|---|
| **Ignored [`ground_pose.py`](file://src/ugv/ground_pose.py):** Wrote generic `H_fw` formulas without noticing the repo already has a floor-contact solver. | [`SkidPoseObserver`](file://src/ugv/ground_pose.py#L25-L150) already implements 4-point planar `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)` + `solvePnPRefineLM` + KLT optical flow + IMU yaw gating—**but it is unwired** from `web.py`/`parking.py`/`autopark.py`, hardcoded to quad skids (`0.254 × 0.3048 m`), and fails with `NaN` on exact noiseless symmetric views in OpenCV 5.0.0. |
| **Claimed fixed `H_fw` gives "cm-level `(x,y)`" at `0.4–2.0 m`:** Assumed constant camera pitch. | Because the car camera is only **`CAM_Z_M = 0.06445 m`** above the floor ([`vision_in_loop.py`](file://simulation/gazebo/vision_in_loop.py#L63)), a realistic **`±1.0°` pitch jitter** (braking/acceleration/carpet seam) shifts the horizon by `16.9 px`, blowing up fixed-`H_fw` range error to **`+373 / -214 mm` at `1.0 m`** and **`+2,366 / -703 mm` at `2.0 m`**. Solving scalar pitch from a 2-point `W_gear` baseline cuts that error **30× (`132.7 → 4.3 mm` p50 at `1.0 m`)**. |
| **Confused `AprilTag3` / `ChArUco` CPU cost (`0.5–2.0 ms`):** Assumed all decoders cost the same. | Measured on host: `DICT_APRILTAG_36h11` with `CORNER_REFINE_APRILTAG` takes **`28.4–152.9 ms/frame`** (`~100–500 ms` projected on Pi 4B—completely breaking 30 Hz), whereas `DICT_4X4_50` (`SUBPIX`) takes **`2.07–3.80 ms`** and `DICT_APRILTAG_36h11` (`SUBPIX`) takes **`2.19–3.96 ms`**. |
| **No `< 0.45 m` blind-spot analysis:** Missed why existing tags fail close-in. | At `VFOV/2 = 20.40°` (`atan(360/968)`), a flat belly tag (`z = 0.345 m`) exits the top of the frame at **`d = 0.754 m`** (`0%` det `≤ 0.75 m`), an un-wedged V-tail fin tag (`z = 0.245 m`) clips by **`d ≤ 0.50 m`**, and [`autopark.py`](file://src/ugv/autopark.py#L31) hard-crops `CROP_Y1 = 440` (severing floor wheel contacts inside `d < 0.42 m`). |
| **No quantitative incidence / `Board SQPNP` comparison:** Treated `IPPE_SQUARE` flips as a hand-wavy note. | Single-tag `IPPE_SQUARE` on a swept `40°` inverted V-tail hits **`45.0%` flips (`173.4 mm` p50)** at `2.0 m`. A **`20°` aft-canted 3-tag wedge constellation (`z = 0.115–0.140 m`) + `cv2.aruco.Board` (`SOLVEPNP_SQPNP`)** achieves **`100.0%` detection from `0.22 m` to `2.00 m`**, **`0.0%` flips**, and **`0.7–11.4 mm` p50 error**. |

---

## 2. What the Codebase Actually Implements Today

From inspecting [`src/ugv/aruco.py`](file://src/ugv/aruco.py), [`src/ugv/ground_pose.py`](file://src/ugv/ground_pose.py), [`src/ugv/autopark.py`](file://src/ugv/autopark.py), [`src/ugv/parking.py`](file://src/ugv/parking.py), and [`simulation/gazebo/vision_in_loop.py`](file://simulation/gazebo/vision_in_loop.py):

1. **Camera & intrinsics:** `1280×720` grayscale (`L8`), nominal `fx = fy = 968.0 px`, `cx = 640.0`, `cy = 360.0`, zero distortion ([`aruco.py`](file://src/ugv/aruco.py#L14-L18), [`camera_profile.json`](file://simulation/gazebo/camera_profile.json#L7-L25)). Half-FOVs are **`HFOV/2 = 33.47°`** and **`VFOV/2 = 20.40°`**. Optical center sits at **`CAM_Z_M = 0.06445 m`** above floor and **`CAM_FWD_M = 0.07105 m`** ahead of chassis center ([`vision_in_loop.py`](file://simulation/gazebo/vision_in_loop.py#L62-L63)).
2. **Primary deployed vision ([`ArucoFollower`](file://src/ugv/aruco.py#L122-L1358)):** Single-tag `DICT_4X4_50` (`TAG_SIZE_M = 0.050 m`), `perspectiveRemovePixelPerCell = 8`, `minDistanceToBorder = 3`, `CORNER_REFINE_SUBPIX`, curved/fallback bit-redecode, KLT → MOSSE → CV bridging, and per-tag `cv2.SOLVEPNP_IPPE_SQUARE`. Note: [`from_points`](file://src/ugv/aruco.py#L352-L371) does **not** check `np.isfinite(translation).all()` (unlike [`autopark.py`](file://src/ugv/autopark.py#L77) and [`parking.py`](file://src/ugv/parking.py#L611)).
3. **Parking estimator gates ([`ParkingEstimator`](file://src/ugv/parking.py#L508-L630)):** Silently drops any detection with `tracked == True`, `confidence < 0.99` (discarding all `decode-fallback` `0.75–0.90` close-range rescues), or `reprojection_error_px > 2.0`.
4. **Unwired floor-contact observer ([`SkidPoseObserver`](file://src/ugv/ground_pose.py#L25-L150)):**
   - Models a **4-point symmetric floor rectangle** (`clear_width=0.254, bar_length=0.3048, radius=0.03` → `±0.157 m × ±0.1524 m` at `y = 0`).
   - Solves 6-DoF pose via `cv2.solvePnPGeneric(..., cv2.SOLVEPNP_IPPE)` + `cv2.solvePnPRefineLM`.
   - Enforces contour area `>= 500 px²`, height `-0.5 < position[1] < 0` (camera above floor), `rotation[1, 1] >= 0.7`, `reprojection_px <= 8`, and IMU heading delta `<= 8°` across `44×44` KLT patch tracks.
   - **Never wired** into the runtime loop (`web.py`/`parking.py`/`autopark.py`).

---

## 3. Ground-Plane Homography (`z = 0`) on `STRL001` Wheel & Skid Contacts

### 3.1 Exact Visibility & Pitch-Sensitivity Mechanics

For a floor contact point $\mathbf{X}_w = [X_w, Y_w, 0]^T$ (camera frame aligned with floor when pitch $\theta = 0$, roll $\phi = 0$, camera height $h = \text{CAM\_Z\_M} = 0.06445\text{ m}$), pixel coordinates are:
$$u = c_x + f_x \frac{X_w}{Y_w}, \qquad v = c_y + f_y \frac{h}{Y_w}.$$

- **Floor visibility limit (bottom edge `v = 720`):** $d_{\min,\text{floor}} = \frac{h}{\tan(\text{VFOV}/2)} = \frac{0.06445}{360/968} = \mathbf{0.173\text{ m}}$. Floor contacts stay visible inside the `1280×720` image **2.6×–4.4× closer** than elevated fin/belly tags (`0.45–0.75 m` blind spot).
- **Wide-track horizontal clip (`u ∈ [3, 1277]`):** For `STRL001` H-A main wheels ($W_{\text{gear}} = 0.56\text{ m}$ at $\pm 0.28\text{ m}$, per [`03_strix_plane_geometry_model.md`](file://docs/council_v2/03_strix_plane_geometry_model.md#L173)), both wheels stay in frame down to $d_{\min,\text{pair}} = \frac{0.28 \times 968}{637} = \mathbf{0.425\text{ m}}$. For single-wheel, centerline nose wheel, or inner tail-skid contacts, visibility extends to **`0.173 m`**.
- **Why fixed-extrinsic `H_fw` fails under `±1°` pitch jitter:** Differentiating range $Y_w = f_y h / (v - c_y)$ with respect to vertical pixel error $\delta v$ gives:
  $$\sigma_{Y_w} \approx \frac{Y_w^2}{f_y h}\,\sigma_v, \qquad \Delta v_{\text{pitch}}(\delta\theta) \approx f_y \tan(\delta\theta) = 968 \tan(1^\circ) = \mathbf{16.90\text{ px}}.$$
  Because the camera sits only $64.45\text{ mm}$ off the floor, at $Y_w = 1.0\text{ m}$ the contact point is only $62.4\text{ px}$ below the horizon (`v = 422.4 px`), and at $2.0\text{ m}$ only $31.2\text{ px}$ below (`v = 391.2 px`). A **`1°` nose-down pitch** eats **54% of the horizon offset at `2.0 m`**!

### 3.2 Clever Fix: 2-Point `W_gear` Pitch Self-Calibrating Homography

Instead of trusting a fixed pitch `θ = 0` (or requiring 4 coplanar points like [`SkidPoseObserver`](file://src/ugv/ground_pose.py#L25)), observe **any two floor contacts** $(u_L, v_L)$ and $(u_R, v_R)$ separated by a known aircraft track $W_{\text{gear}}$ (e.g. $0.56\text{ m}$ main wheels, or $1.15\text{ m}$ tail-tip skids at long range).

For candidate camera pitch $\theta \in [-8^\circ, +8^\circ]$ and known mount height $h = 0.06445\text{ m}$, intersect both camera rays with $Z_w = 0$ to obtain ground points $\mathbf{P}_L(\theta), \mathbf{P}_R(\theta) \in \mathbb{R}^2$, and solve the **1D monotonic scalar equation**:
$$g(\theta) = \bigl\|\mathbf{P}_L(\theta) - \mathbf{P}_R(\theta)\bigr\|_2 - W_{\text{gear}} = 0.$$
Because horizontal separation scales with range while $g(\theta)$ is strictly monotonic in $\theta$, **24 bisection steps take `< 0.02 ms`**, are **SE(2)-invariant** (exact for any car yaw $\psi$ and translation), recover instantaneous car pitch $\hat{\theta}$ to **`0.03°`**, and transfer depth precision from the tiny $62\text{ px}$ vertical height baseline to the **`542 px`** horizontal wheel-track baseline!

### 3.3 Empirical Accuracy Table (`exp_homography_and_vtail.py` — Experiment 1)

Simulation over 400 pixel-noise trials and 300 Monte Carlo trials ($\delta\theta \sim \mathcal{U}(-1^\circ, +1^\circ)$, $\psi \sim \mathcal{U}(-15^\circ, +15^\circ)$, $\sigma_{uv} = 1.0\text{ px}$, $W_{\text{gear}} = 0.56\text{ m}$):

| Range $d$ (m) | Contact Row $v$ (px) | Fixed $H_{fw}$ p50 @ $0.5\text{ px}$ | Fixed $H_{fw}$ p50 @ $1.0\text{ px}$ | Fixed $H_{fw}$ Pitch $-1^\circ / +1^\circ$ (mm) | MC Fixed $H_{fw}$ p50 / p90 (mm) | **2-pt $W_{\text{gear}}$ Self-Cal p50 / p90 (mm)** | Pitch Est MAE (°) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.25** | 609.6 | 0.3 mm | 0.7 mm | +19.4 / -16.9 | *(pair clipped)* | *(single-pt + IMU)* | — |
| **0.35** | 538.3 | 0.7 mm | 1.4 mm | +37.9 / -31.3 | *(pair clipped)* | *(single-pt + IMU)* | — |
| **0.45** | 498.6 | 1.1 mm | 2.1 mm | +63.7 / -49.9 | 27.2 / 51.1 | **1.1 / 2.4** | **0.030°** |
| **0.60** | 464.0 | 1.9 mm | 3.5 mm | +117.8 / -84.8 | 49.8 / 90.7 | **1.7 / 3.9** | **0.030°** |
| **0.80** | 438.0 | 3.5 mm | 7.6 mm | +222.7 / -143.4 | 80.8 / 164.8 | **2.9 / 6.0** | **0.033°** |
| **1.00** | 422.4 | 5.6 mm | 11.3 mm | **+373.0 / -214.0** | **132.7 / 277.3** | **4.3 / 10.5** | **0.031°** |
| **1.25** | 409.9 | 7.6 mm | 17.1 mm | +641.5 / -317.0 | 224.8 / 507.0 | **7.7 / 19.8** | **0.035°** |
| **1.50** | 401.6 | 11.6 mm | 26.3 mm | +1028.2 / -434.1 | 281.4 / 752.7 | **9.6 / 31.1** | **0.034°** |
| **2.00** | 391.2 | 21.5 mm | 41.5 mm | **+2366.0 / -703.4** | **503.1 / 1397.0** | **25.8 / 97.8** | **0.042°** |

**Key takeaway:** Fixed-extrinsic `H_fw` is **unsafe** above `0.45 m` whenever acceleration/braking pitches the car by `±1°` (`+64 mm` error even at `0.45 m`, exceeding the `35 mm` corridor margin). The 2-point `W_gear` pitch self-calibrator reduces error **30×** (`132.7 → 4.3 mm` at `1.0 m`) using only 2 wheel-bottom pixels. Below `0.425 m` (when the `0.56 m` main wheels clip horizontally), holding the last filtered $\hat{\theta}$ (plus IMU pitch/nose-wheel contact) keeps single-wheel homography at **`0.7–1.4 mm`** (`1.0 px` noise).

### 3.4 Direct Audit of [`src/ugv/ground_pose.py`](file://src/ugv/ground_pose.py) (`SkidPoseObserver`)

Running [`SkidPoseObserver`](file://src/ugv/ground_pose.py#L25) directly at `CAM_Z_M = 0.06445 m` on (a) its native quad rectangle (`0.314 × 0.305 m`) and (b) a generalized `STRL001` trapezoidal floor footprint (`±0.28 m` mains at `+0.19 m`, `±0.575 m` tail-tip skids at `-0.19 m`):

| Config | $d$ (m) | Area ($\text{px}^2$) | All 4 in FOV | Exact Noiseless `ok` | `0.5 px` p50 (mm / °) | `1.0 px` p50 (mm) | `±1°` Pitch + `0.5 px` p50 (mm) |
|---|---:|---:|:---:|:---:|---:|---:|---:|
| `QuadSkids` | 0.45 | 84,382 | No | Yes | 0.4 / 0.03° | 1.0 | 0.5 |
| `QuadSkids` | 0.60 | 31,751 | Yes | **No (`NaN` bug)** | 0.8 / 0.04° | 1.7 | 0.9 |
| `QuadSkids` | 1.00 | 6,274 | Yes | Yes | 2.4 / 0.06° | 5.0 | 2.6 |
| `QuadSkids` | 1.50 | 1,810 | Yes | Yes | 5.9 / 0.11° | 10.2 | 5.2 |
| `QuadSkids` | 2.00 | 756 | Yes | Yes | 9.5 / 0.14° | 18.6 | 10.0 |
| `STRL001_Trap` | 0.80 | 49,261 | No (`1.15 m` wide) | **No (`NaN` bug)** | 3.5 / 0.16° | 9.8 | 4.2 |
| `STRL001_Trap` | 1.25 | 11,548 | No (`1.15 m` wide) | **No (`NaN` bug)** | 5.2 / 0.20° | 11.0 | 7.1 |
| `STRL001_Trap` | 1.50 | 6,516 | Yes | Yes | 6.3 / 0.17° | 13.5 | 6.3 |
| `STRL001_Trap` | 2.00 | 2,675 | Yes | Yes | 8.4 / 0.17° | 19.0 | 8.5 |

**Four concrete findings on [`ground_pose.py`](file://src/ugv/ground_pose.py):**
1. **Pitch immunity when all 4 points are visible:** Because `SOLVEPNP_IPPE` estimates full 6-DoF pose, `±1°` pitch jitter adds `< 0.5 mm` error (`2.4 → 2.6 mm` at `1.0 m`).
2. **OpenCV 5.0.0 `SOLVEPNP_IPPE` `NaN` failure:** On exact noiseless projections (`QuadSkids` @ `0.60 m`, `STRL001_Trap` @ `0.80 m` and `1.25 m`), `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)` returns `[nan, nan, nan]` on **both** branches before LM refinement even runs, causing [`SkidPoseObserver.solve`](file://src/ugv/ground_pose.py#L60-L67) to return `None` (`100%` pass under `0.5 px` perturbation!). Fix: add a `cv2.SOLVEPNP_SQPNP` fallback whenever `IPPE` yields non-finite poses.
3. **Hardcoded symmetric quad geometry:** [`SkidPoseObserver.__init__`](file://src/ugv/ground_pose.py#L26-L32) must accept an arbitrary `4×3` (or `N×3`) `object_points` array so `STRL001`'s trapezoidal wheel+skid footprint can be passed cleanly.
4. **Wide-footprint FOV clipping:** Including the `1.15 m` fin-tip skids clips the horizontal FOV for `d < 1.3 m`; `STRL001` should use `[left_main, right_main, nose_wheel, tail_center]` (`0.56 m` max width) or switch from 4-point `SkidPoseObserver` (`d ≥ 0.6 m`) to 2-point `W_gear` homography (`d ∈ [0.43, 1.5 m]`) to single-point/nose-wheel homography (`d < 0.43 m`).

---

## 4. Inverted V-Tail (`38°–45°`) vs. Vertical Fin (`90°`) vs. `15°–25°` Aft-Canted 3D Wedge

### 4.1 Why Flat Fin & Belly Tags Fail on `STRL001`

Inspecting [`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png) (`CAM_Z_M = 0.06445 m`, camera approaching along centerline at `y = -d`):
- **Normal vertical fin (`90°`):**
  - If centered (`x = 0`), normal `(±1, 0, 0)` is strictly orthogonal to approach (`β = 90.0°`), giving **`0.0 px/cell`** and **`0.0%` detection at all ranges**.
  - Even on an inboard face at `x = -0.30 m, z = 0.245 m` facing `+X`, `β = 68.7°–81.5°` across `0.75–2.00 m`, and it exits the `±33.47°` HFOV completely for **`d ≤ 0.50 m`**.
- **Inverted V-tail back fin (`~40°` anhedral, `z = 0.245 m, x = -0.30 m`):**
  - The downward-tilted underside normal `(±sin 40°, 0, -cos 40°)` points **toward the floor**, making it strictly more visible from a low camera than an upward-V fin (whose outer face points away into the sky). However, from an aft centerline approach, its incidence is still **`β = 66.4°–80.6°`** (`2.4 → 0.6 px/cell` at `1.0–2.0 m`, well below [`aruco.py`](file://src/ugv/aruco.py#L403)'s `8 px/cell` warp target) and it leaves the FOV for **`d ≤ 0.50 m`**.
  - Adding `20°` aft sweep improves `β` to `46.8°–60.7°` (`100%` decode at `0.75–2.00 m`), **but** still leaves **`0.0%` detection at `d ≤ 0.50 m`**, and single-tag `IPPE_SQUARE` suffers **`45.0%` yaw flips (`173.4 mm` p50)** at `2.00 m`.
- **Flat under-belly tag (`z = 0.345 m`, normal `(0, 0, -1)`):**
  - Exits the top of the `1280×720` frame (`VFOV/2 = 20.40°`) at **`d_exit = (0.345 - 0.06445) / tan(20.40°) = 0.754 m`**, producing **`0.0%` detection for ALL `d ≤ 0.75 m`**!

### 4.2 Empirical `ArucoFollower.detect_gray` + Photometric DR Results (Experiment 3)

20 photometric-DR trials (`sample_dr` from [`vision_in_loop.py`](file://simulation/gazebo/vision_in_loop.py#L239)) per distance cell with `50 mm` `DICT_4X4_50` tags decoded via the real [`ArucoFollower.detect_gray`](file://src/ugv/aruco.py#L441):

| Mount Configuration | `d = 0.22 m` | `d = 0.30 m` | `d = 0.40 m` | `d = 0.50 m` | `d = 0.75 m` | `d = 1.00 m` | `d = 1.50 m` | `d = 2.00 m` |
|---|---|---|---|---|---|---|---|---|
| `Vertical_Fin_Center_90deg` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `Vertical_Fin_Inner_90deg` | 0% | 0% | 0% | 0% | 100% (`6.8mm`) | 100% (`24.3mm`) | 25% (`71.5mm`) | 0% |
| `InvV_Flat_40deg_NoSweep` | 0% | 0% | 0% | 0% | 100% (`8.7mm`) | 100% (`16.9mm`) | 25% (`27.4mm`) | 0% |
| `InvV_Flat_40deg_Swept20` | 0% | 0% | 0% | 0% | 100% (`11.1mm`) | 100% (`19.1mm`) | 100% (`65.5mm`, 5% flip) | 100% (`173.4mm`, **45% flip**) |
| `Belly_Flat_0.345m` | 0% | 0% | 0% | 0% | 0% | 50% (`41.0mm`, 10% flip) | 85% (`116.3mm`) | 0% |
| `Wedge20_Boom_Low_0.13m` (1-tag IPPE) | 0% (top clip) | 95% (`8.4mm`) | 100% (`8.4mm`) | 100% (`13.1mm`) | 100% (`31.8mm`) | 100% (`45.9mm`) | 100% (`155.6mm`) | 100% (`156.7mm`) |
| **`Wedge20_3Tag_Constellation` (`Board SQPNP`)** | **100% (`5.4mm / 1.03°`)** | **100% (`0.8mm / 0.01°`)** | **100% (`0.7mm / 0.03°`)** | **100% (`2.5mm / 0.19°`)** | **100% (`3.0mm / 0.04°`)** | **100% (`5.1mm / 0.05°`)** | **100% (`4.5mm / 0.06°`)** | **100% (`11.4mm / 0.11°`)** |

**Why the `20°` Aft-Canted 3-Tag 3D Wedge Eliminates the `< 0.45 m` Blind Spot:**
1. **Low mounting height (`z = 0.115–0.140 m`):** Lowering the center aft-canted tag (`ID 22`) onto a rigid bracket under the tail boom / fin apex (`z = 0.115 m`, only `50 mm` above the lens) keeps its upper edge inside `VFOV/2 = 20.40°` down to **`d ≈ 0.21 m`** (`100%` decode at `0.22 m` vs `0%` for `z ≥ 0.13 m` or `z = 0.245–0.345 m`).
2. **Aft-canted `15°–25°` wedge normal:** Aligning the wedge face `(0, -cos 20°, -sin 20°)` holds incidence at **`β = 3.4°–18.1°`** (`cos β ≥ 0.95`) across the entire `0.22–2.00 m` corridor, keeping `px/cell = 3.8–35.1`.
3. **3D Non-Coplanar `cv2.aruco.Board` + `SOLVEPNP_SQPNP`:** Combining `ID 22` (center `z = 0.115 m`) with `ID 20` / `ID 21` (`±0.14 m`, `z = 0.140 m`, canted `±15°` outward + `20°` downward) breaks single-plane IPPE reflection symmetry completely: **0.0% flips across all 160 trials**, **`0.7–5.4 mm` p50** inside `1.5 m`, and **`11.4 mm / 0.11°` at `2.0 m`** (matching Agent 11's independent `1.5 mm` / `0.14°` `Board+SQPNP` finding in [`11_fin_angle_experiment_results.md`](file://docs/council_v2/11_fin_angle_experiment_results.md#L25)).

---

## 5. Fiducial & Keypoint Hybrid Comparison (`DICT_4X4_50` vs. `AprilTag3` vs. `ChArUco` vs. Keypoint CNN)

Experiment 4 ran all OpenCV detectors under identical photometric DR on the `20°` aft-canted wedge (`z = 0.115 m`, 1 core on Ryzen 7 5700X; Pi 4B Cortex-A72 is ~`3.5×` slower per [`10_pi_edge_vs_3090_compute_budget.md`](file://docs/council_v2/10_pi_edge_vs_3090_compute_budget.md#L85)):

| Detector (on `20°` low wedge) | `0.18 m` | `0.22 m` | `0.28 m` | `0.35 m` | `0.50 m` | `1.00 m` | `1.50 m` | Host ms (Pi proj.) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`DICT_4X4_50` (`SUBPIX`)** | 0% | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **2.1–3.8 ms** (~7–13 ms) | **Best primary fiducial.** Fast, fits 30 Hz, full range `0.22–2.0 m`. |
| `APRILTAG_36h11` (`APRILTAG` refine) | 0% | 100% | 100% | 100% | 100% | 100% | 100% | **28.4–152.9 ms** (**100–535 ms**) | **Do NOT use on Pi 4B** — `CORNER_REFINE_APRILTAG` blows 30 Hz budget 4×–15×. |
| `APRILTAG_36h11` (`SUBPIX`) | 0% | 100% | 100% | 100% | 100% | 100% | 100% | **2.2–4.0 ms** (~8–14 ms) | Usable if higher bit-distance needed, ~15% slower than `4X4_50` and smaller cell pitch (`8×8` vs `6×6`). |
| `ChArUco 3×3` (`75×75 mm`, `18 mm` tags) | 0% | 0% | **100%** (`7.1mm`) | **100%** (`10.4mm`) | **100%** (`16.5mm`) | 86.7% (`420.7mm`) | 0% | **2.6–4.2 ms** (~9–15 ms) | **Only for mid/close `0.28–0.6 m`**; `18 mm` inner markers vanish at `1.5 m` and `75 mm` board clips at `0.22 m`. |
| **Floor Homography (`z = 0`, 2-pt `W_gear`)** | **100%*** | **100%*** | **100%*** | **100%*** | **100%** (`1.1mm`) | **100%** (`4.3mm`) | **100%** (`9.6mm`) | **`< 0.02 ms`** (solver) | **Essential `< 0.22 m` & anti-spoof anchor** (*single-wheel + IMU below `0.425 m`). |
| **Keypoint CNN (`160×120` FP32)** | **~100%** | **~100%** | **~100%** | **~100%** | **~100%** | **~100%** | **~100%** | **1.4–1.5 ms** ([**5.2 ms**](file://docs/council_v2/10_pi_edge_vs_3090_compute_budget.md#L19)) | **Sensor-diverse fallback** for `< 0.22 m`, tag occlusion, & wheel-contact localization ([`12_belly_vision_experiment_results.md`](file://docs/council_v2/12_belly_vision_experiment_results.md#L19)). |

**Why the Multi-Plane Fiducial + Ground-Plane Homography + Keypoint Hybrid wins:**
1. **Far-to-near (`2.0 m → 0.22 m`):** The `20°` aft-canted 3-tag `DICT_4X4_50` wedge (`ID 20, 21, 22`) solved jointly with `cv2.aruco.Board` + `SOLVEPNP_SQPNP` delivers **`100%` detection, `0%` flips, and `0.7–11.4 mm` accuracy** without needing any neural network in the happy path.
2. **Ultra-close (`< 0.22 m` or occluded tags):** Even when the low wedge exits the top FOV inside `0.21 m` (or gets occluded by props/wires/shadow), **floor-contact wheel/skid pixels (`z = 0`) remain in the lower frame down to `0.173 m`**. Feeding `left_wheel_contact`, `right_wheel_contact`, and `nose_wheel` from the `160×120` keypoint net (or cheap ROI wheel-bottom edge tracker seeded by the last tag fix) into the **2-point `W_gear` pitch-compensated homography** bridges the final `22 cm` with **`~1 mm` geometric precision**.
3. **Zero false-target lock:** Pairwise gating requires **both** semantic/ID identity (`ID ∈ {20, 21, 22}`) **and** physical `STRL001` geometry: `|d_LR - W_gear| ≤ 30 mm` (`0.56 m`), positive signed triangle chirality `det([p_L - p_tail, p_R - p_tail]) > 0` ([`09_lr_disambiguation_and_belly_shield.md`](file://docs/council_v2/09_lr_disambiguation_and_belly_shield.md#L98-L100)), and IMU heading agreement (`≤ 8°`).

---

## 6. Actionable Codebase Integration Checklist (When Moving Beyond Analysis)

1. **Replace single-tag `IPPE_SQUARE` with `cv2.aruco.Board` + `SOLVEPNP_SQPNP`** in [`aruco.py`](file://src/ugv/aruco.py#L352-L358) and [`parking.py`](file://src/ugv/parking.py#L604-L606) whenever $\ge 2$ tags of the `STRL001` constellation are co-visible; add `np.isfinite(translation).all()` to [`ArucoFollower.from_points`](file://src/ugv/aruco.py#L359).
2. **Fix [`autopark.py`](file://src/ugv/autopark.py#L31) `CROP_Y1 = 440`:** Floor wheel contacts sit at `v = 422–610 px` for `0.25–1.0 m` (`v > 440` inside `0.78 m`), so any floor-contact or low-skid tracker must operate on the lower ROI (`v ∈ [360, 720]`), not the top `440` crop.
3. **Upgrade & wire [`ground_pose.py`](file://src/ugv/ground_pose.py#L25-L71):**
   - Parameterize `SkidPoseObserver(..., object_points=...)` to support `STRL001` wheel/skid footprints;
   - Add a `SOLVEPNP_SQPNP` fallback when `SOLVEPNP_IPPE` returns `NaN` on exact symmetric views;
   - Add the **2-point `W_gear` pitch self-calibrating homography** as the lightweight `< 0.02 ms` floor-contact mode when only 2 wheels are visible (`0.43–1.50 m`), falling back to IMU-pitch-compensated single-contact homography inside `0.43 m`.
4. **Mount hardware per verified geometry:** Place `ID 22` (`50 mm` `DICT_4X4_50`) on a **`20°` aft-canted rigid bracket at `z ≈ 0.115 m`** under the tail boom/apex, flanked by `ID 20` (left) and `ID 21` (right) on `20°` aft-canted V-tail brackets at `z ≈ 0.140 m, x = ±0.14 m` with `≥ 5 mm` matte quiet zones, and calibrate `K` + radial distortion (`config/parking.json` currently has `camera_intrinsics_confirmed: false`, `distortion: [0,0,0,0,0]`).
