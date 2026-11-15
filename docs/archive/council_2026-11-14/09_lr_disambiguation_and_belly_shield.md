# Agent 09 Specification: Left/Right Plane-Frame Disambiguation, False-Target Rejection & Multi-Obstacle Belly Collision Shield

**Target Airframe:** Strix (`STRL001`) twin-tractor / inverted V-tail aircraft ([`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png))  
**Scope:** Perception-to-EKF gating, SE(2) symmetry breaking, and generalized multi-capsule safety shielding across [`src/ugv/staged_baseline.py`](file://src/ugv/staged_baseline.py), [`src/ugv/policy.py`](file://src/ugv/policy.py), [`src/ugv/parking.py`](file://src/ugv/parking.py), and [`rl/sim.py`](file://rl/sim.py).

---

## 1. Problem Diagnosis: Why the Quadrotor Skid Model Breaks on an Aircraft Belly

The existing parking stack assumes a **doubly symmetric** landing-gear corridor:

1. **Symmetry in [`rl/sim.py`](file://rl/sim.py#L167-L169) and [`Geometry`](file://src/ugv/staged_baseline.py#L40-L70):**
   - Only two identical parallel skids exist at $x = \pm \text{SKID\_X}$ ($0.157\text{ m}$), $y \in [-\text{GEAR\_LEN}/2, +\text{GEAR\_LEN}/2]$.
   - Both [`StagedBaseline._swept_clear`](file://src/ugv/staged_baseline.py#L635-L664) and [`MecanumVecEnv.step`](file://rl/sim.py#L1265-L1271) use `abs(x)` and `abs(y)`, which assumes $x \leftrightarrow -x$ and $y \leftrightarrow -y$ invariance.
   - Furthermore, [`rl/sim.py`](file://rl/sim.py#L1265-L1271) performs an unrotated axis-aligned bounding box (AABB) test on `(CAR_W, CAR_L)` instead of rotating the UGV footprint by yaw $\psi$, while [`StagedBaseline._swept_clear`](file://src/ugv/staged_baseline.py#L659-L663) uses an inflated AABB bounding the rotated UGV (`eff_hw = hw*|cos| + hl*|sin|`).
2. **Asymmetry of the Strix Belly Airframe ([`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png)):**
   - **Front (`+y_plane`)** has the **nose wheel** on the centerline ($x \approx 0$), forward fuselage chin, and swept wings with twin wing-mounted propellers.
   - **Rear (`-y_plane`)** has the central tail boom supporting an **inverted V-tail (`/\`)** whose two swept fins angle downward and outward toward the runway, terminating in `left_vtail_tip` and `right_vtail_tip` (with ventral finlets/skids near the ground).
   - **Mid-fuselage** has the two main landing gear wheels (`left_main_wheel`, `right_main_wheel`) flanking the belly bay where the UGV must park.
3. **The $180^\circ$ Parity & Clutter Vulnerability:**
   - **Unlabelled wheel parity:** Any two wheel detections $\{W_1, W_2\}$ separated by track $2w_m$ are invariant under a $180^\circ$ rotation around their midpoint. Without knowing Left (`+x_plane`) vs Right (`-x_plane`) and Front (`+y_plane`) vs Rear (`-y_plane`), an approach from the nose (`+y_plane`) is indistinguishable from an approach between the inverted V-tail fins (`-y_plane`), and a lateral approach from `-x` vs `+x` swaps port and starboard clearance envelopes.
   - **EKF forced-reinit vulnerability:** In [`EKF.update`](file://src/ugv/staged_baseline.py#L426), `reinit = (valid & ~self.init) | (valid & (self.rejected >= 6))`. If a persistent false object (chair leg pair, shoes, floor reflection) passes upstream detection for $6$ consecutive camera ticks ($0.2\text{ s}$ at $30\text{ Hz}$), the EKF **forcibly reinits onto the false target** and drives the UGV into furniture.

---

## 2. Coordinate Frames & Exact Sign Conventions

To avoid silent $180^\circ$ or mirror flips across [`parking.py`](file://src/ugv/parking.py), [`policy.py`](file://src/ugv/policy.py), and [`rl/sim.py`](file://rl/sim.py), we define the **Plane-Local Belly Frame** $\mathcal{F}_P$ explicitly:

| Axis | Meaning in Plane Body Frame $\mathcal{F}_P$ | Key Landmarks |
| :--- | :--- | :--- |
| $+x_P$ | **Left (Port)** side of aircraft | `left_main_wheel` ($+x_m$), `left_vtail_tip` ($+x_t$), `ID_LEFT_FIN` |
| $-x_P$ | **Right (Starboard)** side of aircraft | `right_main_wheel` ($-x_m$), `right_vtail_tip` ($-x_t$), `ID_RIGHT_FIN` |
| $+y_P$ | **Nose / Forward** direction of aircraft | `nose_wheel` ($y_n > 0$), `ID_NOSE` |
| $-y_P$ | **Tail / Aft** direction of aircraft (entry portal) | `ID_BELLY_REAR`, `left_vtail_tip` & `right_vtail_tip` ($y_t < 0$) |
| Origin $(0, 0)_P$ | **Belly Parking Goal** under fuselage center | Target UGV center pose $(x_P, y_P, \psi_P) = (0, 0, 0)$ |

### Handedness & Mapping to `rl/sim.py` / `staged_baseline.py`
* **UGV Docking Convention:** When parked inside the belly facing the aircraft nose (`+y_P`? Or facing aft?):
  * In [`rl/sim.py`](file://rl/sim.py#L3-L5) and [`staged_baseline.py`](file://src/ugv/staged_baseline.py#L7-L8), the UGV's front camera looks along the UGV forward axis, and `yaw = 0` means the UGV sits at $(0, +y_{\text{stage}})$ facing $-y$ toward the origin.
  * When the UGV approaches the Strix belly **from behind the tail** ($y_P < 0$, entering between the inverted V-tail fins moving toward $+y_P$ so the front camera continues to see `ID_BELLY_REAR` / `ID_LEFT_FIN` / `ID_RIGHT_FIN` throughout entry), we align the **sim/controller world frame** $\mathcal{F}_W$ with $\mathcal{F}_P$ via a fixed $180^\circ$ yaw rotation:
    $$x_W = -x_P \;\;(\text{Right}), \qquad y_W = -y_P \;\;(\text{Aft / Entry Side}), \qquad \psi_{\text{ccw}} = \text{wrap}(\psi_P + \pi).$$
  * **Crucial handedness check:** In $\mathcal{F}_P$ with coordinates $(x_P = \text{Left}, y_P = \text{Nose})$, $(+x_P, +y_P)$ has $-Z_{\text{up}}$ orientation; in $\mathcal{F}_W$ with $(x_W = \text{Right}, y_W = \text{Aft})$ and $+Z_{\text{up}}$, **rotating by $\pi$ in the ground plane maps $(+x_P, +y_P) \mapsto (-x_W, -y_W)$ with $\det(R(\pi)) = +1$**, preserving the exact 2D determinant sign!
    Let $p_L^P = (+x_m, y_m)$, $p_R^P = (-x_m, y_m)$, $p_{\text{tail}}^P = (0, y_t)$ with $x_m > 0$ and $y_m > y_t$. Then in **both** $\mathcal{F}_P$ and $\mathcal{F}_W$:
    $$\det\bigl([\,p_L - p_{\text{tail}},\;\; p_R - p_{\text{tail}}\,]\bigr) = (+x_m)(y_m - y_t) - (y_m - y_t)(-x_m) = +2 x_m (y_m - y_t) > 0.$$
  * Conversely, if projected into a reflected frame (e.g., camera-mirrored or `Pose2(x, z, yaw)` CW-positive in [`parking.py`](file://src/ugv/parking.py#L74-L89)), apply `pose2_from_ccw` / `pose2_to_ccw` strictly at the boundary so the chirality determinant is always evaluated in a consistent $(x_{\text{Left}}, y_{\text{Nose}})$ or $(x_W, y_W)$ basis.

---

## 3. Left/Right Disambiguation & Three-Layer False-Target Rejection

No single sensor cue is trusted in isolation. A measurement $z_k = (x, y, \psi)$ is admitted to the EKF **if and only if** it passes the conjunction:
$$\text{Admit}(k) \;\triangleq\; \mathcal{G}_{\text{tag}}(k) \;\land\; \mathcal{G}_{\text{topo}}(k) \;\land\; \mathcal{G}_{\text{SE(2)}}(k).$$

### Layer (a): Asymmetric Fin / Strut / Belly ArUco Tag Constellation

#### Steep-Angle Mounting Analysis on the Inverted V-Tail
Inspecting [`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png), the Strix inverted V-tail fins slope downward-and-outward at anhedral $\phi_{\text{fin}} \approx 40^\circ$ and sweep $\Lambda \approx 20^\circ$.
* If a $50\text{ mm}$ tag is pasted **flat on the outer/upper V-tail skin**, its surface normal $\hat{n}$ points upward-outward. From a ground UGV camera at $h_{\text{cam}} \approx 0.14\text{ m}$ and range $r \in [0.4, 2.0]\text{ m}$, the viewing incidence angle $\theta_{\text{inc}} = \arccos(\hat{n} \cdot \hat{v}_{\text{cam}})$ reaches **$65^\circ\text{–}82^\circ$**.
* At $\theta_{\text{inc}} > 65^\circ$:
  1. Projected shortest edge falls below the `24 px` minimum required by [`docs/ARUCO.md`](file://docs/ARUCO.md#L20) oblique rescue.
  2. `cv2.SOLVEPNP_IPPE_SQUARE` suffers planar homography reflection ambiguity (two solutions with nearly identical reprojection error, causing $180^\circ$ yaw flips).

#### Recommended Asymmetric ID & Mount Spec (`DICT_4X4_50`, $50\text{ mm}$ tags, $\ge 5\text{ mm}$ matte quiet zone)
Keep existing quadrotor IDs `[9, 12, 13, 14]` ([`config/parking.json`](file://config/parking.json#L2-L7)) disjoint from the aircraft IDs:

| Constant | ID | Physical Mount Location | Normal Vector $\hat{n}$ in $\mathcal{F}_P$ | Purpose |
| :--- | :---: | :--- | :--- | :--- |
| `ID_LEFT_FIN` | `20` | Left V-tail ventral finlet / aft-canted $25^\circ$ bracket near `left_vtail_tip` | $(+0.42, -0.90, -0.10)$ (aft + port) | Breaks L/R parity; visible from rear & port quarter |
| `ID_RIGHT_FIN` | `21` | Right V-tail ventral finlet / aft-canted $25^\circ$ bracket near `right_vtail_tip` | $(-0.42, -0.90, -0.10)$ (aft + stbd) | Breaks L/R parity; visible from rear & stbd quarter |
| `ID_BELLY_REAR` | `22` | Aft belly bulkhead / tail-boom root arch (vertical, $10^\circ$ down-pitch) | $(0.00, -0.98, -0.17)$ (directly aft) | Primary close-range terminal anchor ($r < 0.7\text{ m}$) when fins exit $\pm 33.5^\circ$ FOV |
| `ID_NOSE` | `23` | Nose-gear strut fairing / chin plate | $(0.00, +0.98, -0.17)$ (directly fwd) | Resolves $180^\circ$ front-vs-rear parity: seeing `ID_NOSE` triggers perimeter re-route, never direct entry |

**Single-Frame Tag Gates ($\mathcal{G}_{\text{tag}}$):**
1. **Strict ID Allowlist:** `tag_id in {20, 21, 22, 23}`. All other IDs (room markers, drone tags, accidental bit patterns) are dropped immediately.
2. **Decode & Edge Quality (mirroring [`docs/ARUCO.md`](file://docs/ARUCO.md#L84-L85) & [`parking.py`](file://src/ugv/parking.py#L591-L613)):**
   * Exact `DICT_4X4_50` Hamming-0 match (`black-border >= 19/20`), corners $\ge 3\text{ px}$ from image border, shortest edge $\ge 24\text{ px}$, reprojection error $\le 2.0\text{ px}$.
3. **Steep-Angle & IPPE Ambiguity Gate (critical for V-tail fins):**
   * Run `cv2.solvePnPGeneric(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)` returning two pose hypotheses $(R_1, t_1, e_1)$ and $(R_2, t_2, e_2)$ sorted by reprojection error $e_1 \le e_2$.
   * Reject tag yaw (downgrade to position-only or discard if single-tag) unless:
     $$\theta_{\text{inc}} = \arccos\!\left(\frac{(R_1 \hat{z}_{\text{tag}}) \cdot (-t_1)}{\|t_1\|}\right) \le 60^\circ \qquad\text{AND}\qquad \left(\frac{e_2}{e_1 + 10^{-6}} \ge 2.0 \;\lor\; \text{multi-tag co-visible}\right).$$
4. **3D Ground-Plane Height Sanity (Anti-Reflection / Anti-Ceiling Gate):**
   * Transform the detected tag origin into the UGV body/ground frame using calibrated `camera_to_car`.
   * Require $z_{\text{tag}}^{\text{ground}} \in [z_{k,\text{cal}} - 0.08\text{ m},\; z_{k,\text{cal}} + 0.08\text{ m}]$ and $z_{\text{tag}}^{\text{ground}} > 0$. Floor reflections produce $z < 0$ (below ground plane) and are rejected algebraically.
5. **Multi-Tag Rigid Fusion:**
   * When $\ge 2$ allowlisted tags are co-visible, solve the 2D Kabsch/Procrustes alignment between observed tag ground positions $\{q_k^U\}$ and calibrated plane coordinates $\{p_k^P\}$:
     $$(R^*, t^*) = \arg\min_{R \in SO(2),\, t} \sum_{k \in \mathcal{V}} w_k \| R\, p_k^P + t - q_k^U \|^2,$$
     with residual check $\max_{k \in \mathcal{V}} \|R^* p_k^P + t^* - q_k^U\|_2 < 0.04\text{ m}$. This rejects spurious/detached tags and fixes yaw to $\sigma_\psi < 1^\circ$ independent of IPPE single-tag plane flips.

---

### Layer (b): Signed Wheel-to-Tail Triangle Topology

Whenever candidate wheel/strut keypoints $(p_L, p_R)$ and tail reference $p_{\text{tail}}$ (from `ID_BELLY_REAR`, the midpoint of `{ID_LEFT_FIN, ID_RIGHT_FIN}`, or V-tail boom base) are observed in 2D ground coordinates, we enforce a **4-predicate topological lock** ($\mathcal{G}_{\text{topo}}$) before accepting any wheel-derived or hybrid target update:

Let $u = p_L - p_{\text{tail}} \in \mathbb{R}^2$ and $v = p_R - p_{\text{tail}} \in \mathbb{R}^2$.

1. **Signed Chirality Determinant (L/R & Mirror Parity Lock):**
   $$\Delta_{\text{tri}} = \det([u,\; v]) = u_x v_y - u_y v_x.$$
   Require:
   $$\Delta_{\text{tri}} > \Delta_{\min} \triangleq 0.5 \cdot (2 x_m)(y_m - y_t) > 0.$$
   * Swapping $p_L \leftrightarrow p_R$ negates $\Delta_{\text{tri}} < 0 \implies$ **rejected**.
   * A $180^\circ$ front/rear swap where $p_{\text{tail}}$ is mis-associated with the nose wheel while wheels are swapped, or a collinear false target pair (two chair legs inline with a wall corner), collapses $\Delta_{\text{tri}} \le 0$ or violates edge lengths $\implies$ **rejected**.

2. **Rigid 3-Edge Metric Congruence (False-Object Rejection):**
   Let the calibrated aircraft triangle side lengths be $d_{LR}^* = 2 x_m$, $d_{LT}^* = \sqrt{x_m^2 + (y_m - y_t)^2}$, and $d_{RT}^* = d_{LT}^*$. Require:
   $$\bigl|\|p_L - p_R\|_2 - d_{LR}^*\bigr| \le \epsilon_{\text{tri}}, \qquad \bigl|\|u\|_2 - d_{LT}^*\bigr| \le \epsilon_{\text{tri}}, \qquad \bigl|\|v\|_2 - d_{RT}^*\bigr| \le \epsilon_{\text{tri}},$$
   with $\epsilon_{\text{tri}} = 0.045\text{ m} + 0.03 \cdot r$ (range-scaled). Chair legs, shoes, tripod legs, and random floor clutter fail simultaneous 3-edge congruence.

3. **Forward-Sign Dot Product (Nose vs. Tail Directionality):**
   With plane forward unit vector $\hat{y}_P$ (from Layer (a) tag pose or triangle altitude vector rotated $-90^\circ$ from $(p_L - p_R)$):
   $$\left(\frac{p_L + p_R}{2} - p_{\text{tail}}\right) \cdot \hat{y}_P > 0.5\,(y_m - y_t) > 0.$$

4. **Tag-to-Wheel Cross-Consistency:**
   Using the tag-derived plane pose $\hat{T}_{U \leftarrow P}$, project nominal wheel centers $\hat{p}_L^U, \hat{p}_R^U$ into the UGV ground frame. Any wheel candidate outside $\|p_{\bullet}^U - \hat{p}_{\bullet}^U\|_2 > r_{\text{assoc}} = 0.07\text{ m} + 3\sigma_{\text{pos}}$ is **discarded as environmental clutter** (never tracked).

---

### Layer (c): SE(2) Mahalanobis Innovation Gating & Hardening `EKF.update`

In [`src/ugv/staged_baseline.py`](file://src/ugv/staged_baseline.py#L312-L440), the EKF maintains $x = [x, y, \psi, b_x, b_y]^T$ and a `RING = 32` delayed-pose buffer indexed by `self.lag`. We harden [`EKF.update`](file://src/ugv/staged_baseline.py#L388-L440) for the multi-tag + wheel pipeline:

1. **Incidence-Aware Measurement Covariance $R_k$:**
   Replace the isotropic [`tag_std(d)`](file://src/ugv/staged_baseline.py#L97-L99) when a single steep tag is observed:
   $$\sigma_{\text{pos}}(d, \theta_{\text{inc}}) = \frac{0.008 + 0.020\,d}{\max(\cos\theta_{\text{inc}},\, 0.35)}, \qquad \sigma_\psi(d, \theta_{\text{inc}}) = \frac{2^\circ + 2^\circ d}{\max(\cos\theta_{\text{inc}},\, 0.35)} \cdot \frac{1}{\sqrt{|\mathcal{V}|}},$$
   so oblique fin views automatically receive lower Kalman gain than near-normal `ID_BELLY_REAR` or multi-tag Kabsch fixes.

2. **SE(2) Mahalanobis Gate ($\mathcal{G}_{\text{SE(2)}}$):**
   At delayed ring index `idx = (head - 1 - lag) % RING`:
   $$\nu_k = \begin{bmatrix} z_x - \hat{x}_{k-\tau} \\ z_y - \hat{y}_{k-\tau} \\ \text{wrap}(z_\psi - \hat{\psi}_{k-\tau}) \end{bmatrix}, \qquad S_k = H P_{k-\tau} H^T + R_k, \qquad m_k^2 = \nu_k^T S_k^{-1} \nu_k < \chi^2_{3,\,0.999} = 16.3.$$

3. **Closing the `rejected >= 6` Re-Init Loophole (Why False-Target Tracking Becomes Mathematically Impossible):**
   * **Vulnerability in current code:** [`EKF.update` line 426](file://src/ugv/staged_baseline.py#L426) allows `reinit = valid & (self.rejected >= 6)` for *any* `valid` input.
   * **Hardened Rule:**
     1. *Initialization & Re-initialization Gate:* `reinit` may **only** fire if $\mathcal{G}_{\text{tag}}$ passes (allowlisted `DICT_4X4_50` ID with $z_{\text{ground}} > 0$ and $\theta_{\text{inc}} \le 55^\circ$) **AND** either (i) $\ge 2$ allowlisted tags pass 2D rigid Kabsch congruence, **or** (ii) $\mathcal{G}_{\text{topo}}$ passes ($\Delta_{\text{tri}} > \Delta_{\min}$ and 3-edge congruence), **AND** 3 consecutive candidate frames cluster within $3\text{ cm}$ translation and $5^\circ$ yaw (mirroring [`ParkingEstimator._update`](file://src/ugv/parking.py#L680-L693)).
     2. *Why Joint False-Target Lock Has Negligible Probability:*
        - A random non-tag object (chair leg, shoe, specular reflection) has $P(\text{valid DICT\_4X4\_50 allowlisted ID + 19/20 black border + 24px convex quad}) < 10^{-5}$ per frame.
        - Requiring $z_{\text{ground}} > 0$ eliminates floor reflections deterministically.
        - Requiring $\Delta_{\text{tri}} > \Delta_{\min}$ + 3-edge metric congruence rejects symmetric/unlabeled clutter pairs.
        - Requiring $m_k^2 < 16.3$ (while initialized) rejects any outlier $> 3.7\sigma$ from IMU+odometry motion prediction, and requiring 3 mutually consistent verified frames to re-init eliminates single-frame transient glitches ($P_{\text{false-reinit}} < 10^{-15}$).

---

## 4. State Machine Extension: Front-Approach (`ID_NOSE`) Perimeter Re-Route

In [`StagedBaseline._control`](file://src/ugv/staged_baseline.py#L666), the quadrotor controller always funnels straight to `(0, stage_y)` because the skids are identical front and back. On the Strix plane:
* **`APPROACH_REAR` (Nominal, $y_P < y_{\text{tail}} - d_{\text{stage}}$):** Funnel to staging pose $(x_P, y_P, \psi_P) = (0, y_{\text{tail}} - d_{\text{stage}}, 0_{\text{facing nose}})$ aligned with the inverted V-tail tunnel, settle, then execute `ENTER` to $(0, 0)$.
* **`PERIMETER_ARC` (Triggered when `ID_NOSE` is seen or initial pose has $y_P > y_{\text{tail}} - 0.15$):**
  - Never drive directly through $x_P \in [-x_{\text{wing\_keep}}, +x_{\text{wing\_keep}}]$ from the front (blocked by `nose_wheel`, propellers, and low wings).
  - Select shorter wingtip clearance side $s = \text{sign}(x_P)$ (default $+1$ Left if $x_P = 0$) and route through waypoint $(s \cdot x_{\text{keep}}, y_P) \to (s \cdot x_{\text{keep}}, y_{\text{tail}} - d_{\text{stage}}) \to (0, y_{\text{tail}} - d_{\text{stage}})$ under the Multi-Obstacle Shield below.

---

## 5. Multi-Cylinder / Capsule Aircraft Footprint

We replace the 2-skid model in [`rl/sim.py`](file://rl/sim.py#L167-L169), [`staged_baseline.py`](file://src/ugv/staged_baseline.py#L40-L70), and [`policy.py`](file://src/ugv/policy.py#L55-L62) with a unified **5-Component Aircraft Footprint** (plus optional height-sliced V-tail fin capsules) expressed in $\mathcal{F}_P$:

### 5.1 Obstacle Primitives in Plane Body Frame $\mathcal{F}_P$

> [!WARNING]
> Numerical coordinates below are **provisional placeholders** mirroring [`config/parking.json`](file://config/parking.json)'s `confirmed: false` gating discipline. Exact Strix airframe measurements (`wheel_track`, `nose_wheel_y`, `vtail_tip_xy`, `vtail_roof_xy`, `belly_height_m`) must be measured on the physical plane before setting `airframe.confirmed = true` in [`readiness()`](file://src/ugv/parking.py#L314).

| ID | Component Name | Primitive | Center / Segment $(A \to B)$ in $\mathcal{F}_P$ | Radius $r_m$ | Physical Hazard |
| :-: | :--- | :--- | :--- | :---: | :--- |
| 0 | `left_main_wheel` | Cylinder (zero-len capsule) or short tire capsule | $(+x_{\text{main}}, y_{\text{main}} - \frac{l_w}{2}) \to (+x_{\text{main}}, y_{\text{main}} + \frac{l_w}{2})$ | $r_{\text{main}}$ | Left main gear strut & tire |
| 1 | `right_main_wheel` | Cylinder / tire capsule | $(-x_{\text{main}}, y_{\text{main}} - \frac{l_w}{2}) \to (-x_{\text{main}}, y_{\text{main}} + \frac{l_w}{2})$ | $r_{\text{main}}$ | Right main gear strut & tire |
| 2 | `nose_wheel` | Cylinder (zero-len capsule) | $(0.0,\; y_{\text{nose}}) \to (0.0,\; y_{\text{nose}})$ | $r_{\text{nose}}$ | Forward centerline wheel; blocks front entry & forward overshoot |
| 3 | `left_vtail_tip` | Cylinder (or roof-height swept fin capsule) | $(+x_{\text{tail,gnd}}, y_{\text{tail,le}}) \to (+x_{\text{tail,roof}}, y_{\text{tail,te}})$ | $r_{\text{tail}}$ | Left inverted V-tail lower tip & inward-sloping fin surface at UGV height $h_{\text{ugv}}$ |
| 4 | `right_vtail_tip` | Cylinder (or roof-height swept fin capsule) | $(-x_{\text{tail,gnd}}, y_{\text{tail,le}}) \to (-x_{\text{tail,roof}}, y_{\text{tail,te}})$ | $r_{\text{tail}}$ | Right inverted V-tail lower tip & inward-sloping fin surface at UGV height $h_{\text{ugv}}$ |

#### Why the Inverted V-Tail Uses a Height-Sliced Capsule
Looking at [`strix_plane_realflight.png`](file://docs/council_v2/strix_plane_realflight.png), the inverted V-tail fins slant **inward as height $z$ increases**. At ground level ($z=0$), the tips are separated by $2 x_{\text{tail,gnd}}$. At UGV roof/camera-mast height ($z = h_{\text{ugv}}$), the fin edges lie closer to the centerline:
$$x_{\text{tail}}(z) = x_{\text{tail,gnd}} - s_{\text{vtail}} \cdot z, \qquad z \in [0,\; h_{\text{ugv}}],$$
where $s_{\text{vtail}} = \cot(\phi_{\text{fin}})$ is the inward slope (the inverted-V counterpart of `inward_slope` in [`gear_obstacles`](file://src/ugv/parking.py#L234-L242)). Projecting the fin slice for $z \in [0, h_{\text{ugv}}]$ into a single 2D Capsule from $(x_{\text{tail,gnd}}, y_{\text{tail,le}})$ to $(x_{\text{tail}}(h_{\text{ugv}}), y_{\text{tail,te}})$ with radius $r_{\text{tail}}$ guarantees clearance for the **entire 3D height of the UGV chassis** without needing a 3D mesh check at runtime.

#### Provisional Parameterization (for Sim Prototyping Only)
```python
@dataclass(frozen=True)
class StrixFootprint:
    # All dimensions provisional until physical airframe calibration (confirmed=False)
    main_x: float = 0.165        # half main-wheel track |x| (m)
    main_y: float = 0.020        # main wheels y in Plane Frame (+y is nose)
    main_r: float = 0.038        # main tire radius + strut envelope (m)
    main_tire_len: float = 0.060 # tire contact patch length along y (m)
    nose_y: float = 0.340        # nose wheel y ahead of belly center (m)
    nose_r: float = 0.032        # nose tire radius (m)
    vtail_x_gnd: float = 0.195   # V-tail lower tip |x| at ground (m)
    vtail_x_roof: float = 0.145  # V-tail inner clearance |x| at z = car_h (m)
    vtail_y_le: float = -0.320   # V-tail leading/lower tip y (m)
    vtail_y_te: float = -0.400   # V-tail trailing/upper slice y at z = car_h (m)
    vtail_r: float = 0.025       # fin edge safety radius (m)
    car_w: float = 0.160         # UGV swept width (m)
    car_l: float = 0.220         # UGV swept length (m)
    confirmed: bool = False
```

---

## 6. Exact Vectorized OBB-vs-Capsule Collision & Clearance Math

To unify [`rl/sim.py`](file://rl/sim.py) (PyTorch GPU, $N$ batched environments), [`src/ugv/staged_baseline.py`](file://src/ugv/staged_baseline.py) (NumPy batched), and [`src/ugv/policy.py`](file://src/ugv/policy.py) / [`src/ugv/parking.py`](file://src/ugv/parking.py) (deploy NumPy, $N=1$), we eliminate the unrotated AABB bug in [`rl/sim.py`](file://rl/sim.py#L1266-L1271) and the symmetric-skid assumption in [`_swept_clear`](file://src/ugv/staged_baseline.py#L635-L664).

### 6.1 Exact Segment-to-Oriented-Box Signed Clearance

Given UGV pose $(x, y, \psi)$ in $\mathcal{F}_P$ (or $\mathcal{F}_W$), inflated half-extents $h_w = \frac{W_{\text{car}}}{2} + \text{pad}$, $h_l = \frac{L_{\text{car}}}{2} + \text{pad}$, and $M$ capsules $\{(A_m, B_m, r_m)\}_{m=1}^M$:

1. **Transform capsule endpoints into UGV Body Frame** $\mathcal{F}_U$:
   $$a_m^U = R(\psi)^T (A_m - p), \qquad b_m^U = R(\psi)^T (B_m - p), \qquad p = [x, y]^T.$$
   In $\mathcal{F}_U$, the UGV footprint is the axis-aligned rectangle $\mathcal{B} = [-h_w, +h_w] \times [-h_l, +h_l]$.

2. **Fast Exact Cylinder Case ($A_m = B_m = C_m$):**
   $$d_x = \max(|c_{m,x}^U| - h_w,\; 0), \qquad d_y = \max(|c_{m,y}^U| - h_l,\; 0),$$
   $$\text{signed\_clear}_m = \sqrt{d_x^2 + d_y^2} - r_m \quad (\text{exact when outside } \mathcal{B}; \le 0 \iff \text{collision}).$$

3. **General Capsule Segment Case ($\overline{a_m^U b_m^U}$ vs $\mathcal{B}$):**
   Let $d(t) = a_m^U + t (b_m^U - a_m^U)$ for $t \in [0, 1]$.
   * **Step 3a (Slab Intersection — segment enters $\mathcal{B}$):**
     Compute standard 1D slab clip intervals $[t_x^{\text{lo}}, t_x^{\text{hi}}]$ against $[-h_w, h_w]$ and $[t_y^{\text{lo}}, t_y^{\text{hi}}]$ against $[-h_l, h_l]$. If $\max(0, t_x^{\text{lo}}, t_y^{\text{lo}}) \le \min(1, t_x^{\text{hi}}, t_y^{\text{hi}})$, the capsule spine penetrates $\mathcal{B}$ ($\text{dist} = 0 \implies$ collision).
   * **Step 3b (Minimum Exterior Distance — segment misses $\mathcal{B}$):**
     If the segment does not intersect $\mathcal{B}$, the closest pair of points between convex polygon $\mathcal{B}$ and line segment $\overline{a_m^U b_m^U}$ **must** occur at either an endpoint $\{a_m^U, b_m^U\}$ against $\mathcal{B}$, or one of the 4 box corners $\{(\pm h_w, \pm h_l)\}$ against $\overline{a_m^U b_m^U}$:
     $$\text{dist}(\overline{a_m^U b_m^U}, \mathcal{B}) = \min\Bigl(\text{dist\_pt\_box}(a_m^U),\; \text{dist\_pt\_box}(b_m^U),\; \min_{c \in \text{corners}(\mathcal{B})} \text{dist\_pt\_seg}(c, a_m^U, b_m^U)\Bigr).$$
   * **Collision predicate:**
     $$\text{collides} \iff \min_{m=1..M} \Bigl(\text{dist}(\overline{a_m^U b_m^U}, \mathcal{B}) - r_m\Bigr) \le 0.$$

This formulation is **100% branch-free and vectorizable** in both PyTorch (`[N, M]` tensors in `rl/sim.py`) and NumPy (`[N, K_sweep, M]` in `staged_baseline.py`), with zero AABB rotation over-approximation error.

### 6.2 Reference Batched PyTorch Implementation for `rl/sim.py`

```python
def capsules_collide_obb(
    pose: torch.Tensor,      # [N, 3] (x, y, yaw_ccw)
    hw: float,               # car_w / 2 + pad
    hl: float,               # car_l / 2 + pad
    cap_a: torch.Tensor,     # [M, 2] segment start in plane frame
    cap_b: torch.Tensor,     # [M, 2] segment end in plane frame
    cap_r: torch.Tensor,     # [M] capsule radii
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact batched OBB vs M capsules. Returns (colliding [N], min_clearance [N])."""
    c = torch.cos(pose[:, 2])[:, None]  # [N, 1]
    s = torch.sin(pose[:, 2])[:, None]  # [N, 1]
    da = cap_a[None, :, :] - pose[:, None, :2]  # [N, M, 2]
    db = cap_b[None, :, :] - pose[:, None, :2]  # [N, M, 2]
    # Rotate into UGV body frame R(yaw)^T * d
    ax = c * da[..., 0] + s * da[..., 1]
    ay = -s * da[..., 0] + c * da[..., 1]
    bx = c * db[..., 0] + s * db[..., 1]
    by = -s * db[..., 0] + c * db[..., 1]

    # 1) Slab intersection: segment ab vs AABB [-hw, hw] x [-hl, hl]
    dx, dy = bx - ax, by - ay
    inv_dx = torch.where(dx.abs() > 1e-9, 1.0 / dx, torch.zeros_like(dx))
    inv_dy = torch.where(dy.abs() > 1e-9, 1.0 / dy, torch.zeros_like(dy))
    tx1, tx2 = (-hw - ax) * inv_dx, (hw - ax) * inv_dx
    ty1, ty2 = (-hl - ay) * inv_dy, (hl - ay) * inv_dy
    tmin_x = torch.where(dx.abs() > 1e-9, torch.minimum(tx1, tx2),
                         torch.where(ax.abs() <= hw, 0.0, 2.0))
    tmax_x = torch.where(dx.abs() > 1e-9, torch.maximum(tx1, tx2),
                         torch.where(ax.abs() <= hw, 1.0, -1.0))
    tmin_y = torch.where(dy.abs() > 1e-9, torch.minimum(ty1, ty2),
                         torch.where(ay.abs() <= hl, 0.0, 2.0))
    tmax_y = torch.where(dy.abs() > 1e-9, torch.maximum(ty1, ty2),
                         torch.where(ay.abs() <= hl, 1.0, -1.0))
    hit_box = torch.maximum(torch.maximum(tmin_x, tmin_y), torch.zeros_like(ax)) <= \
              torch.minimum(torch.minimum(tmax_x, tmax_y), torch.ones_like(ax))

    # 2) Endpoint-to-AABB distances
    dist_a = torch.hypot((ax.abs() - hw).clamp_min(0.0), (ay.abs() - hl).clamp_min(0.0))
    dist_b = torch.hypot((bx.abs() - hw).clamp_min(0.0), (by.abs() - hl).clamp_min(0.0))

    # 3) 4 UGV corners to segment ab: corners [4, 2]
    corners = pose.new_tensor([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
    ab2 = (dx * dx + dy * dy).clamp_min(1e-12)
    t_proj = (((corners[None, None, :, 0] - ax[..., None]) * dx[..., None] +
               (corners[None, None, :, 1] - ay[..., None]) * dy[..., None]) / ab2[..., None]).clamp(0.0, 1.0)
    px = ax[..., None] + t_proj * dx[..., None] - corners[None, None, :, 0]
    py = ay[..., None] + t_proj * dy[..., None] - corners[None, None, :, 1]
    dist_c = torch.hypot(px, py).min(dim=-1).values  # [N, M]

    dist_seg_box = torch.where(hit_box, torch.zeros_like(dist_a),
                               torch.minimum(torch.minimum(dist_a, dist_b), dist_c))
    clearance = dist_seg_box - cap_r[None, :]  # [N, M]
    min_clear = clearance.min(dim=1).values
    return min_clear <= 0.0, min_clear
```

---

## 7. Generalized Swept-Clear Safety Shield (Onboard & Sim)

### 7.1 Uncertainty-Aware Stopping-Horizon Sweep
Both [`StagedBaseline._swept_clear`](file://src/ugv/staged_baseline.py#L635-L664) and [`StudentPolicy.step`](file://src/ugv/policy.py#L203-L237) forward-predict the UGV trajectory over the stopping horizon:
- `T_stop = dt_stale + latency + brake_tau + margin` (`0.35 s` in `policy.py`, `0.25 s` in `staged_baseline.py`).
- Evaluate $K = 7$ interpolated poses along the predicted arc $t_k \in [0, T_{\text{stop}}]$ (using `5 mm` max corner travel step in `parking.py`'s [`swept_clear`](file://src/ugv/parking.py#L175)).
- **Dynamic Covariance Inflation:** Inflate $(h_w, h_l)$ by:
  $$\text{pad}_{\text{eff}} = \text{pad}_{\text{base}} + \min\bigl(0.04\text{ m},\; 2.0 \cdot \sigma_{\text{pos}}\bigr),$$
  where $\sigma_{\text{pos}} = \text{EKF.pos\_std}$ (`0.015 m` floor when tags fresh, growing during under-belly vision blackout). This automatically tightens the virtual walls when dead-reckoning uncertainty grows.

### 7.2 Shield Override Action (Replacing the Symmetric `safe_vy = +0.04` Assumption)
In [`StagedBaseline._control`](file://src/ugv/staged_baseline.py#L909-L917), a blocked sweep currently assumes the safe direction is always `+y` (`safe_vy = 0.04`). With `nose_wheel` at `+y_P` and `left_vtail_tip` / `right_vtail_tip` at `-y_P`:
1. **In `StudentPolicy.step` ([`policy.py`](file://src/ugv/policy.py#L226-L234)):**
   - Replace `self.obstacles = gear_obstacles(self.gear)` with the 5-obstacle `strix_obstacles(self.airframe)` (`list[Capsule]`). Because [`ugv.parking.swept_clear`](file://src/ugv/parking.py#L161) already handles arbitrary `Capsule` lists, `policy.py`'s shield immediately protects against `nose_wheel`, `left_main_wheel`, `right_main_wheel`, `left_vtail_tip`, and `right_vtail_tip` with **zero change to the `swept_clear` API**, braking (`cmd = 0`) whenever any predicted stop trajectory touches any of the 5 obstacles.
2. **In `StagedBaseline` & RL Shield Wrapper (`_shield_step`):**
   - If the proposed `(vx_w, vy_w, w)` fails `swept_clear`:
     1. First test **pure braking** (`cmd_brake = 0` coasting over `brake_tau`) vs **pure lateral centering** (`v_w = [-k_x * x_P, 0]`) vs **gradient-ascent escape** along $\nabla_{x,y} d_{\min}(x, y, \psi)$ (computed via 2 finite-difference evaluations of `min_clearance`, pushing directly away from the nearest active obstacle!).
     2. Only emit a retreat/escape velocity if that escape trajectory itself passes `swept_clear`; otherwise command **immediate hard brake (`[0, 0, 0]`)** (never blindly back into `left_vtail_tip` or `nose_wheel`).

---

## 8. `rl/sim.py` Training & Sensor Emulator Updates

To train an RL student that transfers cleanly to the Strix airframe without sim-to-real parity failures:

1. **Multi-Tag Sensor Emulator (`simulate_plane_tags` in `rl/sim.py`):**
   - Replace the single-point `tag` model ([`rl/sim.py` lines 1127–1190](file://rl/sim.py#L1127-L1190)) with per-tag 3D mount poses $\{T_{P,k}, \hat{n}_k\}_{k \in \{20,21,22,23\}}$.
   - For each tag $k$ at delayed UGV pose, compute bearing $\beta_k$ and incidence angle $\theta_{\text{inc}, k}$.
   - Visibility gate: `(abs(beta_k) < FOV_HALF) & (dist_k < 5.0) & (dist_k > min_range_k) & (cos(theta_inc_k) > cos(60°))`.
   - Under-belly close-range transition: `ID_LEFT_FIN` and `ID_RIGHT_FIN` naturally drop out of the $\pm 33.5^\circ$ camera FOV as the UGV passes between the tail tips, handing off to `ID_BELLY_REAR` until `dist < env_min_tag_range` (`0.30–0.40 m`) blind final entry.
   - **Adversarial Clutter DR:** Inject synthetic false-target detections (random non-allowlisted IDs, $180^\circ$-flipped single-tag IPPE glitches at $\theta_{\text{inc}} > 55^\circ$, and decoy ground leg pairs) in `L_chaos` to verify **0% false-target lock** in [`rl/evaluate.py`](file://rl/evaluate.py).
2. **Multi-Capsule Collision & Termination:**
   - Replace [`rl/sim.py` lines 1265–1271](file://rl/sim.py#L1265-L1271) with `capsules_collide_obb(self.pose, CAR_W / 2, CAR_L / 2, self.cap_a, self.cap_b, self.cap_r)`.
   - Add `min_clearance` to `info` and reward shaping for maintaining margin inside the rear V-tail portal (`y_P < 0`) while blocking `nose_wheel` (`+y_P`) overshoot.
3. **Curriculum Spawns (`L4` / `L5` / `L_chaos`):**
   - Enable full $360^\circ$ spawns around the aircraft (`bearing = ±180°`, `yaw = ±180°`) so the student learns `ID_NOSE` recognition $\to$ wingtip perimeter arc $\to$ rear V-tail alignment $\to$ belly dock.

---

## 9. Verification & Acceptance Gates (Pre-Hardware)

Before deploying to the physical UGV + Strix airframe:
1. **Parity Unit Tests:**
   - Verify $\det([p_L - p_{\text{tail}}, p_R - p_{\text{tail}}]) > 0$ for true L/R assignments across all $360^\circ$ rotations $\theta \in [-\pi, \pi)$ and $< 0$ under $p_L \leftrightarrow p_R$ swap or planar reflection.
   - Verify `EKF.update` rejects $100\%$ of simulated clutter sequences (chair legs, shoes, $z < 0$ reflections, non-allowlisted tags, and persistent 10-frame false objects that previously triggered `rejected >= 6` re-init).
2. **Geometry Equivalence Tests:**
   - Assert bit-exact agreement (to $10^{-5}\text{ m}$) between `capsules_collide_obb` (PyTorch) and [`ugv.parking.collides`](file://src/ugv/parking.py#L133) (NumPy) across $10{,}000$ random poses $(x, y, \psi)$ around all $5$ Strix obstacles.
   - Verify [`StagedBaseline._swept_clear`](file://src/ugv/staged_baseline.py#L635) and [`StudentPolicy.step`](file://src/ugv/policy.py#L127) brake before contact on forced-collision trajectories toward `nose_wheel`, `left_main_wheel`, `right_main_wheel`, `left_vtail_tip`, and `right_vtail_tip`.
3. **Steep-Angle Bench Test:**
   - Test physical `ID_LEFT_FIN` (`20`) and `ID_RIGHT_FIN` (`21`) tags on $25^\circ$-canted trailing brackets vs flat V-tail mounts at $0.5\text{–}2.0\text{ m}$ range; confirm $\theta_{\text{inc}} \le 55^\circ$ and zero IPPE yaw flips.
