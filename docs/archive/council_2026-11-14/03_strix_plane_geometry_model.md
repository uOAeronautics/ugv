# Agent 03 — Strix-class fixed-wing (`STRL001`) belly-parking target: parametric 3D geometry model

**Agent:** 03 (Aircraft Geometry & Parametric 3D Belly-Parking Target Modeler)
**Target:** `.` — Freenove 4WD mecanum UGV parking under the nose/belly of the
fixed-wing aircraft loaded in RealFlight 9.5 as `STRL001`
**Evidence:** `docs/council_v2/strix_plane_realflight.png` (screenshot, airspeed 0, gear on the runway)
**Output:** this document only. No source, config, calibration, or `.sdf` files were modified.
**Status:** geometry **specification and generator snippet**; every absolute value is an
*image-derived estimate* (see §2). No airframe was measured; nothing here is a hardware clearance proof.

---

## 1. What the screenshot shows

`Loaded: STRL001` (captured 2026-11-14, RealFlight 9.5, "Naval Air Station"-type runway, prop stopped,
aircraft static on centreline). The airframe is a twin-propeller tactical fixed-wing UAV:

| # | Feature | Observation (pixel coords, 1256×768) |
|---|---------|--------------------------------------|
| 1 | High main wing, straight tapered, black/white/orange checkered upper surface | tip-to-tip ≈ x 292→1003, y≈425; span ≈ 713 px |
| 2 | Twin tractor props on the wing, on pylons/humps **above** the wing, discs blurred in rotation even at rest capture | disc centres ≈ (497, 388) and (752, 388); disc Ø ≈ 66 px; spacing ≈ 255 px (0.358 span) |
| 3 | Central fuselage pod, white, rounded fore end, "chin" with orange trim at the near end | widest ≈ x 580→668 (88 px ≈ 0.124 span) |
| 4 | Black vertical camera/sensor bar on the pod roof (far end) | y≈420–445 in `pod` close-up |
| 5 | Gold/silver turret with canister hanging under the chin | canister radius ≈ 0.18 m-class (est.) |
| 6 | Inverted **V-tail**: two very large flat panels cantilevered from the pod sides, anhedral ≈ 38–40°, white undersides with orange/black/white checkered leading-edge strip; panels slope down-and-out to near the ground | left panel x≈455→575 (root y≈415 → tip y≈585), right mirrored |
| 7 | Small light-coloured pegs with orange tips below each fin tip (tip skids/wheels or antennae, contact unresolved) | left ≈ (574, 598), right ≈ (804, 588) |
| 8 | Thin black vertical member from the chin canister down through the fin apex; white conical fairing at its lower end | does **not** appear to reach the floor (§4.6) |
| 9 | Thin bracing rods splayed from the chin down-out to each fin mid-span | both sides |
| 10 | Two soft shadows on the runway: a wide band (wing) and a large wedge (fins + pod) | sun from upper-right; shadow lower-left |
| 11 | UI text: `Loaded: STRL001`, `Switching to Nose Camera`, two "missing Visual Frame" warnings (`~CS_LMVS`, `~CS_Pivot1`), and an unrelated error dialog | the aircraft file references two missing frames; the model in-sim is **not** the as-built CAD |

### 1.1 View orientation (stated, per Council §06/§09 convention)

The consistent reading — also used by Agents 06 and 09 — is a **rear three-quarter view from above**:
the inverted V-tail (nearest the camera, largest in frame, panels sloping down-and-out) is the **aft**
structure; the pod's far rounded end is the **nose**, pointing away from the camera down the runway.
Aircraft **nose = +y_P (front)**, **tail portal = −y_P**. The council's nominal `APPROACH_REAR`
(entering between the fins) therefore means entering **from the side the camera is on**.

The UI says `Switching to Nose Camera`; the frame shown is not a nose-camera view (it sees the whole
aircraft), so the view label cannot be used to disambiguate sides. Orientation is an inference from
panel/prop geometry, not from the label. See §9.1.

### 1.2 What is NOT resolvable from this image

- Absolute scale (no reference object or HUD tape in frame).
- Which members touch the runway (fin-tip pegs? belly mains? nose and/or tail wheel?).
- Whether the centreline black member reaches the floor.
- Pod cross-section shape (assumed ellipse, 0.30 × 0.225 m based on the 88 px width).
- True dihedral/incidence (outer wing appears flat, maybe 0–1° polyhedral, 2° incidence).
- Propeller diameter exactly (disc blur edges ±10%).

All of these are parameterised in §5 and must be measured on the airframe (§10) before any
hardware-facing use.

---

## 2. Method and scale anchor

No reference object appears in frame, so **all absolute dimensions are derived from ratios to the 713 px
wingspan, anchored to an assumed wingspan S = 2.4 m** (large RC / Group-1 tactical UAV class, matching a
2–3 m man-portable Strix-class airframe). Every value below rescales linearly with `S` except the fixed
50 mm ArUco cards and the gear wheel radii, which are *not* rescaled by `S`.

Pixel ratios used (±10 % read error):

| Feature | px | Ratio of S | Value @ S = 2.4 m |
|---|---|---|---|
| Wingspan | 713 | 1.000 | **2.40 m** (assumed) |
| Prop disc spacing (half-span of disc centres) | 255 | 0.358 | **0.86 m** (±0.43 m) |
| Prop disc diameter | ≈ 66 | 0.093 | **0.22 m** |
| Pod max width | ≈ 88 | 0.124 | **0.30 m** |
| Fin tip lateral offset from centreline | ≈ 180 | 0.253 | **±0.61 m** |
| Small lateral peg offset (H-A main-gear hypothesis, §4.4) | ≈ 83 | 0.233 | **±0.28 m** (track 0.56 m) |
| Fin root → tip vertical drop | ≈ 160 px over ≈ 330 px lateral run | — | anhedral ≈ **38°** |
| Wing tip chord | ≈ 33 | 0.046 | **≈ 0.20 m** (root ≈ 0.42 m, taper ≈ 0.48) |

Uncertainty: ±20–30 % on every absolute value; the geometry ratios themselves are good to ±10 %.

---

## 3. Reference frames and units

**Generator (model-local, matches `drone_model.py` conventions):**

- `X` = fuselage longitudinal axis, **+X = nose**;
- `Y` = aircraft lateral axis, **+Y = aircraft left (port)**;
- `Z` = up, **`Z = 0` is the floor** (runway), origin under the belly centre;
- units: metres, radians; part schema `{name, shape, pos, size, rot, color, collision, ...}` with
  shapes `box | cylinder | mesh` — identical to `drone_assembly()` output so existing consumers
  (`add_drone()`-style SDF insertion, `render_drone.py`, Observatory part overlays) can consume it.

**Mapping to the policy frame** (`rl/sim.py` S frame, `config/parking.json`,
`car enters from +y facing −y`): with the aircraft placed nose toward **−y_S** (tail toward the entering
car), model `+X` (nose) ↔ S `−y`, i.e. nose away, tail portal at `+y` — the `APPROACH_REAR` world
placement from Agent 09. For a nose-first (`APPROACH_FRONT`) placement, yaw the model 180° about Z in
the world pose instead. Model `+Y` (port) ↔ S `−x` in that placement.

**Mapping to Gazebo frames** (as documented in `drone_model.py`): Gazebo G → sim S via
`x_G = −y_S, y_G = +x_S, z_G = z_S`; the snippet's `add_strix_plane()` writes the same
`pose` attribute convention (`x y z rpy`) used by `add_drone()`.

**Yaw sign:** CCW positive around +Z in `autopark.py`/`policy.py`/`rl/sim.py`; `parking.Pose2` is CW
positive. The generator does not cross that boundary.

---

## 4. Exact bounding primitives (reference instantiation, S = 2.4 m)

Coordinates are model-local. `Z` = floor origin. `Collision` = emitted as an `<collision>` in SDF and as
a solid for corridor tests.

### 4.1 High main wing

| Part | Primitive | pos (x, y, z) | size (x/y/z) | collision |
|---|---|---|---|---|
| `wing_panel` | box | (0.05, 0, 0.585) | (0.40, 2.02, 0.055) — panel spans 0.84 × S | yes |
| `wing_tip_l`, `wing_tip_r` | box, tapered chord 0.20, thickness 0.8× panel | (0.07, ±1.105, 0.585) | (0.20, 0.19, 0.044) — each spans the outer 0.08 × S | no (visual) |
| `wing_le`, `wing_te` | edge strips | (0.05, 0, 0.585) | along span, 0.008 | no |

- Span 2.40 m; root chord 0.42 m → built as one 0.40 m average-chord panel plus visual tip boxes;
  tip chord ≈ 0.20 m; taper ≈ 0.48; aspect ratio ≈ 7.7; area ≈ 0.74 m².
- Incidence ≈ 2°, dihedral ≈ 1° (not reproduced in the static model; noted for the sim force model).
- Underside at z = 0.557; top at z = 0.612. **Wing underside is never a corridor constraint** (car
  roof ≈ 0.21 m) and is included as a collision plate only to keep SDF `gz sdf -k` honest.

### 4.2 Twin prop nacelles (tractor) on pylons above the wing

| Part | Primitive | pos | size | collision |
|---|---|---|---|---|
| `nacelle_l`, `nacelle_r` | horizontal cylinder (axis X) | (0.10, ±0.43, 0.655) | r 0.055, L 0.34 | yes |
| `nacelle_pylon_l/r` | box | (0.10, ±0.43, 0.628) | (0.16, 0.075, 0.055) | no |
| `spinner_l`, `spinner_r` | 7-ring cone stack (as `propeller` spinner) | (0.29, ±0.43, 0.685) | r 0.038 → 0.004 | no |
| `motor_can_l/r` | cylinder (axis X) | (0.10, ±0.43, 0.660) | r 0.032, L 0.06 | yes |
| `propeller_l/r` | `assets/drone/propeller.obj` mesh, rotated into plane ⊥ X | (0.315, ±0.43, 0.685) | Ø 0.222 | no (visual; same policy as quad) |

Prop disc plane is vertical, normal ±X, hub at z ≈ 0.685 — disc top ≈ 0.80 m, disc centre above the
wing by ≈ 0.073 m, matching the screenshot (disc centres ~25 px above the wing surface).
Tractor configuration (props forward of the nacelles), consistent with the discs seen face-on from the
rear three-quarter camera. `pusher/tractor` label in the brief resolved to **tractor**, not pusher.

### 4.3 Fuselage belly pod

| Part | Primitive | pos | size | collision |
|---|---|---|---|---|
| `pod_mid` | box (ellipsoid section approximated) | (0.02, 0, 0.465) | (0.62, 0.30, 0.225) | yes |
| `pod_nose` | box, stepped taper | (0.36, 0, 0.475) | (0.20, 0.22, 0.17) | yes |
| `pod_rear` | box, stepped taper | (−0.36, 0, 0.475) | (0.20, 0.22, 0.17) | yes |
| `pod_roof_hump` | cylinder (axis X, squashed) | (0.00, 0, 0.585) | r 0.14, L 0.55 | no |
| `roof_sensor_bar` | box on pod roof (far/nose end) | (0.28, 0, 0.600) | (0.09, 0.16, 0.030) | no |
| `pod_belly_plate` | thin steel box | (0.02, 0, 0.352) | (0.55, 0.26, 0.012) | yes |
| `chin_canister` | vertical cylinder under chin | (+0.33, 0, 0.300) | r 0.055, h 0.105 | yes |
| `chin_turret` | vertical cylinder (sensor ball) | (+0.33, 0, 0.235) | r 0.042, h 0.035 | yes |

**Fuselage belly pod bounding box:** width **0.30 m** (±0.15), height **0.225 m** mid
(z 0.352 → 0.577), length ≈ **1.05 m** including tapers (z range 0.352 → 0.60). Belly underside
**z_belly = 0.352 m** at the steel plate (pose-critical: this, and the canister, bound
`belly_clearance_z`). Chin canister + turret lower face: **z ≈ 0.217 m** under the nose-third only.
For a 0.20 m UGV the cleanest corridor segment (between the mains) keeps the 0.217 m turret clear of
the roof by only ≈ 17 mm — see §6.3 and §10 (measure UGV height).

### 4.4 Left/right main landing gear (hypothesis H-A — primary) and hypothesis H-B

**H-A (mid-belly mains, "flanking the central belly", inferred from the two small lateral pegs and the
brief):**

| Part | Primitive | pos | size/shape | collision |
|---|---|---|---|---|
| `main_strut_l/r` | rod | (±0.145, z 0.375) → (±0.28, z 0.085) | r 0.020, angled down-out | yes |
| `main_wheel_l/r` | cylinder, axis = Y | (±0.28, 0, 0.085) | r 0.085, w 0.060 | yes |
| `main_hub_l/r` | cylinder, axis = Y | (±0.28, 0, 0.085) | r 0.030, w 0.062 | no |

Main-gear **track (wheel_spacing) = 0.56 m** centre-to-centre; wheel inner faces at ±0.25 →
**inner clear width ≈ 0.50 m**; wheel top z = 0.17.

**H-B (fin-tip outrigger mains; alternative reading of the low outboard pegs):** fin-tip pegs are
load-bearing tip wheels at lateral **±0.58 m**, track **1.16 m**, inner clear ≈ **1.10 m**. H-B cannot
be excluded from the image: the pegs hang below the fin tips exactly where contact would be. The model
keeps the outrigger volumes as collision volumes in both cases (they are physical objects either way).
**The corridor verdicts in §6 are reported for both hypotheses.** Which one is real is decidable only by
a tape measure / side photograph with a scale reference (§10).

### 4.5 Nose wheel / skid

| Part | Primitive | pos | size | collision |
|---|---|---|---|---|
| `nose_strut` | rod | (+0.42, z 0.36) → (+0.42, z 0.075) | r 0.016 | yes |
| `nose_wheel` | cylinder, axis = Y | (+0.42, 0, 0.075) | r 0.075, w 0.050 | yes |
| `nose_fork` | box | (+0.42, 0, 0.14) | (0.05, 0.045, 0.12) | no |

Nose gear is on the **centreline** at x = **+0.42 m** ahead of belly centre; top of tyre z = 0.15;
blocks centreline passage at the front in the gear-down state (§6.4, finding F1).

### 4.6 Inverted V-tail (the distinctive feature) + tail centreline member

Root at pod sides z ≈ 0.46, lateral ±0.12, longitudinal x ≈ −0.33; anhedral ≈ **38°**; tips at lateral
**±0.60 m**, z_tip ≈ **0.075 m**; root chord 0.55 m at a station sweep of ≈ 15° (tip trailing edge aft).

| Part | Primitive | pos (mid-panel) | size (chord, span, thick) | rot (rx, ry, rz) | collision |
|---|---|---|---|---|---|
| `fin_inner_l/r` | box | (−0.28, ±0.24, 0.345) | (0.46, 0.42, 0.045) | (∓0.66, 0, ±0.26) | yes |
| `fin_outer_l/r` | box | (−0.36, ±0.47, 0.175) | (0.31, 0.28, 0.038) | (∓0.66, 0, ±0.26) | yes |
| `fin_tip_skid_l/r` | cylinder, axis = X | (−0.38, ±0.575, 0.045) | r 0.022, L 0.16 | — | yes |
| `fin_le_strip_l/r` | thin box (checkered LE) | — | visual only | — | no |
| `tail_strut` | rod | (−0.46, z 0.33) → (−0.46, z 0.16) | r 0.017, black | — | yes |
| `tail_fairing` | cone stack (3 rings) | (−0.46, 0, 0.150) | r 0.030 → 0.010 | — | yes |
| `tail_brace_l/r` | thin rod, chin → fin mid | — | r 0.008 | — | no |

Fin true panel span (root→tip) ≈ 0.60 m; tip chord ≈ 0.31 m; the fins are the **only** structures lateral
of |y| > 0.3 below z = 0.36. Fin surface height above floor as a function of lateral offset
(root 0.46 @ ±0.12 → tip 0.075 @ ±0.60):

| lateral | 0.0 | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | 0.58 |
|---|---|---|---|---|---|---|---|---|---|
| fin z (m) | 0.46 | 0.46 | 0.40 | 0.36 | 0.32 | 0.28 | 0.24 | 0.17 | 0.10 |

A 0.20 m-wide (±0.10 m) UGV at centreline never approaches a fin; fins only bind the corridor for
lateral excursions beyond ±0.35 m — they are portal-side landmarks, not roof.

**Centreline black member (the single most decision-relevant unknown).** The member descends from the
chin canister area to a white conical fairing at the fin apex; its lowest visible point is at roughly the
apex z ≈ **0.15–0.20 m** and it does **not** appear to continue to the floor (fin tips and pegs project
lower). In the model it is `tail_strut` + `tail_fairing`, bottom z ≈ **0.141 m** (fairing tip). If this
is literal (a fixed pitot/antenna or a tail-wheel leg end), it sits inside the 0.20 m UGV envelope and
**blocks the rear centreline portal** (§6.4, finding F2). If measured to terminate at z ≥ 0.23 m, the
rear portal is open for the UGV. The three cases are evaluated in §6.4.

### 4.7 Marker set (ArUco, DICT_4X4_50, 50 mm, real-paper card bounds)

Proposed IDs (pending board assignment — must not clash with the deployed quad set 14/6/3 or the
`config/parking.json` list 9/12/13/14):

| ID | Mount | Face normal | pos (x, y, z) | rot (rx, ry, rz) |
|---|---|---|---|---|
| **21** | left fin (port, +Y) | +Y (vertical, ≈ flush, tilt-randomised) | (−0.30, +0.435, 0.245) | (−π/2, 0, 0) |
| **22** | right fin (starboard, −Y) | −Y | (−0.30, −0.435, 0.245) | (+π/2, 0, 0) |
| **23** | belly bottom (under pod mid) | −Z | (−0.02, 0, 0.345) | (π, 0, 0) |

- Card bounds: `REAL_PAPER_BOUNDS` (−0.028, +0.028, −0.0265, +0.043) m as in `drone_model.py` —
  50 mm ink, ~1.5 mm bottom quiet zone.
- Real DICT_4X4_50 bit grids (verified against `cv2.aruco.getPredefinedDictionary`, cv2 5.0.0, and
  against the 14/6/3/9/12/13 grids printed in `drone_model.py`, which match exactly):

```
21: 1011 0000 0010 1011     22: 1100 1100 1101 0101     23: 1101 1101 1000 0010
```

- Fin cards face ±Y so they are readable on approach/departure laterally; the belly card faces the
  floor (−Z) and is read only from below/front-low angle as in the quad's `entry_marker_14`.
  Cards are visual-only for collision (same policy as the quad).
- Tilt randomisation (`randomize_tilt=True`, ±10–25°) applies to all three.

### 4.8 Mass/inertia, scope note

Not in scope. Static obstacle model; no aerial dynamics, no gear retraction animation, no prop spin.

---

## 5. Parameterisation (single-source scaling)

`STRIX_DEFAULTS` in the snippet is the complete list. Only `wingspan_m`, the two wheel radii, the
50 mm card size and the pod section (0.30 × 0.225 m) are not simple multiples of each other; everything
else derives from S. To re-anchor to a measured wingspan: set `wingspan_m` and multiply all `*_m`
lengths by `new_S / 2.4` (the snippet does this via `scale` in `strix_plane_assembly`).
The two wheel radii and card size are pinned (they don't scale with S) — **by design**, matching the
fixed gear/tire hardware.

---

## 6. Belly-parking corridor — definitions, numbers, verdicts

Corridor = the set of poses `(x_P, y_P, ψ)` in which a 0.20 m-wide, 0.149 m-track, ≈ 0.21 m-high UGV
fits under the airframe without touching a collision primitive. Evaluated at the **car reference
height z = 0.20 m** (as `corridor_width_evaluation_height` in the quad JSON) and at the roof z = 0.21 m.

### 6.1 Canonical corridor metrics (quad vs plane)

| Metric | Quad (skid model) | Plane **H-A** | Plane **H-B** |
|---|---|---|---|
| `wheel_spacing` (centre-to-centre) | 0.340 (skids) | **0.56** | **1.16** |
| Inner-face clear width (wheel width 0.06) | 0.310 @ floor | **0.50** | **1.10** |
| Min corridor width @ z = 0.20 m | **0.255** | **≈ 0.50** (mains) | **≈ 1.10** (no constraint) |
| `belly_clearance_z` (floor → lowest roof-threat over corridor) | 0.225 (payload bottom) | **0.217** (turret) / **0.352** (pod plate) | same |
| Corridor length (pod/fin run below roof height) | ≈ 0.48 (skids) | **≈ 1.05** (pod) | same |
| Lateral drift budget @ centre (0.25 − car half 0.10 − margin 0.03) | 0.028 | **0.12** | 0.45 |
| Longitudinal stop margin to far obstruction | none (through-open) | nose gear +0.42 / tail fairing −0.46 | same |
| Portal obstructions | none on centreline | nose gear (front), tail member (rear) | same |
| Tags on entry path | 14 (head-on, pitched 25°) | 23 (belly, −Z) + 21/22 (fins) | same |
| Collision parts in generator | 29 | 27 | 27 |

The plane is **strictly more permissive laterally** (H-A doubles the quad's floor gap; H-B is
essentially open sideways) but **introduces two new hard blockers on the centreline** (F1/F2, §6.4) and
a **lower roof-threat** ahead of mid-corridor (turret 0.217 m vs quad payload 0.225 m).

### 6.2 Collision volumes, left/right mains (model-local, H-A)

- `main_wheel_l`: cylinder axis Y, centre **(+0.28, 0, 0.085)**, r 0.085, w 0.060; wheel-top z = 0.170.
  Swept conservative volume: capsule r 0.085, axis along Y from y 0.25 → 0.31.
- `main_wheel_r`: mirror at **(−0.28, 0, 0.085)**.
- `main_strut_l/r`: capsules r 0.020 from (±0.145, 0, 0.375) to (±0.28, 0, 0.085) — intrudes into the
  upper corridor corners; at z = 0.20 the strut surface is at |y| ≥ 0.247 → not binding for a ±0.10 m car.
- Effective wall for the car at roof height: |y| = 0.25 (wheel inner faces). H-A corridor, two ways
  to state it: geometric obstacle tube |y| ≤ **0.22** (30 mm from the wall), or go/no-go centreline
  tube |y| ≤ **0.12** with the 0.10 m car half-width and 30 mm margin folded in
  (`strix_corridor_metrics()["usable_half_width_margin30mm_m"]`).

### 6.3 `belly_clearance_z` profile along the corridor (centreline x=0 slice)

| Station x (m) | Structure above | Bottom z (m) |
|---|---|---|
| +0.42 | nose wheel leg | 0.150 (wheel top) — hard blocker, F1 |
| +0.33 | chin canister + turret | **0.217** — roof-threat, 7 mm below quad payload |
| +0.20 → −0.20 | pod belly plate | **0.352** — generous |
| −0.28 → −0.40 | fin panels (centreline) | ≈ 0.46 → 0.42 (not threatening) |
| −0.46 | tail strut + fairing | fairing tip **0.141** — roof-threat / blocker, F2 |
| wing underside | 0.557 — never binding |

Lowest full-width headroom (belly plate): **0.352 m**. Lowest centreline roof-threat: **0.217 m**
(turret). Lowest structural element, period: tail fairing tip ≈ 0.141 m; fin-tip skids ≈ 0.030–0.075 m
(outboard, non-binding at centre).

### 6.4 Entry analysis — rear V-tail portal vs front nose portal

**Finding F1 — front (nose) portal.** The nose gear leg at x = +0.42 (z 0.075–0.360, r ≤ 0.025, tyre
top z = 0.150) is a **centreline hard blocker**: a 0.21 m UGV cannot pass over or under it. Nose-first
entry can only approach until the bumper reaches x ≈ +0.40 m and must stop — **it can never reach the
belly-centred dock at x ≈ 0 with gear down**. (If the airframe's gear retracts in flight and is simply
*depicted* down here, this finding flips — it cannot be resolved from the image.)

**Finding F2 — rear (V-tail) portal.** The centreline member descends to the fin apex fairing,
bottom z ≈ **0.141 m** as modelled (unmeasured; plausible range 0.10–0.23 m). Same consequence: a rear
centreline approach hits it before reaching the dock. To the outboard sides the portal is wide open
(fin-tip skids at ±0.575 m, bottom ≈ 0.030 m, but the car never goes there), and the pod roof there is
0.46 m — the portal *structure* is fine, the *centreline* is the problem.

**Resolution cases for F2 (must be measured before gates are set):**

| Measured tail-member bottom z | Verdict for 0.21 m UGV |
|---|---|
| ≥ 0.23 m | rear portal open — `APPROACH_REAR` (Agents 06/09) valid as written |
| 0.18–0.23 m | marginal — allowed only with roof-margin monitoring and abort-on-contact |
| < 0.18 m | rear portal blocked — **the aircraft as parked cannot be entered from either end**; dock only from the side (blocked by mains, F1-style) or stage the car beside the airframe and use a non-drive-through mate |

**Finding F3 — conflict with the council's nominal approach.** Agent 06 (keypoints) and Agent 09
(shield: `APPROACH_REAR` staging at `y_P < y_tail − d_stage`, then drive to `(0,0)`) both assume the
rear portal is enterable. Both models are invalid with the centreline member at its modelled height.
The cheapest disambiguator is a single tape measure under the apex, or one side photo with a reference
(ruler/phone) resting on the runway under the fairing. This document does not change 06/09; it flags
that their nominal path is gated on one unmeasured number.

**Finding F4 — gear-down drive-through is impossible on this airframe as parked.** There is no
end-to-end drive-through corridor (F1 front, F2 rear). The viable "parking" geometries are:
(a) stop under the belly as close as the approach-side blocker allows (docking = final approach +
hold, not threading), or (b) require the real airframe to be gear-up/retracted for the task, or
(c) measure and clear the two centreline members. All three are airframe/ops decisions, not UGV ones.

**Nose entry vs tail entry, scored** (ignoring F1/F2 blockers, i.e. treating them as removable):

| Criterion | Nose (+X) entry | Tail (−X) entry |
|---|---|---|
| Path clearance to slab | turret 0.217 m — single pinch point | fin panels 0.46 → 0.42 m — no pinch point |
| Portal width | pod 0.30 m wide flanks, turret centre — cramped entry sight picture | ±0.60 m fin-tip walls, ~1.1 m — ceremonial width |
| Visual approach (camera fwd) | sees turret + pod plate: tag 23 (belly) face-on low angle; far tag 14-style entry cue available | sees fin anhedral interior; 21/22 fin tags flank the path head-on at 25–40° tilt; belly tag 23 visible ahead-low |
| Alignment aids | single centreline feature | bilateral fins = strong L/R disambiguation (Agent 09's parity invariant preserved) |
| Contact risk if blocked | turret glass (expensive, fragile) | tail fairing (paint, cheap) |
| Verdict (if blockers clear) | acceptable, tighter | **preferred** — matches council `APPROACH_REAR` |

---

## 7. Comparison summary vs the quadcopter skid model (`drone_model.py`)

| Aspect | Quad `photo_quadcopter` (real, 2026-11-13) | STRL001 Strix-class (this doc) |
|---|---|---|
| Class | multirotor, 2 splayed legs + 2 skids | fixed-wing, gear-down static |
| `skid_center_spacing` / `wheel_spacing` | 0.340 m | 0.56 m (H-A) / 1.16 m (H-B) |
| Floor gap between supports | 0.310 m | 0.50 m (H-A inner faces) / 1.10 m (H-B) |
| `minimum_center_corridor_width` @ z=0.2 | 0.255 m | ≈ 0.50 m (H-A) / ≈ 1.10 m (H-B) |
| Lowest solid over corridor | 0.225 m (payload silver) | **0.217 m** (chin turret) / 0.352 m (pod plate) |
| Visual-only lowest | 0.168 (cards) / 0.195 (cable) | 0.141 (tail fairing, modelled solid), 0.030 (fin-tip skid pads) |
| Supports block centreline? | **No** (both legs splayed, centre open) | **Yes** (nose leg + tail fairing) — findings F1/F2 |
| Drive-through | yes, end-to-end | **no** (as parked) |
| Tags | 14 (entry, pitched), 6, 3 (sides) | 21 (port fin), 22 (starboard fin), 23 (belly) |
| Tag readable on approach heading | 14 head-on | 23 fore/aft-low; 21/22 lateral |
| Frame origin | floor under deck centre | floor under belly centre |
| SDF insertion | `add_drone()` | `add_strix_plane()` (snippet §8) |
| 2D overlay footprint | 0.48 × 0.31 m skid rectangle | 2.40 × ~0.62 m envelope; portal walls ±0.58 m |
| 3D overlay consumption | `drone_assembly()` parts list | `strix_plane_assembly()` parts list (same schema) |

Both targets share the `drone_assembly()` part schema, the `DRONE_COLORS` material table (the snippet
reuses it and adds `orange`), the `REAL_PAPER_BOUNDS` card, and the "collision only where solid; cards,
propellers, fine wire visual-only" policy — except that the fin-tip skids and the tail fairing are
promoted to collision parts because unlike the quad's cards/wires they are solid ground-reachable mass.

---

## 8. Ready-to-use generator snippet

Self-contained; do **not** edit `drone_model.py`. Place alongside it if/when adopted
(e.g. `simulation/gazebo/strix_plane_model.py`). Mirrors `drone_assembly()` / `add_drone()` interfaces,
prints corridor metrics on `__main__`. The snippet was extracted from this document, compiled and run
(`parts: 80`, `collision: 27`; tilt-randomisation and re-anchor paths exercised), and checked for a
clearances invariant: every part's lowest z is ≥ 0 (the three wheels touch exactly 0). Metrics match §6.1.
**Not executed in Gazebo here** (Agent 00/02 own world-level validation; run `gz sdf -k` on any world
using it).

```python
"""Strix-class (STRL001) fixed-wing belly-parking target — parametric static model.

Model-local frame: X = fuselage longitudinal (+X nose), Y = aircraft left (+Y port),
Z up, origin on the floor under the belly centre. Units SI. Part schema identical to
drone_model.drone_assembly(): dicts with name/shape/pos/size/rot/color/collision (+mesh).

All lengths derive from STRIX_DEFAULTS['wingspan_m'] via `scale`; wheel radii, card
size are pinned. Absolute values are image-derived estimates (see
docs/council_v2/03_strix_plane_geometry_model.md); re-measure before hardware use.
"""
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
S_NOM = 2.4  # assumed wingspan anchor for the reference island

STRIX_DEFAULTS = {
    "wingspan_m": S_NOM,
    "scale": 1.0,  # set new_S/2.4 to re-anchor; pinned parts below stay fixed
    # --- pinned (do not scale) ---
    "wheel_radius_main": 0.085, "wheel_width_main": 0.060,
    "wheel_radius_nose": 0.075, "wheel_width_nose": 0.050,
    "wheel_radius_tail": 0.050, "wheel_width_tail": 0.040,
    "card_size": 0.050,   # DICT_4X4_50 50 mm black square
    # --- scaled (reference @ S = 2.4 m) ---
    "wing_chord_root": 0.42, "wing_chord_tip": 0.20, "wing_thick": 0.055,
    "wing_z": 0.585, "wing_x": 0.05,
    "nacelle_y": 0.43, "nacelle_r": 0.055, "nacelle_z": 0.655,
    "prop_diameter": 0.222, "prop_hub_z": 0.685, "prop_x": 0.315,
    "pod_w": 0.30, "pod_h": 0.225, "pod_z": 0.465, "pod_belly_z": 0.352,
    "turret_bottom_z": 0.217, "pod_length": 1.04,
    "fin_root_z": 0.46, "fin_root_lat": 0.12, "fin_tip_lat": 0.60, "fin_tip_z": 0.075,
    "fin_anhedral_deg": 38.0, "fin_root_chord": 0.46, "fin_tip_chord": 0.31,
    "fin_thick": 0.045, "fin_sweep_deg": 15.0,
    "tail_member_bottom_z": 0.141,   # F2 — MEASURE THIS
    "track_main_HA": 0.56, "track_main_HB": 1.16,
    "main_strut_top_lat": 0.145,
    "nose_wheel_x": 0.42, "tail_x": -0.46,
}

STRIX_COLORS = {  # extends DRONE_COLORS; import and .update() if co-located with drone_model
    "white": (0.93, 0.94, 0.90), "carbon": (0.027, 0.032, 0.035),
    "carbon_edge": (0.012, 0.016, 0.019), "black": (0.006, 0.008, 0.009),
    "steel": (0.28, 0.32, 0.34), "silver": (0.58, 0.63, 0.66),
    "orange": (0.85, 0.32, 0.02), "copper": (0.55, 0.22, 0.075), "gold": (0.67, 0.40, 0.08),
}

# DICT_4X4_50 grids, rows top->bottom, '1' = black bit (matches drone_model.py layout).
# 14/6/3/9/12/13 verified against drone_model.py; 21/22/23 generated with
# cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) + generateImageMarker, borderBits=1.
# 21/22/23 are PROPOSED IDs pending board assignment.
STRIX_MARKER_GRIDS = {
    21: ("1011", "0000", "0010", "1011"),
    22: ("1100", "1100", "1101", "0101"),
    23: ("1101", "1101", "1000", "0010"),
}
REAL_PAPER_BOUNDS = (-0.028, 0.028, -0.0265, 0.043)  # left, right, bottom, top [m]


def strix_plane_assembly(params=None, randomize_tilt=False, rng=None, tilt_deg=(10.0, 25.0)):
    p = dict(STRIX_DEFAULTS)
    if params:
        p.update(params)
    k = p["scale"]  # linear re-anchor for non-pinned lengths
    L = lambda v: v * k
    parts = []

    def add(name, shape, pos, size, color, rot=(0, 0, 0), collision=False, **extra):
        parts.append(dict(name=name, shape=shape, pos=tuple(pos), size=tuple(size),
                          color=color, rot=tuple(rot), collision=collision, **extra))

    def box(n, xyz, size, c, rot=(0, 0, 0), collision=False):
        add(n, "box", xyz, size, c, rot, collision)

    def cyl(n, xyz, r, h, c, rot=(0, 0, 0), collision=False):
        add(n, "cylinder", xyz, (r, h), c, rot, collision)

    def rod(n, a, b, r, c, collision=False):
        dx, dy, dz = [b[i] - a[i] for i in range(3)]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        cyl(n, [(a[i] + b[i]) / 2 for i in range(3)], r, length, c,
            (0, math.acos(max(-1, min(1, dz / length))), math.atan2(dy, dx)), collision)

    zW = L(p["wing_z"]); half_span = L(p["wingspan_m"]) / 2
    # --- high wing: one average-chord panel + visual tapered tips ---
    box("wing_panel", (L(p["wing_x"]), 0, zW), (L(0.40), L(p["wingspan_m"]) * 0.84, L(p["wing_thick"])), "white", collision=True)
    for s in (-1, 1):
        box(f"wing_tip_{'l' if s > 0 else 'r'}", (L(p["wing_x"]) + 0.02, s * half_span * 0.92, zW),
            (L(p["wing_chord_tip"]), half_span * 0.16, L(p["wing_thick"] * 0.8)), "white")
    # --- twin prop nacelles above the wing, tractor ---
    for s in (-1, 1):
        y = s * L(p["nacelle_y"])
        box(f"nacelle_pylon_{s}", (L(0.10), y, zW + 0.040), (L(0.16), L(0.075), L(0.055)), "white")
        cyl(f"nacelle_{s}", (L(0.10), y, L(p["nacelle_z"])), L(p["nacelle_r"]), L(0.34), "white",
            (0, math.pi / 2, 0), collision=True)
        cyl(f"motor_can_{s}", (L(0.10), y, L(p["nacelle_z"]) + 0.005), L(0.032), L(0.06), "black",
            (0, math.pi / 2, 0), collision=True)
        cyl(f"prop_hub_{s}", (L(p["prop_x"]), y, L(p["prop_hub_z"])), L(0.020), L(0.03), "silver",
            (0, math.pi / 2, 0))
        for j in range(5):  # spinner cone from ring stack, no external mesh dependency
            cyl(f"spinner_{s}_{j}", (L(p["prop_x"]) + 0.030 + j * 0.006, y, L(p["prop_hub_z"])),
                L(0.030) * (1 - j / 6) ** 0.7, 0.006, "silver", (0, math.pi / 2, 0))
        # propellers: visual mesh, same policy as quadcopter propellers
        # add(f"propeller_{s}", "mesh", (L(p['prop_x'] - 0.02), y, L(p['prop_hub_z'])), (1, 1, 1),
        #     "white", (0, 0, 0), mesh="assets/drone/propeller.obj")
    # --- fuselage belly pod ---
    box("pod_mid", (0.02, 0, L(p["pod_z"])), (L(0.62), L(p["pod_w"]), L(p["pod_h"])), "white", collision=True)
    box("pod_nose", (L(0.36), 0, L(p["pod_z"]) + 0.01), (L(0.20), L(0.22), L(0.17)), "white", collision=True)
    box("pod_rear", (L(-0.36), 0, L(p["pod_z"]) + 0.01), (L(0.20), L(0.22), L(0.17)), "white", collision=True)
    box("pod_belly_plate", (0.02, 0, L(p["pod_belly_z"])), (L(0.55), L(0.26), 0.012), "steel", collision=True)
    box("roof_sensor_bar", (L(0.28), 0, L(p["pod_z"]) + L(p["pod_h"]) / 2 + 0.008), (L(0.09), 0.16, 0.030), "black")
    cyl("chin_canister", (L(0.33), 0, L(0.300)), L(0.055), L(0.105), "white", collision=True)
    cyl("chin_turret", (L(0.33), 0, L(p["turret_bottom_z"]) + 0.018), L(0.042), 0.036, "gold", collision=True)
    # --- main gear H-A (mid-belly), wheels pinned radius ---
    for s in (-1, 1):
        top = (s * L(p["main_strut_top_lat"]), 0, 0.375 * k)
        bot = (s * L(p["track_main_HA"]) / 2, 0, p["wheel_radius_main"])
        rod(f"main_strut_{s}", top, bot, 0.020, "carbon", collision=True)
        cyl(f"main_wheel_{s}", bot, p["wheel_radius_main"], p["wheel_width_main"], "black",
            (math.pi / 2, 0, 0), collision=True)
        cyl(f"main_hub_{s}", bot, 0.030, p["wheel_width_main"] + 0.002, "silver", (math.pi / 2, 0, 0))
    # H-B outrigger volumes at fin tips (keep as collision in both hypotheses)
    for s in (-1, 1):
        cyl(f"fin_tip_skid_{s}", (L(-0.38), s * L(0.575), 0.045), 0.022, 0.16, "carbon_edge",
            (0, math.pi / 2, 0), collision=True)
    # --- nose gear (centreline, pinned radius) ---
    rod("nose_strut", (L(p["nose_wheel_x"]), 0, 0.36 * k), (L(p["nose_wheel_x"]), 0, p["wheel_radius_nose"]),
        0.016, "carbon", collision=True)
    cyl("nose_wheel", (L(p["nose_wheel_x"]), 0, p["wheel_radius_nose"]), p["wheel_radius_nose"],
        p["wheel_width_nose"], "black", (math.pi / 2, 0, 0), collision=True)
    # --- inverted V-tail ---
    anh = math.radians(p["fin_anhedral_deg"]); swp = math.radians(p["fin_sweep_deg"])
    for s in (-1, 1):
        rx = -s * anh; rz = s * swp
        lat_in = (L(p["fin_root_lat"]) + L(p["fin_tip_lat"])) / 2 * 0.66
        z_in = (L(p["fin_root_z"]) + L(p["fin_tip_z"])) / 2 + 0.077
        lat_out = (L(p["fin_root_lat"]) + L(p["fin_tip_lat"])) / 2 + 0.11
        z_out = (L(p["fin_root_z"]) + L(p["fin_tip_z"])) / 2 - 0.093
        box(f"fin_inner_{s}", (L(-0.28), s * lat_in, z_in), (L(p["fin_root_chord"]), L(0.42), L(p["fin_thick"])),
            "white", (rx, 0, rz), collision=True)
        box(f"fin_outer_{s}", (L(-0.36), s * lat_out, z_out), (L(p["fin_tip_chord"]), L(0.28), L(p["fin_thick"] * 0.85)),
            "white", (rx, 0, rz), collision=True)
        box(f"fin_tip_cap_{s}", (L(-0.40), s * L(p["fin_tip_lat"]), L(p["fin_tip_z"])),
            (L(0.16), 0.06, 0.05), "orange", (rx, 0, rz))
    # --- tail centreline member (F2) ---
    rod("tail_strut", (L(p["tail_x"]), 0, 0.33 * k), (L(p["tail_x"]), 0, L(p["tail_member_bottom_z"]) + 0.02),
        0.017, "black", collision=True)
    for j, r in enumerate((0.030, 0.022, 0.012)):
        cyl(f"tail_fairing_{j}", (L(p["tail_x"]), 0, L(p["tail_member_bottom_z"]) + 0.025 + j * 0.018),
            r, 0.018, "white", collision=True)
    rod("tail_tie_l", (L(0.33), 0, L(0.30)), (L(-0.26), L(0.24), L(0.36)), 0.008, "carbon")
    rod("tail_tie_r", (L(0.33), 0, L(0.30)), (L(-0.26), L(-0.24), L(0.36)), 0.008, "carbon")
    # --- ArUco tags: 21 port fin, 22 starboard fin, 23 belly ---
    grids = STRIX_MARKER_GRIDS
    tilt_rng = (rng if rng is not None else random.Random(0)) if randomize_tilt else None

    def card(name, pos, rot, marker_id):
        lb, rb, bb, tb = REAL_PAPER_BOUNDS

        def transform(v):
            x, y, z = v
            y, z = y * math.cos(rot[0]) - z * math.sin(rot[0]), y * math.sin(rot[0]) + z * math.cos(rot[0])
            x, z = x * math.cos(rot[1]) + z * math.sin(rot[1]), -x * math.sin(rot[1]) + z * math.cos(rot[1])
            x, y = x * math.cos(rot[2]) - y * math.sin(rot[2]), x * math.sin(rot[2]) + y * math.cos(rot[2])
            return tuple(pos[i] + (x, y, z)[i] for i in range(3))

        box(name + "_paper", transform((0.5 * (lb + rb), 0.5 * (bb + tb), 0.0)),
            (rb - lb, tb - bb, 0.0008), "white", rot)
        cell = p["card_size"] / 6
        box(name + "_ink", transform((0, 0, 0.0006)), (p["card_size"],) * 2 + (0.0003,), "black", rot)
        for row, line in enumerate(grids[marker_id], 1):
            for col, bit in enumerate(line, 1):
                if bit == "1":
                    box(f"{name}_px_{row}_{col}", transform(((col - 2.5) * cell, (2.5 - row) * cell, 0.0009)),
                        (cell,) * 2 + (0.0003,), "white", rot)

    def _jit(rot):
        if tilt_rng is None:
            return rot
        lo, hi = tilt_deg
        sgn = lambda: math.radians(tilt_rng.choice((-1.0, 1.0)) * tilt_rng.uniform(lo, hi))
        rx, ry, rz = rot
        return (rx + sgn(), ry, rz + sgn())

    card("fin_marker_21", (L(-0.30), L(0.435), L(0.245)), _jit((-math.pi / 2, 0, 0)), 21)
    card("fin_marker_22", (L(-0.30), L(-0.435), L(0.245)), _jit((math.pi / 2, 0, 0)), 22)
    card("belly_marker_23", (L(-0.02), 0, L(p["pod_belly_z"]) - 0.001), _jit((math.pi, 0, 0)), 23)
    return parts


def strix_corridor_metrics(params=None):
    p = dict(STRIX_DEFAULTS); p.update(params or {})
    half_track = p["track_main_HA"] / 2
    inner_clear = p["track_main_HA"] - p["wheel_width_main"]
    return {
        "wheel_spacing_m": p["track_main_HA"],
        "inner_clear_width_m": inner_clear,
        "half_track_m": half_track,
        "belly_clearance_z_pod_m": p["pod_belly_z"],
        "belly_clearance_z_roof_threat_m": p["turret_bottom_z"],
        "tail_member_bottom_z_m": p["tail_member_bottom_z"],
        "fin_tip_z_m": p["fin_tip_z"],
        "usable_half_width_margin30mm_m": inner_clear / 2 - 0.10 - 0.030,
    }


def add_strix_plane(world, node, pose=(0.0, 0, 0, 0, 0, 0), params=None,
                    randomize_tilt=False, rng=None, tilt_deg=(10.0, 25.0)):
    """Insert static Strix model; mirror of add_drone(). `node` is the XML helper from generate.py."""
    model = node(world, "model", name="strix_plane")
    node(model, "static", "true")
    node(model, "pose", " ".join(map(str, pose)))
    link = node(model, "link", name="airframe")
    for part in strix_plane_assembly(params, randomize_tilt, rng, tilt_deg):
        for kind in ("visual", "collision") if part["collision"] else ("visual",):
            obj = node(link, kind, name=part["name"] + "_" + kind)
            node(obj, "pose", " ".join(map(str, (*part["pos"], *part["rot"]))))
            shape = node(node(obj, "geometry"), part["shape"])
            if part["shape"] == "box":
                node(shape, "size", " ".join(map(str, part["size"])))
            elif part["shape"] == "mesh":
                node(shape, "uri", part["mesh"]); node(shape, "scale", "1 1 1")
            else:
                node(shape, "radius", part["size"][0]); node(shape, "length", part["size"][1])
            if kind == "visual":
                mat = node(obj, "material")
                rgba = " ".join(map(str, (*STRIX_COLORS[part["color"]], 1)))
                node(mat, "ambient", rgba); node(mat, "diffuse", rgba)
    return model


if __name__ == "__main__":
    print(json.dumps(strix_corridor_metrics(), indent=2))
    print(f"parts: {len(strix_plane_assembly())}")
```

**Proposed `strix_plane_dimensions.json`** (NOT written to the repo; review before adopting):

```json
{
  "units": "metres",
  "status": "Screen estimates from one RealFlight 9.5 still; S assumed 2.4 m; re-measure before use",
  "wingspan": 2.4, "wing_root_chord": 0.42, "wing_tip_chord": 0.20, "wing_z": 0.585,
  "nacelle_spacing": 0.86, "prop_diameter": 0.222,
  "pod_width": 0.30, "pod_height": 0.225, "pod_belly_z": 0.352, "turret_bottom_z": 0.217,
  "wheel_spacing_HA": 0.56, "wheel_spacing_HB": 1.16,
  "wheel_radius_main_hA": 0.085, "track_inner_clear_m": 0.50,
  "fin_root_z": 0.46, "fin_tip_z": 0.075, "fin_tip_lat": 0.60, "fin_anhedral_deg": 38.0,
  "tail_member_bottom_z": 0.141,
  "marker_ids": {"left_fin": 21, "right_fin": 22, "belly": 23},
  "marker_size_m": 0.05,
  "corridor_width_evaluation_height": 0.2,
  "usable_half_width_margin30mm_m": 0.12
}
```

**Observatory 2D overlay** (tools/observatory replay canvas): consume `strix_plane_assembly()` parts of
shape `box`/`cylinder`, project to the S-frame floor plan (model +X → S −y for the `APPROACH_REAR`
world yaw), draw: envelope rect (2.40 × 0.62 m) as ghost outline, fins as two swept quads, mains as two
0.17 × 0.06 m rects at ±0.28, nose/tail centreline blockers hatched, docking target circle r = 0.15 m
at (0, 0). **3D overlay**: feed the parts list straight into the existing drone-overlay part renderer
(same schema; add `orange` to its palette). The corridor polygons to draw in either view are the
inner faces |y| ≤ 0.25 minus 30 mm margin, from nose gear to tail fairing.

**Gazebo SDF**: `add_strix_plane()` under the same `generate.py` `node()` helper produces the static
model for a `strix_docking_world.sdf`; remember `gz sdf -k` and a visual pass before use, and note the
`photo_quadcopter` precedent: *static obstacle only, no dynamics.*

---

## 9. Ambiguities and disagreements to resolve

### 9.1 View orientation (§1.1) — unresolved, low impact

Reading A (used here): rear-3/4 view, nose away (consistent with Agents 06/09 frames and the fin
geometry). Reading B: front view with forward-swept fins — rejected (fins would sweep forward of the
wing; drag impossible geometry). Reading C: `Switching to Nose Camera` is literally active — rejected
(the frame shows the whole aircraft; nose cam cannot). Cost of being wrong: nose/tail swapped in tags
21–23 and in F1/F2 assignment — fixable by rotating the world pose 180°; **corridor numbers unchanged**.

### 9.2 Main-gear hypothesis H-A vs H-B (§4.4) — must be measured

Track 0.56 m vs 1.16 m. Both are modelled; Fin-tip outrigger volumes are collision parts in both.
Decision rule: tape the axle track on the airframe, or one side photo with a runway reference (tie-down
rings, runway edge stripe, 0.3 m ruler in-frame). Until then **H-A governs the corridor report** because
it is the conservative (narrower) option.

### 9.3 Tail centreline member height (F2) — the gating number

Modelled at 0.141 m. Range allowed by the image: 0.10–0.23 m. If ≥ 0.23 m: council `APPROACH_REAR`
stands; if lower: rear entry is blocked; see §6.4. **One tape measure settles F2 and F3.**

### 9.4 Where the UGV docks if blockers are real (F4)

Not this document's call: it is an airframe-ops decision (gear retraction, member removal, or
stop-short dock definition) that belongs to Agents 00/09 and the operator. This document's role is to
surface it with numbers attached.

### 9.5 `STRL001` scale

`S = 2.4 m` is an assumption, not a measurement. Wingspan re-anchors the model linearly; corridors
change sub-linearly (wheels/cards pinned). Re-run `strix_corridor_metrics()` after re-anchoring.

### 9.6 Marker IDs 21/22/23

Proposed, collision-free with deployed sets. Board (Agent 00) must assign officially and propagate to
`tools/l7/l7_run.py`, `config/parking.json` tag lists, and `rl/` marker config before any vision run.
Grids included here are the genuine DICT_4X4_50 patterns (cv2 5.0.0 verified), not placeholders.

### 9.7 Missing Visual Frames (screenshot text)

`~CS_LMVS` and `~CS_Pivot1` are absent from the loaded aircraft file — the in-sim model is already a
partial stand-in. Geometry reverse-engineered from it inherits that error; treat as another reason to
measure the real airframe.

---

## 10. Measurement plan (to promote every `confirmed: false` to true)

1. Wingspan tip-to-tip (tape) → re-anchor `wingspan_m`.
2. Main wheel axle track → resolve H-A/H-B; set `track_main_*`.
3. Floor → chin turret underside (calliper at lowest point) → `turret_bottom_z`.
4. Floor → tail member fairing tip → resolve F2 (the decision-relevant measurement).
5. Floor → pod belly plate → `pod_belly_z` (directly sets `belly_clearance_z`).
6. Fin tip z and fin lateral at tip → `fin_tip_z`, `fin_tip_lat`.
7. Pod width/height (calipers or width photo with scale) → `pod_w/h`.
8. UGV height (roof, antenna down) + track → combine with (2–5) into the signed corridor sheet.
9. Tag positions on fins/belly + photograph for pose calibration; side IDs confirmed by board.
10. Update `STRIX_DEFAULTS`, regenerate, `gz sdf -k` the SDF, visual render, and only then re-run
    corridor gates / L7 checks.

## 11. Scope and safety

No source, config, calibration, SDF, or systemd files were modified; this document is the only write.
No hardware was contacted; `docs/SAFETY.md` levels unchanged. All values are provisional estimates in
the same `confirmed: false` discipline as `config/parking.json`. Simulation success (this model in
Gazebo) would **not** constitute hardware parking clearance; the gear-down through-passage finding
(F1/F2/F4) is an airframe fact, not a modelling artifact, and cannot be "passed" by a better policy.
Evidence rules apply: record any measurement with its date and photo/log in the board entry.
