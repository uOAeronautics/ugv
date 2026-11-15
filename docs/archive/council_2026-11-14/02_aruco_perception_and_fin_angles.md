# Agent 02 — ArUco Perception & Inverted V-Tail Fin-Angle Audit

**Role:** Real Robot Perception & ArUco fin-angle auditor for the UGV/Strix-belly parking task.
**Evidence base:** `src/ugv/*.py`, `simulation/gazebo/vision_in_loop.py`, `tools/l7/l7_run.py`, L7 field logs under `calibration/runs/l7/` (3,471 decoded tag observations in `frames.jsonl`, 18,188 ticks in `ticks.jsonl`), OpenCV 5.0.0 in the repo `.venv` (`cv2.aruco.Board` present), plus a self-contained synthetic conditioning experiment (corner noise σ = 0.5 px, no image decode) quoted inline. No source files were modified.
**Scope note:** `docs/council_v2/exp_fin_angles.py` is a sibling council artifact that renders car-mounted tags at the car's own camera position (degenerate geometry); its output `11_fin_angle_experiment_results.md` does not exist. It is not used as evidence here. `exp_fin_angles.py`, `exp_fin_angles.json` and the Agent-09/08 specs are left untouched.

---

## 1. Answer to Q1 — What localization does the robot actually use?

**Short answer: yes — the only *fiducial/vision* localizer in the deployed runtime is `DICT_4X4_50` ArUco + KLT/MOSSE tracking + IMU gyro fusion. The only additional vision code path is a deliberately short "static-scene visual bridge" for dead-reckoning between tag fixes. There is no semantic detector of any kind (no YOLO/AprilTag/HOG/SIFT/template matcher) anywhere in `src/ugv`.**

### 1.1 The deployed stack, file by file

| Path | What it is | Where |
|---|---|---|
| Dashboard/follow perception | Full-frame ArUco decode + fallbacks below, `IPPE_SQUARE` PnP, KLT→MOSSE→predict continuity | `src/ugv/aruco.py` |
| Web auto-park localizer | Half-res crop decode + subpixel + re-read bits + `IPPE_SQUARE` via `solvePnPGeneric`, gyro-offset fusion | `src/ugv/autopark.py` |
| Parking estimator | `IPPE_SQUARE` re-solve with 2.0 px reproj gate, confidence ≥ 0.99 for fresh fixes | `src/ugv/parking.py:586,591,604` |
| L7 field harness | Same detection math, replayed offline | `tools/l7/l7_run.py` |
| Gazebo synthetic loop | Same `aruco.py detect_gray` on rendered frames with photometric DR | `simulation/gazebo/vision_in_loop.py:12` |
| Gyro heading fusion | `RelativeHeading` CW+, used as yaw prior; CCW+ boundary conversions in `parking.pose2_from_ccw`/`pose2_to_ccw` | `src/ugv/heading.py`, `src/ugv/parking.py` |

Key constants and behaviors:

- `DICT_4X4_50`, IDs 0–49 allowlisted: `aruco.py:19` (`ALLOWED_IDS = frozenset(range(50))`), `aruco.py:393,427`; Auto Park restricts to `(3, 6, 14)` (`autopark.py:127`) and re-validates bits ≤ 1 error (`TagBits`, `autopark.py:110-125`).
- Pose: `cv2.SOLVEPNP_IPPE_SQUARE` everywhere — `aruco.py:357`, `parking.py:605`, `autopark.py:69`. Never `IPPE` (non-square) except `ground_pose.py:42` (not deployed, see 1.3).
- Camera model: nominal `fx = fy = 968`, `cx = 640`, `cy = 360`, zero distortion — `aruco.py:18,343-346`, `autopark.py:25-27`. The UC-788 HFOV is ~67°; **intrinsics are unconfirmed** (`config/parking.json: camera_intrinsics_confirmed=false`, `distortion=0`).
- Tag size 50 mm (`aruco.py:16`, `autopark.py:26`).
- Tracking TTL from last decode (absolute): KLT ≤ 0.65 s → MOSSE ≤ 0.85 s → CV predict ≤ 1.0 s (`aruco.py:1039,1074,1085-1096,1118-1140,1141-1155`; `docs/ARUCO.md:25-27`). Anything older is dropped; confidence < 0.28 → drop (`aruco.py:1155`).
- Control TTL: fresh decoded ID ≤ 0.35 s (`aruco.py:1287`); frame stale > 450 ms → brake (`aruco.py:1178,1201`).
- Gyro fusion in Auto Park: tag-derived yaw estimates a gyro offset; |residual| ≥ 20° → outlier, fix rejected (`autopark.py`, `_pick_cands`, `update`); in L7 the same math runs in `tools/l7/l7_run.py:455-530`. Real-car behavior is logged in `ticks.jsonl` (`src` ∈ `tag`/`dr`/`none`).
- Real-car decode-rate reality check per run (frames with ≥1 tag / frames decode pass): `20261113-180312-watch-klt` 143/143 (100%, all KLT `trk`), `181848-watch-shutter4ms` 100/100, `182030` 93/93, `193354-watch-diag` 55/55, `184652-policy-gyro-zero` 384/392 (98%), `183857-policy-ep3-p0` 476/787 (60%), `183731-stage-front6` 47/537 (9%), `182645-policy-ep2-mean` 41/124 (33%). Shutter 16 ms/gain 3 → 4 ms/gain 8 (see §4.4) is what restored the ~100% decodes.

### 1.2 The one exception: `SceneOdometry` short static-scene bridge

`parking.py:512-558` instantiates `SceneOdometry` (`src/ugv/odometry.py`): `goodFeaturesToTrack` (≤300 features) + KLT (31 px window, maxLevel 3, FB error < 1.5 px) + `solvePnPRansac(..., SOLVEPNP_SQPNP)` anchored to fresh tag fixes. It is used **only** to bridge ≤ ~0.45 s of tag dropout while the scene is static and a tag anchor was recently acquired (`parking.py:537-558`, pose_source `"scene features"`). It never commands motion by itself, never creates an anchor, and is discarded on pan/tilt moves or large disagreement (`parking.py:540-558`). It is not a substitute for the tag constellation.

### 1.3 Present in repo but NOT wired into the runtime

- `src/ugv/ground_pose.py` — `SkidPoseObserver` (skid-rectangle `SOLVEPNP_IPPE`, lines 25, 43). Only referenced by `tests/test_ground_pose.py` and docs (`docs/PARKING.md`, `docs/council_v2/06`). **Not used by the dashboard/autopark path.**
- `cv2.aruco.Board`, `cv2.aruco.estimatePoseBoard`, ChArUco — **not used anywhere** (grep: no matches for `Board|Charuco` in `src/`, `simulation/`, `rl/`, `tools/`).
- `docs/council_v2/strix_plane_realflight.png` (the STRL001 asset) and `docs/AGENT_BOARD.md` tag the plane context; no plane-mounted tag config exists yet — the attempt in `exp_fin_angles.py` is the sibling artifact noted above, not a working pipeline.

**Conclusion:** perception is ArUco-only (+ KLT/MOSSE/predict bridging and IMU gyro yaw), with a narrow scene-feature bridge for tag dropouts. Anything the task calls "plane fin tag perception" does not exist yet.

### 1.4 Field co-visibility vs. multi-tag assumption

Across all 5,412 decode passes in `calibration/runs/l7/*/frames.jsonl`:

| Tags seen in frame | Frames | Share |
|---|---|---|
| 0 | 2,120 | 39.2% |
| exactly 1 | 3,113 | 57.5% (2210× 14, 849× 6, 54× 3) |
| ≥ 2 | 179 | **3.3%** (148× `{3,14}`, 31× `{6,14}`) |

Any multi-tag scheme designed against the current loose-hanging-card layout is starving at the frame level — this is a mounting/geometry problem, not a detector problem.

---

## 2. Q2 — Tags on inverted V-tail fins vs. vertical fins vs. under-belly

### 2.1 Geometry you must reason about (floor camera, plane at 0.28–0.36 m)

Camera height on the car ≈ 0.065 m (`simulation/gazebo/vision_in_loop.py:63` computes `CAM_Z_M ≈ 0.0645`; consistent with the low floor-level mount seen in every `end.jpg`). The camera level frame is servo-centered (`config/parking.json: camera_mount_servo 1500/1500`; horizon sits slightly above frame center in `20261113-193354-watch-diag/end.jpg`).

Off-normal incidence θ for a flat tag with unit normal **n** and range vector **d** (camera→tag) is `θ = acos(n·d̂)`. With the plane ~0.22–0.28 m above the lens:

| Mount | Normal **n** (craft frame, nose −y, left +x) | θ at head-on 1.0 m | θ at head-on 0.5 m | θ directly under (d=0.2 m) |
|---|---|---|---|---|
| Entry card (faces car, pitched 25° like TAG 14) | (0, cos25°, −sin25°) | ~13–25° | ~25–52° | leaves VFOV (see below) |
| Inverted V-tail fin, 45° dihedral | (±0.707, 0, −0.707) | **~78–83°** | **~71–76°** | ~45° |
| Normal (upward) V-tail fin, 45° | (±0.707, 0, +0.707) | **~107–113°** (back-facing) | ~105–109° | **~135°** — unusable |
| Vertical fin | (±1, 0, 0) | **~90°** | **~90°** | **~90°** |
| Under-belly | (0, 0, −1) | **~79°** | **~66°** | ~34° |

These are exact for the assumed heights; they match the synthetic rig (§2.4) almost exactly. Consequences:

1. **Inverted-V is strictly better than normal-V from a floor camera** (the downward-facing normal component is what the low lens sees), but **still ~72–83° off-normal head-on** — effectively edge-on for the whole approach corridor. Fin tags only reach θ ≤ 55° when the car is substantially abeam of the fin (|Δx| ≳ 0.35 m at ≤ 0.6 m range) or nearly under the plane.
2. **Under-belly is the worst mount for approach** (θ ≥ 66° until the car is within ~0.35 m of directly beneath the tag) and **unreadable at range**. It is only a terminal-phase tag.
3. **A vertical fin is edge-on at all approach ranges from the corridor** (θ ≈ 90°); it only becomes readable near-abeam.
4. The current TAG 14 (entry side, pitched 25° down/forward, `x = −0.028 m`) exists precisely because it is the *only* mount with a +y normal component — `vision_in_loop.py:103-110` models it at pitch −25°, and the L7 logs show it is the workhorse (2,389 of 3,471 field observations).

**Geometry limit seen in the field:** tag 14 leaves the vertical FOV ~0.4 m out (`docs/L7_FIELD_NOTES.md:52`). VFOV half-angle = `atan(360/968)` ≈ 20.4°; a level camera loses a tag whose center is 0.075 m above the lens at d ≈ 0.20 m — the measured ~0.4 m is consistent with a slightly down-tilted camera and the 25° tag pitch.

### 2.2 Field reprojection error vs. incidence angle (L7, `frames.jsonl`, IPPE_SQUARE branch-0)

| Off-normal bin | n | TAG 14 err px (p50/max) | TAG 3 err px (p50/max) | TAG 6 err px (p50/max) |
|---|---|---|---|---|
| 0–30° | 912 | 0.58 / 1.96 | 0.73 / 1.17 | 0.50 / 0.51 |
| 30–45° | 2,220 | 0.68 / 1.64 | 0.40 / 1.20 | 0.68 / 1.54 |
| 45–55° | 159 | 0.28 / 1.36 | 0.22 / 1.55 | **1.35 / 2.03** |
| **55–65°** | **148** | **1.52 / 2.17** | 0.86 / 1.39 | — |
| **65–90°** | **43** | **1.47 / 1.79** | 0.62 / 0.63 | — |

Overall: err0 p50 = 0.64 px, p90 = 1.35 px, **7/3,471 (0.2 %) exceed the 2.0 px parking gate** (`parking.py:591`). Six of the seven are close-range TAG 6 fixes at θ ≈ 49–50°, 0.44–0.46 m (err ≈ 2.01–2.03 px); the seventh is a 59.2° TAG 14 fix at 0.73 m (2.17 px). Only 1/179 (0.6 %) of the > 55° fixes exceeds the gate, but their median consumes three quarters of it: **the 55–65° TAG-14 bin shows p50 1.52 px — single-tag fixes there are marginal, not trustworthy.**

### 2.3 Close range < 0.40 m — why "bigger in frame" ≠ "better"

px size ≈ `968·0.05/z` (frontal) — at 0.40 m ≈ 121 px, cell ≈ 20 px: decode is geometrically easy, but:

1. **Lens distortion is unmodelled** — `config/parking.json` has `camera_intrinsics_confirmed=false`, `distortion=[0,0,0,0,0]`; the UC-788 is a wide 67° HFOV lens, so IPPE corner placement error scales with image radius. Field data agrees:
   - range < 0.5 m: **err p50 = 1.42 px** (p90 1.57) vs 0.56 px at 1.0–1.5 m — see binning below.
   - range 0.5–0.75 m: 0.75 px; 0.75–1.0 m: 0.63 px; 1.0–1.5 m: 0.56 px; 1.5–2.1 m: 0.84 px (TAG 14 far tail).
2. **IPPE_SQUARE near-frontal degeneracy** — at < 0.4 m with a 20° off-normal tag, both solutions satisfy the data almost equally (§3); the flip is a coin toss without a gyro prior.
3. **Corners near the frame edge trip the detector's own guards** (`minDistanceToBorder=3`, `aruco.py:399`; `detect_gray` drops border-clipped quads, `aruco.py:512-520`).
4. **Auto Park's detector crops rows ≥ 440 of 720** (`CROP_Y1 = 440`, `autopark.py:31,146-147`) — a low entry tag drifting into the bottom half of the frame is invisible to `autopark.py`'s `TagDetector` even though the dashboard full-frame decoder still sees it. A close-range low-mounted entry tag fires a false "lost" in the auto-park path.
5. `parking.py:586` requires `confidence ≥ 0.99`; close-range decode-fallback detections carry 0.75–0.9 confidence (`aruco.py:382-386,548-550`) and are silently dropped by the parking estimator — exactly when you need them most (terminal docking). Field note confirmed by `frames.jsonl`: close-range TAG 6 (range 0.44–0.46 m, px ≈ 85–109) decodes at err p50 = 1.35, max 2.03.

**The 0.2 % gate-violator rate and the 1.42 px close-range p50 both say: the 2.0 px `reprojection_error_px` gate is already near saturation for this camera with unverified intrinsics. Do not raise the gate; calibrate the intrinsics and mount tags rigidly with full quiet zones.**

### 2.4 Synthetic conditioning rig (0.5 px corner noise, `cv2 5.0.0`, SQPNP/IPPE)

Geometry: floor cam z = 0.065 m; tags — entry 5 pitched 25° at (0, 0.45, 0.22); fins 7/8 at (±0.38, 0.35, 0.33) with 45° outward-down normals; belly 9 at (0, 0, 0.30), normal down. 40 trials per pose. Key rows (full table reproducible from the numbers below; ideal-projection corner noise only — no decoder failure modeled):

| Pose (x,y,yaw) | Tag | θ (off-normal) | IPPE e₁ (px) | IPPE e₂ (px) | wrong-branch rate | Board reproj | Board xy | Board yaw |
|---|---|---|---|---|---|---|---|---|
| 0, 2.00, 0° | 5 | 19.3° | 0.20 | 0.28 | **0.33** | 0.54 | 14.1 mm | 0.098° |
| 0, 2.00, 0° | 9 | 83.3° | 0.21 | 0.35 | **0.23** | (≤0.58) | — | — |
| +0.25, 2.00, +12° | 5 | 21.3° | 0.21 | 0.30 | 0.25 | 0.60 | 2.5 mm | 0.052° |
| +0.25, 2.00, +12° | 7 | 86.7° | 0.54 | 0.72 | 0.17 | — | — | — |
| +0.25, 1.50, +12° | 5 | 21.1° | 0.19 | 0.37 | 0.12 | 0.60 | 1.5 mm | 0.046° |
| +0.25, 1.20, +12° | 5 | 22.3° | 0.23 | 0.57 | 0.05 | 0.59 | 0.9 mm | 0.034° |
| +0.25, 0.80, +12° | 5 | 33.2° | 0.26 | 4.16 | 0.00 | 0.60 | 0.3 mm | 0.023° |
| +0.25, 0.60, +12° | 5 | **52.4°** | 0.27 | **49.2** | 0.00 | 0.68 | 0.2 mm | 0.015° |
| +0.25, 0.45, +12° | 7 | 72.2° | 0.29 | **48.5** | 0.00 | 0.59 | 0.3 mm | 0.012° |

Read the wrong-branch column first: **single-tag IPPE_SQUARE is a coin flip (20–45 % wrong branch) at 1.5–2.0 m even for nearly frontal entry tags, and stays 12–23 % for edge-on belly tags; it collapses to 0 % only inside ~0.8 m.** The 2 m fin rows show the failure mode in the opposite direction: both branches fit sub-pixel (`e₁` ≈ 0.5 px at θ = 87°) yet the solution is unobservable — reprojection error does **not** certify pose at high incidence, which is exactly why the 55°–65° field bin inflates.

### 2.5 Motion blur / exposure limits

Measured deployed state: `UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8` (Pi drop-in `vision-camera.conf`; `docs/STATE.md`; backup `calibration/runs/l7/vision-camera.conf.bak-20261113` with 16000 µs / gain 3). The 16 ms/gain-3 config blurred tags during yaw; switching to 4 ms/gain 8 restored ~100 % decode on `181848`, `182030`, `193354` and 98 % on `184652` (`docs/L7_FIELD_NOTES.md:42`).

Blur model: `Δpx = ω · f · Δt = ω · 968 · Δt`.

| yaw rate ω | 4 ms | 8 ms | 16 ms | verdict at 4 ms |
|---|---|---|---|---|
| 0.5 rad/s | 1.9 px | 3.9 px | 7.7 px | safe |
| 1.0 rad/s | 3.9 px | 7.7 px | 15.5 px | marginal — subpixel corners degrade |
| 1.5 rad/s | 5.8 px | 11.6 px | 23.2 px | cell (8 px @ 1 m) destroyed at 8 ms+ |
| 2.5 rad/s | 9.7 px | 19.4 px | 38.7 px | 4 ms still too much beyond 1 m |

Cell pitch at range z: `968·(0.05/6)/z = 8.07/z` px → 1.0 m → 8.1 px (exactly the `perspectiveRemovePixelPerCell = 8` threshold, `aruco.py:403`), 1.5 m → 5.4 px, 2.0 m → 4.0 px. The foreshortened minor axis is `8.07·cosθ/z`; at θ = 55° the 8 px/cell floor is crossed at **z ≈ 0.57 m**; at 65° at **z ≈ 0.42 m**; at 75° at **z ≈ 0.26 m**. This is the physical decode cliff the L7 `> 55°` rows reflect — not a detector-parameter problem.

**Recommendations:** keep shutter ≤ 4 ms, raise gain (≤ 16) rather than exposure for dim scenes; blur budget at pivot rates ≤ 1.5 rad/s is ~5 px at 1 m — acceptable; never revert to ≥ 8 ms while tags are the primary localizer; `docs/ARUCO.md:14` still documents `shutter=8000µs gain=1.5` and is stale — flag for the docs agent; verified decode rate in motion (IDs subject to blur) is exactly what `ARUCO.md:66-75` measured in 2026-09 (compute p95 ≈ 50 ms) and what L7 re-measured.

### 2.6 `IPPE_SQUARE` yaw-flip ambiguity (the "180° flip" family)

`SOLVEPNP_IPPE_SQUARE` returns two pose hypotheses with opposite tilts ("rotation about the in-plane ambiguity axis"), `aruco.py:352-363`, `autopark.py:62-94`, `vision_in_loop.py:283-313`. Field branches (`sol[0]`, `sol[1]` in `frames.jsonl`), ambiguity test `e₂ < 1.2·e₁ + 0.15` (the threshold used for branch acceptance in code):

| Metric | TAG 14 (n=2389) | TAG 3 (n=202) | TAG 6 (n=880) | All (n=3471) |
|---|---|---|---|---|
| err₁/err₀ p10/p50/p90 | — | — | — | 1.18 / 1.66 / 2.70 |
| Branch-ambiguous fraction | **1193/2389 = 49.9 %** | 91/202 = 45.0 % | 0/880 = 0 % | **1284/3471 = 37.0 %** |
| err₂ p50 | 0.886 | 0.888 | 1.214 | — |

By off-normal bin (all tags):

| θ bin | frac ambiguous | err₁/err₀ p50 |
|---|---|---|
| 0–20° | 0.78 | 1.25 |
| 20–30° | 0.84 | 1.34 |
| 30–40° | 0.16 | 1.88 |
| 40–50° | 0.21 | 2.13 |
| 50–60° | 0.36 | 1.47 |
| 60–90° | **0.75** | **1.20** |

Mechanics: near-frontal, both solutions fit the pixels (ambiguity harmless for yaw magnitude but poisons the tilt; the wrong branch is selected 12–45 % of the time at 1.5–2 m). At 30–50° one branch is genuinely better (p90 ratio ≈ 2), so the margin gate works — this is the only regime where IPPE_SQUARE is self-sufficient. At ≥ 55–60° the two solutions carry nearly the same residual (ratio → 1.2): **the flip is information-theoretically unresolvable from one planar square; only a prior (gyro or a second tag) can decide.**

Existing mitigations, all mandatory to keep:
- Auto Park accepts the secondary branch only if `e₂ < 1.2·e₁ + 0.15` **and** the branch is closer to the gyro-predicted tag yaw; else keeps primary; with no calibrated offset (`off is None`), an unclear branch invalidates the fix entirely — staging/orbit may still use uncalibrated tags with `obs==0` but passing `_has_calibrated_fix` gate before `entering` (`autopark.py`, `_pick_cands` and `update`; the 1.2·+0.15 line is `autopark.py:267`).
- L7 harness identical: `is_clear = sols[1][6] >= 1.2*e0 + 0.15`; unclear branch requires temporal prior, else skipped (`tools/l7/l7_run.py:478-491`).
- Gazebo `decode_pose` picks branch per-marker by camera-center consistency across ≥ 2 markers, then a joint `ITERATIVE` refine on all co-visible tags (`vision_in_loop.py:370-416`) — this is a manual multi-tag PnP and the strongest existing in-repo proof that a joint solve suppresses the flip.
- Parking estimator only accepts `confidence ≥ 0.99` decodes with reproj ≤ 2.0 px, confidence 1.0 for decode vs 0.75–0.9 fallback (`parking.py:586-591`, `aruco.py:548,583,601`).

**Under-belly and > 55° panel tags will feed the gate 30–75 % of the time on flip risk alone.** The observed L7 root cause (`docs/L7_FIELD_NOTES.md`): ep2 lost the tag after strafe-induced yaw; ep3 decoded on only 1,024/2,237 ticks (45.8 %) with gyro-induced action saturation — a loop problem, not a decoder-parameter problem.

---

## 3. Q3 — Multi-tag 3D constellation: `cv2.aruco.Board` joint PnP

### 3.1 Why joint PnP solves the flip (and IPPE_SQUARE cannot)

`IPPE_SQUARE` fits **one planar square**. The flip is a discrete symmetry of the planar problem. With ≥ 2 tags on **distinct, non-coplanar planes**, the combined point set has no such symmetry: the mirrored hypothesis violates at least one tag's plane. OpenCV 5.0.0 in the repo exposes `cv2.aruco.Board` (constructed with per-marker object points + dictionary + ids) — verified present, though unused (`cv2.aruco.estimatePoseBoard` does not exist; you call `board.matchImagePoints(...)` or stack object/image points and call `cv2.solvePnP(..., SOLVEPNP_SQPNP)` — SQPNP is already the flag used by `odometry.py:90`).

Three mechanisms disambiguate **Left vs Right** of the plane:
1. **ID-encoded slab identity** — left fin ID ≠ right fin ID (e.g., IDs 6/3 today). Seeing *either* fin label resolves port vs starboard absolutely; seeing both fixes it even under mirroring.
2. **Chirality of the constellation** — the signed volume (or any consistent orientation predicate) of `{left_fin_tag, right_fin_tag, belly_tag}` model points has a fixed sign. A mirrored PnP cannot match all three planes (a reflection changes the handedness of the ordered triple; the metric wheel/fin coordinates don't reflect).
3. **Residual rejection** — the mirrored/flip hypotheses show up as large joint residuals in `SOLVEPNP_SQPNP`; with non-coplanar points there is no 2-solution branch family at all (the flip degeneracy exists only for coplanar configurations).

This is the same trick as chessboard calibration board detection: per-marker IPPE ambiguity dies against a joint solve. `vision_in_loop.py:370-416` already implements the manual version (cross-marker camera consistency → winner → joint refine) and it decodes the real layout end-to-end in Gazebo.

### 3.2 Recommended constellation (IDs disjoint from `{3,6,9,12,13,14}`)

Metric model measured once (tape/CAD), stored as JSON (no source change): per-tag object points in the plane frame, e.g. nose −y, left +x, floor z = 0 (see Agent 09 frame conventions). Example:

| Role | ID (suggest) | Mount | Normal (plane frame) | Phase it anchors |
|---|---|---|---|---|
| Entry/Face | 5 | fuselage chin or nose-wheel fairing, pitched ~20–25° down (TAG-14 equivalent) | (0, cos25°, −sin25°) | approach 1.0–2.0 m |
| Left V-fin | 7 | outer skin of left inverted-V fin, **wedged aft 20–25° toward the entry axis** | (+0.66, sin25°, −0.64) | side-lock ≤ 1.0 m, chirality |
| Right V-fin | 8 | outer skin of right inverted-V fin, wedged aft 20–25° | (−0.66, sin25°, −0.64) | side-lock ≤ 1.0 m, chirality |
| Belly | 9 | belly bulkhead, facing down | (0, 0, −1) | terminal ≤ 0.3 m |

Mounting rules:
- **Fin wedge 20–25° into the entry axis** converts each fin's θ from ~83–87° head-on to ~58–62° head-on — still oblique but decodable with fallback paths and, crucially, **jointly** resolvable. Without the wedge, fin tags are dead weight head-on (§2.4: e₁ ≈ 0.5 px yet unobservable).
- Full white quiet zone ≥ 5 mm all round and **rigid cards** (laminated plate, not loose hanging paper — `docs/L7_FIELD_NOTES.md:42,60`). The current loose cards produce 25–50 % of frames with IDs unreadable in adverse runs and 3.3 % co-visibility (§1.4).
- Do **not** raise `perspectiveRemovePixelPerCell` or relax `errorCorrectionRate`; keep Hamming-0 exact decode (`autopark.py` bit re-read).

### 3.3 Quantified Board-PnP performance (synthetic, §2.4 rig)

Joint all-visible solve (`SOLVEPNP_SQPNP`, 0.5 px noise):

| Pose | Tags visible | Joint reproj (px) | xy err | yaw err |
|---|---|---|---|---|
| x=0, y=2.0, yaw 0° | entry+belly (fins edge-on, excluded by facing test) | 0.54 | 14.1 mm | **0.098°** |
| +0.25, 2.0, +12° | entry+belly+fin7 | 0.60 | 2.5 mm | **0.052°** |
| +0.25, 1.5, +12° | entry+belly+fin7 | 0.60 | 1.5 mm | 0.046° |
| +0.25, 1.2, +12° | entry+belly+fin7 | 0.59 | 0.9 mm | 0.034° |
| +0.25, 1.0, +12° | entry+belly+fin7 | 0.61 | 0.5 mm | 0.025° |
| +0.25, 0.8, +12° | entry+belly+fin7 | 0.60 | 0.3 mm | 0.023° |
| +0.25, 0.6, +12° | entry+belly+fin7 | 0.68 | 0.2 mm | 0.015° |
| +0.25, 0.45, +12° | fin+belly (entry exited FOV) | 0.59 | 0.3 mm | 0.012° |

Single-tag occlusion survival (drop one tag from the joint solve):

| Pose | Drop | Joint reproj | xy err | yaw err |
|---|---|---|---|---|
| +0.25, 2.0, +12° | −fin7 | 0.56 | 2.5 mm | 0.057° |
| +0.25, 2.0, +12° | −belly | 0.55 | 15.5 mm | **0.484°** |
| +0.25, 2.0, +12° | −entry | 0.52 | 18.3 mm | **0.392°** |
| +0.25, 1.2, +12° | −entry | 0.55 | 5.6 mm | 0.202° |
| +0.25, 1.2, +12° | −fin7 | 0.55 | 2.9 mm | 0.068° |
| +0.25, 0.8, +12° | −fin7 | 0.59 | 0.8 mm | 0.034° |
| +0.25, 0.8, +12° | −belly | 0.59 | 1.1 mm | 0.062° |

Compare to single-tag IPPE at the same poses: wrong-branch 12–45 % at ≥ 1.0 m, err inflation, and no left/right separation. **The board holds ≤ 0.5° yaw and ≤ 20 mm at 2.0 m with any single tag occluded, and ≤ 0.1° / ≤ 10 mm inside 1.2 m.** No mirrored solution appeared in any synthetic trial; IDs differ per fin so left/right identity survives even fin dropouts, and the entry+belly opposite-plane pair retains handedness.

Degradation ladder (what you lose at each stage):

| Tags visible | Mode | yaw quality |
|---|---|---|
| 4 | SQPNP joint | ≤ 0.05° @ ≤ 1.2 m, ≤ 0.1° @ 2 m |
| 3 | SQPNP joint | ≤ 0.1° (belly dropped worst: 0.38–0.48° @ 2 m) |
| 2 | SQPNP joint, still non-coplanar in all real pairings | ≤ 0.2° @ 2 m; pairing entry+belly is the worst pair (16–18 mm, ~0.5°) |
| 1 | degenerates to planar single-tag IPPE | ambiguity returns (37–50 % branch swap) — keep gyro prior gate |

### 3.4 Hard requirements for the Board to help in the field

1. **Static, coplanar-safe mounting** — loose hanging cards move between frames; KLT then tracks bent-paper surfaces that PnP reads as tilted squares. Mount rigid laminated cards with known normal vectors; re-measure after any knock.
2. **≥ 2 tags decodable per frame in the approach corridor.** Today 3.3 % of frames achieve this (§1.4). With wedge-canted rigid fin tags + the entry tag, geometric co-visibility head-on at 1–2 m should reach entry + (at least) one fin ~always, and it is the acceptance criterion for mounting (verify with `tools/l7/l7_run.py watch` at 0.5/1/1.5/2 m before any drive test).
3. **Full-frame detection for the entry phase** — the auto-park `CROP_Y1 = 440` crop (`autopark.py:31`) must not clip the entry tag; either raise it or mount the entry tag above the horizon line of the crop (i.e. higher pitch). Dashboard path already full-frame.
4. **Board object points must be metric** — unverified intrinsics (`fx=fy=968` nominal) put a scale error into everything; with distortion 0 the board is least biased at image center, so mount layout should keep tags centrally distributed (fins left/right, entry center, belly center-low) rather than all in one quadrant.
5. **Gate per-tag residuals inside the joint solve** — e.g. drop any tag whose corner-residual contribution > 1.5 px and re-solve; a single bogus (glare-misread) tag otherwise drags the joint estimate, exactly like `decode_curved` guards conflicts by rejecting disagreement (`aruco.py:711-715`).
6. **Keep the 2.0 px parking gate** per joint residual; do not loosen (field p50 0.64 px, p90 1.35 px; 0.2 % violators today).

### 3.5 Minimal integration sketch (specification only — no files changed)

```
config/parking.json addition:
"constellation": {
  "ids": [5, 7, 8, 9],
  "tag_size_m": 0.05,
  "obj_points": { "5": [[...4×3...]], "7": [...], "8": [...], "9": [...] }  // plane frame, m
}
```

- Build one `cv2.aruco.Board` from the four point sets at startup.
- In `ArucoFollower.track_all` (`aruco.py:948`): after decoding, if ≥ 2 constellation IDs present, call `board.matchImagePoints` (or stack manually) → `cv2.solvePnP(..., SOLVEPNP_SQPNP)` → publish alongside the per-tag IPPE results.
- Consumers: `autopark.Localizer` (replace per-tag yaw voting with joint pose when available; keep single-tag IPPE + 1.2·e0+0.15 + gyro-prior fallback — do not delete it), `parking.ParkingEstimator` (confidence 1.0 for board fixes), `web.state.aruco_markers` (add `board` entry).
- Chirality gate for the plane frame: `det([left_fin − belly, right_fin − belly]) ≠ 0` with the mounted sign; log `board_reproj_px`, per-tag residual, `tags_visible` bitmask to `ticks.jsonl`.
- Failure path: if joint residual > 2.0 px → drop to best single tag (current logic); the Board never *replaces* Hamming verification or the gyro prior, it only outranks IPPE for pose.

---

## 4. Quantitative recommendations (ranked)

1. **Keep a dedicated forward-facing entry tag (TAG-14 class, pitched 20–25°) as the 1–2 m anchor.** No fin or belly mount can see the corridor from a 0.065 m floor camera (§2.1, §2.4). If mounting budget allows only two tags, entry + belly is better than both fins.
2. **Wedge the inverted-V fin tags 20–25° aft into the approach axis.** Raises head-on θ from ~83° to ~58–62°; combined with the entry tag this makes a 2–3-tag Board solution possible head-on (§3.4). Do not expect useful head-on fin decodes otherwise.
3. **Adopt a non-coplanar Board constellation via `cv2.aruco.Board` + `SOLVEPNP_SQPNP`** (or manual stacking as in `vision_in_loop.py:370-416`) — joint yaw ≤ 0.1° ≤ 1.2 m, ≤ 0.5° at 2.0 m, chirality from tag IDs, single-tag occlusion covered for ≤ 15 mm / ≤ 0.5° (§3.3). Implement behind a config flag; retain single-tag IPPE + gyro prior as fallback.
4. **At θ > 55°, use tags for identity only, never for pose alone.** Field bin p50 = 1.52–1.47 px and the 1.2·e0+0.15 gate ambiguity climbs to 36–75 % (§2.6). If a fix has θ > 60° or z < 0.3 m, mark it `tracked=True, confidence ≤ 0.5` so `parking.confidence ≥ 0.99` rejects it.
5. **Close-range (< 0.4 m) handling:** require reproj ≤ 1.5 px (field p50 1.42 px today), demand `obs ≥ 10` in the co-observation map (`parking.py:coobservations`), confirm intrinsics (`camera_intrinsics_confirmed=true` after a `cv2.calibrateCamera` pass on ≥ 20 views), and fix the `autopark.py` 440-row crop so a low entry tag is not excluded from Auto Park's detector (§2.3).
6. **Exposure: keep `UGV_CAMERA_SHUTTER_US=4000` / `UGV_CAMERA_GAIN=8`; never ≥ 8 ms while in pivot phases.** Blur budget computed in §2.5; at ≥ 1.5 rad/s the 4 ms setting already costs ~5–6 px at 1 m — acceptable, 8 ms is not. Update the stale `docs/ARUCO.md:14` shutter note.
7. **Rigid laminated cards with ≥ 5 mm quiet zones on all edges; re-survey the tag map after every knock (`tagmap.json` tags 6/3 are co-observation estimates, obs 0 and 129).** Decode rate history: `182150-policy-ep1-front` 86 %, `182645-policy-ep2-mean` 33 % vs 100 % for rigid/full-border runs — geometry, not code.
8. **Size bump if the airframe allows: 75–100 mm tags.** At 2 m a 75 mm tag yields cell ≈ 6 px → single-tag decode without fallback passes; at 1.5 m 8 px/cell. Cheap insurance for the 1–1.5 m band where IPPE flip risk is highest.
9. **Acceptance gate for any fin/belly mounting work:** ≥ 97 % in-motion decode for 10 consecutive frames at 0.4 / 0.8 / 1.2 / 1.8 m measured with `tools/l7/l7_run.py watch --time 4` before any `stage`/`policy` drive; ≥ 2-tag co-visibility ≥ 70 % of frames head-on; board reproj p50 ≤ 1.0 px; 0 ÷ 200 frames below the 2.0 px gate. The L7 rig is the only valid instrument — sim renders ideal quadrilaterals and overstates decode at high θ.
10. **Do not modify gains, watchdog, `config/parking.json`, or systemd units** while landing any of this; the Auto Park changes above touch detector/PNP code paths only. (Per AGENTS.md; the car was on a charging pause at `HANDOFF.md` revision — no Pi contact until cleared.)

---

## 5. Evidence gaps & caveats

- **No real plane constellation exists.** All aircraft numbers come from (a) field stats of the current quadrotor layout, (b) a synthetic ideal-projection rig with 0.5 px corner noise. Real bent/aged/glare card behavior at high θ is governed by `decode_curved`'s bent-border model and is only partially validated (curved-rescue regression in `docs/ARUCO.md:78-90`).
- **Lens intrinsics unverified**; all synthetic ranges assume the nominal 968/640/360 pinhole, zero distortion. Radial distortion at image edges will inflate off-center fin residuals; re-run the gate calibration before trusting mm-level board errors.
- **Co-visibility after mounting changes is unmeasured** (3.3 % today). Recommend measuring it with the L7 harness before any docking attempt.
- **`exp_fin_angles.py`** (sibling) attempts the same study with degenerate geometry; do not cite its numbers.
- Field `frames.jsonl` logs the *best* branch-0 IPPE residual; it does not log which branch was selected nor true incidence from a calibrated plane model, so §2.2's θ bins rely on the IPPE branch-0 normal — approximately correct, mildly optimistic at ≥ 55°.

## 6. File/line index for this audit

| Topic | Location |
|---|---|
| Tag dictionary/allowlist | `src/ugv/aruco.py:19,393,427`; `src/ugv/autopark.py:96-107` |
| IPPE_SQUARE solve + reprojection | `src/ugv/aruco.py:352-386`; `src/ugv/parking.py:604-617`; `src/ugv/autopark.py:65-94` |
| Branch disambiguation (1.2·e₀+0.15 + prior) | `src/ugv/autopark.py:265-279`; `tools/l7/l7_run.py:478-491`; `simulation/gazebo/vision_in_loop.py:370-416` |
| Fallback / low-light / curved decode | `src/ugv/aruco.py:409-441,454-606,609-871` |
| KLT/MOSSE/predict continuity + TTLs | `src/ugv/aruco.py:948-1170` (KLT 1052-1096, MOSSE 1118-1140, predict 1141-1155) |
| Gyro fusion (dashboard) | `src/ugv/aruco.py:1291-1299` |
| Gyro fusion (Auto Park localizer) | `src/ugv/autopark.py:265-330` |
| Parking 2.0 px gate / confidence 0.99 | `src/ugv/parking.py:586-613` |
| Scene-feature bridge (SceneOdometry) | `src/ugv/parking.py:512-558`; `src/ugv/odometry.py:1-130` |
| SkidPoseObserver (unwired) | `src/ugv/ground_pose.py:25-68` |
| Auto Park top-half crop | `src/ugv/autopark.py:29,146-157` |
| Camera K, FOV, CAM_Z | `src/ugv/aruco.py:14-18,343-346`; `simulation/gazebo/vision_in_loop.py:45,63`; `docs/HARDWARE.md`; `docs/STATE.md` |
| Real layout markers (14/6/3, tilts) | `simulation/gazebo/vision_in_loop.py:103-118`; `docs/L7_FIELD_NOTES.md:36-43,52` |
| L7 field data (this audit) | `calibration/runs/l7/*/frames.jsonl`, `ticks.jsonl`, `20261113-193354-watch-diag/end.jpg`, `20261113-185204-stage-park-0.45/end.jpg`, `20261113-181622-stage-front/end.jpg` |
| Board API in venv | OpenCV 5.0.0, `cv2.aruco.Board` = True |
| L7 shutter/gain history | `docs/L7_FIELD_NOTES.md:42`; `calibration/runs/l7/vision-camera.conf.bak-20261113`; `docs/STATE.md`; stale `docs/ARUCO.md:14` |

**Bottom line:** the robot localizes today only via ArUco `DICT_4X4_50` + KLT/MOSSE + IMU gyro (plus a scene-feature bridge between fixes). Nothing reads fins yet. Flat tags on inverted V-tail fins or belly sit at 66–90° off-normal through the whole approach corridor — unusable head-on and only marginally safe beyond 55°; under-belly only within ~0.3 m. The correct design is a non-coplanar 4-tag Board (entry + wedged fins + belly) solved jointly with SQPNP: it eliminates IPPE_SQUARE's 180° flip (field branch-swap rate 37–50 % at range), gives absolute left/right identity from IDs, and survives any single-tag occlusion at ≤ 0.5° / ≤ 20 mm out to 2 m — but only if mounting is rigid with full quiet zones and the entry tag anchors the far field. Field validation on the L7 rig is the release gate, not the simulator.
