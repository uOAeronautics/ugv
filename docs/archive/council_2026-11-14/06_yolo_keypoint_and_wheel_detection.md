# Agent 06 — Aircraft keypoint detection: classical vs learned, PyTorch → ONNX → Pi

Supersedes the earlier draft of this file (written by `beam-free`). Scope: how the car's
camera should localise the STRL001 aircraft (`strix_plane_realflight.png`) well enough to
park in its belly. Analysis and design only. No source file, config, systemd unit, gain,
watchdog or motor path was changed, and the Pi was not contacted.

Measured evidence used here:
- `exp_06_constellation_check.py` → `exp_06_out/constellation_check_sigma{0.5,2,4}.json`
  (synthetic, **illustrative geometry**, 2000 views each; run ids `20261114-111930`,
  `-111726`, `-111728`, all `agent06`).
- Agent 12 belly experiment, run `20261114-111515-belly12` (current file on disk; the
  `111311` figures in the earlier version are superseded).
- Agent 13 ORT / `cv2.dnn` desktop bench (`exp_onnx_out/bench_results.json`).
- Agent 11 fin-angle results (`11_fin_angle_experiment_results.md`).

Nothing below is a measured number from the Pi, from real images, or from a trained
YOLO model. Those are listed as open items in §11.

---

## 0. Verdict

1. **The keypoint-constellation idea is right; the earlier draft's specifics are not.**
   It named a different 8-point set, chose NCNN (the repo deploys ONNX via `cv2.dnn` or
   `onnxruntime`), and gave Pi latencies (12–20 ms for YOLO11n INT8) with no source.
2. **Primary model: a small spatial-soft-argmax keypoint CNN with a presence head**
   (the Agent 12 architecture), trained in PyTorch on the 3090, exported to ONNX and run
   with `onnxruntime` (`cv2.dnn` as the zero-dependency fallback). Measured on the desktop
   only: 1.34 ms per frame on one thread at 160×120 for the Agent 12 CNN (118,514 params);
   Agent 13's 106k-param 8-keypoint net runs in 0.76 ms (ORT) vs 1.57 ms (`cv2.dnn`) on one
   host thread. Pi figures are ×3.5 projections, not measurements.
   **YOLO11n-pose (or v8n-pose) is an ablation on the same split**, not the default. It
   is worth it only if the CNN misses at range on real images.
3. **HOG is rejected** (50% distractor FPR, 65% recall in Agent 12). Hough/ellipse wheel
   detection is rejected (no nose/tail, many circular clutter hits). `ground_pose.py` is
   reused only for its gating pattern: its 4-point IPPE skid rectangle does not fit an
   aircraft.
4. **Left/right side, fins and clutter are all solved by one thing: a labelled 3D
   constellation with ≥6 points, a reprojection gate, and a chirality test.** Measured
   results (§5) show it rejects swapped labels with 0% acceptance at 8 points, but
   accepts about 10% of swaps with only 4 points. So the 4-point mode is not allowed to
   drive control.
5. **"Zero false positives" is not a claim this design can make.** It can only state a
   measured upper bound on clutter (rule of three; §8).

---

## 1. Audit of the earlier draft

| # | Earlier draft says | Problem | Fix here |
|---|---|---|---|
| 1 | Keypoints `nose_tip`, `belly_center`, `left/right_fin_root/tip` | Not the spec. No ground contacts (nose gear, main wheels), so the z=0 constraint cannot be used | Use the 8-point spec in §2 |
| 2 | Deploy via NCNN INT8 | Adds a runtime the repo does not have. Deploy path is ONNX via `cv2.dnn` (root dep, already installed) or `onnxruntime` (`council_v2/13`) | ONNX, §7 |
| 3 | YOLO11n INT8 ~12–20 ms at 320×256 on Pi 4B | No measurement, no source. Council 13 says desktop → Pi is 5–10×, so this is not credible. The only measured Pi-class number is for a different model | Latency is an open item (§11); no figure is claimed |
| 4 | "`solvePnP` on wheel/keypoints is the same as `ground_pose.py`" | `ground_pose.py` is 4 coplanar points + IPPE (a homography). The aircraft is 8 non-coplanar points; different solver requirement | SQPNP on the 3D constellation (§4) |
| 5 | Chirality via "signed volume or planar signed area" | Vague. A 3D signed volume needs a fourth non-coplanar point and only holds if the labels are right; the image-space test is the one to use | §5.2: image sign, calibrated once |
| 6 | Reprojection RMS < 3 px | A single fixed threshold. Measured: the 3 px gate passes 100% of correct 8-point solutions at 0.5 px noise, 91.5% at 2 px, and 6.5% at 4 px (§5.3) | Threshold scales with measured detector noise |
| 7 | Class score floor 0.35–0.50 "tuned on data" | No data existed. Agent 12 CNN (presence p > 0.99) still gives 10% distractor FPR | The presence threshold is a measurement; the geometric gates carry the FP load (§6) |
| 8 | Claims a `StagedBaseline` "freshness" budget (0.35 s / 1.0 s) | `aruco.py` has the 0.35 s decode-age cutoff. The 1.0 s value and the χ² gate (`CHI2_GATE = 16.3`, `staged_baseline.py:313`) were not checked beyond that | Stated as the existing EKF gate; the 1.0 s is unverified |
| 9 | Assumes wheel contacts are visible and on the ground | Agent 03 §1.2: contacts for this airframe are **unresolved** in the screenshot (fin-tip pegs may be the ground contacts). The rigid model is not yet grounded | Blocker, §3 |
| 10 | Treats the airframe as generic | Agent 03 and 09: wings, props and fins are close to mirror-symmetric. Mirror ambiguity is the central risk and the earlier draft did not analyse it | §5.1 analyses it directly |

---

## 2. Keypoint spec and frame

Aircraft frame F_P (matches Agents 03/09): **+y = nose**, **+x = aircraft left**,
**+z = up**, origin = ground point under the fuselage centre (to be fixed in CAD, not
yet measured). Keypoint order is fixed for training, export and PnP:

| idx | Name | On ground (z=0)? | Notes |
|---|---|---|---|
| 0 | `nose_gear` | yes (if tricycle gear is confirmed) | Fixes forward direction |
| 1 | `left_main_wheel_contact` | yes | |
| 2 | `right_main_wheel_contact` | yes | |
| 3 | `belly_turret` | no | Under chin canister; visible from low camera |
| 4 | `left_vtail_root` | no | Fuselage side |
| 5 | `left_vtail_tip` | maybe | Agent 03 item 7: fin-tip peg may touch ground |
| 6 | `right_vtail_root` | no | |
| 7 | `right_vtail_tip` | maybe | As above |

**Left/right flip index** (for the training flip augmentation and for any image mirror):
`flip_idx = [0, 2, 1, 3, 6, 7, 4, 5]`. A horizontal image flip is a mirror view, so the
labels must swap as well. Without this, flip augmentation teaches the network a mirror
contradiction.

Model geometry is **not measured**. The synthetic model in `exp_06_constellation_check.py`
only respects the layout (tricycle gear, belly under the chin, fins splayed to near the
ground). Its numbers are used only to test the logic in §5.

---

## 3. Blockers before any real model

1. **Which points touch the ground on STRL001?** If the fin-tip pegs and nose gear are
   the contacts, the aircraft has five coplanar ground points (nose, both wheels, both
   tips). That gives a full planar homography and a direct check on z=0. If only three
   contacts exist, a homography is impossible (IPPE needs ≥4 coplanar points). Then the
   z=0 constraint must come from the 3D model alone. Resolve in the sim CAD first.
2. **Dimensions.** Agent 03's values are image-derived with ±20–30% scale error and an
   assumed 2.4 m span. The fin dihedral (~38–40°) and the gear track are the two numbers
   that matter most. Measure on the airframe before setting `confirmed` anywhere.
3. **Visibility at close range.** Camera HFOV is 67° (f = 968 px at 1280 px). On the
   optical axis, the half-width visible at depth d is 0.661·d. Consequences for the
   illustrative geometry:
   - main contacts at lateral ±0.28 m are in frame from d ≥ 0.42 m;
   - fin tips at ±0.61 m are in frame only from d ≥ 0.92 m;
   - so **at close range the fin tips leave the frame**, and the 6-point (no-tips) subset
     becomes the operating mode there.
   These thresholds use the illustrative geometry; redo with measured dimensions.

---

## 4. Solver: SQPNP on the labelled constellation

- Input: detector keypoints (2D, pixels at the 1280×720 frame) with confidences.
- Solver: `cv2.solvePnP(..., flags=cv2.SOLVEPNP_SQPNP)` on the visible subset. Refine with
  `solvePnPRefineLM`. Reject any solution with a model point behind the camera.
- Points on the ground (z=0) are part of the 3D model, so no separate homography is
  needed for pose. The homography is only a cross-check, and only if §3.1 gives ≥4
  coplanar ground points.
- Output: `(R, t)` → camera centre in F_P, plus RMS and per-point residuals.

### 4.1 Left vs right side (the plane side the car is on)

The car is on the aircraft's **left** if the camera centre in F_P has x > 0, and on the
**right** if x < 0. This is a property of the fitted pose, not of a classifier's label.
A mislabelled L/R pair produces a pose on the wrong side and is caught by §5.

### 4.2 Fin angles

Fin dihedral is a fixed property of the airframe, so the pose already contains it. The
detector's job is to localise fin roots and tips under steep views. Evidence that this
works: Agent 11 found single-tag IPPE on a 55° dihedral fin had 53–62% flip rate at 1.0 m
(`B_Dihedral55_Swept35`), while a 3-tag board with SQPNP had 0% flips at 0.25–0.5 m
(Condition C). The constellation extends the board result from tags to airframe
keypoints.

---

## 5. Measured checks (`exp_06_constellation_check.py`, synthetic, illustrative geometry)

Noise: independent Gaussian keypoint noise σ in pixels. Gate: SQPNP RMS < 3 px.

### 5.1 Q1 — can reprojection reject swapped L/R labels? (mirror ambiguity)

| subset | points | σ | correct labels pass 3 px gate | **swapped labels accepted < 3 px** | accepted swaps that are the mirror pose |
|---|---|---|---|---|---|
| all8 | 8 | 0.5 | 100.0% | **0.0%** | 0.0% |
| all8 | 8 | 2 | 91.5% | **0.0%** | 0.0% |
| all8 | 8 | 4 | 6.5% | **0.0%** | 0.0% |
| no_fin_tips | 6 | 0.5 | 100.0% | **0.0%** | 0.0% |
| no_fin_tips | 6 | 2 | 96.0% | **0.0%** | 0.0% |
| no_fin_tips | 6 | 4 | 21.4% | **0.0%** | 0.0% |
| contacts+belly | 4 | 0.5 | 98.5% | **10.2%** | 0.0% |
| contacts+belly | 4 | 2 | 97.1% | **10.1%** | 0.0% |
| contacts+belly | 4 | 4 | 65.4% | **8.3%** | 0.0% |

Reading:
- With the full non-coplanar constellation, a swapped labelling does not fit within 3 px
  at any tested noise level. The reprojection gate is therefore a valid L/R test **only
  with the full set**.
- With four points, about 10% of swapped labellings fit within 3 px. So the 4-point mode
  must not drive control.
- Accepted swaps are never the exact mirror pose in this test. The swapped solution is a
  different wrong pose, so checking the side sign (§4.1) also catches it.

### 5.2 Q2 — image-space chirality sign

Test: sign of det([p_L − p_nose, p_R − p_nose]) in pixel coordinates, with camera height
z ∈ [0.05, 0.6] m (always above the ground plane).

- Correct labels: sign constant in **100.0%** of 2000 views (majority = +1 in this pixel
  convention; the sign is a convention and must be calibrated once on real labelled images).
- Swapped labels: the sign is the opposite in 100.0% of views (majority-sign match 0.0%).
- Ground-plane determinant of the model: −0.364 m² (aircraft frame). Its sign is fixed.

Use: reject any labelling whose image sign differs from the calibrated sign. This is an
independent check from reprojection and costs nothing. It assumes the camera is above the
ground plane; reject views where the camera pitch or mounting puts it below.

### 5.3 Threshold scales with detector noise

The "3 px" gate is only right for about 1.5 px keypoint noise. Correct-label pass rates
at the 3 px gate (8 points): 100.0% at σ = 0.5, 91.5% at σ = 2, 6.5% at σ = 4. The gate
must be set from **measured** residuals on real labelled frames (target: pass ≥ 95% of
correct frames), not copied from here.

### 5.4 Q3 — outliers (clutter)

- One keypoint moved 40 px (chair-leg-sized error): full 8-point RMS < 3 px in **0.0%**
  of trials.
- Leave-one-out refit identifies the bad keypoint in **100.0%** of trials at σ = 0.5 and
  99.8% at σ = 4 (500 trials). So a leave-one-out step can drop one bad keypoint before
  the final fit.

### 5.5 Q4 — random 8-point sets

3000 uniformly random 8-point sets: **0.0%** fit the model within 3 px.

**What these checks do not cover:** real model mismatch (CAD vs airframe), occlusion,
correlated detector errors, label noise in training data, and real lighting. They show
the geometric logic is sound; they do not show the detector is good.

---

## 6. Detector: what to train and why

### 6.1 Options compared

| option | status in repo | strengths | weaknesses | decision |
|---|---|---|---|---|
| HOG + linear SVM | Agent 12 | CPU only, simple | 50% distractor FPR; 65% recall; 17.5 ms/frame desktop 1 thread; box only | Reject (baseline only) |
| Hough / ellipse wheels | not built | Cheap | Many circular clutter hits; no nose/tail; no 6-DoF | Reject |
| ArUco on airframe tags | `src/ugv/aruco.py` | Metric, IDs, tested | Agent 12: 0.8% recall overall, 1.0% on frames with a tag in view; cannot see wheels; fails edge-on | Keep as independent cross-check and fallback |
| `ground_pose.py` skid IPPE | `src/ugv/ground_pose.py` | Good gating pattern (below-plane reject, yaw continuity) | 4 coplanar points; drone skids, not aircraft | Reuse pattern only |
| **Keypoint CNN + presence head (SSA)** | Agent 12 (measured) | 98.6% recall; 98.1% at ≥40° off-axis; 1.34 ms desktop 1 thread; 118k params; soft-argmax gives sub-pixel points | 10% distractor FPR before geometry; trained on synthetic only so far | **Primary** |
| YOLO11n-pose / YOLOv8n-pose | not built | Pretrained backbone; detection + keypoints in one head; Ultralytics export path | Pretrained on people, not this airframe; export has ops that need `onnxruntime`; Pi latency unknown | **Ablation** on the same split |
| NanoDet-Plus + keypoint head | not built | Light detector | Two networks, two failure modes | Not pursued |

### 6.2 Primary architecture (SSA keypoint CNN)

- Input: 1280×720 → resize to 320×180 → pad to 320×192 (divisible by 16).
- Backbone: 4 stride-2 conv blocks → 1/16 resolution feature map. Size: ~118k params, the
  Agent 12 size.
- Per keypoint: one 1×1 conv heatmap → spatial softmax → soft-argmax (x, y) and a
  confidence = peak mass. Eight keypoints total.
- Presence head: global average pool → 2-layer MLP → p(aircraft). Trained with hard
  negatives.
- Export: `torch.onnx.export(..., dynamo=False, opset_version=17)`, the same pattern as
  `rl/export.py --onnx` (`council_v2/13`).

### 6.3 Ablation architecture (YOLO-pose)

- Ultralytics pose model, 1 class (`aircraft`), `kpt_shape: [8, 3]`,
  `flip_idx: [0, 2, 1, 3, 6, 7, 4, 5]`, `imgsz: 320`.
- Visibility per keypoint: 0 = unlabelled, 1 = occluded, 2 = visible.
- Export to ONNX and run with `onnxruntime`. Council 13 reports that `cv2.dnn` hits
  unsupported-op and reshape problems on C2f/SiLU/Split graphs, so it is not used here.

### 6.4 Training data and negatives

- Positives: real frames of STRL001 (or the physical airframe) from the car camera at
  0.3–2.0 m and yaw up to ~60°, plus Gazebo renders (`simulation/gazebo/`) with the same
  intrinsics.
- Keypoint labels: CAD-projected for sim; hand-annotated for real, with visibility flags.
  Check L/R labels on every image. One mislabelled example can teach a mirror (Agent 03,
  09).
- Negatives (essential): chairs and chair legs, table legs, shoes, tripods, floor
  reflections, stray ArUco tags, boxes, dark discs. Mine hard negatives from the current
  detector's false positives.
- Split: by session, not by frame, so no near-duplicate leaks into validation.
- Airframe asymmetry: Agent 09 proposes asymmetric marker tags on the fins to break the
  mirror symmetry. If adopted, add it to the training distribution and to the 6-point
  subset in §5.

### 6.5 Losses and training

- Keypoints: L1 on soft-argmax coordinates, visible points only; plus a heatmap spread
  regulariser to stop collapse.
- Visibility: BCE per keypoint.
- Presence: BCE with hard negatives weighted up.
- Augment: brightness/gain, glare, blur, noise, partial occlusion, scale jitter. **Flip
  augmentation only with `flip_idx` applied.**
- Compute: 3090 (`rl/` CUDA environment). This shell has no CUDA torch, so training
  was not run here.

---

## 7. Export and Pi runtime

- Train in PyTorch (3090). Export ONNX. Gate on parity: ORT and `cv2.dnn` outputs vs
  PyTorch, max abs error < 1e-5 (same gate as `council_v2/13`).
- Deploy with **`onnxruntime` (ARM64 CPU) as primary**: Council 13 measured it 2.07×
  faster than `cv2.dnn` for the 8-keypoint net (0.759 ms vs 1.574 ms, one host thread).
  It is not torch, so the torch-free rule holds. It needs one small optional wheel; this
  is a dependency decision for the user, not made here.
- **`cv2.dnn.readNetFromONNX` as the zero-dependency fallback** (already a root dependency,
  `opencv-contrib-python-headless>=4.13`). Known footgun from Council 13: a bare
  `net.forward()` on a multi-output graph returns only the first output and drops the rest
  silently. Always request outputs by name.
- No torch on the Pi. The root package stays torch-free.
- INT8: not used until an FP32 model misses the frame budget on the Pi.
- **Latency: not measured.** Pi numbers must come from running `exp_onnx_cv2dnn_bench.py`
  on the Pi at the car's real load (camera + control threads running), median and p95.
  Until then, no control rate is claimed.

---

## 8. Gates and false-positive statistics

Keypoint presence and geometry are gates in series. A frame yields a pose only if all pass:

1. **Presence:** p(aircraft) above a threshold set on a calibration split.
2. **Keypoint confidence:** at least 6 points above a confidence floor, including ≥2
   fin points for the side decision (§5.1).
3. **SQPNP residual:** RMS below the noise-scaled threshold (§5.3).
4. **Chirality:** image sign equals the calibrated sign (§5.2).
5. **Metric check:** wheel spacing (|p_L − p_R| vs model width, 3 edges of the contact
   triangle) within tolerance. Chair or table legs are rejected because they do not
   reproduce the airframe's distances.
6. **Temporal lock:** K = 3 consecutive frames within 3 cm and 5° and the same side label,
   then the EKF innovation under the existing χ² gate (`CHI2_GATE = 16.3`,
   `staged_baseline.py:313`). A rejected frame never updates the estimator.

**Statistics.** "Zero false positives" becomes a measured upper bound:

- Frame level: 0 accepted poses in N negative frames → rate < 3/N at 95% (rule of three).
  N = 30,000 negative frames gives < 1.0e-4 per frame.
- Lock level (what matters for control): 0 locks in T hours of clutter video → rate < 3/T
  per hour at 95%. Target: ≥10 h of distractor video with 0 locks, i.e. < 0.3 per hour.
- Both are measured on held-out clutter, never on the training clutter.

The earlier figure of "< 1% of frames" is replaced by these bounds.

---

## 9. Integration (design only, no code yet)

Proposed module `src/ugv/aircraft_pose.py` (torch-free; loads the ONNX via `cv2.dnn`):

```
frame 1280x720 (latest-frame capture, unchanged)
  -> resize/pad 320x192
  -> keypoint CNN (cv2.dnn)
  -> presence + 8 keypoints + confidences
  -> gates 1-5 (§8)
  -> AircraftPoseObservation(t_capture, pose_F_P, side, rms_px, n_points, kp_conf, valid)
  -> temporal lock (gate 6)
  -> fuse with ArUco observation: prefer the lower-residual valid one, log both
  -> EKF update at 50 Hz using the latest valid observation and its age
```

- Pose frames: convert F_P to the policy frame (x, y, yaw) through one rigid transform,
  defined in one place. Yaw is CCW positive in the policy (`autopark.py`, `policy.py`,
  `tools/l7/l7_run.py`, `rl/sim.py`). `Pose2` (CW) only at the parking boundary, via
  `pose2_from_ccw` / `pose2_to_ccw`.
- Age: observations older than the existing freshness limit are not used for control.
  `aruco.py` already stops driving at `age_s > 0.35`.
- **Pose is an observation, not a parked car.** Parked still requires the end frame
  (`docs/SAFETY.md`).

---

## 10. Measured evidence used (run ids)

| source | run id / file | key result |
|---|---|---|
| Agent 12 belly, current | `20261114-111515-belly12` | CNN recall 98.6%, close 90.6%, steep 98.1%, distractor FPR 10.0%, 1.34 ms desktop 1 thread. HOG 64.8% recall, 50.0% FPR. ArUco 0.8% recall |
| Constellation σ=0.5 | `20261114-111930-agent06` | swaps accepted: 0% (8pt), 0% (6pt), 10.2% (4pt); chirality 100% constant |
| Constellation σ=2 | `20261114-111726-agent06` | correct pass 91.5% (8pt); swaps 0% (8pt), 10.1% (4pt) |
| Constellation σ=4 | `20261114-111728-agent06` | correct pass 6.5% (8pt); swaps 0% (8pt), 8.3% (4pt); outlier ID 99.8% |
| Agent 13 ORT vs cv2.dnn | `13_pytorch_to_onnx_pi_pipeline.md`, `exp_onnx_out/bench_results.json` | 106k-param 8-keypoint net: ORT 0.759 ms vs cv2.dnn 1.574 ms, one host thread (Pi values are ×3.5 projections); bare cv2.dnn `forward()` returns only the first output |
| Agent 11 fins | `11_fin_angle_experiment_results.md` | single-tag IPPE flip 53–62% at 1.0 m (55°); 3-tag board SQPNP 0% flip at 0.25–0.5 m |

---

## 11. Open items (ordered; each one blocks the next)

1. **Ground-contact question** (§3.1): which airframe points touch the runway. Resolve in
   the sim CAD and from the real airframe.
2. **Measure the airframe** (§3.2): gear track, nose-gear position, fin root/tip, fin
   dihedral. Set nothing `confirmed` until measured.
3. **Real labelled set**: start with ~1000 frames across range and yaw, with clutter
   negatives, labelled L/R checked. Record the run id.
4. **Calibrate the gate thresholds** from measured residuals on real frames (§5.3) and
   the chirality sign (§5.2).
5. **Train both architectures** (CNN primary, YOLO-pose ablation) on the same split on the
   3090, with the same metrics: keypoint error (px), pose error (cm/deg), side accuracy,
   lock-level FPR on clutter.
6. **Pi latency**: measure median and p95 for the chosen model on the Pi under real load.
   Until this is done, no control rate is claimed.
7. **Safety gate**: the first real-world run stays at the level in `docs/SAFETY.md`
   (read-only for perception). No motion is authorised by this document.
