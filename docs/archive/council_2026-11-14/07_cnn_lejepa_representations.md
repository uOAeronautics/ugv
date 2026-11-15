# Agent 07 — Visual JEPA / LeJEPA + Lightweight Spatial CNN Encoders for Under-Belly Parking

Scope: the council question "cnn lejepa? hog/yolo? some other clever approach?" for the Freenove
mecanum UGV docking under the fixed-wing target (`docs/council_v2/strix_plane_realflight.png`:
high-wing Cessna-class airframe, inverted-V tail, fixed gear with wheels, on a runway). The same
analysis maps onto this repo's actual lab target — the quadcopter in `simulation/gazebo/DRONE_MODEL.md`
(two foam skids with red-ringed end-caps, clear floor gap ≈ 310 mm, deck ≈ 255 mm at ≈ 200 mm height) —
because the sensing problem is identical: a low monochrome camera sees an under-body target whose
**structure** (two rails + center belly + tail apex) is stable while its **appearance** (floor, glare,
clutter) is not.

This document is an analysis and architecture blueprint. No source files were modified. It complements
Agent 06 (`docs/council_v2/06_yolo_keypoint_and_wheel_detection.md`), which covers supervised HOG/YOLO/
pose detectors; §9 relates the two. Every measured number below is cited to a repo path or a run id.
Estimates are labelled `[est]` and unverified extrapolations are labelled `[inferred]`. Nothing here
is a release gate; the only success gate is `rl/evaluate.py`, 1000 episodes, seed 11, plus the L7
procedure in `docs/SAFETY.md`.

---

## 1. Premise correction (read first)

The task brief describes `rl/models.py` as already holding "a 1D Temporal Conv `HistoryEncoder` +
`jepa_pred` latent dynamics predictor on the 50-step observation history". That is not what the code
contains today, and the report would be fiction without saying so:

- `rl/models.py` `Student` (`rl/models.py:44`) is **GRU-64** (`nn.GRU`, input `STUDENT_DIM = 13`,
  hidden 64) + 2×64 tanh MLP head. There is no temporal-convolution encoder and no module named
  `jepa_pred` anywhere in the repo (verified by repository-wide search).
- The 50-step history is `HIST = 50` "1 s at 50 Hz" (`rl/sim.py:171`). Student obs frame =
  tag(x,y,yaw) + age + valid + imu_w + wrapped imu_yaw + prev_act(3) + EMA tag velocity(3) = 13 dims
  (`rl/sim.py:174-175`, mirrored in `src/ugv/policy.py:14-15`).
- The JEPA-*like* machinery that exists is **train-only latent supervision on the GRU hidden state**,
  added under the `rl-curric-and-jepa` / `jepa-rma-calibration` board items
  (`docs/AGENT_BOARD.md:49-50`): `Student.aux_head = Linear(h, 6)` predicting privileged
  `[pose(3), vel(3)]` + anti-collapse `0.01·relu(1 − h.std)` term, and `Student.dyn_head = Linear(h, DYN_DIM=12)`
  (`rl/models.py:63-69`) predicting the privileged dynamics vector. `export.py` ships **only `gru.*` and
  `head.*` keys** (`rl/export.py:50`), so the deployed `.npz` and `src/ugv/policy.py` forward are unchanged.
  This is RMA/LeWM-flavoured auxiliary supervision, not a visual JEPA and not a masked/future-image predictor.
- The one logged experiment in that family, `student_L1_jepa_s0` (`rl/runs/student_L1_jepa_s0.json`,
  archive `student_L1_jepa_s0`, 2026-11-13T19:02:15Z), **failed its gate**: L1, 1000 eps, seed 11,
  success 0, collisions 56 (5.6%), timeouts 944 (94.4%), median 45.0 s. The failure was dominated by
  timeouts/no-progress, not instability; the aux head did not buy success. It is a cautionary data
  point: "add a latent-prediction loss" is not by itself a fix for the L1 gate, and its cause was never
  isolated (collision rate did improve 866→56 vs the no-curriculum predecessor).

Consequences for this study:

1. "Visual JEPA for the UGV" is **new ground**, not an extension of an existing module.
2. The `HistoryEncoder` the visual front-end must feed is, in reality, the **existing GRU-64** over the
   13-dim frame (or a resized input). The design below is phrased against that reality.
3. Prior art in-repo already establishes the deploy discipline any visual addition must obey:
   training-only heads, export strips them, NumPy forward, sha-pinned `.npz`, ~0 added Pi cost — or, for
   a visual encoder, an explicitly budgeted and measured cost (§8).

---

## 2. The sensing gap this addresses

Measured, not hypothetical:

- Tags are the *primary* localiser today (`docs/ARUCO.md`), and they go blind exactly where parking is
  decided: Tag 14 leaves the vertical FOV ≈ 0.4 m out (`docs/L7_FIELD_NOTES.md:52`); L3+ levels model tag
  loss under the drone with `under_drone_loss` (`rl/sim.py`); real L7 runs went 0/3 parked with
  oscillation while drifting (`docs/L7_FIELD_NOTES.md` — ep1 42.96 s timeout, ep2 `lost` at 11.31 s after
  strafe-induced yaw, ep3 45.98 s timeout, drift 0.57→1.10 m).
- Stock ArUco found Tag 6 in only ~25% of frames on the real car (quiet-zone defect on one card edge)
  (`docs/L7_FIELD_NOTES.md`, vision section).
- The Gazebo camera path reproduces the same blindness: L6 0/20 for `student_L1_latw5_s0`
  (`docs/AGENT_BOARD.md:58`), and the `--lighting-dr` probe decoded at 5.2% vs 98.3% clean (ibid.).
- Pulled Pi frames (`HANDOFF.md` §8) show specular floor glare, foam skid end-caps, and lab clutter that
  the tags-only synthetic frames lack. This is the sim-to-real visual gap in one sentence.
- There is already a classical, non-tag skid observer in-repo: `src/ugv/ground_pose.py`
  (`SkidPoseObserver`, `solvePnPGeneric(SOLVEPNP_IPPE)` on four foam-skid end-cap centres, LM refinement,
  mirrored-pose rejection; `docs/PARKING.md:37`). It recovered an oblique 35° approach at ≈ 5.7→4.4 px
  reprojection error as an *analysis*, with no control authority. It is the strongest evidence that
  **under-body geometry alone is sufficient to localise the dock**, and the cleanest baseline the learned
  encoder must beat.

The visual endgame therefore needs a representation that (i) works from ≈ 0.3–2.0 m, (ii) survives
glare/darkness/clutter/blur, (iii) survives the last ≈ 0.4 m where every tag is out of frame or shadowed,
and (iv) costs ≤ a few ms on a Pi 4B that is already spending p50 ≈ 44–50 ms/frame fetch+detect against a
33.3 ms budget (`HANDOFF.md` §10, `vision_ms_p50 = 50.0 ms` in the watch-diag run).

---

## 3. Why pixel-reconstruction autoencoders lose, and why JEPA/LeJEPA wins here

### 3.1 The capacity accounting argument

An under-belly approach image is dominated by two kinds of energy:

- **High-frequency, task-irrelevant**: concrete speckle, asphalt grain, sun/specular glare streaks,
  shadow boundaries, dust, label print on the floor, compression artefacts. The Gazebo frame generator
  does not model these at all (`HANDOFF.md` §8), so they appear only at deployment.
- **Low/mid-frequency, task-relevant**: the two-point symmetry of skids/wheels, the center belly mass,
  the V-tail apex high in the frame, the horizon/runway vanishing line, the drone's shadow.

A pixel MSE decoder must drive *every* pixel toward its target. Gradient mass concentrates where pixel
variance is largest — floor texture and glare — because that is where the loss is cheapest to reduce
first. The encoder is paid to *transmit* light, not *structure*. Result: features that encode
"concrete brightness at (312, 480)" alongside nose azimuth, entangled such that a lighting change reads
as a pose change. On this exact problem, that is the failure mode behind every "works in Gazebo, blind
on the lab floor" report in `docs/AGENT_BOARD.md`.

A joint-embedding objective changes the question from "reconstruct the pixels" to "make the embeddings
of two compatible views mutually predictable". In a synthetic DR pipeline, the two views (same pose,
independent photometric draw) share **only** geometry. The encoder is therefore paid to *discard*
glare/texture/blur and *keep* shape — the invariance is constructed, not hoped for.

### 3.2 The JEPA family, precisely

- **I-JEPA** (Assran et al. 2023, arXiv:2301.08243): context block(s) → predictor → predict the
  *target encoder's* embedding of held-out target blocks. Non-generative, no pixel decoder; semantic
  structure emerges without hand-crafted augmentations. Target encoder is EMA of context encoder.
- **V-JEPA** (Bardes et al. 2024, arXiv:2404.08471): the same idea on video tubes; temporal prediction
  of latent features.
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906): explicit anti-collapse with three plain
  terms — invariance (MSE between view embeddings), variance (hinge, per-dimension std ≥ 1), covariance
  (off-diagonal penalty). No negatives, no EMA, no stop-grad; two-three loss weights to tune
  (classically 25/25/1).
- **LeJEPA** (Balestriero & LeCun 2025, arXiv:2511.08544): proves the **isotropic Gaussian** is the
  optimal embedding distribution for JEPA downstream risk, and replaces the heuristic stack (EMA teacher,
  stop-grad, schedulers) with **SIGReg** — *Sketched Isotropic Gaussian Regularization*: project
  embeddings onto random 1-D directions (sketches) and penalise deviation from N(0, I) with a normality
  test statistic, averaged over sketches. Single trade-off hyperparameter λ, linear time/memory, reported
  stable across ResNet/ViT/ConvNet architectures and notably on **small, domain-specific corpora**
  (medical, galaxy imagery) — our regime is ≈ 10⁴–10⁵ renders, not ImageNet. ~50 lines on top of any
  backbone.
- **LeWM / JEPA-dynamics** (internal board ids `2603.19312`, `2610.39302` — "LeWM: stable e2e JEPA =
  next-embedding prediction + Gaussian-latent regulariser, no EMA/pretrain, ~15M params";
  "Platonic-JEPA: two views of one scene → same representation up to rotation, 42× sample efficiency vs
  next-token-prediction"): internal agent notes, **unverified independently**; cite as motivation only.

Why LeJEPA specifically for this project rather than VICReg or vanilla I-JEPA:

1. **Heuristic-free collapse control.** EMA/stop-grad are the knobs that silently break at small scale
   and small batch. SIGReg is the one term; there is no teacher to diverge.
2. **One sensitive hyperparameter.** DR-render training will need LR/λ sweeps on a single 3090 with one
   GPU job at a time (`rl/AGENTS.md`). Fewer coupled knobs = fewer full retrains.
3. **Architecture agnosticism.** We want to compare a 150k custom ConvNet against distilled MobileNetV4/
   RepViT backbones under one objective — LeJEPA is explicitly validated across backbone families.
4. **Small-data claim.** The paper's small-corpus results are exactly our regime. This is the paper's
   claim, not our measurement; treat as priors, verify with the §7 ablations.

VICReg remains the mandatory **ablation**: it is the strongest simple baseline and costs nothing to try
once the data loader exists.

### 3.3 What the latent actually learns, concretely, for this scene

With DR pairs = same camera pose, independent draws of {floor albedo/specular, key/fill intensity and
direction, shadow gradient, glare blobs, gamma, noise, defocus, motion-blur kernel, clutter placement},
the Bayes-optimal predictor cannot use texture/glare (they are independent across the pair) and must use:

- the **two-rail vanishing geometry** (skids / wheels-fore pair) — saliency at mid-gray edges stable
  across photometrics;
- the **center belly silhouette** against the corridor — vertical extent in image ∝ clearance/range;
- the **inverted-V tail** — the one strong high-contrast apex; its image position is an unambiguous
  bearing+pitch cue and its L/R symmetry ordering gives chirality (resolves left/right, cf. Agent 06 §3.1);
- **shadow softness vs hard edge** — a weak but monotone range cue.

Those are precisely the features ArUco cannot see (it needs a printed decode) and YOLO must be
*labelled* to see. It is also precisely what `ground_pose.py` exploits manually with four points; a
LeJEPA encoder learns a dense, multi-cue version of the same invariants, with confidence calibrated by
the aux head (§5.4).

---

## 4. Candidate encoders on the Pi 4B: honest budget math

Target envelope: fit inside the *existing* vision pipeline (latest-frame-only capture at 30 Hz,
`docs/ARUCO.md` discipline), run on the same cores as the 50 Hz control loop, zero change to the 600 ms
watchdog or the PWM cap, and leave the fetch+detect p50 (~44 ms) intact or reduced.

| Candidate | Params | MACs (input) | Published latency | Pi 4B `[est]` FP32 | Verdict |
|---|---|---|---|---|---|
| MobileNetV4-Conv-Small (ECCV 2024, Qin et al.) | 3.77–3.8 M (timm `mobilenetv4_conv_small`, arXiv:2404.10518) | ≈ 0.2 GMACs @ 224² | 2.4 ms Pixel 6 CPU; 73.8–74.6% IN-1K | 60–150 ms even with NEON; INT8/ncnn maybe 15–30 ms | **Too heavy per-frame.** Acceptable as a 5–10 Hz asynchronous thread or as a *teacher* to distil. |
| RepViT-M0.6 (THU-MIG, CVPR 2024, arXiv:2307.09283) | not published in repo README; M0.9 = 5.1 M / 0.8 GMACs / 0.9 ms iPhone 12 Core ML; M0.6 = 74.1% top-1, ~0.6 ms `[est ~2.5–3 M, ~0.4–0.5 GMACs]` | ≈ 0.3–0.5 GMACs | ~0.6 ms iPhone 12 NPU/ANE | 40–120 ms optimized; 5–15 ms INT8/ncnn at 224² `[est]` | **Borderline.** Only as async thread or distillation source. The Core ML number does not transfer to Cortex-A72. |
| Custom 4-layer strided depthwise-separable ConvNet + spatial soft-argmax (§5) | **~140–160 k** | **~30–60 MMACs @ 160×120 mono** | n/a (ours) | **`[est]` 3–8 ms FP32-NEON (cv2.dnn/ncnn), 8–25 ms naive NumPy im2col; <3 ms only with INT8 + downscaled 128×96 input** | **The right tier.** The "<3 ms on Pi 4B" figure in the brief is achievable only under INT8 + small input + optimized runtime, and is **unverified on this hardware**; measure with `tools/perf.py` before believing it. |
| HOG+SVM / sliding window | n/a | hundreds of evals/frame | n/a | 100s of ms @ 1280×720 | Rejected for primary use; see Agent 06 §5. |
| YOLO11n-pose (Agent 06) | ~2.6 M | ~6.5 GMACs @ 320² | 12–20 ms NCNN INT8 | 12–30 ms | Supervised complement; see §9 and Agent 06. |

Structural notes:

- **Spatial soft-argmax** (Levine et al. 2016) converts a small heatmap into sub-pixel (u, v) coordinates
  via softmax-weighted expectation. It is fully differentiable (aux pose head can backprop through it),
  needs no NMS, gives per-point confidence (peakedness of the softmax), and the overlays are free debug
  artifacts for field logs. Grid of 20×15 (160×120 → stride 8) is enough for ≤ 1 px effective
  localization with temperature annealing; it is coarse for PnP but excellent as a *pose feature*.
- Input **grayscale 160×120** (or 128×96 for INT8) — the camera is natively L8 (`camera_profile.json`),
  and 4:3 keeps the 1280×720 downscale a pure, fast `cv2.resize`. Decode glare as *intensity*, not color.
- Parameter budget check for the custom net (depthwise-separable 3×3, expansion 1):
  block1 1→16 s2 (≈160) + dw, block2 16→32 s2 (≈4.7k), block3 32→64 s2 (≈18.5k),
  block4 64→64 s2 (≈37k) + pointwise tails ≈ **60 k** conv params; attention/avg-pool → 128-d z head
  (≈8–16 k); K=8 soft-argmax heads via shared 1×1 (≈4 k); pose MLP head 3-layer 128→64→(6) (≈10 k);
  LeJEPA predictor MLP 128→128→128 SiLU ×2 (≈50 k, **training-only, not exported**). Total deployed ≈ 90–110 k;
  with the predictor, ≈ 150–160 k. Both fit the brief's "~150 k, <3 ms *aspiration*" — treat the latency
  figure as a target to be measured, not a fact.

---

## 5. Proposed architecture: a LeJEPA belly-geometry front-end

### 5.1 Diagram

```
L8 frame 1280x720 (existing capture, latest-frame-only)
  └─ crop ROI (track-gated, see §5.5) → resize 160x120 gray (cv2, ~0.2 ms)
       └─ Encoder E  (4× strided depthwise-separable Conv blocks, BN+SiLU, ~60 k conv params)
            ├─ f ∈ R^{64 x 20 x 15}            (stride-8 feature map)
            ├─ z = AttnPool(f) ∈ R^{128}       (global embedding, SIGReg-regularized)
            └─ K=8 spatial soft-argmax maps    → (u_k, v_k, conf_k)
                 └─ heads:
                      • z-projection 128→128    (JEPA predict target; train-only)
                      • keypoint decode 1x1     (train+deploy; conf = softmax peakedness)
                      • pose MLP [z ‖ kpts] → (dx, dy, dz, roll, pitch, yaw) + σ²  (train+deploy)

JEPA branch (training only):
  context = E(view_a) ; target = E(view_b)   (same pose, independent photometric DR draw,
                                              or frame t, t+Δt pair from a 50 Hz clip)
  predictor P: z_a → ẑ_b  ;  L_pred = MSE(ẑ_b, sg(z_b))  [stop-grad optional; SIGReg carries stability]
  L_sigreg  = SIGReg({z_a} ∪ {z_b}, λ)        (single λ; see §5.3)
  L_pose    = Huber(pose(z_a), gt_pose)       (synthetic labels only; NEVER on real frames)
  L_kpts    = soft-argmax CE/coord-L1 + conf BCE
  L_total   = L_pred + λ·L_sigreg + w_p·L_pose + w_k·L_kpts
```

### 5.2 Views and target definition

Two viable target constructions; use both in ablation:

1. **Photometric pair (recommended primary)**: render the same (pose, airframe) under two independent
   DR draws. This is a *pure invariance* task — exactly the failure axis of glare/texture. It needs no
   temporal coherence and is trivially batch-parallel on the 3090.
2. **Temporal pair (V-JEPA/LeWM-style)**: frames Δt = 1–3 ticks apart (20–60 ms) from rendered
   approach clips; predictor sees motion. More sample-efficient for dynamics but couples representation
   quality to control jitter; keep as secondary.

I-JEPA-style **masked** prediction (predict masked belly patches from visible ones) is a third option;
keep it as the ablation that isolates "spatial context" from "invariance", but do not make it the
deployed objective — it spends capacity on reconstructing *locations*, not on docking-relevant semantics.

### 5.3 SIGReg mechanics and collapse diagnostics

SIGReg (LeJEPA §3): draw M ≈ 1024 random unit projections `u_m`; for each, compute a normality
statistic of `{u_m·z_i}` against N(0,1) (Epps–Pulley-type); average. Practical recipe for this project:

- λ (LeJEPA) — sweep {0.01, 0.05, 0.1, 0.2} on a 2 k-render validation subset; report downstream probe,
  not train loss.
- Batch ≥ 256 for stable sketch statistics (3090 OK: model is tiny; render throughput is the bottleneck).
- **Collapse tripwires**, checked every N iters and logged: per-dim embedding std (hinge if any dim
  < 0.05), effective rank / singular-value spectrum of a 4096-sample embedding matrix (flag if
  stable rank < 8), mean pairwise cosine (flag if > 0.9), and kNN pose-probe error (the metric that
  cannot be gamed by a collapsed code).
- VICReg ablation: invariance + variance-hinge(μ=1) + covariance, weights 1/25/25 classic — same probes.
- If both collapse: fall back to nature's baseline — supervised-only pose head (§5.4) — rather than ship
  an untested SSL variant. An honest negative result is still a gate.

### 5.4 Auxiliary 3D pose / keypoint head on synthetic labels

Synthetic renders carry exact camera→airframe ground truth; exploit it:

- **Pose head** `[z ‖ kpts] → (dx, dy, dz, roll, pitch, yaw, log σ²)`: directly supervise the quantity
  the controller wants. dz (height above/below nominal) is observable here and *not* observable from the
  tags — a genuine capability add for the bent-airframe / gear-sag case.
- **Keypoint head** K = 8, matching Agent 06's semantic set for continuity with that analysis:
  `nose_tip, left/right wheel_contact, belly_center, left/right V-fin_root, left/right V-fin_tip`
  (drone analogue: `nose_equiv, left/right skid end-cap (near+far), belly_center, + 2 gyro/mast cues`).
  Coordinates in the **airframe frame**, metric via the known model (wheel spacing / skid spacing ≈
  0.254–0.340 m measured-operator vs model — see discrepancy note in `docs/PARKING.md:19,37`; use one
  frame definition and record it).
- Metric scale from monocular input is only valid because the object model is metric — same argument as
  Agent 06 §2. No metric claim is made for clutter or non-target objects.
- Pose head is deployed; SIGReg/predictor are not. The deployed artifact remains a tiny feed-forward
  net, NumPy-portable exactly like `src/ugv/policy.py` (export path: torch → `.npz`, parity < 1e-5,
  root package stays torch-free).

### 5.5 Region of interest and negative frames

- Track-gated crop: once any observation (ArUco, skid observer, or this net) seeds a track, crop to a
  window around the projected target (±40%); cold-start uses full-frame but downscaled harder (128×96).
  This keeps FLOPs flat and starves clutter false-positives.
- **Clutter negatives** (chairs, table legs, shoes, cable reels — the `ROOM_DETAILS.md` inventory) must
  be in the DR set with a `valid=0` label so the pose head learns "no dock here" instead of regressing
  garbage poses. This is the single most important difference between "works on renders" and "safe on a
  cluttered bench".
- Glare-only epochs (drive sun straight at the lens): the invariance term should absorb them; if the
  pose-probe error explodes on glare-only val, the DR photometrics are under-sampled — fix data, not loss.

---

## 6. Integration with the existing 50-step history (`rl/models.py`)

Three options, in increasing risk. **Option A is the recommendation; B is the research path; C is out of scope.**

**Option A — encoder as a virtual sensor (recommended first step).**
The front-end emits a tag-equivalent observation
`(x, y, yaw, valid, age, source=belly)` + covariance, fused into the same estimator path ArUco uses
(`ArucoFollower` / EKF in `src/ugv/aruco.py` / `tools/l7/l7_run.py`), timestamped the same way. The
student contract (`STUDENT_DIM = 13`, `HIST = 50`, `rl/models.py`, `src/ugv/policy.py`, `export.py`) is
**untouched**. Benefits: zero RL retrain to get first value; A/B directly against tag-only with the
existing 1000-episode gate; the shield, watchdog, `swept_clear`, and parking thresholds are untouched.
Costs: the encoder sees nothing of the GRU — it is a better sensor, not a better policy. Pose error
~2–5 cm / 2–4° `[est]` would already close the 0.4 m tag-loss gap during final approach.

**Option B — embedding into the history (the "LeJEPA front-end for the HistoryEncoder").**
Append a compact embedding `e_t ∈ R^{32–48}` to the 13-dim frame: `STUDENT_DIM → 45–61`, GRU input resized.
Training requires *rendered observations inside the RL loop* — `rl/sim.py` has no pixels, so either
(i) train the RL student on Option-A sensor outputs emulated with noise (no end-to-end gradient through
the encoder; encoder frozen or EMA), or (ii) move visual RL training into the Gazebo bridge
(`simulation/gazebo/run_policy_episodes.py`, known 0/20 gate state and ~5.2% camDR decode rate — do not
plan around it), or (iii) precompute per-step embeddings for a fixed demonstration set and do a
sequence-level BC/AWR warm start (breaks the on-policy PPO loop; not recommended). Realistic sequencing:
A first (weeks), then B with encoder frozen and only GRU resized, with the L1/L2 gates as the arbiter.
Option B's deploy delta: encoder forward (~3–8 ms `[est]`) + slightly larger GRU input GEMM
(13→45 inputs: ≈ 3.5× input GEMM, +0.1–0.2 ms stateful) — still NumPy-portable; `export.py` gains
`vision.*` keys shipped **separately** from `gru.*`/`head.*`; `ugv.policy.StudentPolicy` grows an
optional vision sidecar; parity harness extended exactly as for the GRU.

**Option C — jepa-predictor dynamics head (LeWM-style planning).** Predict next embedding
`P(z_t, a_t, Δt) → z_{t+1}`; use it for MPC or option-value estimation. Highest ceiling, highest risk,
zero existing harness. Record as future work; do not start before Option A passes its gate.

In all options the deploy rules from `rl/AGENTS.md` apply verbatim: torch only in `rl/`, root torch-free,
one GPU job at a time, no editing `sim.py`/`models.py` mid-run, checkpoints to `rl/runs/`, and every
number from `evaluate.py --seed 11`.

---

## 7. Training recipe (3090) and evaluation protocol

**Data.** 40–80 k synthetic frames, split **by scene/airframe instance**, not by frame (no leakage).
Renders from the Gazebo/Blender path (`simulation/gazebo/`, `drone_model.py`, plus an aircraft model
matching the Strix geometry or measurement-estimated from the photos in `docs/council_v2/`), with DR over:
key/fill intensity ratio (including 10:1 and moonlit), light azimuth/elevation, temperature, floor albedo
{concrete, asphalt, polished epoxy with GGX glare lobes, wet-look}, shadow gradient across frame, gamma
0.6–1.6, Gaussian noise σ 0–12, defocus σ 0–3, motion-blur kernel 1–15 px at 15–60°, glare blob
(specular streak toward sun), partial occlusion (legs/cables intruding ≥ 20% of body), clutter 0–8
distractors from the room inventory, exposure 4 ms / 16 ms (the two real settings on the Pi,
`HANDOFF.md` §1), ±3% intrinsics jitter. Plus ~300–500 real Pi frames, **eval-only**, for the sim-to-real
gap measurement (never trained on; KLT anchors as pseudo-labels only with `ground_pose.py` cross-check).

**Training.** AdamW, lr 3e-4 cosine → 3e-5, wd 0.05, batch 256, bf16, ~150–300 epochs over the renders
(≈ 30–60 min on the 3090 for the 150k model; under 20 min if the dataset fits in RAM). λ(LeJEPA) per §5.3.
Losses: `L_pred + λ·SIGReg + 1.0·Huber_pose + 0.5·kpts`, pose/kpts annealed in after epoch ~10 (warm the
invariance first). No EMA teacher. Checkpoint each best-probe epoch; never overwrite gate-passing
checkpoints.

**Ablations (each one row in the results table).**
(1) supervised-only (pose+kpts, no SSL); (2) VICReg; (3) LeJEPA; (4) I-JEPA-masked; (5) pixel-AE
(+ same aux heads — the honest baseline for §3.1); (6) LeJEPA without floor-glare DR (train) → glare DR
(eval): the invariance stress test; (7) distill MobileNetV4-Conv-Small → 150k student vs scratch.

**Metrics.**
- Offline: kNN (k=20) pose regression error (cm / deg); linear-probe same; keypoint PCK @ 0.05/0.10;
  conf-AUC for `valid`; glare-only and dark-only splits; effective rank of embeddings.
- Pose solver: 8-point `solvePnP` (mirror-safe as in `src/ugv/ground_pose.py`) with RANSAC; report RMS
  reprojection px.
- Downstream (the only gates that matter): Option A in sim — re-run `evaluate.py --level L1/L2/L3
  --episodes 1000 --seed 11` with the belly sensor fused (inject pose w/ measured noise model replacing
  tag obs); pass = success ≥ baseline, collisions ≤ baseline, and last-0.4 m "blind segment" completion
  rate improves. Gazebo L6 stays as-is (do not build on the 5.2% decode path). L7 only per
  `docs/SAFETY.md` with operator authorization, measured gear clearance, and end-frame confirmation.

**Stopping/rollback criteria.** If LeJEPA < supervised-only on the glare split, ship supervised-only and
record the negative result — the ablation table is still the deliverable. If effective rank collapses
with λ ∈ {0.05..0.2}, increase λ or switch to VICReg; if both fail, ship the supervised head.

---

## 8. Deploy-time reality on the Pi 4B (budget and verification)

Constraints (measured, `HANDOFF.md` §10 + §5): Pi 4B 2 GB (1.8 Gi total, ~1.3 Gi available, desktop
stack ~250 MB), `uvicorn` ~160–164 MB RSS, camera 1280×720 L8 30 Hz, fetch+detect p50 ≈ 44 ms, p95 ≈ 52 ms,
`vision_ms_p50 = 50 ms`, load 1.4–1.6, temp 60 °C, no swap pressure. Frame budget for the *existing*
pipeline is already blown; the right target is not "add 30 ms" but "add ≤ 5 ms async".

Plan:

- Run encoder in the existing async vision worker (latest-frame-only), never in the 50 Hz control path;
  publish timestamped observations into the same estimator the tags feed.
- INT8 quantization (ncnn or cv2.dnn INT8 on ARM NEON) of the 160×120 model; FP32 fallback before
  quantization sign-off. Parity vs torch: < 2 px mean keypoint shift, < 1 cm pose bias on the DR val set.
- Measure with the already-wired `PerfMonitor` (`src/ugv/perf.py`): add `belly_ms` alongside the existing
  stack; report p50/p95 under the HANDOFF §10 high-pressure test (30 s loop, expect ≥ 400 frames).
  Gate: p95 `belly_ms` ≤ 8 ms and no growth in `uvicorn` RSS over 500 requests.
- The 600 ms watchdog, PWM cap 1800/4095, `config/parking.json`, and systemd units are untouchable —
  this is a sensor, not an actuator. Zero changes to motors.
- Export discipline: `.npz` sidecar for the vision net (sha-pinned), `tools/l7/deploy.sh` must sync it
  alongside `runs/student.npz` (and both must be preserved on Pi deploy, per `HANDOFF.md` §2).

Expected end-to-end: Option A adds ≤ 30 Hz pose updates with ~2–5 cm / 2–4° `[est]` error during the
tag-blind final 0.4 m — the single most valuable regime for the parked-not-estimated distinction, and
complementary to, not a replacement for, ArUco (which remains better at range with valid tags).

---

## 9. Direct answer to "cnn lejepa? hog/yolo? some other clever approach?"

Short form:

1. **HOG/sliding-window**: rejected for Pi for primary use (Agent 06 §5.1) — no invariance, no metric
   pose, hundred-ms frame times.
2. **YOLO11n-pose (Agent 06)**: strong, supervised, deployable via NCNN INT8 at 12–20 ms `[est]`, but
   buys accuracy with labels and AGPL-adjacent tooling; it does not give lighting invariance for free.
3. **LeJEPA/SSL ConvNet front-end (this doc)**: reduces label cost by an order of magnitude (pose labels
   come free from renders; the SSL objective does not need them), gives lighting/glare invariance *by
   construction*, detects "nothing to lock onto" via confidence, and fits Pi with headroom. Its risk is
   the sim-to-real DR gap, which is measurable offline before any car moves.
4. **Clever composition (recommended)**: **LeJEPA-pretrained features + Agent-06-style supervised
   keypoint/pose heads** (the aux heads in §5.4 are exactly that), with ArUco as independent fallback and
   `ground_pose.py` as the classical baseline to beat. SSL buys the features; supervision buys the metric
   pose; geometry gates buy false-positive suppression; ArUco keeps everyone honest at range.
5. **Not yet**: e2e visual RL in Gazebo (L6 0/20), LeWM planning (Option C), MobileNetV4-class models
   per-frame.

HOG/YOLO answer the question "what is in this frame?"; LeJEPA answers "what stays true about this scene
when the light changes?" Parking needs the second answer more than the first.

---

## 10. Risks, honest uncertainties, and what would falsify this

| # | Risk / unknown | Evidence | Mitigation / falsifier |
|---|---|---|---|
| 1 | Premise mismatch in brief (no HistoryEncoder/jepa_pred exists) | `rl/models.py:44,60-69` | This doc; §1. |
| 2 | Gazebo/Blender renders may inherit the L6 gap (0/20; camDR 5.2% decode) | `docs/AGENT_BOARD.md:58` | Real-Pi eval-only split; DR breadth; Option A first (no RL dependency); if renders don't transfer, the ablation table shows it before any deploy. |
| 3 | Glare-specific invariance is *assumed*, not measured, for LeJEPA here | LeJEPA paper claims (small corpora, ResNet/ConvNet); no robotics glare benchmark exists | Glare-only val split as a first-class metric; supervised-only fallback; record negative results. |
| 4 | <3 ms Pi claim unverified | Pi p50 detect 44–50 ms; no CNN measured yet | `tools/perf.py` measurement; INT8; 128×96; or async 10 Hz thread. Claim stands only after measurement. |
| 5 | Monocular scale drift on bent/deformed target | object-model metric argument (Agent 06 §2); gear-sag noted in repo | Known-model scale; `dz` supervision; cross-check vs ArUco at range; abstain (`valid=0`) on gross misfit. |
| 6 | Clutter false positives (chair legs ≈ skids) | `ROOM_DETAILS.md` 423 parts; real lab clutter in pulled frames | Clutter-negatives training + 5-gate suppression (Agent 06 §4); temporal Mahalanobis gate; abstain, never guess. |
| 7 | SIGReg collapse at tiny batch / mis-set λ | LeJEPA theory; small-batch sketch noise | §5.3 tripwires; VICReg ablation; supervised fallback. |
| 8 | `student_L1_jepa_s0` precedent: aux latent losses did not move the L1 gate | `rl/runs/student_L1_jepa_s0.json` | Option A first — decouples *sensor* value from *policy* value; no RL retrain needed to see value. |
| 9 | Texture shortcut: net learns "glossy floor = near plane" rather than geometry | DR correlation risk | Train-without-glare-DR → eval-with-glare; shuffle floor draws per-pose; report glare-split probe. |
| 10 | Deployed-net drift vs training torch net | export discipline in repo | `.npz` sha-pin + parity harness (as GRU), add `vision.*` to `export.py` parity check. |

Falsifiers worth stating plainly: if the LeJEPA/VICReg probes are not ≥ supervised-only on the
glare-only real-frame eval, SSL is not paying for itself here — drop to supervised. If Option A fused
EKF does not improve blind-segment completion in sim L1/L3 vs tag-only, the front-end is a curiosity,
not a product. If p95 `belly_ms` cannot hold ≤ 8 ms on the Pi, the approach stays off the car.

---

## 11. Concrete next steps (numbered; no code changes implied)

1. Build the DR renderer path (Gazebo or headless Blender off `drone_model.py` / an aircraft surrogate);
   40–80 k frames with the §7 list; scene-level split; eval-only real Pi frame set.
2. Implement E + heads in `rl/` (torch, 3090) — training-only module; nothing exported yet. Log
   tripwires per §5.3.
3. Run ablations §7 (supervised / VICReg / LeJEPA / pixel-AE / no-glare). Publish the table, win or lose;
   negative result = supervised head ships.
4. Export Option A sidecar (`.npz`), parity < 1e-5, `tools/perf.py` on the Pi bench (no motion).
   Preserve `student.npz`/`vision-camera.conf` on deploy per `HANDOFF.md` §2.
5. Shadow-mode on the Pi (S0/S1 read-only): encoder outputs logged, zero control authority; compare vs
   `SkidPoseObserver` and ArUco on the same frames; tune conf gates.
6. Sim A/B in `rl/sim.py` (behind a new observation *source* flag, not a behavior change): tag-only vs
   tag+belly on L1/L2/L3, 1000 eps, seed 11; require gate pass or record the negative.
7. Only then: Option B behind a flag, and only with operator authorization and `docs/SAFETY.md` gates for
   any L7 run. End-frame confirmation remains the definition of parked, not an estimate.

---

## 12. References

**Papers.** Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture* (I-JEPA), arXiv:2301.08243 · Bardes et al., *Revisiting Feature Prediction for Learning
Visual Representations from Video* (V-JEPA), arXiv:2404.08471 · Balestriero & LeCun, *LeJEPA: Provable and
Scalable Self-Supervised Learning Without the Heuristics* (SIGReg), arXiv:2511.08544
(https://arxiv.org/abs/2511.08544) · Bardes, Ponce, LeCun, *VICReg*, arXiv:2105.04906 · Qin et al.,
*MobileNetV4*, arXiv:2404.10518 (ECCV 2024; Conv-Small 3.8 M / 0.2 GMACs / 2.4 ms Pixel 6) ·
Wang et al., *RepViT*, arXiv:2307.09283 (CVPR 2024; M0.6 74.1%, ~0.6 ms iPhone 12; M0.9 5.1 M/0.8 GMACs) ·
Levine et al., *Spatial-Softmax*, 2016.

**Internal context (AGENT_BOARD scout notes; unverified externally).** `2603.19312` "LeWM" (e2e JEPA, no
EMA, ~15M), `2610.39302` "Platonic-JEPA" (good-views=2, 42× sample-eff vs NTP), `2107.04034` RMA;
`rl-curric-and-jepa`, `jepa-rma-calibration` entries, `docs/AGENT_BOARD.md:49-50`.

**Repo evidence.** `rl/models.py:44-79` (Student/GRU/aux/dyn heads) · `rl/sim.py:171-175,1188-1230,1277-1300`
(HIST=50, 13-dim frame, sensor emulator) · `rl/export.py:43-51` (gru+head-only export) ·
`src/ugv/policy.py:12-26,48-98` (NumPy GRU-64 forward, HIST=50/DIM=13) · `docs/RL_PARKING.md` (gates,
blind last 0.30 m) · `docs/L7_FIELD_NOTES.md` (0/3 parked; 25% Tag-6 detection; 0.4 m FOV limit;
gyro-noise mismatch) · `docs/ARUCO.md` · `docs/PARKING.md:19,31-41` (skid dims; `ground_pose.py` observer,
4.4 px) · `simulation/gazebo/DRONE_MODEL.md` (310 mm gap, 480 mm skids) · `camera_profile.json` (L8
1280×720 30 Hz, fx=fy=968 nominal) · `HANDOFF.md` §5, §8, §9, §10 (Pi state, perf, vision_ms_p50) ·
`docs/AGENT_BOARD.md:58` (Gazebo L6 0/20, camDR 5.2%) · `rl/runs/student_L1_jepa_s0.json` (aux-latent L1
fail) · `docs/council_v2/06_yolo_keypoint_and_wheel_detection.md` (supervised sibling analysis).

*Prepared as council Agent 07. Analysis only; no source files modified; no Pi contact made (Pi is paused
charging per HANDOFF.md §1).*
