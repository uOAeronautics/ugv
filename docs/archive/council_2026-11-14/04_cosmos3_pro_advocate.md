# Agent 04 — NVIDIA Cosmos 3 (`Cosmos3-Edge` 4B & `Cosmos3-Nano-Sim-Transfer` 15B): Pro-Advocate & Critical Technical Evaluation

**Agent:** 04 (NVIDIA Cosmos 3 / Cosmos3-Edge Pro-Advocate & Critical Technical Evaluator)  
**Target Stack:** Training on single **NVIDIA RTX 3090 (24 GB, Ampere SM86)** → ONNX / `cv2.dnn` (or NCNN) deployment on **Raspberry Pi 4B (`ARM Cortex-A72` CPU, 2 GB RAM on `adv034`)** for UGV autonomous parking under the **`STRL001` fixed-wing aircraft belly** ([`strix_plane_realflight.png`](strix_plane_realflight.png)).  
**Evaluated Checkpoints ([`cosmos3_huggingface.png`](cosmos3_huggingface.png)):** `nvidia/Cosmos3-Nano-Sim-Auto` (15B), `nvidia/Cosmos3-Nano-Transfer-Auto` (15B), `nvidia/Cosmos3-Nano-Sim-Bimanual` (15B), `nvidia/Cosmos3-Nano-Sim-Transfer` (15B), `nvidia/Cosmos3-Edge-FP8` (4B), `nvidia/Cosmos3-Nano-FP8` (16B).  
**Constraint Honor:** Documentation/analysis only. No source files outside `docs/council_v2/` modified; no Pi contact; no changes to `config/parking.json`, safety gates, or RL checkpoints.

---

## 0. Executive Verdict: Where Cosmos 3 Genuinely Wins vs. Where It Fails

Neither uncritical enthusiasm (the earlier `beam-free` draft of this document) nor blanket dismissal ([`05_cosmos3_skeptical_audit.md`](05_cosmos3_skeptical_audit.md)) gets the engineering tradeoff right. **Cosmos 3 has zero place on the Raspberry Pi 4B and zero place inside the `rl/sim.py` PPO loop—Agent 05's proof on both points is airtight.** However, Agent 05 evaluated Cosmos 3 primarily against the legacy **ArUco-only** pipeline (`50 mm` planar fiducials, `StudentPolicy` GRU-64 on state vectors). Under the Council v2 perception pivot—training a **lightweight PyTorch Keypoint / LeJEPA visual student (`~107k–160k` params at `160×120` grayscale, or `YOLO11n-pose`)** to localize the `STRL001` airframe when ArUco fails (`<0.45 m` close-range and `≥40°` steep off-axis angles)—two specific Cosmos 3 checkpoints have **bounded, falsifiable, offline-only utility on the RTX 3090**:

| Checkpoint ([`cosmos3_huggingface.png`](cosmos3_huggingface.png)) | Size | Verdict for `.` | Exact Role (If Gated In) |
|---|---:|---|---|
| **`nvidia/Cosmos3-Nano-Sim-Transfer`** | 15B | **CONDITIONAL GO (Offline 3090 Only)** | Depth/Canny/Segmentation-conditioned Sim2Real **photometric appearance transfer** (`~20%` of training bank) generating **dual-view LeJEPA invariance pairs** (specular hangar glare, wet/epoxy concrete, cast wing shadows, unmodeled clutter) over exact Blender 3D geometry. |
| **`nvidia/Cosmos3-Edge-FP8`** | 4B | **TARGETED GO (Offline 3090 Triage Only)** | Offline VLM **hard-example miner & false-positive critic** over unlabeled real Pi / hangar footage (flagging glare, motion blur, clutter distractors) + optional 3-logit auxiliary scene-condition distillation. **Never** a metric 3D keypoint teacher; **never** on Pi. |
| `nvidia/Cosmos3-Nano-Sim-Auto` | 15B | **NO-GO** | Autonomous-driving street priors conflict with indoor/runway belly geometry. |
| `nvidia/Cosmos3-Nano-Transfer-Auto` | 15B | **NO-GO** | Driving-domain transfer; strictly dominated by `Sim-Transfer`. |
| `nvidia/Cosmos3-Nano-Sim-Bimanual` | 15B | **NO-GO** | Tabletop bimanual manipulation priors; irrelevant to SE(2) under-belly docking. |
| `nvidia/Cosmos3-Nano-FP8` | 16B | **NO-GO** | Base world model without `Sim-Transfer` multi-control conditioning; too heavy for triage vs `Edge` 4B. |

---

## 1. Critical Audit of Prior Council Documents (`04` Draft vs. `05` Audit)

Before arguing the pro case, we correct factual errors and blind spots in both prior documents:

### 1.1 Errors in the Prior `beam-free` Draft of `04_cosmos3_pro_advocate.md`
1. **Stale Quadrotor-Only Framing (Missed `STRL001` & Keypoint/LeJEPA Student):** The prior draft focused almost exclusively on quadrotor foam skids (`0.254 m` gap), `Tag 6/14/3`, and distilling a `<1M` glare-severity flag classifier for OpenCV ArUco. It ignored the `STRL001` fixed-wing belly-parking target ([`strix_plane_realflight.png`](strix_plane_realflight.png), [`03_strix_plane_geometry_model.md`](03_strix_plane_geometry_model.md)) and the core Council v2 student architecture: a **PyTorch Keypoint + LeJEPA CNN** (`107k–160k` params, `160×120` mono, [`07_cnn_lejepa_representations.md`](07_cnn_lejepa_representations.md), [`10_pi_edge_vs_3090_compute_budget.md`](10_pi_edge_vs_3090_compute_budget.md), [`12_belly_vision_experiment_results.md`](12_belly_vision_experiment_results.md)) or `YOLO11n-pose` ([`06_yolo_keypoint_and_wheel_detection.md`](06_yolo_keypoint_and_wheel_detection.md)).
2. **False Codebase Claim on Gazebo Control Buffers:** The prior draft claimed *"Our Gazebo stack already produces the four control modalities it consumes (`simulation/gazebo/environment.py`, `camera_model.py`, `render_drone.py`): depth buffer, segmentation mask, Canny edge of the scene, and a blurred low-frequency pass."* **This is false:**
   - [`simulation/gazebo/environment.py`](../../simulation/gazebo/environment.py) is a 50-line SDF XML builder defining `ground`, `metric_test_pad`, `key_light`, and `fill_light`—no camera or buffer export exists there.
   - `simulation/gazebo/camera_model.py` does not exist in the repository.
   - [`simulation/gazebo/render_drone.py`](../../simulation/gazebo/render_drone.py) and [`simulation/gazebo/render_blender.py`](../../simulation/gazebo/render_blender.py) render Blender Cycles/EEVEE **RGB only**; they do not emit Z-depth, segmentation masks, Canny edges, or blur passes today. Adding Blender render passes (`ViewLayer` `use_pass_z`, `Object Index` / `Cryptomatte`, plus OpenCV `Canny` on clean passes) is straightforward engineering (~60 lines of `bpy`), **not** an existing feature.
3. **Broken 3090 VRAM Accounting:** The prior draft stated *"Sim-Transfer is 16B BF16 (32 GB) ... Run in 8-bit ... ≈16 GB weights + 4–6 GB activations ≈ 20–22 GB, fits."* First, `Cosmos3-Nano-Sim-Transfer` is **15B** (16B is `Cosmos3-Nano-FP8`). Second, 15B refers to the **DiT backbone alone**—omitting the **text encoder** (`~4.8–9.5 GB`), **3D causal video VAE** (`~1.5–2.5 GB`), and **ControlNet/multi-control embeddings** (`~1–3 GB`). Naively loading all components in 8-bit simultaneously on a `24 GB` RTX 3090 **OOMs** unless text embeddings are pre-cached or DiT weights are `W4A16`-quantized (§4).

### 1.2 Where [`05_cosmos3_skeptical_audit.md`](05_cosmos3_skeptical_audit.md) Is Right vs. Over-Broad
- **Indisputably Right:**
  - `Cosmos3-Edge-FP8` (4B) cannot run on `adv034` (Pi 4B, `2 GB` total RAM / `~1.3 GiB` free, `BCM2711` Cortex-A72 with no FP8, `12.8 GB/s` LPDDR4 peak → `312–800 ms` weight-read floor vs `< 5 ms` policy budget and `33.33 ms` 30 Hz camera period).
  - `Cosmos3-Nano` (15B/16B) cannot replace `rl/sim.py`'s `~243,000 steps/sec` kinematic PPO loop.
  - Video diffusion VAE latent compression (`8×`/`16×`) destroys `50 mm` `DICT_4X4_50` ArUco interior bits (`5.4 px/bit` at `1.5 m`).
- **Where Agent 05 Over-Generalized:**
  - Agent 05 argued that diffusion `2–5 cm` boundary drift invalidates *all* `Sim-Transfer` synthetic training data. That holds for **50 mm planar ArUco decoding** and **thin 8 mm rods**, **not** for a **`160×120` Keypoint/LeJEPA CNN** conditioned on exact Blender depth+segmentation+Canny controls and deployed with `solvePnPRansac` + metric consistency gates (§2.2).
  - Agent 05 also omitted the fact that **Blender Cycles PBR + real Pi background compositing** is already a strong baseline, meaning Cosmos `Sim-Transfer` only earns its compute if it closes the *unmodeled photometric & clutter gap* better than Blender PBR alone—a testable hypothesis (§6).

---

## 2. Evaluation (a): Offline Photorealistic Sim2Real Data Generation (`Cosmos3-Nano-Sim-Transfer`)

### 2.1 Why the `STRL001` Keypoint / LeJEPA Student Needs Photometric Realism
As documented in [`12_belly_vision_experiment_results.md`](12_belly_vision_experiment_results.md) (`exp_belly_vision.py`, `160×120` grayscale, `fx=fy=121.0 px`), ArUco-only localization collapsed to **0.0% recall at `<0.45 m`** and **1.9% recall at `≥40°` off-axis**, whereas a **118.5k-parameter Keypoint CNN + LeJEPA aux** achieved **85.4% overall and 85.4% steep-angle recall**. However, that synthetic benchmark also revealed the exact failure modes a student faces when moving from stylized OpenCV/Blender renders to real cameras:
- **12.0% distractor false-positive rate (FPR)** on chair legs, boxes, shoes, and dark discs.
- **Photometric domain gap** ([`HANDOFF.md`](../../HANDOFF.md) §8, [`07_cnn_lejepa_representations.md`](07_cnn_lejepa_representations.md) §2): real Pi frames exhibit intense **specular concrete floor glare**, overhead fluorescent reflections, harsh cast shadows under the wing/fuselage, and complex workshop clutter. On `STRL001` ([`strix_plane_realflight.png`](strix_plane_realflight.png)), we additionally have checkered black/white/orange wing and inverted-V leading edges, glossy white composite fuselage skin, a metallic gold/silver chin sensor turret, and asphalt/runway speculars.

### 2.2 Depth + Canny + Segmentation Conditioning: Does Label Preservation Actually Hold?
To train the PyTorch student (`8` semantic keypoints from [`06_yolo_keypoint_and_wheel_detection.md`](06_yolo_keypoint_and_wheel_detection.md): `nose_tip`, `left_wheel_contact`, `right_wheel_contact`, `belly_center`, `left_fin_root`, `left_fin_tip`, `right_fin_root`, `right_fin_tip`, plus aircraft `valid` flag), `Sim-Transfer` must alter **lighting, speculars, floor texture, and background clutter** without moving the projected 2D pixel coordinates of those 8 keypoints.

**Spatial Budget Analysis (`480p` Generation → `160×120` Student Input):**
1. **Why ArUco fails while Keypoints survive:** At `1.5 m`, a `50 mm` ArUco marker is `32.3 px` wide at `1280×720` (`5.4 px/bit`). Under `8×` VAE spatial compression (`160×90` latent), each bit grid square is `< 0.7` latent cells—guaranteed bit corruption. By contrast, the `STRL001` airframe spans `~0.6–2.4 m` (`150–700 px` at `480p`, or `20–90` latent cells), and our target student downsamples the camera frame to **`160×120` grayscale** (`10_pi_edge_vs_3090_compute_budget.md` §4.1: `5.2 ms` projected on Pi 4B, vs `18.0 ms` at `320×240`).
2. **Downsampling absorbs small diffusion edge jitter:** Generating at `480p` (`640×480` or `848×480` cropped to 4:3) under joint **Depth + Segmentation + Canny Edge** conditioning anchors rigid object boundaries to `~1–2 px` at `480p`. When downsampled `4×` (`640×480 → 160×120`), **`1–2 px` of 480p boundary jitter shrinks to `0.25–0.50 px` at `160×120`**.
3. **Metric translation of `0.25–0.50 px` drift (`fx = fy = 121.0 px` at `160×120`):**
   $$\Delta X_{\text{metric}} = \frac{Z}{f_x} \Delta u \implies \text{at } Z = 1.0\text{ m},\;\; 0.35\text{ px} \approx \mathbf{2.9\text{ mm}};\quad \text{at } Z = 0.5\text{ m},\;\; 0.35\text{ px} \approx \mathbf{1.4\text{ mm}}.$$
   A `1.4–2.9 mm` per-keypoint label perturbation is **10×–20× smaller** than the `±30 mm` lateral parking gate and is naturally smoothed out by `8`-point `solvePnPRansac` ([`06_yolo_keypoint_and_wheel_detection.md`](06_yolo_keypoint_and_wheel_detection.md) §4). **For `YOLO11n-pose` at `320×256` (`fx ≈ 242 px`), `1–2 px` at `480p` is `0.5–1.0 px` (`2.1–4.1 mm` at `1 m`), still well inside tolerance.**

### 2.3 Exact Cases Where `Sim-Transfer` Label Preservation Breaks (and Hard Mitigations)
Even with multi-control conditioning, three failure modes will corrupt labels if left unchecked:
1. **Thin Sub-Pixel Structural Members:** Agent 03's model ([`03_strix_plane_geometry_model.md`](03_strix_plane_geometry_model.md) §4.6) includes `tail_brace_l/r` (`r = 0.008 m`) and `tail_strut` (`r = 0.017 m`). At `1.5 m`, these project to `< 2 px` at `480p`—below the `8×` VAE spatial Nyquist limit—causing Cosmos to break or thicken them. **Rule:** Never place keypoints on thin braces or antennas; anchor keypoints solely on macroscopic intersections (`pod` nose/belly, `60 mm`-wide main wheels, `fin_root`/`fin_tip`).
2. **Dark Wheel-to-Shadow Merging (`wheel_contact` Drift):** Under high-contrast prompts (*"harsh overhead hangar light, dark cast shadow"*), diffusion can bleed a black rubber tire into its cast runway shadow, shifting the apparent floor contact patch by `1–3 cm`. **Mitigation:** Feed **both** clean Canny edges (drawn from Blender's albedo + surface-normal discontinuity, which preserves the wheel-floor tangent even when dark) **and** binary part segmentation masks (`wheel_l`, `wheel_r`, `floor`), then run an **automated post-generation label gate** (§2.4).
3. **ArUco Tag Interiors:** Never train `TagDetector` or hybrid ArUco+Keypoint models on diffusion-rendered ArUco bits. If tags (`20–23` per Agent 09) are present in training frames, either mask them out in the control pass or **paste analytically projected `DICT_4X4_50` quads** using exact ground-truth homography *after* `Sim-Transfer` renders the lighting/glare around them.

### 2.4 The Killer Use Case: Dual-View LeJEPA Photometric Pairs + Automated Quality Gate
Where `Cosmos3-Nano-Sim-Transfer` provides a capability **Blender PBR cannot easily match** is synthesizing **realistic unmodeled appearance diversity** (weathered concrete stains, epoxy specular glare streaks, workshop clutter textures, camera bloom/halos) while holding 3D geometry strictly constant:

1. **Blender Ground-Truth Control Pass:** Render a scene from Agent 03's parametric `STRL001` assembly in Blender (`640×480`) to emit exact `Z-depth`, `part_seg`, `canny_edge` (from normal+albedo edges), and ground-truth 2D/3D coordinates for the `8` keypoints.
2. **Dual-View `Sim-Transfer` Generation (for LeJEPA `L_pred + λ·SIGReg`):** Pass the **exact same `(depth, seg, canny)` control tensor** through `Sim-Transfer` with two divergent lighting/environment prompts:
   - **View A (Benign/Clean):** *"overcast outdoor asphalt runway, soft diffused lighting, dry matte concrete, neutral background"*
   - **View B (Adversarial Glare/Hangar):** *"indoor robotics hangar, polished epoxy floor with intense specular fluorescent glare streaks, deep under-wing shadows, cluttered lab benches and cables in background"*
   Because both views share **identical Blender geometry and ground-truth pose**, the Bayes-optimal LeJEPA encoder (`07_cnn_lejepa_representations.md` §3.3) is mathematically forced to discard specular glare, floor texture, and background clutter while retaining only airframe geometry.
3. **Automated Label-Preservation Gate (Zero Manual Inspection):** Before admitting any `Sim-Transfer` pair `(I_A, I_B)` into the PyTorch training set, run a `~2 ms` CPU check:
   - Extract Canny edges on `I_A` and `I_B` inside a `12 px` dilation band around the Blender ground-truth `canny_edge` of the wheels and fin roots; require **edge chamfer distance `≤ 1.5 px` at `480p`** (`≤ 0.38 px` at `160×120`).
   - Optionally check segmentation consistency (e.g., foreground-background contrast/SSIM on the fuselage mask). Reject frames failing the gate (~5–15% expected on extreme glare prompts).

### 2.5 Honest Tradeoff vs. Blender Cycles PBR + Real Background Compositing
**Do NOT use `Sim-Transfer` for 100% of the dataset.** On an RTX 3090 (OptiX), **Blender Cycles PBR + HDRI environment maps + procedural GGX floor roughness** renders `640×480` frames at **`~2.5–10 FPS`** with **`0.00 px` geometric drift** and exact ray-traced reflections. Meanwhile, `Cosmos3-Nano-Sim-Transfer` (15B) runs at **`~0.15–0.5 FPS`** (§4).
- **Optimal Split:** Generate **`80%` (`~40,000` frames)** with fast Blender Cycles PBR + real Pi hangar floor/clutter background plates, and **`20%` (`~5,000–10,000` frames)** via `Cosmos3-Nano-Sim-Transfer` dual-view pairs targeted specifically at extreme specular glare, metallic reflection on the chin turret/props, and photorealistic hangar clutter diversity.

---

## 3. Evaluation (b): Teacher Distillation & Hard-Example Mining (`Cosmos3-Edge` 4B)

`nvidia/Cosmos3-Edge-FP8` (4B) is a compact Vision-Language Model (VLM) tuned for physical-AI perception and reasoning. On our stack, it has a sharp boundary between **high-ROI offline use** and **fundamentally broken use**.

### 3.1 Where `Cosmos3-Edge` (4B) Fails as a Teacher
1. **No Metric 3D Keypoint Distillation:** Prompting a 4B VLM to output pixel coordinates `(u, v)` of `left_wheel_contact` or relative pose `(x, y, yaw)` produces quantized text-token coordinates with `15–50 px` variance and no chirality guarantee. Training a `107k` student on VLM-hallucinated coordinates would destroy the `±30 mm` / `±5°` parking accuracy. **Exact 3D/2D keypoint labels must come exclusively from synthetic 3D geometry (Blender) and calibrated Pi/RealFlight transforms.**
2. **No Onboard Deployment:** Even at 4B FP8, it consumes `~4.5–6 GB` minimum (OOM on `2 GB` `adv034`) and has a `312–800 ms` LPDDR4 bandwidth floor vs `< 5 ms` policy budget ([`05_cosmos3_skeptical_audit.md`](05_cosmos3_skeptical_audit.md) §1.3).

### 3.2 Where `Cosmos3-Edge` (4B) Genuinely Adds Value on the RTX 3090
1. **Offline Hard-Negative Mining to Crush the 12% Distractor FPR:**
   The single worst metric in [`12_belly_vision_experiment_results.md`](12_belly_vision_experiment_results.md) is the Keypoint CNN's **12.0% distractor FPR** (triggering on chair legs, boxes, and shoes). When we capture unlabeled real Pi walkaround clips in the lab (`calibration/runs/l7/*/frames/`) or outdoor runway tests:
   - Run our trained `118k` PyTorch Keypoint CNN (or `YOLO11n-pose`) in inference over all real frames.
   - Run `Cosmos3-Edge` (4B) on the RTX 3090 over the flagged detections (`p_aircraft > 0.5`) with a structured binary/categorical prompt: *"Does this image contain a fixed-wing twin-propeller aircraft or drone landing gear, or only room furniture/legs/boxes/floor glare?"*
   - Every frame where the student fires `p > 0.85` and `Cosmos3-Edge` confirms `aircraft_present = false` (or `clutter_distractor = true`) is **automatically harvested as a verified `valid = 0` hard negative** for student retraining—zero manual scrubbing across tens of thousands of real video frames.
2. **Failure-Mode Bucketing & Targeted `Sim-Transfer` Prompt Synthesis:**
   Run `Cosmos3-Edge` offline across real Pi failure trajectories to emit structured JSON metadata per frame: `{glare_level: 0..3, motion_blur: bool, under_belly_shadow: bool, occlusion_source: str, visible_side: "rear_vtail"|"nose"|"side"|"none"}`. This gives two concrete wins:
   - Stratified evaluation splits on real frames (measuring student PCK / pose drift specifically on `glare_level == 3` real frames, addressing Risk #3 in [`07_cnn_lejepa_representations.md`](07_cnn_lejepa_representations.md) §10).
   - Automated prompt generation for `Sim-Transfer` (§2.4): whenever real failures cluster under a specific condition (e.g., *"low-angle sunlight reflecting off concrete beside left V-tail fin"*), `Cosmos3-Edge`'s description becomes the next synthetic batch prompt.
3. **Optional 3-Logit Scene-Condition Auxiliary Distillation (`+0.02 ms` Pi Delta):**
   If desired, `Cosmos3-Edge`'s semantic ratings (`glare_heavy`, `heavy_occlusion`, `motion_blur`) on real + synthetic frames can supervise a tiny `3`-output linear branch off the student's `128`-dim `AttnPool(f)` embedding (`3 × 128 = 384` extra params, `< 0.02 ms` on Pi CPU). The 50 Hz EKF ([`src/ugv/staged_baseline.py`](../../src/ugv/staged_baseline.py)) can use `glare_heavy` purely to **inflate measurement covariance `R`** (trusting IMU/gyro prediction slightly longer during transient glare spikes)—never as motion authority.

---

## 4. Evaluation (c): Exact VRAM & Quantization Reality on RTX 3090 (INT8 / W4A16 vs. FP8)

### 4.1 Ampere SM86 Hardware Reality Check
Verified on the local workstation GPU: `NVIDIA GeForce RTX 3090`, `24,576 MiB` (`24.0 GB`) GDDR6X (`384-bit`, `936.2 GB/s`), **Compute Capability `8.6` (Ampere)**.

| Precision / Kernel Path | Native SM86 Tensor Core? | Peak Throughput (3090) | Weight Bytes / Param | Actual Behavior on RTX 3090 (`SM 8.6`) |
|---|---|---:|---:|---|
| **FP16 / BF16 (`HMMA`)** | **Yes** | **142 TFLOPS** | 2.0 B | Native baseline; fastest compute per byte unpacked, zero quantization loss, but `15B` DiT alone = `30 GB` (OOM). |
| **Native FP8 (`E4M3`/`E5M2`)** | **NO** (Added in Ada SM89 / Hopper SM90) | **0 TFLOPS** | 1.0 B | `torch._scaled_mm` / TransformerEngine FP8 GEMMs fail SM86 capability checks (`8.6 < 8.9`). Requires software dequant (`W8A16`) or upcast to BF16. |
| **FP8 Stored + `W8A16` Marlin Dequant** | **No** (dequants to `FP16` MMA) | `≤ 142 TFLOPS` | 1.0 B | Saves `2×` weight VRAM; **2× faster on bandwidth-bound decoding** (`Edge` 4B token gen), but **10–25% slower than FP16** on compute-bound DiT attention/convs due to bit-unpack overhead. |
| **`W8A8` INT8 (`IMMA` / SmoothQuant)** | **Yes** | **284 TOPS** | 1.0 B | **True 2× SM86 compute + 2× VRAM win** on compute-bound linear layers (`torch._int_mm` / CUTLASS), *if* activation outliers are smoothed (SmoothQuant/INT8-DiT). Note: `bitsandbytes` `load_in_8bit` (`LLM.int8()`) uses slow mixed-precision outlier decomposition—often slower than BF16. |
| **`W4A16` (AWQ / GPTQ / Marlin 4-bit)** | **Dequants to `FP16` MMA** | `≤ 142 TFLOPS` | **~0.55 B** (w/ g128 scales) | **3.6× memory bandwidth gain** + **~3.5× weight VRAM reduction**. Optimal way to fit `15B` `Sim-Transfer` DiT on 24 GB without PCIe offload—provided patchify, AdaLN, VAE, and output heads stay FP16/BF16. |

### 4.2 Component-by-Component VRAM Budgets on 24 GB RTX 3090

Ignoring the text encoder and 3D video VAE (as the prior draft did) produces dangerously wrong VRAM estimates. Below is the realistic single-GPU memory accounting (`CUDA context + PyTorch caching allocator ≈ 1.2 GB` included):

#### A. `Cosmos3-Edge-FP8` (4B VLM — Triage & Hard-Example Mining)
| Component | BF16 (Recommended) | `W8A16` (FP8→FP16 Marlin) | `W4A16` (Marlin) |
|---|---:|---:|---:|
| 4B LLM + Vision Tower Weights | `8.2 GB` | `4.4 GB` | `2.6 GB` |
| KV Cache + Vision Tokens (`batch=1–4`, `720p`) | `1.8–3.0 GB` | `1.8–3.0 GB` | `1.8–3.0 GB` |
| CUDA Context + Allocator Headroom | `1.2 GB` | `1.2 GB` | `1.2 GB` |
| **Total Peak VRAM** | **`11.2–12.4 GB` ✓** | **`7.4–8.6 GB` ✓** | **`5.6–6.8 GB` ✓** |
| **3090 Verdict** | **Run in BF16** (12 GB free; zero quant risk) | Use only for large-batch (`batch ≥ 8`) | Unnecessary unless sharing GPU |

#### B. `Cosmos3-Nano-Sim-Transfer` (15B Control-Conditioned Diffusion — `480p` Sim2Real)
| Component | BF16 (Naive) | Naive 8-Bit (All Resident) | **Config 1: Cached-T5 + `W8A8`/`W8A16` DiT (Quality)** | **Config 2: Cached-T5 + `W4A16` Marlin DiT (Speed)** |
|---|---:|---:|---:|---:|
| 15B DiT + Multi-Control Conditioning | `30.0 GB` | `15.5 GB` | `15.5 GB` (8-bit DiT) | **`8.8 GB`** (`W4A16` attn/MLP + FP16 AdaLN/patchify) |
| Text Encoder (T5-XXL-class) | `9.5 GB` | `4.8 GB` | **`0.0 GB`** (pre-encoded & cached on disk!) | **`0.0 GB`** (pre-encoded & cached on disk!) |
| 3D Causal Video VAE | `2.0 GB` | `2.0 GB` | **`2.0 GB`** (or CPU-swapped post-denoise = `0 GB`) | **`2.0 GB`** (stays GPU-resident) |
| Denoising Activations (`640×480`, `1–9` frames, `SDPA`) | `4.5 GB` | `4.5 GB` | `3.8–4.8 GB` (1–5 frames, tiled VAE) | `4.2–6.0 GB` (1–17 frames) |
| CUDA Context + Allocator Headroom | `1.2 GB` | `1.2 GB` | `1.2 GB` | `1.2 GB` |
| **Total Peak VRAM** | **`47.2 GB` (OOM)** | **`28.0 GB` (OOM)** | **`22.5–23.5 GB` (Tight, 1–5 frames)** *(or `20.5 GB` w/ VAE swap)* | **`16.2–18.0 GB` ✓ (Comfortable)** |
| **Est. 3090 Throughput (`480p`, 4–8 steps)** | N/A | N/A | **`~0.12–0.30 FPS`** (`3–8 s/frame`) | **`~0.25–0.50 FPS`** (`2–4 s/frame`) |

**Key Quantization Takeaway for RTX 3090:**
1. **Never keep the Text Encoder resident during DiT denoising.** Pre-encoding our `~50–100` hangar/runway lighting prompts into `.pt` tensors once frees `4.8–9.5 GB` of VRAM permanently.
2. **Avoid `accelerate` layer-by-layer CPU offload for `Sim-Transfer`.** Streaming `15 GB` across PCIe 4.0 x16 (`~24 GB/s` vs `936 GB/s` GDDR6X — a `39×` bandwidth penalty) per diffusion step collapses throughput to `< 0.03 FPS`. Either use **Config 2 (`W4A16` Marlin on DiT linear blocks + FP16 sensitive layers)** so the entire denoising loop + VAE stays in `16–18 GB` VRAM, or use **Config 1 (`W8A8` INT8 / `W8A16` DiT)** with sequential 2-stage swap (text encode once → denoise → VAE decode).
3. **Never confuse `FP8` checkpoint format with `SM86` capability:** An `-FP8` checkpoint on HuggingFace is just a `1 byte/param` storage format; on an RTX 3090 it executes via software dequant (`W8A16`) or must be converted to `W8A8 INT8` (true `284 TOPS` SM86 Tensor Cores) or `W4A16` Marlin (`~16–18 GB` peak).

---

## 5. End-to-End Offline-to-Pi Architecture (Grounded in Repo Specs)

If gated in, Cosmos 3 plugs strictly into the **offline 3090 data & mining loop**, completely decoupled from the Pi runtime:

```text
[RTX 3090 — Offline Stage 1: Synthetic Geometry & Controls]
  Blender + STRL001 Model (Agent 03)
    ├─► 80% Fast Cycles PBR + HDRI + Real Pi Floor Plates (0.00 px drift, ~3–10 FPS)
    │     → 40k frames + exact 8-keypoint 2D/3D GT + valid=1/0 clutter negatives
    └─► 20% Control Passes (Z-depth + Part Seg + Clean Canny Edges)
          └─► Cosmos3-Nano-Sim-Transfer (15B, W4A16/W8A8, cached prompts)
                ├─ View A (clean runway) + View B (epoxy glare / shadow / clutter)
                ├─ Quality Gate: 480p Canny edge chamfer ≤ 1.5 px (≤ 0.38 px @ 160×120)
                └─► 5k–10k Dual-View LeJEPA Invariance Pairs (exact same GT pose!)

[RTX 3090 — Offline Stage 2: Hard-Negative Mining on Real Pi / Hangar Clips]
  Unlabeled Real Frames (calibration/runs/l7/*/frames/ + STRL001 captures)
    └─► Student Keypoint CNN flags p_aircraft > 0.5
          └─► Cosmos3-Edge (4B, BF16) audits false positives & glare severity
                ├─ Confirmed distractors (chair legs/shoes/glare) → valid=0 hard negatives
                └─ Failure descriptions → new prompts for Stage 1 Sim-Transfer

[RTX 3090 — Offline Stage 3: PyTorch Student Training]
  PyTorch (rl/.venv, 1-GPU rule) trains 107k–160k Keypoint + LeJEPA CNN (160×120 L8)
    (or YOLO11n-pose) on 80% PBR + 20% Sim-Transfer Dual-View + Mined Hard Negatives
    └─► torch.onnx.export(..., opset_version=17) + ORT parity < 1e-5 (Agent 13)

[Raspberry Pi 4B (adv034) — Real-Time Deploy (ZERO Cosmos Weights on Pi)]
  1280×720 L8 → 160×120 crop/resize → cv2.dnn ONNX (~5.2 ms projected)
    → 8 Keypoints + Chirality & Metric Gates (Agents 06/09) → solvePnPRansac
    → 50 Hz EKF + StudentPolicy (.npz, 98.2 kB, <0.5 ms) + swept_clear + 600 ms watchdog
```

---

## 6. Falsification Gates (No Gate Pass = Drop Cosmos 3)

Per project discipline, Cosmos 3 must justify its complexity empirically against **pure Blender Cycles PBR + classical DR**. Every stage has an explicit kill switch:

1. **Gate 1 — 3090 Fit & Throughput Gate (`Sim-Transfer` 15B):**
   Pre-encode prompts, load `Cosmos3-Nano-Sim-Transfer` in `W4A16` (Marlin) or `W8A8`/`W8A16` with tiled FP16 VAE on the local RTX 3090. **Kill criterion:** if peak VRAM exceeds `23.0 GB` or sustained throughput at `480p` drops below **`0.10 FPS`** (`>10 s/frame`), abandon `Sim-Transfer` and rely 100% on Blender Cycles PBR + real-background compositing.
2. **Gate 2 — Sub-Pixel Label Preservation Gate:**
   Evaluate `200` `Sim-Transfer` frames over Blender `STRL001` controls using edge chamfer distance on `left/right_wheel_contact` and `left/right_fin_root`. **Kill criterion:** if median edge drift exceeds **`2.0 px` at `480p`** (`0.50 px` at `160×120`, `~4 mm` at `1 m`), or if `> 25%` of glare frames fail the consistency filter, `Sim-Transfer` introduces more geometric noise than it cures.
3. **Gate 3 — Downstream Sim2Real Student Lift Gate (The Decisive Test):**
   Train two identical `107k–160k` Keypoint/LeJEPA PyTorch students (same seed, same compute budget):
   - **Baseline:** `100%` Blender Cycles PBR + standard photometric DR (`gamma`, Gaussian noise, blur, synthetic glare blobs).
   - **Treatment:** `80%` Blender Cycles PBR + `20%` `Sim-Transfer` dual-view LeJEPA pairs + `Cosmos3-Edge` mined real hard negatives.
   Evaluate both on **held-out real Pi / `STRL001` frames** (split by glare severity `0–3` and distractor-only clutter frames). **Kill criterion:** if Treatment does not measurably reduce held-out real-frame **distractor FPR** (from the `12.0%` baseline in [`12_belly_vision_experiment_results.md`](12_belly_vision_experiment_results.md) toward `< 2%`) or improve **heavy-glare keypoint PCK / pose MAE**, drop Cosmos 3 from the pipeline and ship the pure Blender PBR + LeJEPA student.