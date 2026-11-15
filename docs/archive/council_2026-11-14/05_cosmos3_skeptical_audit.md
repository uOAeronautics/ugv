# Agent 05 — Skeptical Systems & Latency Audit: NVIDIA Cosmos 3 for Sub-5 cm UGV Under-Belly Parking

**Auditor:** Agent 05 (Skeptical Systems & Latency Auditor)  
**Target Stack:** Training on single **NVIDIA RTX 3090 (24 GB, Ampere SM86)** $\rightarrow$ Deployment at **50 Hz (`< 5 ms` policy budget)** on **Raspberry Pi 4B (`ARM Cortex-A72` CPU, 2–8 GB LPDDR4, no GPU/NPU/FP8 Tensor Cores)**  
**Evaluated Artifacts:** [`cosmos3_huggingface.png`](cosmos3_huggingface.png) (`nvidia/cosmos3-examples` collection: `Cosmos3-Edge-FP8` 4B, `Cosmos3-Nano-Sim-Auto` 15B, `Cosmos3-Nano-Transfer-Auto` 15B, `Cosmos3-Nano-Sim-Transfer` 15B, `Cosmos3-Nano-FP8` 16B)  
**Verdict:** **HARD REJECT (NO-GO) for onboard deployment, RL training loop integration, and ArUco-bearing `Sim-Transfer`.** At most a **low-priority, offline-only background-clutter augmentation experiment**—and even there, strictly inferior to calibrated PBR/Gazebo rendering or analytical homography compositing for our sub-5 cm tolerance regime.

---

## Executive Summary

NVIDIA's `Cosmos3` collection (`4B` Edge, `15B` Nano-Sim/Transfer, `16B` Nano-FP8) targets **Ada/Hopper/Blackwell datacenter/workstation GPUs with native FP8 Tensor Cores** and **high-clearance autonomous driving / bimanual manipulation** (decimeter-scale tolerances, multi-hundred-millisecond or offline planning). Every architectural assumption in Cosmos 3 collides head-on with the physics and compute envelope of `.`:

1. **Pi 4B Deployment Failure (100×–2,000× over latency budget, 2×–4× over RAM):** `Cosmos3-Edge-FP8` (4B) requires native FP8 tensor hardware that `ARM Cortex-A72` does not have. Bound by LPDDR4-3200 memory bandwidth (`12.8 GB/s` peak, `~4–6 GB/s` sustained CPU) and NEON FP32 compute (`96–115 GFLOP/s` peak), a single 4B forward step takes **312–800 ms** in pure weight streaming alone and **2,000–10,000 ms (0.1–0.5 FPS)** with visual tokens/diffusion—compared to our **50 Hz (`20 ms` total loop, `< 5 ms` policy budget)** and our current [`StudentPolicy`](../../src/ugv/policy.py) (`GRU-64`, `~23k` params, `98.2 kB`, `< 2 ms` on Pi). On our actual test unit (`adv034`, **2 GB Pi 4B** with `~1.3 GiB` free RAM per [`HANDOFF.md`](../../HANDOFF.md)), or even a 4 GB Pi 4B, 4B weights + activations trigger immediate **OOM / swap thrashing**.
2. **RTX 3090 Training & Sim Throughput Collapse (5–7 orders of magnitude slower):** RTX 3090 is **Ampere SM86**—it has **zero native FP8 Tensor Cores** (introduced in SM89 Ada / SM90 Hopper). Running 15B–16B `Cosmos3-Nano` models on 24 GB VRAM requires BF16/FP16 fallback (`30–32 GB` weights alone $\rightarrow$ OOM without CPU offload) or lossy INT4/INT8 quantization + offload, collapsing rollout throughput to **~0.02–0.5 frames/sec** versus **`~243,000 steps/sec`** in our vectorized PyTorch simulator [`MecanumVecEnv`](../../rl/sim.py).
3. **Sub-3 cm Geometric Hallucination Risk:** Under-belly drone parking has **`4.7 cm` per-side physical clearance** (`0.254 m` skid gap vs `0.16 m` car width) and a **`±30 mm` lateral / `±50 mm` longitudinal / `±5°` yaw** gate ([`docs/RL_PARKING.md`](../RL_PARKING.md)). Video diffusion/world-model `Sim-Transfer` compresses `1280×720` frames into coarse `8×`/`16×` spatial latents—destroying `50 mm` ArUco `DICT_4X4_50` bit grids (`5.4 px/bit` at `1.5 m`) and warping wheel/skid contact geometry by **2–5 cm**, which consumes **67%–167% of our entire parking error budget**.

---

## 1. Raspberry Pi 4B Deployment Audit: `Cosmos3-Edge-FP8` (4B)

### 1.1 Hardware Baseline (`BCM2711` / `adv034`)
| Parameter | Pi 4B Spec (4 GB / 8 GB) | Measured Project Pi (`adv034`, [`HANDOFF.md`](../../HANDOFF.md)) | Current [`StudentPolicy`](../../src/ugv/policy.py) |
| :--- | :--- | :--- | :--- |
| CPU | 4× ARM Cortex-A72 @ 1.5–1.8 GHz (ARMv8.0-A) | Same (`cores=4`) | Pure NumPy (no Torch/ORT) |
| SIMD / Precision | NEON 128-bit (FP32, INT8 scalar/4-way; **no FP8, no native FP16 arith, no `i8mm`/`dotprod`**) | Same | FP64 / FP32 |
| RAM Available | ~3.4 GB (4 GB) / ~7.4 GB (8 GB) | **1.8 GiB total (`~1.3 GiB` free)** with camera + dashboard | `160 MB` dashboard + `67 MB` `rpicam-vid` |
| Memory Bandwidth | `12.8 GB/s` peak theoretical (32-bit LPDDR4-3200); **`~4.0–6.0 GB/s` sustained CPU** | Shared with MJPEG camera DMA & OS | `~98.2 kB` `.npz` fits in L2 cache (`1 MB`) |
| FP32 Peak Compute | `4 cores × 1.5 GHz × 16 FLOPs/cyc = 96 GFLOP/s` (`115.2` @ 1.8 GHz) | Sustained `~30–50 GFLOP/s` | `~0.1 MFLOPs/step` (`< 2 ms` Pi CPU, `0.90 ms` desktop) |

### 1.2 Memory Footprint & OOM Analysis
- **Weights:** `4B` parameters at 1 byte/param (FP8 `E4M3`) = **`~4.0 GB`** on disk. Because Cortex-A72 has **no FP8 ALU** (`E4M3`/`E5M2` cannot execute in NEON), weights must either be:
  - Upcast on-the-fly in small tiles (adds software bit-unpacking/shift instructions on every MAC), or
  - Pre-dequantized to FP16/FP32 (`8.0 GB` – `16.0 GB`).
- **KV Cache, Visual Patch Embeddings & Activations:** Even a modest context/visual buffer (e.g., `512–2,048` spatio-temporal tokens across 28–32 transformer layers) adds **`0.5–2.0 GB`** of activations and KV state.
- **Total RAM:** **`4.5–6.0 GB`** minimum (with on-the-fly custom FP8 $\rightarrow$ FP32 kernels) or **`9–18 GB`** in standard FP16/FP32 runtimes.
  - On **2 GB Pi 4B (`adv034`)**: **Immediate OOM kill** before model load finishes.
  - On **4 GB Pi 4B**: Leaves `< 0 GB` after Linux + `rpicam-vid` + `uvicorn` (`~540 MiB` baseline per [`HANDOFF.md`](../../HANDOFF.md)) $\rightarrow$ **swap thrashing on SD/eMMC** (`10–50×` slowdown).
  - On **8 GB Pi 4B**: Fits in RAM *only* if quantized to custom INT4/FP8 kernels, still leaving the fundamental bandwidth/FLOPs wall below.

### 1.3 Exact Memory-Bandwidth & FLOPs Bounds vs. 50 Hz Control
Our RL loop ([`rl/sim.py`](../../rl/sim.py)) runs at **`dt = 0.02 s` (50 Hz)** with a **`< 5 ms` policy budget** (because ArUco vision already consumes `vision_ms_p50 = 44–50 ms` asynchronously at 20–30 Hz, per [`HANDOFF.md`](../../HANDOFF.md) §99, and [`PerfMonitor.BUDGET_MS`](../../src/ugv/perf.py#L27) tracks the `33.33 ms` camera cadence).

Let's bound `Cosmos3-Edge-FP8` (4B) per inference call:

1. **Memory-Bandwidth Bound (Single-Step / Single-Token Floor):**
   Reading `4.0 GB` of 8-bit weights once per forward pass (even assuming **zero** KV-cache traffic and **zero** dequantization overhead):
   $$t_{\text{mem, peak}} = \frac{4.0\text{ GB}}{12.8\text{ GB/s}} = \mathbf{312.5\text{ ms}} \quad (3.2\text{ Hz max})$$
   $$t_{\text{mem, real}} = \frac{4.0\text{ GB}}{5.0\text{ GB/s}} = \mathbf{800\text{ ms}} \quad (1.25\text{ Hz max})$$
   Even an **unachievable, 100%-bandwidth-efficient 1-token forward pass** exceeds our `5 ms` policy budget by **62×–160×** and violates the `20 ms` 50 Hz period by **15×–40×**.

2. **Compute Bound (Visual Tokens / World-Model Rollout):**
   A `4B` dense transformer performs $\approx 2 \times 4\times 10^9 = 8\text{ GFLOPs}$ per token. Any image-conditioned world/VLA model must process visual patch tokens $N_{\text{tok}}$ (e.g., a downsampled `320×180` frame with `14×14` or `16×16` patches gives $N_{\text{tok}} \approx 220\text{–}260$ tokens, plus history/action tokens $\rightarrow$ conservatively $N_{\text{tok}} \in [128, 512]$):
   $$\text{FLOPs}_{\text{frame}} \approx N_{\text{tok}} \times 8\text{ GFLOPs} = 1.02\text{–}4.10\text{ TFLOPs}$$
   At Cortex-A72's theoretical FP32 peak (`96 GFLOP/s`) and realistic sustained efficiency (`~40 GFLOP/s`, accounting for thermal throttling above `60°C`—already `60.3°C` idle on `adv034`—and software FP8 $\rightarrow$ FP32 unpack):
   $$t_{\text{compute}} = \frac{1.02\text{–}4.10\text{ TFLOPs}}{40\text{ GFLOP/s}} = \mathbf{25.5\text{–}102.5\text{ s/frame}} \quad (0.01\text{–}0.04\text{ FPS})$$
   Even with aggressive token pruning / single-frame `64`-token encoding + 1-step action decode (`~512 GFLOPs` at `40–50 GFLOP/s`, or **0.1–0.5 FPS** $\approx$ **2,000–10,000 ms** per control update), the Pi 4B CPU is **400×–2,000× too slow** for `< 5 ms` / 50 Hz control. At 0.2 FPS (`5.0 s` blind latency), our UGV moving at even `0.30 m/s` travels **`1.50 m`** blind—smashing straight through the `0.3048 m` drone skids before the first neural action returns.

---

## 2. RTX 3090 Training & Simulation Audit: `Cosmos3-Nano` (15B / 16B)

### 2.1 Architectural Mismatch on Ampere SM86 (24 GB VRAM)
From [`cosmos3_huggingface.png`](cosmos3_huggingface.png), the domain-specialized models are **`15B`** (`Cosmos3-Nano-Sim-Auto`, `Cosmos3-Nano-Transfer-Auto`, `Cosmos3-Nano-Sim-Bimanual`, `Cosmos3-Nano-Sim-Transfer`) and **`16B`** (`Cosmos3-Nano-FP8`).

| Resource / Metric | RTX 3090 (Ampere SM86, 24 GB) | `Cosmos3-Nano` 15B / 16B Requirement | Vectorized [`rl/sim.py`](../../rl/sim.py) Baseline |
| :--- | :--- | :--- | :--- |
| Native FP8 Tensor Cores | **None** (FP8 added in Ada SM89 / Hopper SM90) | Built for native FP8 Tensor Cores | Pure FP32 PyTorch CUDA |
| Weights Footprint | `24.0 GB` total VRAM | `15–16 GB` (INT8/FP8 stored) or **`30–32 GB` (BF16/FP16)** | `< 1 MB` (`Student`: `23k` params; `Actor`: `3×256`) |
| Training / Inference Fit | `~18–20 GB` usable after CUDA + VAE + activations | **OOM in BF16/FP16**; requires CPU offload or 4-bit QLoRA | `~10 GB` at `N=1024` envs (leaves 14 GB headroom) |
| Simulation Throughput | `35.6 TFLOP/s` FP32 / `142 TFLOP/s` FP16 TC | **`~0.02–0.5 frames/sec`** (offloaded diffusion/AR) | **`~243,000 steps/sec`** (`4,096–8,192` envs) |

### 2.2 What Actually Happens on a Single RTX 3090
1. **No Native FP8 Acceleration:** On SM86, `torch._scaled_mm` / TransformerEngine FP8 GEMMs fail hardware capability checks (`compute capability 8.6 < 8.9`). Running `Cosmos3-Nano-FP8` (16B) forces either software dequantization to FP16/BF16 (destroying the 2× bandwidth saving and exploding peak memory) or unsupported-op errors.
2. **VRAM Cliff at 24 GB:** A `15B` diffusion/world model in BF16 needs `30 GB` just for parameters, `60 GB` with AdamW states + gradients (`120 GB` for full fine-tuning). Even for **inference-only** video generation (`Sim-Transfer`), adding the video tokenizer/VAE (`~1.5–3 GB`), T5/text/conditioning encoder (`~2–5 GB`), and `720p` spatio-temporal latent activations (`~4–8 GB`) blows past `24 GB` unless weights are aggressively CPU-offloaded or 4-bit quantized—cutting effective memory bandwidth from `936 GB/s` (GDDR6X) down to PCIe 4.0 x16 (`~24 GB/s`, a **39× bandwidth collapse**).
3. **5–7 Order-of-Magnitude RL Throughput Collapse:**
   - Our current `rl/sim.py` (`SE(2)` Bayesian mecanum dynamics + sensor emulator) runs **`~243,000 env steps/sec`** on a single RTX 3090 (`300` PPO student iterations finish in minutes).
   - Generating synthetic training frames with a `15B` world/transfer model at `~0.1–0.5 FPS` (`2–10 s/frame`) means **a single 10-second episode at 30 Hz (`300` frames) takes 10–50 minutes to synthesize**. Collecting the `~10–50 million` transitions needed for L2–L4 RL generalization would take **years** on one RTX 3090.

---

## 3. Geometric Hallucination & Sub-5 cm Precision Failure

Even if compute were free, using generative video world models (`Cosmos3-Nano-Sim-Transfer` / `Sim-Auto`) in the perception or sim-to-real loop introduces fatal metric errors for under-belly parking.

### 3.1 Physical Clearance & ArUco Pixel Budget
From [`rl/sim.py`](../../rl/sim.py#L165-L179), [`config/parking.json`](../../config/parking.json), [`src/ugv/aruco.py`](../../src/ugv/aruco.py#L14-L18), and [`docs/RL_PARKING.md`](../RL_PARKING.md):
- **Car dimensions:** `0.16 m` wide × `0.22 m` long.
- **Drone landing skids:** `0.254 m` clear inner gap (`bar_radius = 0.03 m`, `bar_length = 0.3048 m`).
  - **Hard collision margin:** `(0.254 - 0.160) / 2 =` **`4.7 cm` (`47 mm`)** per side (with zero yaw error; any yaw $\theta$ shrinks clearance by $\frac{L}{2}\sin|\theta| \approx 1.9\text{ mm/deg}$).
- **Success gate:** **`|dy| <= 30 mm` (3.0 cm)** lateral, **`|dx| <= 50 mm` (5.0 cm)** longitudinal, **`|yaw| <= 5°`**.
- **Visual Fiducial Budget (`DICT_4X4_50`, `50 mm` tag, `1280×720`, `fx = fy = 968 px`):**
  - At `0.50 m` range: tag width = `96.8 px` $\rightarrow$ `~16.1 px` per ArUco grid cell.
  - At `1.50 m` range (L1–L3 spawn envelope): tag width = `32.3 px` $\rightarrow$ **`~5.4 px` per ArUco grid cell**.
  - At `4.00 m` range (L4 outer edge): tag width = `12.1 px` $\rightarrow$ **`~2.0 px` per ArUco grid cell**.
  - OpenCV `solvePnP` requires **`< 0.5 px` corner precision** to keep lateral pose noise below `~8–15 mm` at `1.0–1.5 m`.

### 3.2 Failure Modes of Diffusion `Sim-Transfer` on Fiducials & Contact Geometry
1. **VAE Latent Aliasing Destroys `4×4` ArUco Bits:**
   Latent video diffusion models encode `1280×720` RGB into `8×` or `16×` spatially downsampled latent tokens (effective resolution `160×90` or `80×45`). At `1.5 m`, our entire `32 px` ArUco marker collapses into **`2×2` to `4×4` latent pixels**. Decoding back to pixel space hallucinates blurred edges, curved bit squares, or flipped interior bits—causing `cv2.aruco` decode dropouts or corrupted PnP corners right in the range where our real pipeline needs clean fixes.
2. **2–5 cm Perspective & Wheel/Skid Contact Drift:**
   Unconstrained or text/video-prompted `Sim-Transfer` routinely shifts object boundaries, floor-plane shadows, and wheel/skid contact lines by **`2–5 cm` (`20–50 mm`)** across frames to satisfy photorealistic priors (e.g., smoothing concrete glare or hallucinating skid foam thickness). A `3 cm` lateral hallucination on a skid edge or tag shift is **100% of our `±30 mm` goal tolerance** and **64% of the physical collision gap**.
3. **Why Depth/Edge ControlNets Still Fall Short:**
   Adding depth + Canny edge ControlNets suppresses large spatial drift, *but*:
   - Depth maps from Gazebo/Sim at `1.5 m` still quantize or smooth thin `50 mm` planar paper tags (zero depth relief against the drone battery/belly!), so depth conditioning cannot constrain ArUco bit fidelity.
   - Edge conditioning still passes through the noisy latent denoising loop (`5–20` DDIM/flow steps on RTX 3090, making generation even slower) and does not fix **our actual sim-to-real failure modes** documented in [`HANDOFF.md`](../../HANDOFF.md) (§4, §8, §10):
     1. Turn-channel deadzone / scrub breakaway friction (`600–800 PWM`),
     2. Uncalibrated Tag 6 (`obs=0` in `tagmap.json`),
     3. CW+ vs CCW+ IMU sign conventions, and
     4. On-Pi ArUco CPU latency (`44–50 ms` per frame in `detect_jpeg`).

---

## 4. Summary Scorecard & Better Alternatives

| Dimension | `Cosmos3-Edge-FP8` (4B) / `Nano` (15B–16B) | Current [`ugv`](../../rl/sim.py) + [`StudentPolicy`](../../src/ugv/policy.py) | Recommended Alternative |
| :--- | :--- | :--- | :--- |
| **Pi 4B Policy Latency** | **2,000–10,000 ms** (`0.1–0.5 FPS` CPU) | **`< 2 ms`** (`> 500 Hz` NumPy `GRU-64`) | Keep `GRU-64` `.npz` (`0` Torch/ORT) |
| **Pi 4B RAM Usage** | **4.5–18 GB** (OOM on 2 GB/4 GB Pi) | **`98.2 kB`** weights + `0 MB` extra RSS | Keep `< 1 MB` footprint |
| **RTX 3090 Sim Speed** | **0.02–0.5 FPS** (offloaded, no SM86 FP8) | **`243,000 steps/s`** (`4096–8192` envs) | Keep vectorized `sim.py` + `L_real` DR |
| **Metric / Tag Fidelity** | **2–5 cm** drift; latent-blurred ArUco bits | Exact SE(2) kinematics + calibrated noise | **Gazebo PBR + analytical tag homography** (zero generative tag pixels) |

### What to Do Instead (Concrete, High-ROI Path)
1. **Do NOT put Cosmos 3 anywhere in the Pi 4B runtime or the `rl/sim.py` PPO loop.** The hardware mismatch (no FP8 on SM86 or Cortex-A72; 2 GB RAM on `adv034`; `< 5 ms` budget) is fundamental, not a tuning issue.
2. **To close the Gazebo-vs-real visual gap noted in [`HANDOFF.md`](../../HANDOFF.md) §8** (specular concrete floor, grey-capped foam skids, lab clutter):
   - Upgrade `freenove_world.sdf` in Gazebo Harmonic with **procedural specular concrete textures, realistic skid end-caps, and background clutter meshes**, or
   - Use **2D/3D PBR domain randomization + analytical ArUco homography compositing** (render/composite real Pi floor backgrounds from `calibration/runs/l7/*/frames/` *behind* analytically projected, pixel-exact ArUco tags—giving 100% realistic glare/clutter with **0.0 mm** geometric hallucination and **1000×** faster generation on RTX 3090).
3. **Spend the compute budget on the real bottleneck:** Profiling and optimizing the `44–50 ms` on-Pi OpenCV `TagDetector` path (`PerfMonitor` in [`src/ugv/perf.py`](../../src/ugv/perf.py)), completing Tag 6 multi-tag calibration, and scaling `L_real` / `L_chaos` PPO training in [`rl/sim.py`](../../rl/sim.py).
