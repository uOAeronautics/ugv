# Agent 13 — End-to-End PyTorch (RTX 3090) → ONNX → Raspberry Pi 4B Edge Pipeline

**Verification script:** [`docs/council_v2/exp_onnx_cv2dnn_bench.py`](file://docs/council_v2/exp_onnx_cv2dnn_bench.py)  
**Raw benchmark JSON:** [`docs/council_v2/exp_onnx_out/bench_results.json`](file://docs/council_v2/exp_onnx_out/bench_results.json)  
**Reproduce (from `rl/` venv):**
```bash
cd rl && uv run python ../docs/council_v2/exp_onnx_cv2dnn_bench.py
```
*(No existing repo source files outside `docs/council_v2/` were modified.)*

---

## 1. Executive Answer: Where "No Torch" Came From & Why ONNX Is the Right Bridge

There was never a restriction against **training** with PyTorch. Inspecting the repo makes the boundary explicit:

- **Training lab (RTX 3090, [`rl/pyproject.toml`](file://rl/pyproject.toml)):** `ugv-rl` is 100% PyTorch (`torch 2.14.1+cu126`, occupying **1.7 GB** on disk in `rl/.venv` for `torch` alone, plus CUDA libraries).
- **Pi deployment ([`pyproject.toml`](file://pyproject.toml), [`rl/AGENTS.md`](file://rl/AGENTS.md#L45-L46)):** `"Pi stays torch-free"` exists solely so the Raspberry Pi 4B (`BCM2711`, 2 GB RAM, SD card) does not install a **~200–500 MB+ CPU `torch` wheel** (or 1.7 GB desktop wheel) just to run inference.
- **Why `.npz` was used originally ([`src/ugv/policy.py`](file://src/ugv/policy.py)):** When the only neural network on the Pi was `Student` (**23,683** deployable parameters: a single-layer GRU-64 + 2×64 tanh MLP = **97.3 kB** `.npz`), hand-unrolling 12 lines of NumPy (`_step_gru` + `_head`) was trivial, needed zero new dependencies, and ran in **0.0256 ms** (`~0.09 ms` projected on Pi 4B).
- **Why `.npz` fails for vision — and why ONNX solves it cleanly:** Now we want PyTorch-trained **vision** models on the RTX 3090 (a **LeJEPA + Spatial-Soft-Argmax 8-keypoint aircraft belly detector**, or **YOLOv11n-Pose** / distilled CNNs). Hand-porting `Conv2d` (`im2col`), `BatchNorm2d`, `ReLU`/`SiLU`, `Softmax`, `ReduceMean`, and `Spatial-Soft-Argmax` into pure NumPy for every architectural iteration is slow (10–50× slower than C++/NEON) and guarantees parity bugs.
- **ONNX is the exact compiler bridge:** Train freely in PyTorch on the 3090 → strip training-only heads (`aux_head`, `dyn_head`, `jepa_pred`) automatically via `torch.onnx.export` → execute on the Pi CPU in C++/NEON via **`onnxruntime`** (74 MB unpacked, **2.07× faster than `cv2.dnn`** in our benchmark: **0.759 ms** desktop / **~2.66 ms** projected Pi 4B for the 8-keypoint LeJEPA detector) or **`cv2.dnn.readNetFromONNX`** (already installed via `opencv-contrib-python-headless>=4.13` in [`pyproject.toml`](file://pyproject.toml#L9) = **0 MB** incremental footprint, **1.574 ms** desktop / **~5.51 ms** projected Pi 4B).

| Stack Layer | Location | Framework / Runtime | Disk Footprint (`rl/.venv` measured) | Role |
| :--- | :--- | :--- | ---: | :--- |
| **RL + Vision Training** | RTX 3090 (`rl/`) | PyTorch `2.14.1+cu126` | 1,700 MB (`torch`) + CUDA | PPO, distillation, LeJEPA (`SIGReg`), Spatial-Soft-Argmax, YOLO-Pose |
| **Policy Deploy (`Student`)** | Pi 4B (`src/ugv/policy.py`) | **Keep NumPy `.npz`** (or ORT) | 0 MB extra (`numpy`) | 23.7k-param stateful GRU+MLP (`0.09 ms` Pi proj) |
| **Vision Deploy (Primary)** | Pi 4B perception worker | **`onnxruntime` CPU / XNNPACK** | 74 MB unpacked (~15–20 MB ARM64 wheel) | 106k-param 8-KP LeJEPA CNN (`2.66 ms` Pi proj @1T, `1.51 ms` @2T) |
| **Vision Deploy (Zero-Dep Fallback)** | Pi 4B perception worker | **`cv2.dnn.readNetFromONNX`** (`cv2 5.0.0`) | 0 MB extra (already in `pyproject.toml`) | Same `.onnx` file (`5.51 ms` Pi proj @1T, `3.12 ms` @2T) |

---

## 2. Codebase & Prior-Draft Audit

### 2.1 Repo Codebase Findings
1. [`rl/models.py`](file://rl/models.py#L55-L98):
   - `Student` has **24,853** total parameters: `gru` + `head` (**23,683** deploy params) and training-only `aux_head` (`Linear(64, 6)`, 390 params) + `dyn_head` (`Linear(64, 12)`, 780 params) = **1,170** training-only params.
   - `Student.forward(hist)` only traverses `self.gru` and `self.head`, so PyTorch's ONNX exporter naturally excludes `aux_head`/`dyn_head`.
2. [`rl/export.py`](file://rl/export.py#L33-L75):
   - Already supports `--onnx` (`torch.onnx.export(..., dynamo=False)`), because `onnxscript` is not installed in `rl/.venv` (PyTorch 2.9+ defaults to `dynamo=True` which fails without `onnxscript`).
   - **Three gaps in `rl/export.py` today:**
     1. Parity is asserted (`err < 1e-5`) ONLY for `.npz` vs `StudentPolicy.forward`, **not** for `--onnx` vs `onnxruntime`.
     2. Only exports windowed `(1, 50, 13) -> (1, 3)` (`Student.forward`), whereas the Pi deploy path uses `stateful=True` single-step `(1, 1, 13) + h0 -> (act, h1)` (`0.0256 ms` vs `0.8442 ms` windowed — **33× faster**).
     3. `torch.load(args.ckpt, map_location="cpu")` omits `weights_only=False` (older checkpoints with metadata dicts can fail under stricter PyTorch defaults if `weights_only=True` is enforced).
3. [`src/ugv/policy.py`](file://src/ugv/policy.py#L39-L202):
   - `StudentPolicy` supports both `stateful=False` (`forward(self.hist)` over 50 frames) and `stateful=True` (`_step_gru(frame, self._h)` + `_head(self._h)`).
   - The `swept_clear` collision shield (`~0.18 ms` desktop / `~0.63 ms` Pi proj per Agent 10) dominates the policy step anyway, so switching `StudentPolicy` from `.npz` to `.onnx` is optional; **vision is where ONNX delivers massive leverage**.
4. [`pyproject.toml`](file://pyproject.toml#L7-L16):
   - Root depends on `opencv-contrib-python-headless>=4.13` (installed `5.0.0.93`, 72 MB), meaning `cv2.dnn.readNetFromONNX` is already available on the Pi with zero package changes. Adding `onnxruntime` as an optional or root dependency on Pi adds ~74 MB unpacked (vs 1,700+ MB for `torch`) and cuts CNN inference time in half.

### 2.2 Flaws Corrected from Prior `muse-spark-low` Draft
The previous draft (`v1`) had five technical deficiencies that are now fixed in [`exp_onnx_cv2dnn_bench.py`](file://docs/council_v2/exp_onnx_cv2dnn_bench.py):
1. **Wrong vision model & resolution:** Tested a toy 4-conv `96×96` 4-keypoint net (`60,484` params) with no `BatchNorm2d`, no LeJEPA embedding/predictor stripping, and no pose/chirality/confidence heads. Upgraded to a realistic **`160×120` `(1, 1, 120, 160)` LeJEPA + Spatial-Soft-Argmax 8-Keypoint Aircraft Belly Detector** (`119,990` total params, `106,486` deploy params, `13,504` training-only LeJEPA params stripped).
2. **Missing stateful GRU & 50-step rollout check:** Only benchmarked windowed `(1, 50, 13)` `Student` and omitted `cv2.dnn` latency (`"ok"` without timing) and `stateful=True` `(1, 1, 13) + h0` mode.
3. **Missed `cv2.dnn` multi-output truncation footgun:** Calling bare `net.forward()` on a multi-output ONNX graph in OpenCV 5.0.0 silently returns **only the first output** (`keypoints`), dropping `kp_conf`, `pose_xy_sc`, `left_right_sign`, and `target_confidence`.
4. **Missed pixel-space parity scaling:** Unnormalized pixel keypoints (`[0..160]`) accumulate ~`3.05e-5 px` FP32 rounding error across the `300`-element soft-argmax dot product (`1.91e-7` normalized), which tripped a naive unscaled `1e-5` absolute check.
5. **No distribution stats or quantization test:** Only reported mean latency; omitted `p50`/`p95`/`max`, 2-thread scaling, and empirical verification of why naive `quantize_dynamic` degrades small spatial-soft-argmax ConvNets.

---

## 3. Benchmark Results (Measured Single-Core Host + Pi 4B Projection)

**Host:** AMD Ryzen 7 5700X (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`, `cv2.setNumThreads(1/2)`, ORT `intra_op=1/2`, `ORT_ENABLE_ALL`).  
**Pi 4B Projection:** `×3.5` compute-bound scaling from Ryzen 7 5700X to BCM2711 Cortex-A72 @1.8 GHz, matching [`rl/runs/pi_inference_bench.json`](file://rl/runs/pi_inference_bench.json) and Agent 10 §1.

### 3.1 Model 1 — `Student` GRU (`rl/models.py`, real weights `student_L2_cons_adr_s0.pt`)

- **Parameters:** `24,853` total → **`23,683` deployable** (`1,170` train-only `aux_head` + `dyn_head` verified stripped from `.onnx`).
- **Artifact sizes:** `.pt` checkpoint = `1,285.9 kB` | `.npz` = `97.34 kB` | `student_windowed.onnx` = `97.28 kB` | `student_stateful.onnx` = `96.18 kB`.

| Mode | Runtime | Parity vs PyTorch (max abs err) | 50-Step Rollout Parity | Host p50 (ms) | Host p95 (ms) | Host max (ms) | Pi 4B Proj p50 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Windowed `(1, 50, 13)`** | PyTorch CPU (1T) | ref | ref | 1.2569 | 1.4447 | 1.8739 | 4.40 |
| **Windowed `(1, 50, 13)`** | **ORT CPU (1T)** | `2.98e-08` ✓ | — | **0.0567** | 0.0863 | 0.1219 | **0.20** |
| **Windowed `(1, 50, 13)`** | `cv2.dnn` 5.0.0 (1T) | `8.94e-08` ✓ | — | 0.2901 | 0.3140 | 0.4886 | 1.02 |
| **Windowed `(1, 50, 13)`** | NumPy `.npz` (`StudentPolicy.forward`) | `4.15e-08` ✓ | — | 0.8442 | 0.9288 | 1.3602 | 2.95 |
| **Stateful `(1, 1, 13)+h0`** | PyTorch CPU (1T) | ref | ref | 0.1068 | 0.1502 | 0.1855 | 0.37 |
| **Stateful `(1, 1, 13)+h0`** | **ORT CPU (1T)** | — | `2.98e-08` ✓ | **0.0123** | 0.0173 | 0.0558 | **0.04** |
| **Stateful `(1, 1, 13)+h0`** | `cv2.dnn` 5.0.0 (1T) | `2.98e-08` (act) / `1.19e-07` (h1) ✓ | — | 0.0190 | 0.0213 | 0.0343 | 0.07 |
| **Stateful `(1, 1, 13)+h0`** | NumPy `.npz` (`_step_gru + _head`) | — | `4.15e-08` ✓ | **0.0256** | 0.0291 | 0.0658 | **0.09** |

**Takeaway:** For `Student`, ORT CPU is **14.9× faster** than NumPy on the 50-step window (`0.0567 ms` vs `0.8442 ms`), and **2.1× faster** on the stateful single step (`0.0123 ms` vs `0.0256 ms`). However, in `stateful=True` deploy mode (`0.09 ms` Pi proj), `StudentPolicy` NumPy is already 7× smaller than the `swept_clear` shield (`0.63 ms` Pi proj), so **keeping `.npz` for `StudentPolicy` is fine**, while `.onnx` is ready whenever the policy architecture grows.

### 3.2 Model 2 — `BellyLeJEPADetector` (`160×120` Grayscale, 8 Keypoints + Pose + L/R + Confidence)

- **Input:** `(1, 1, 120, 160)` (`H=120, W=160` normalized grayscale ROI).
- **Trunk:** 5 `Conv2d + BatchNorm2d + ReLU` blocks (`1→24(s2)→32(s2)→48(s1)→64(s2)→80(s1)` → `15×20` stride-8 feature map) + `1×1` heatmap conv (`8` channels) + Spatial-Soft-Argmax + global avg-pool + MLP head (`104→96→6`).
- **8 Keypoints:** `[left_wheel, right_wheel, nose_gear, belly_center, left_fin_root, left_fin_tip, right_fin_root, right_fin_tip]`.
- **Outputs:** `keypoints (1, 8, 2)`, `kp_conf (1, 8)`, `pose_xy_sc (1, 4)` (`x, y, sin_yaw, cos_yaw`), `left_right_sign (1, 1)`, `target_confidence (1, 1)`.
- **Parameters & Stripping:** `119,990` total → **`106,486` deployable**; training-only LeJEPA `emb` + `jepa_pred` (`13,504` params) **verified absent** from `.onnx` (`train_only_leaked: []`); all 5 `BatchNorm2d` layers **folded into `Conv`** (`batchnorm_nodes_remaining: []`).
- **ONNX Size:** `432.85 kB` (multi-output, `43` nodes, opset 17) | `433.84 kB` (packed `(1, 30)`, `46` nodes).

#### Parity vs PyTorch FP32 Reference (`max-abs-err`)

| Output Tensor | Range / Units | ORT Multi (`max-abs-err`) | `cv2.dnn` Multi (`max-abs-err`) | Gate Status |
| :--- | :--- | ---: | ---: | :--- |
| `keypoints` | `[0..160, 0..120]` px | `3.05e-05 px` (`1.91e-07` norm) | `3.05e-05 px` (`1.91e-07` norm) | ✓ `< 1e-5` normalized |
| `kp_conf` | `[0, 1]` | `4.66e-10` | `6.98e-10` | ✓ `< 1e-5` |
| `pose_xy_sc` | `x, y, sin, cos` | `2.98e-08` | `1.49e-08` | ✓ `< 1e-5` |
| `left_right_sign` | logit | `7.45e-09` | `0.00e+00` | ✓ `< 1e-5` |
| `target_confidence` | logit | `1.49e-08` | `1.49e-08` | ✓ `< 1e-5` |

#### Latency Comparison (Ryzen 7 5700X Measured → Pi 4B Projected)

| Runtime & Variant | Threads | Host p50 (ms) | Host p95 (ms) | Host max (ms) | Pi 4B Proj p50 (ms) | Share of 33.33 ms (30 Hz) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch CPU FP32 | 1 | 1.5610 | 1.9936 | 2.1902 | 5.46 | 16.4 % |
| **`onnxruntime` CPU FP32 (multi)** | **1** | **0.7591** | **0.8648** | **1.0709** | **2.66** | **8.0 %** |
| **`onnxruntime` CPU FP32 (multi)** | **2** | **0.4307** | **0.5373** | **0.8572** | **1.51** | **4.5 %** |
| `onnxruntime` CPU FP32 (packed) | 1 | 0.7818 | 0.9043 | 1.1069 | 2.74 | 8.2 % |
| `cv2.dnn` 5.0.0 FP32 (multi) | 1 | 1.5742 | 1.7045 | 2.2045 | 5.51 | 16.5 % |
| `cv2.dnn` 5.0.0 FP32 (multi) | 2 | 0.8922 | 1.1172 | 1.5841 | 3.12 | 9.4 % |
| `cv2.dnn` 5.0.0 FP32 (packed) | 1 | 1.6249 | 1.9320 | 2.9993 | 5.69 | 17.1 % |
| `onnxruntime` Dynamic INT8 *(rejected)* | 1 | 3.5171 | 4.2116 | 5.2924 | 12.31 | 36.9 % *(20.19 px drift!)* |

**Key Architectural Finding:**
- **ORT beats `cv2.dnn` by 2.07×** (`0.759 ms` vs `1.574 ms` @1T; `0.431 ms` vs `0.892 ms` @2T) and beats PyTorch CPU by **2.06×**!
- In Agent 10's 4-core Pi 4B budget ([`10_pi_edge_vs_3090_compute_budget.md`](file://docs/council_v2/10_pi_edge_vs_3090_compute_budget.md#L202)), Core 1 (perception) budgeted `5.2 ms` for a `160×120` keypoint CNN using `cv2.dnn`. Using **`onnxruntime` cuts that to `~2.66 ms`**, returning **~2.5 ms of free jitter headroom** per 33.3 ms frame on Core 1.

---

## 4. Production Gotchas & Larger Vision Backbones (YOLOv11n-Pose)

1. **`cv2.dnn` bare `net.forward()` drops outputs 2–5:** Verified directly in `exp_onnx_cv2dnn_bench.py`: calling `net.forward()` without output names on `belly_lejepa_multi.onnx` returned shape `(1, 8, 2)` (`keypoints` only) and silently discarded `kp_conf`, `pose_xy_sc`, `left_right_sign`, and `target_confidence`. Always pass `out_names = net.getUnconnectedOutLayersNames(); outs = dict(zip(out_names, net.forward(out_names), strict=True))` (or export a packed tensor).
2. **Naive `quantize_dynamic` breaks both accuracy and speed on small ConvNets:** `onnxruntime.quantization.quantize_dynamic(..., QInt8)` ballooned the graph from `43` to `93` nodes (`DynamicQuantizeLinear` + `ConvInteger`), shifted `keypoints` by **`20.19 px`** (uncalibrated softmax-logit quantization), and ran **4.6× slower** (`3.517 ms` vs `0.759 ms` FP32). Furthermore, Pi 4B's Cortex-A72 is ARMv8.0-A (no `SDOT`/`VDOT` dot-product instruction). **Use FP32 NEON/XNNPACK** for `~100k`-param models.
3. **Exporter flags (`dynamo=False, opset_version=17`):** PyTorch `2.14.1` warns that the dynamo exporter is now default, but `rl/.venv` does not include `onnxscript`. `dynamo=False, opset_version=17` exports cleanly, folds `BatchNorm2d`, and produces clean opset-17 graphs (`Conv`, `Relu`, `Softmax`, `MatMul`, `ReduceMax`, `ReduceMean`, `Gemm`).
4. **If scaling up to YOLOv11n-Pose (~2.6M params):** Do **not** rely on `cv2.dnn` for modern YOLO-Pose graphs (`C2f`/`C3k2`, `SPPF`, `Split`, `SiLU`, multi-scale concat heads frequently hit unsupported-op or reshape issues in `cv2.dnn`). Export from PyTorch/Ultralytics with `opset=17`, `simplify=True`, and run via **`onnxruntime` ARM64** (or **NCNN INT8** with calibration if 15–25 ms async inference is needed, per Agent 06).

---

## 5. Concrete Integration Pattern (3090 Train/Export → Pi Inference)

### 5.1 RTX 3090 Training & Gated ONNX Export (`rl/`)

Mirror the `rl/models.py` + `rl/export.py` contract so training-only LeJEPA heads never ship, and add the ONNX Runtime parity gate:

```python
# 1) During 3090 training: call model.forward_train(x_views) -> computes SIGReg + jepa_pred MSE + pose/kp losses.
# 2) At export time: call torch.onnx.export on `model` (uses `forward()`, excluding `emb` and `jepa_pred`).

def export_and_gate_vision_onnx(model, onnx_path: str, opset: int = 17) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    model.eval().cpu()
    dummy = torch.randn(1, 1, 120, 160, dtype=torch.float32)
    out_names = ["keypoints", "kp_conf", "pose_xy_sc", "left_right_sign", "target_confidence"]

    with torch.no_grad():
        refs = [t.numpy() for t in model(dummy)]
        torch.onnx.export(
            model,
            (dummy,),
            onnx_path,
            input_names=["image"],
            output_names=out_names,
            dynamo=False,
            opset_version=opset,
            do_constant_folding=True,
        )

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    leaked = [i.name for i in m.graph.initializer if "jepa_pred" in i.name or i.name.startswith("emb.")]
    assert not leaked, f"Train-only LeJEPA weights leaked: {leaked}"

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    got = sess.run(out_names, {"image": dummy.numpy()})

    for name, ref, val in zip(out_names, refs, got, strict=True):
        err = float(np.abs(ref - val).max())
        norm_err = err / 160.0 if name == "keypoints" else err
        assert norm_err < 1e-5, f"ONNX parity gate failed on {name}: max_abs={err:.2e}"
```

### 5.2 Pi Edge Loader (`src/ugv/` Perception Process)

Prefer `onnxruntime` (2× faster on CPU), with an explicit `cv2.dnn` fallback that maps multi-output tensors by name:

```python
import cv2
import numpy as np


class OnnxBellyVision:
    """Torch-free Pi vision runner for the 160x120 8-keypoint LeJEPA belly detector."""

    OUT_NAMES = ("keypoints", "kp_conf", "pose_xy_sc", "left_right_sign", "target_confidence")

    def __init__(self, onnx_path: str, threads: int = 1, prefer_ort: bool = True):
        self.backend = "cv2"
        self.sess = None
        self.net = None
        if prefer_ort:
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sess = ort.InferenceSession(
                    str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
                )
                self.in_name = self.sess.get_inputs()[0].name
                self.backend = "ort"
                return
            except ImportError:
                pass

        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.cv2_out_names = self.net.getUnconnectedOutLayersNames()

    def infer(self, gray_160x120: np.ndarray) -> dict[str, np.ndarray]:
        # Match training normalization: (img/255 - 0.5) / 0.25 -> float32 (1, 1, 120, 160)
        x = ((gray_160x120.astype(np.float32) * (1.0 / 255.0) - 0.5) * 4.0)[None, None, :, :]
        if self.backend == "ort":
            vals = self.sess.run(list(self.OUT_NAMES), {self.in_name: x})
            return dict(zip(self.OUT_NAMES, vals, strict=True))

        self.net.setInput(x)
        vals = self.net.forward(self.cv2_out_names)  # MUST pass names; bare forward() drops outputs 2-5
        by_name = dict(zip(self.cv2_out_names, vals, strict=True))
        return {k: by_name[k] for k in self.OUT_NAMES}
```

---

## 6. Final Recommendation

1. **Train everything in PyTorch on the RTX 3090** (`rl/`): keep `Student` RL/distillation + LeJEPA / Spatial-Soft-Argmax / YOLO-Pose training completely in PyTorch.
2. **Keep `StudentPolicy` on `.npz`** (`src/ugv/policy.py`, `stateful=True`, `0.09 ms` Pi proj) — zero dependencies, already parity-gated in `rl/export.py`.
3. **Adopt `.onnx` (`opset=17, dynamo=False`) + `onnxruntime` ARM64 (`CPUExecutionProvider`) as the standard vision deployment bridge** (`2.66 ms` Pi proj @1T / `1.51 ms` @2T for the 8-KP LeJEPA detector), keeping `cv2.dnn.readNetFromONNX(..., out_names)` as a built-in zero-dependency fallback (`5.51 ms` Pi proj @1T).
4. **Add the ONNX parity assert (`norm_err < 1e-5`) to `rl/export.py --onnx`** (and any new vision exporter), and **verify real Pi 4B `BCM2711` `p50/p95` latencies** once an `onnxruntime` ARM64 wheel is tested on `adv034`.
