# Agent 10 — Pi 4B / RTX 3090 Compute Split & Real-Time Edge Budget

**Author:** Agent 10 (Raspberry Pi 4B / 3090 Compute Split & Real-Time Edge Systems Architect)
**Scope:** exact ms budget for the 4 Cortex-A72 cores of `adv034` (Pi 4B, `BCM2711`, 2 GB) under the
proposed split — Core 0/1 capture + ROI ArUco + KLT + keypoint CNN, Core 2 = 50 Hz EKF + `StudentPolicy`
+ swept-clear shield, Core 3 = FastAPI/WebSocket + MJPEG — with 3090-side training and a Pi-side
inference-only contract.
**Inspected:** [`src/ugv/aruco.py`](../../src/ugv/aruco.py), [`src/ugv/policy.py`](../../src/ugv/policy.py),
[`src/ugv/parking.py`](../../src/ugv/parking.py), [`src/ugv/staged_baseline.py`](../../src/ugv/staged_baseline.py),
[`src/ugv/perf.py`](../../src/ugv/perf.py), [`rl/runs/pi_inference_bench.json`](../../rl/runs/pi_inference_bench.json),
[`rl/export.py`](../../rl/export.py), [`rl/models.py`](../../rl/models.py), [`HANDOFF.md`](../../HANDOFF.md) §99,
[`docs/AGENT_BOARD.md`](../AGENT_BOARD.md), [`docs/HARDWARE.md`](../HARDWARE.md), [`docs/COMMANDS.md`](../COMMANDS.md).
**Verdict:** **The split is feasible, but not in the shape proposed.** Measured on this host, a 107k-param
keypoint CNN costs **1.49 ms** at `160x120` and **6.36 ms** at `384x256` on one core, and **0.79 / 3.27 ms**
on two; projected onto Cortex-A72 that is **5.2 ms** vs **22.3 ms** of *work per frame* (a second core
divides latency, not work). A `320x240`/`384x256` keypoint net **cannot share one core with ROI ArUco at
30 Hz** (18.0–22.3 ms for the net + 14.1 ms ROI ArUco + 2.0 ms glue = 34.1–38.4 ms > 33.33 ms). Split over
both perception cores it fits at 81 % — but only by halving the full-frame acquisition rate and leaving
**4.2 ms** of jitter margin, against **10.5 ms** for `160x120` (§4.2). Run the keypoint net at **`160x120`
(5.2 ms, 66 % of one core)**. Full-frame ArUco at 30 Hz does not fit on any single core either (52.9 ms
projected > 33.3 ms): decode full-frame only for *acquisition* at 3–5 Hz, then stay in ROI.

---

## Executive summary

1. **Measured, not assumed.** Every number below except the ones marked `[Pi]` was measured today on one
   pinned Ryzen 7 5700X core (`taskset -c 0`, `OMP_NUM_THREADS=1`) with the repo's own code paths
   (`ArucoFollower.detector()`, `StudentPolicy.load(...)`, `staged_baseline.EKF`). Raw JSON is in
   `/tmp/opencode/edge/*.json`; the scripts are `/tmp/opencode/edge/bench_{vision,net,kp,core2}.py`.
2. **ArUco does not scale past 2 cores.** `detectMarkers` at 1280×720: **15.10 ms** on 1 core → **10.73 ms**
   on 2 cores (1.41×) → no further gain from `cv2.setNumThreads(2)` (10.56 ms). The repo comment at
   [`aruco.py:199-201`](../../src/ugv/aruco.py#L199) ("two workers reduced median and p95 vs four") is
   consistent with this; there is nothing to win from a third or fourth worker.
3. **Corner refinement is free; the threshold sweep is the cost.** 1280×720 `detectMarkers` with
   `CORNER_REFINE_SUBPIX` **15.10 ms** vs `CORNER_REFINE_NONE` **15.59 ms** — refining corners costs
   *nothing* on this scene, so `parameters.cornerRefinementMethod` is not a knob worth spending. Resolution
   is the only knob that matters: 1280×720 **15.10 ms** → 640×360 ROI **4.04 ms** (3.7× for 4× fewer px).
4. **ROI decode is the only way the 30 Hz loop survives.** The production path
   (`ArucoFollower.detect_gray`, with the fallback/low-light passes) costs **75.2 ms** full-frame and
   **22.7 ms** on a 640×360 deck ROI on this host — **3.31×**. Applied to the real on-Pi measurement
   (`fetch+detect 45.5 ms mean / 44.3 p50 / 52.4 p95`, 473 frames, [`HANDOFF.md`](../../HANDOFF.md) §99)
   that is **≈13.4–15.1 ms on the Pi**, i.e. a ROI decode fits one core at 30 Hz but a full-frame one does not.
5. **The keypoint CNN as briefed (`320x240`/`384x256`) does not fit on one core, but does fit across two.**
   `cv2.dnn` fp32 on ONNX: 107k params → `120x160` **1.486 ms** (1 core) / **0.790 ms** (2 cores),
   `240x320` **5.139 / 2.677 ms**, `256x384` **6.356 / 3.265 ms** (28.0 / 113.9 / 142.2 MMAC). Projected ×3.5
   onto A72: `120x160` **5.2 ms**, `240x320` **18.0 ms**, `256x384` **22.3 ms** of *work*, per frame. On a
   single core, `320x240` + ROI ArUco + glue = **34.1 ms > 33.33 ms** — it does not fit. Split across both
   perception cores with `cv2.dnn` 2-thread convs (1.9× measured) it fits at **81 % of the pair**, but only
   if the full-frame acquisition rate is halved, and it leaves **4.2 ms** of jitter margin on the decode core
   versus 10.5 ms for `160x120` (§4.2). `384x256` at 30 Hz does not fit even then (95 %). Hence the
   recommendation: **`160x120`; jitter margin is worth more than keypoint resolution here.**
6. **INT8 buys almost nothing on this SoC and that should change the plan.** Cortex-A72 is Armv8.0-A: no
   `SDOT`/`VDOT` (Armv8.2 dot-product), no native FP16 arithmetic. Weight-only int8 measured **0.854 ms**
   vs **1.117 ms** fp32 for the same net in torch (1.31×), because the arithmetic path is unchanged. The
   0.43 MB of weights already sit in L2, so there is no bandwidth win either. **Plan for fp32 NEON, not INT8.**
7. **`StudentPolicy` is a rounding error, and that is the good news.** With the deployed weights
   (`student_L0_lat_s1.npz`, sha 6137a3f8b1d993ea, 23 683 params, 98.2 kB): stateful step + swept-clear
   shield **0.090 ms p50 / 0.096 p95 / 0.995 max** (5 000 iters) → **0.32 ms projected on one A72 core**,
   against the repo's recorded `0.47–0.54 ms` from the same 3.5–4× assumption. Windowed-50 is **0.891 ms**
   (10× worse) — the deploy path must stay `stateful: True`, exactly as [`policy.py:196-199`](../../src/ugv/policy.py#L196) already does.
8. **The 50 Hz core is ~14 % loaded and the whole stack is ~42 % — which matches the car.** EKF over 60
   particles predict+update **0.399 ms**, `AdaptiveYaw.observe+turn` **0.085 ms**, in-place PWM/I2C ~0.6 ms
   → **2.8 of 20.0 ms per 50 Hz tick**. Summed per 30 Hz frame the split needs **56.5 of 133.3 core-ms
   (42 %)**; [`HANDOFF.md`](../../HANDOFF.md) §9 measured the live dashboard at load **1.43–1.56 of 4 cores
   (36–39 %) streaming 30 fps**. The projection method is corroborated by an independent live measurement.
9. **Hard architectural blocker: one Python process = one GIL.** [`web.py`](../../src/ugv/web.py) runs the
   ArUco follower as an asyncio task in the uvicorn process. A 14 ms ROI decode inside that task stalls the
   event loop and the 50 Hz control tick. "Core 2 for control" is only real if the control loop is a
   **separate process** (`multiprocessing` / second `systemd` unit) with `chrt -f 40` and, ideally,
   `isolcpus=3` minus the web worker; otherwise the cores are a fiction and the budget is one serial
   timeline of ~20 ms.
10. **The MJPEG stream, not the CPU, is the first thing that breaks.** 1280×720 q80 is **385 kB/frame**
    (**11.6 MB/s ≈ 92 Mbit/s at 30 Hz**, 46 Mbit/s at 15 Hz; q95 is 659 kB/frame ≈ **158 Mbit/s** at 30 Hz).
    The arithmetic for the NN is comfortable; the radio is not. Keep 1280×720 on the Pi and stream **640×360 @15 Hz** to the dashboard.

---

## 1. Method, and what a `[Pi]` number means

| Item | Value |
| :--- | :--- |
| Host | AMD Ryzen 7 5700X (Zen 3), 1 core pinned via `taskset -c 0`, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` |
| Runtimes | root env `cv2 5.0.0` + `numpy 2.5.2`; `rl/.venv` `torch 2.14.1+cu126` (CPU only, `set_num_threads`) |
| Pi projection factor | **×3.5** compute-bound, **×3.0** bandwidth-bound — the factor already used and recorded in [`rl/runs/pi_inference_bench.json`](../../rl/runs/pi_inference_bench.json) ("Cortex-A72 1.8 GHz ≈ 3.5–4× slower per core than Ryzen 7 5700X") |
| Roofline cross-check | A72 @1.5 GHz: 2 NEON pipes × 4 FP32 lanes × FMA = **24 GFLOP/s** peak; at 40 % efficiency the 56.1 MFLOP `160x120` net = **5.8 ms** — matches ×3.5 (5.2 ms). At 1.8 GHz, 4.9 ms. |
| On-Pi anchors (`[Pi]`, from the repo) | `detect_jpeg` fetch+detect **45.5 mean / 44.3 p50 / 52.4 p95 ms**, 473 frames, no actuation ([`HANDOFF.md`](../../HANDOFF.md) §99); dashboard load **1.43–1.56** of 4 cores at 30 fps streaming + IMU; RSS `uvicorn 163 MB` + `rpicam-vid 67 MB`, `MemTotal 1.9 GB`, `available 1.3 GB`, `throttled=0x0`, `60.3 °C` |
| Pi contact | **None.** [`HANDOFF.md`](../../HANDOFF.md) records `adv034` paused and charging, so per [`docs/SAFETY.md`](../SAFETY.md) nothing was ssh'd, deployed or moved. Every `[Pi]` cell is either a citation of a recorded measurement or explicitly labelled *projected*. |

**Nothing in this document modifies the repo.** Scripts live under `/tmp/opencode/edge/`; the only writes
are this file.

Benchmark scene (deterministic, `seed=3`): 1280×720 mono, illumination gradient + σ=6 sensor noise, three
`DICT_4X4_50` tags at **140 / 55 / 22 px** with 6 px quiet zones, all inside the deck ROI
(`x∈[480,1120)`, `y∈[300,660)`) so full-frame and ROI runs see the *same* markers. The production path on
this scene is deliberately harsher than the live one (many rejected quads → the 48-crop low-light loop), so
its 75.2 ms is a **worst case**, not the live 44 ms.

---

## 2. Measured latency (`p50` / `p95` ms, one pinned core unless stated)

### 2.1 ArUco / vision primitives — [`aruco.py:389-460`](../../src/ugv/aruco.py#L389)

| Operation | iters | p50 | p95 | max | Note |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `detectMarkers` **1280×720**, `CORNER_REFINE_SUBPIX` (production params) | 200 | **15.104** | 17.386 | 26.259 | ids found: 3/3 |
| `detectMarkers` 1280×720, `CORNER_REFINE_NONE` | 200 | 15.589 | 16.966 | 20.359 | refinement is free — the sweep dominates |
| `detectMarkers` **640×360 deck ROI** | 400 | **4.037** | 4.187 | 4.504 | 3.74× faster than full frame |
| `detectMarkers` 640×360 whole-frame resize | 400 | 1.568 | 2.012 | 2.567 | no fallback-safe (small tags lost) |
| `detectMarkers` 1280×720 on **2 cores** (`OMP=2`) | 200 | 10.733 | 11.707 | 25.190 | **1.41×**, saturates here |
| `detectMarkers` 640×360 ROI on 2 cores | 400 | 2.757 | 3.011 | 3.205 | 1.46× |
| `detectMarkers` 1280×720, 2 cores **+** `setNumThreads(2)` | 200 | 10.561 | 11.228 | 24.902 | **no gain over `OMP=2`** |
| `ArucoFollower.detect_gray` 1280×720 (full fallback path) | 30 | **75.23** | 86.531 | 90.362 | synthetic worst case |
| `ArucoFollower.detect_gray` 640×360 ROI | 30 | **22.685** | 23.912 | 24.392 | **3.31×** |
| `cv2.imdecode` JPEG→gray 1280×720 q80 (385.2 kB) | 200 | 3.886 | 4.290 | 4.485 | MJPEG capture cost |
| `cv2.imdecode` JPEG→gray 1280×720 q95 (658.7 kB) | 200 | 4.891 | 5.325 | 5.730 | quality drives the radio, not the CPU much |
| `calcOpticalFlowPyrLK` 12 pts, win 35, 4 levels | 200 | 1.226 | 1.523 | 1.589 | KLT as used at [`aruco.py:1052`](../../src/ugv/aruco.py#L1052) |
| `calcOpticalFlowPyrLK` 48 pts | 200 | 2.337 | 2.661 | 2.978 | |

### 2.2 Keypoint CNN — 4 strided convs (5×5 s2, 3×3 s2, 3×3 s2, 3×3 s2) + spatial soft-argmax + 2-layer head

Exported to ONNX (`opset 17`) and run through `cv2.dnn` **and** torch CPU; `cv2.dnn` is 33 % slower than
torch's CPU kernels on the same graph (1.486 vs 1.117 ms), which is the honest number to plan with.

| Variant | params | conv MMAC | input | dnn fp32 1 core p50 | dnn fp32 **2 cores** p50 | torch fp32 1 core p50 |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| `k107_120x160` | 106 886 | 28.04 | 120×160 | **1.486** | **0.790** (1.9×) | 1.117 |
| `k130_120x160` | 124 182 | 33.26 | 120×160 | **1.649** | 0.879 (1.9×) | — |
| `k107_240x320` | 106 886 | 113.93 | 240×320 | **5.139** | **2.677** (1.9×) | — |
| `k107_256x384` | 106 886 | 142.15 | 256×384 | **6.356** | **3.265** (1.95×) | — |
| weight-only int8 (torch, linear only) | 106 886 | — | 120×160 | — | — | **0.854** (1.31× vs fp32) |

ONNX artefact: **0.433 MB** (fp32). Interpolating the two bracketing variants: the "≈120 k params" design
target costs **≈1.55 ms** on this host → **≈5.4 ms** on one A72 core.

Two things follow, and the second is the one that matters for the budget. Unlike ArUco (1.41×), the conv
net **does** scale on a second core (1.9×, `cv2.dnn` parallelises output tiles). But parallel speed-up
divides *latency*, not *work*: 5.14 ms on 1 core and 2.68 ms × 2 cores on 2 cores are both **≈5.2 core-ms
per frame**. The 4-core budget in §4 is therefore counted in **core-ms**, and threading only decides how
much of the 33.33 ms period the work occupies.

### 2.3 Core-2 primitives — [`policy.py`](../../src/ugv/policy.py), [`staged_baseline.py:312-441`](../../src/ugv/staged_baseline.py#L312)

| Operation | iters | p50 | p95 | max |
| :--- | ---: | ---: | ---: | ---: |
| `StudentPolicy.step` **stateful** + swept-clear shield (deploy path) | 5 000 | **0.090** | 0.096 | 0.995 |
| GRU-64 step + head only (no shield) | 5 000 | 0.024 | 0.025 | 0.116 |
| `StudentPolicy.step` **windowed-50** + shield | 2 000 | 0.891 | 0.907 | 1.860 |
| `forward()` windowed-50 only | 2 000 | 0.802 | 0.959 | 5.721 |
| `swept_clear` alone (46 interp. steps, 3 capsules) | 20 000 | 0.179 | 0.202 | 0.444 |
| `EKF.predict+update`, n=1 | 3 000 | 0.227 | 0.238 | 0.348 |
| `EKF.predict+update`, **n=60** (full posterior) | 3 000 | **0.399** | 0.484 | 2.031 |
| `EKF.predict` only, n=60 | 3 000 | 0.161 | 0.204 | 1.834 |
| `AdaptiveYaw.observe` + `turn`, n=60 | 3 000 | 0.053 + 0.023 | 0.067 | — |

Repo cross-check: [`rl/runs/pi_inference_bench.json`](../../rl/runs/pi_inference_bench.json) recorded
`stateful_step_plus_shield p50 = 0.136 ms` (10 000 iters). Today's 0.090 ms on the same weights is the
same order; use the **repo's** 0.47–0.54 ms Pi projection for planning because it is the recorded,
gate-relevant figure, and treat mine as the reproducibility check.

---

## 3. Pi 4B projection (`[Pi-proj]`, ×3.5 compute / ×3.0 memory, A72 @1.8 GHz)

| Stage | host p50 | × | Pi p50 | Owner |
| :--- | ---: | ---: | ---: | :--- |
| `imdecode` 1280×720 q80 | 3.886 | 3.0 | **11.7** | Core 0 |
| `imdecode` 1280×720 q95 | 4.891 | 3.0 | 14.7 | Core 0 |
| `detectMarkers` 1280×720 | 15.104 | 3.5 | **52.9** | — (does not fit 33.3 ms) |
| `detectMarkers` 640×360 ROI | 4.037 | 3.5 | **14.1** | Core 1 |
| same, anchored on the live `[Pi]` 44.3 ms × (4.04/15.10) | — | — | **11.8** | Core 1 |
| `detect_gray` full path (worst-case scene) | 75.23 | 3.5 | 263 (not comparable) | — |
| `detect_gray` full path, live `[Pi]` measured | — | — | **44.3 (p50) / 52.4 (p95)** | acquisition only |
| KLT 12 pts | 1.226 | 3.5 | **4.3** | Core 0 |
| Keypoint CNN `120x160` 107k | 1.486 | 3.5 | **5.2** | Core 1 |
| Keypoint CNN `120x160` 124k | 1.649 | 3.5 | 5.8 | Core 1 |
| Keypoint CNN `240x320` 107k | 5.139 | 3.5 | **18.0** | Core 1, or 9.0 split over 0+1 |
| Keypoint CNN `256x384` 107k | 6.356 | 3.5 | **22.3** | Core 1, or 11.2 split over 0+1 |
| `StudentPolicy` stateful + shield | 0.090 | 3.5 | 0.32 (repo: **0.47–0.54**) | Core 2 |
| `StudentPolicy` windowed-50 + shield | 0.891 | 3.5 | 3.1 | Core 2 (reject) |
| `EKF.predict+update` n=60 | 0.399 | 3.5 | **1.40** | Core 2 |
| `AdaptiveYaw` n=60 | 0.085 | 3.5 | 0.30 | Core 2 |
| `swept_clear` alone | 0.179 | 3.5 | 0.63 | (included in the policy row) |

The `detect_gray` worst-case row is **not** to be scaled: it is dominated by the number of rejected quads,
which is scene-dependent (my synthetic scene produces far more than the lab floor). Use the live `[Pi]`
number for that path and the *ratio* (3.31×) to derive the ROI variant.

---

## 4. The exact 4-core budget

Period: camera **30 Hz → 33.33 ms/frame**; control **50 Hz → 20.0 ms/tick**. Total demand per frame across
4 cores = **133.3 core-ms**. Two workable arrangements; **A is recommended**.

### 4.1 Arrangement A — recommended (`160x120` keypoint net, 30 Hz)

| Core | Work | ms / period | Utilisation | Headroom |
| :--- | :--- | ---: | ---: | ---: |
| **0 — ingest** | MJPEG decode 11.7 + KLT 12 pts 4.3 + full-frame ArUco *acquisition* at 3 Hz (52.9×3/30 = 5.3) + crop/ROI bookkeeping 1.5 | **22.8 / 33.33** | **68 %** | 10.5 ms |
| **1 — perception** | ROI ArUco 640×360 14.1 + keypoint CNN `160x120` 5.2 + quad filter / `solvePnP` / `from_points` glue 2.0 + Kalman on keypoints 0.7 | **22.0 / 33.33** | **66 %** | 11.3 ms |
| **2 — control** | EKF n=60 1.4 + `StudentPolicy` stateful+shield 0.5 + `AdaptiveYaw` 0.3 + I2C/PWM + brake command 0.6 | **2.8 / 20.0** | **14 %** | 17.2 ms |
| **3 — I/O** | uvicorn routing + WS broadcast @30 Hz ~3.0 + MJPEG restream to 1 client ~4.0 (640×360) | **7.0 / 33.33** | **21 %** | 26.3 ms |
| **Total** (Core 2 at 2.8 ms × 1.67 ticks per 33.33 ms frame = 4.7) | | **56.5 / 133.3 core-ms** | **42 %** | **58 %** |

The 42 % aggregate is the same number the car already shows: `load 1.43–1.56` of 4 cores =
36–39 % ([`HANDOFF.md`](../../HANDOFF.md) §9). Two independent routes to the same answer, which is the
strongest evidence in this document.

### 4.2 Arrangement B — only if a `320x240` keypoint net is mandatory

`cv2.dnn` convs at `setNumThreads(2)` scaled **1.9×** on two real cores, so the net can be split across the
perception pair. Work is conserved (≈18.0 core-ms), latency halves — but the work still has to be paid for
from the same two cores that already run decode + ArUco.

| Core | Work | ms / 33.33 | Utilisation |
| :--- | :--- | ---: | ---: |
| 0 — ingest | decode 11.7 + KLT 4.3 + acquisition **at 1.5 Hz** 2.7 + glue 1.5 + CNN share 9.0 | **29.2** | **88 %** |
| 1 — perception | ROI ArUco 14.1 + glue 1.5 + CNN share 9.0 | **24.6** | 74 % |
| 2 — control | as A | 2.8 / 20.0 | 14 % |
| 3 — I/O | as A | 7.0 | 21 % |

Wall clock per frame: decode ends 11.7, ROI ArUco ends 14.1, then the CNN owns both cores for 9.0 ms →
**23.1 ms (69 % of the period)**, 53.8 of 66.7 perception core-ms (**81 %**). The binding constraint is not
the wall clock but **Core 0 at 88 %, with 4.2 ms of jitter margin**.

And it only works with the acquisition rate halved. Keep acquisition at 3 Hz and Core 0 becomes
22.8 + 9.0 = **31.8 ms (95 %, 1.5 ms of margin)** — one JPEG spike or GC pause and the frame is late.
`256x384` at 30 Hz does not fit at all: Core 0 = 31.4 ms (**94 %**) even with acquisition at 1.5 Hz.

So the trade is explicit: `320x240` buys 4× keypoint resolution and costs the full-frame acquisition rate
*and* most of the jitter margin. `160x120` (Arrangement A) keeps **68 % / 66 % with 10.5 / 11.3 ms** of
margin and lets acquisition run at 3 Hz. Recommendation stands at `160x120`; make `320x240` an A/B
experiment that must show a parking-success delta before it is allowed to spend that margin.

**On one core it does not fit at all:** `320x240` **at 30 Hz** (18.0 ms) on top of ROI ArUco (14.1) + glue
(2.0) = **34.1 ms > 33.33 ms**, before jitter; `384x256` is 38.4 ms. Pushing ArUco onto the decode core to
make room gives 11.7 + 14.1 + 4.3 = 30.1 ms = **90 %** — no margin at all.

### 4.3 Degraded ladder (what to drop, in order)

| Frame budget | Drop | Cost recovered |
| :--- | :--- | ---: |
| 25 Hz (40 ms) | nothing | — |
| 20 Hz (50 ms) | full-frame acquisition 3 Hz → 1.5 Hz | 2.6 ms |
| 15 Hz (66.7 ms) | keypoint CNN every other frame (KLT carries the gap) | 2.6 ms |
| tag lost > 1.0 s | CNN off, full-frame acquisition every frame (44–53 ms, still < 66.7) | — |
| thermal `throttled != 0x0` or `temp > 80 °C` | CNN off, ArUco ROI only, control untouched | 5.2 ms |

The control core is never on the degradation list. ArUco loss already brakes through
`"brake: stale/invalid observation"` ([`policy.py:143-145`](../../src/ugv/policy.py#L143)) and the shield
already covers 200 ms of stale pose (`STOP_HORIZON = 0.35`, verified at yaw 0/30/60 in
[`tests/test_policy_and_shield.py`](../../tests/test_policy_and_shield.py)).

---

## 5. 3090-train / Pi-deploy split contract

| Artifact | 3090 side (training) | Pi side (deploy) | Gate |
| :--- | :--- | :--- | :--- |
| `Student` GRU-64 + 2×64 head ([`rl/models.py:54-75`](../../rl/models.py#L54)) | PPO/distillation, GPU, `aux_head`/`dyn_head` are **training-only** | `gru.*`+`head.*` only → `.npz` **98.2 kB**, `StudentPolicy` in pure NumPy | export parity `1e-5` ([`rl/export.py:47`](../../rl/export.py#L47)); recorded 2.97e-7 windowed / 4.82e-7 stateful |
| Keypoint CNN (107–124 k params) | 3090, fp32/AMP, spatial-soft-argmax supervision from ArUco/gear keypoints | ONNX **0.433 MB**, `cv2.dnn` fp32, **no torch** | ONNX fp32 vs torch fp32 ≤ 1e-4 on 200 random frames; Pi parity vs `cv2.dnn` on a 32-frame fixture |
| Perception geometry | — | `aruco.py` `from_points`/`solvePnP` unchanged | `detect_gray` ROI ids = full-frame ids on the field set |
| Dynamics/SysID fits | 3090 offline refit ([`tools/calibration`](../../tools/calibration)) | `calibration/` JSON only, no model | fit residual < 8 % (existing test) |

Rules that this split depends on:
* **No torch on the Pi.** [`src/ugv`](../../src/ugv) must import without torch (repo rule); `cv2.dnn` and
  NumPy are already loaded by [`aruco.py`](../../src/ugv/aruco.py), so the CNN adds **≈0 import cost** and
  ≈0.43 MB of read-only weights (no new process, no ONNX Runtime install, no NCNN cross-build).
* **`stateful=True` only.** Windowed-50 costs 0.891 ms vs 0.090 ms host — 10× — and 3.1 ms projected on the
  Pi, which is 15 % of a 20 ms tick for the same output. [`rl/export.py`](../../rl/export.py) must keep
  writing `meta_stateful`.
* **The 3090 never enters the loop.** Training loss is not success; `rl/evaluate.py` 1000 episodes seed 11
  remains the only gate, and a sim success is not a hardware success.
* **A deploy must preserve** `~/ugv/runs/student.npz` and the `vision-camera.conf` drop-in.

---

## 6. Blockers and corrections to the proposed design

1. **One process, one GIL — the cores are not free.** [`web.py`](../../src/ugv/web.py) runs capture,
   `ArucoFollower`, the parking loop, `PerfMonitor` and the WS broadcast in one uvicorn process. cv2 releases
   the GIL inside a call, so CPU time does overlap, but *the Python glue serialises and a blocking call
   stalls the event loop*. For the 50 Hz core to be real: control loop in its own process, `chrt -f 40`
   priority, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the unit, and pin with `taskset -c 2`.
   Without that, the honest budget is a single serial timeline where ROI ArUco (14 ms) + CNN (5 ms) + control
   (2.8 ms) = 21.8 ms must all fit inside one 20 ms tick — i.e. it does not, and the 50 Hz loop drops to fit.
2. **INT8/FP16 is a dead end on A72.** No dot-product extension, no FP16 arithmetic → weight-only int8 is
   1.31× measured, and fp16 would be *slower* than fp32 (emulation). Budget fp32. If more headroom is ever
   needed, the answer is a smaller input or a cheaper net, not a different dtype.
3. **`320x240`/`384x256` is a two-core decision, not a one-core one** (§4.2). The brief's "15–30 Hz" holds
   for `320x240` at 30 Hz *only* with 2-thread convs across both perception cores, acquisition halved, and
   4.2 ms of margin left; `384x256` does not fit at all at 30 Hz (Core 0 at 94–95 %). On a single core
   `320x240` is 34.1 ms against a 33.33 ms period.
4. **Intra-op threading: worth it for the conv, not for ArUco.** The CNN scaled **1.9×** on two real cores;
   `detectMarkers` scaled only **1.41×** and gained nothing further from an explicit `setNumThreads(2)` on
   top of `OMP=2` ([`aruco.py:199`](../../src/ugv/aruco.py#L199) sets 2 — correct, and there is no point
   going to 4). Parallelise the *pipeline*, and give the conv the second core only when it needs it.
5. **MJPEG is the wire bottleneck, not the CPU** (summary item 10): 11.6 MB/s ≈ 92 Mbit/s at 30 Hz q80 on a
   mini-router. Stream `640x360@15` to the dashboard and keep the 1280×720 decode on the Pi.
6. **`PerfMonitor.BUDGET_MS = 33.33`** ([`perf.py:27`](../../src/ugv/perf.py#L27)) is the only in-repo budget
   signal and it tracks the *camera cadence*, not the control tick. If Core 2 is to be defended, the control
   loop needs its own 20 ms budget counter (`control_ms`, `tick_overruns`), otherwise a regression is invisible.
7. **No `isolcpus` today.** The Pi runs the desktop stack (`labwc`/`wf-panel`/`pcmanfm`/`wayvnc`, ~250 MB,
   [`HANDOFF.md`](../../HANDOFF.md) §9) on the same 4 cores. Pinning a 50 Hz hard-ish loop only means something
   after that stack is stopped in the field image, or the loop is put on the least-loaded core by measurement.

---

## 7. What must be measured on the Pi before any of this is claimed as fact

The car is charging; none of the `[Pi-proj]` cells have been confirmed. In order, once
[`HANDOFF.md`](../../HANDOFF.md) says the car is back:

1. Cheapest sanity check first: copy `bench_vision.py` (`/tmp/opencode/edge/`, nothing in the repo changes),
   then `taskset -c 1 <venv>/python bench_vision.py` on `adv034` → confirm `detectMarkers` 640×360 ROI lands
   in **12–16 ms** (projection 14.1, anchored 11.8). If it is > 20 ms, the ArUco share of Core 1 is wrong and
   Arrangement A collapses to 20 Hz.
2. `cv2.dnn` ONNX on the Pi → confirm the `160x120` keypoint net is **4.5–6.5 ms** (projection 5.2,
   roofline 4.9–5.8) and that `setNumThreads(2)` gives **≥1.6×** (host: 1.9×). This is the single number the
   whole CNN proposal rests on, and the 1.6× scaling is what Arrangement B depends on.
3. Full split under load: 30 fps stream + CNN + ROI ArUco + 50 Hz control, 60 s, with `PerfMonitor` stages;
   record `plan_ms`, `aruco_processing_ms`, `control tick` p50/p95/p99 and `throttled`/`temp`.
4. Loop lag and action-to-duty parity (repo rule: a policy that oscillates is usually a loop problem).
   Measure the tick-to-PWM delay before blaming the policy.
5. End-frame confirmation for any parking claim — an estimate-based "arrived" is not a parked car.

Until 1–3 are recorded in [`docs/AGENT_BOARD.md`](../AGENT_BOARD.md) with a run id, treat §3–§5 as a
design budget, not a measurement.

---

## 8. One-line answers

* **Core 0/1 (33.33 ms each):** decode **11.7** + KLT **4.3** + acquisition **5.3** + glue **1.5** =
  **22.8 ms (68 %)** / ROI ArUco **14.1** + CNN `160x120` **5.2** + PnP/glue **2.7** = **22.0 ms (66 %)**.
* **Core 2 (20.0 ms per 50 Hz tick):** EKF **1.4** + policy+shield **0.5** + yaw ID **0.3** + PWM/I2C
  **0.6** = **2.8 ms (14 %)** — the cheapest core in the system, and the only one that must never be starved.
* **Core 3 (33.33 ms):** uvicorn **3.0** + MJPEG **4.0** = **7.0 ms (21 %)**; the radio, not the CPU, is the
  limit at ≈92 Mbit/s for 1280×720@30 (46 Mbit/s at 15 Hz).
* **3090 side:** trains GRU-64 (98.2 kB) and the 0.433 MB keypoint ONNX; the Pi runs NumPy + `cv2.dnn` fp32
  only, with `1e-5` export parity as the hand-off gate.

## References

* [`src/ugv/aruco.py:389-460`](../../src/ugv/aruco.py#L389) detector parameters, `detect_gray`, live pass ring
* [`src/ugv/aruco.py:199-201`](../../src/ugv/aruco.py#L199) `cv2.setNumThreads(2)` rationale (confirmed: no gain past 2)
* [`src/ugv/aruco.py:1052`](../../src/ugv/aruco.py#L1052) KLT window/levels measured in §2.1
* [`src/ugv/policy.py:14-33`](../../src/ugv/policy.py#L14) `STOP_HORIZON`, `V_*_MAX`, `SEARCH_*`
* [`src/ugv/policy.py:127-237`](../../src/ugv/policy.py#L127) `step()`: stateful branch, shield, `dt_stale`
* [`src/ugv/parking.py:161`](../../src/ugv/parking.py#L161) vectorized `swept_clear`
* [`src/ugv/staged_baseline.py:312-441`](../../src/ugv/staged_baseline.py#L312) batched EKF (n particles, 5-state, delayed ring)
* [`src/ugv/perf.py:27`](../../src/ugv/perf.py#L27) `BUDGET_MS = 33.33`
* [`rl/runs/pi_inference_bench.json`](../../rl/runs/pi_inference_bench.json) Pi projection factor, recorded policy latency, shield stress
* [`rl/export.py:23-47`](../../rl/export.py#L23) parity gate and `gru+head`-only export
* [`HANDOFF.md`](../../HANDOFF.md) §9 (live load/RSS/thermals), §99 (vision 44–52 ms, 473 frames)
* [`docs/SAFETY.md`](../SAFETY.md) S0–S4, one agent owns the car
* [`docs/council_v2/05_cosmos3_skeptical_audit.md`](05_cosmos3_skeptical_audit.md) the 5 ms policy budget / 33.33 ms cadence framing reused here
