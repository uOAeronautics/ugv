---
name: yolo-pose-pi
description: Train, INT8-quantize, benchmark on real held-out frames, and deploy single-stage YOLO-pose ONNX models for STRL-001 airframe detection on Raspberry Pi 4. Use when retraining or evaluating the airframe keypoint detector.
---

# YOLO-Pose for Raspberry Pi 4 (`train/yolo_pose.py` → `YoloPoseDetector`)

## 1. Environment boundary

- `src/ugv` MUST stay torch-free; Pi inference runs pure `onnxruntime` CPU (`src/ugv/pose/detector.py`, `YoloPoseDetector`, `threads=2`).
- Training lives in `train/` with its own CUDA venv (`cd train && uv run python yolo_pose.py ...`).

## 2. Dataset recipe that generalizes to real rooms

Single-stage `yolo11n-pose` at `416×256` grayscale (`runs/pose_train/yolo11n-pose-v3-416-int8`, `5.3 MB`, `135 ms` on Pi 4) beat two-stage MobileNetV3 (`250 ms`, wing-clipping) and `yolo26n-pose` (`149 ms`, `0/15` real bay frames):

- **Pixel-perfect silhouette bboxes** from rendered triangle IDs (`v.ids >= 0`), never keypoint-extents (keypoint boxes clip wings).
- **15 keypoints** (`ugv.pose.mesh.keypoints(..., near=True)`) with symmetric `flip_idx` for horizontal flip augmentation.
- **Diverse backgrounds**: mix ~120 tonemapped 2K HDR panoramas (`runs/cache/hdri_v2/`) with real lab frames so the detector never overfits to one room.
- **Close-range bay views (~35%)**: sample inside/near the main gear where only `gear_nose` + `tyre_left/right/top` are visible.
- **Pose loss weight**: `pose=18.0` tightens wheel/hub keypoint precision.

## 3. INT8 ONNX export for Pi 4

- Export static ONNX (`imgsz=(256, 416)`), then run static QDQ INT8 quantization (`onnxruntime.quantization.quantize_static`, `QDQ`, `CalibrationMethod.MinMax`) **excluding the Pose head nodes (`/model.23/`)** so keypoint coordinates stay FP32-accurate.
- Keep only the active winner ONNX under `adv034:~/ugv/runs/pose_train/` to save SD card space.

## 4. Mandatory real-frame gate (never trust synthetic mAP alone)

Before deploying any new model to the Pi, benchmark on **real held-out frames** (including close-up bay frames from `adv034:~/ugv/runs/autopark/*/frames/`):
- Measure overall detection rate, **close-up bay detection rate** (e.g. `15/15` for `yolo11n-pose-v3-416-int8`), false-positive rate on background clutter, and live Pi 4 `ms` (`threads=2`).
- Example failure caught by this gate: `yolo26n-pose-v1-416-int8` had higher synthetic Pose mAP50-95 (`0.914` vs `0.893`) but dropped to **`0/15` on real bay frames** after INT8 quantization and ran slower (`149 ms` vs `135 ms`).

## 5. Runtime clutter gating (`src/ugv/pose/detector.py`)

- Normal range: `conf >= 0.50`, `kpt >= 0.50`, reject `< 3 pts` (and `< 4 pts` when `score < 0.58`) — rejects rolling chair/table casters while keeping 3-pt side views (`hubRear + gearL + gearR`).
- `close_range=True` (during `ENTER`): relax to `conf >= 0.30`, `kpt >= 0.40`, `>= 3 pts` so nose-wheel-only bay views stay locked.
