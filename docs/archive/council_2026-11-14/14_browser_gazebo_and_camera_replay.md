# 14 — Browser Gazebo Replay, STRL001 Belly Rendering & Live Camera/Training Preview

**Agent:** 14 (Browser Gazebo Replay & Live Training/Vision Preview Architect)  
**Target:** `.` — Observatory web portal (`tools/observatory/`), Gazebo L6 harness (`simulation/gazebo/`), and L7 real-car field runs (`calibration/runs/l7/`)  
**Scope:** Specification & architecture blueprint only (no existing source files modified).

---

## 1. Current Status Diagnosis ("Did the simulations pass Gazebo?" & Why Nothing Shows in Replay)

### 1.1 Gazebo L6 Pass/Fail Ground Truth (`rl/runs/gazebo_l6/`)

| Run file | Controller | Level & Sensing | Result | Verdict |
|---|---|---|---|---|
| `baseline_gazebo_L1.json` | `ekf_staged` (`StagedBaseline`) | L1, `emulated`, 100 ep | `97/100` succ, `0` coll, `3` tout, med `8.4 s` | **PASS** |
| Smoke (`docs/AGENT_BOARD.md:110-111`) | `StagedBaseline` | L3, `emulated`, 5 ep / 3 ep | `5/5` & `3/3` succ | **PASS** |
| `student_L1_latw5_s0_L1_20ep_s11.json` | `StudentPolicy` | L1, `emulated`, 20 ep | `0/20` succ, `0` coll, `20` tout | **FAIL** (physics gap) |
| `student_L1_latw5_s0_L1_camdr_10ep_s11.json` | `StudentPolicy` | L1, `camera_dr`, 10 ep | `0/10` succ, `0` coll, `10` tout (`5.2%` decode) | **FAIL** |
| `student_L2_cons_adr_s0_gazebo_L1_3ep.json` | `StudentPolicy` | L1, `emulated`, 3 ep | `0/3` succ, `0` coll, `3` tout | **FAIL** |

### 1.2 Why Gazebo & Camera Views Are Invisible in the Browser Today

1. **`simulation/gazebo/run_policy_episodes.py` drops trajectories:** `run_episodes()` (`L299-334`) only accumulates `EpisodeResult` scalars (`L70-76`) and writes summary JSON (`L472-474`). Neither `FakeBridge` nor `GazeboBridge` records per-step `ugv_replay_v1` frames or `vision_in_loop` rendered images.
2. **`tools/observatory/server.py` skips `gazebo_l6` in replay:** `_leaderboard()` (`L149-200`) scans `rl/runs/gazebo_l6/*.json`, but `replay_list()` (`L378-391`) and `replay_bundle()` (`L394-416`) only look in `rl/runs/traces/*.json`, `rl/runs/*.traces.json`, and `calibration/runs/l7/*/ticks.jsonl`.
3. **`tools/observatory/replay.py` clobbers geometry & truncates `tag_age`:**
   - `_norm_episodes()` (`L222`) unconditionally executes `data["geometry"] = GEOMETRY`, erasing Gazebo's `GAZEBO_GEOMETRY` (`src/ugv/staged_baseline.py:73`) and any future STRL001 aircraft geometry.
   - `_norm_frame()` (`L200`) casts `tag_age = int(...)`, truncating float seconds (e.g. `0.08 s` latency in Gazebo) to `0` unless stored as `steps_since_valid` or kept float.
4. **`tools/observatory/static/app.js` is static and quadrotor-only:**
   - `loadReplayList()` (`L312-323`) shows raw `bundle_id` strings with no `[gazebo_l6]` / `[l7]` / `[eval]` source tags.
   - `drawReplay()` (`L494-513`, `L564-584`) hardcodes two blue quadrotor skid capsules (`skid_x=0.157`, `gear_len=0.3048`, `gear_r=0.03`) in flat 2D only.
   - `#endframe-card` (`L410-412`) only shows a static `end.jpg` at the end of an L7 run; it ignores `calibration/runs/l7/<run_id>/frames/*.jpg` + `frames.jsonl` (`cap`, `tags[].c`) and has no Gazebo camera POV view.
   - `#auto-refresh` (`L767-769`) only polls `/api/overview` and `/api/leaderboard`, not `/api/curves` or active rollouts/training.

---

## 2. Recording `ugv_replay_v1` Traces & Camera Keyframes in `run_policy_episodes.py`

### 2.1 CLI Extensions (`simulation/gazebo/run_policy_episodes.py`)

Add four optional flags to `main()` (`L355`):
- `--save-traces PATH`: Write a `ugv_replay_v1` JSON bundle. Default auto-path when `--bridge gazebo` and `--out` is used: `<out>/<run_id>.traces.json` (or explicit path, e.g. `rl/runs/gazebo_l6/student_L2_L1.traces.json`).
- `--max-trace-episodes N` (default `20`): Cap recorded episodes per bundle (matching `rl/evaluate.py:96`).
- `--camera-every K` (default `5`, i.e., `10 Hz` keyframes on `50 Hz` loop; `0` disables disk image saving): When `--sensing camera` or `camera_dr`, save JPEG keyframes (`.jpg`, quality `75`, downscaled to `640×360` for browser bandwidth — `~18–28 KB/frame`) to `<trace_stem>_frames/ep_<EEE>/step_<SSSS>.jpg` plus `end.jpg` on termination.
- `--target {quadrotor_skids,strl001_belly}` (default `quadrotor_skids`): Sets `bundle["geometry"]["target_type"]` and obstacle footprint metadata.

### 2.2 Retaining Frame & Detection Telemetry in `vision_in_loop.py`

Currently `VisionSensors.tag()` (`simulation/gazebo/vision_in_loop.py:469-486`) renders the `1280×720` frame, applies `apply_photometric_dr()`, calls `decode_pose()`, and discards `img` and `dets`. To expose this without double-rendering:
1. Refactor `decode_pose()` (`L335`) or add `decode_pose_with_details()` returning `(est, ok, dets_meta)` where `dets_meta` is a compact list: `[{"id": int(d.tag_id), "c": [[u0,v0],...,[u3,v3]]}]`.
2. Store on `VisionSensors`:
   - `self.last_img: np.ndarray | None` (the exact photometric-DR `1280×720` `uint8` frame)
   - `self.last_dets: list[dict]` (raw pixel corners from `ArucoFollower.detect_gray`)
   - `self.last_decode_ok: bool`
   - `self.dr_dict: dict` (serialized `DRParams`: `brightness`, `contrast`, `gamma`, `shadow_strength`, `glare_gain`, `noise_sigma`, `blur_k`)
3. On `GazeboBridge` (`run_policy_episodes.py:128`):
   - Expose `self.last_camera_img`, `self.last_dets`, `self.last_reason`, `self.last_shield_blocked`.
   - Capture the `(act, reason)` tuple returned by `step_fn(...)` in `run_episodes()` (`L329`: currently discards `reason` with `act, _ = step_fn(...)`!). Notice `StudentPolicy.step()` (`src/ugv/policy.py:234`) returns `"brake: swept footprint blocked"` whenever the shield intervenes, and `baseline_controller` (`L289`) returns `"search" | "approach" | "enter" | "hold" | "back"`.

### 2.3 Canonical `ugv_replay_v1` Bundle Schema for Gazebo L6

Compatible with `rl/evaluate.py:269-278` and `tools/observatory/replay.py`:

```json
{
  "schema": "ugv_replay_v1",
  "format": "ugv_replay_v1",
  "source": "gazebo_l6",
  "run_id": "student_L2_cons_adr_s0_gazebo_L1",
  "controller": "student",
  "bridge": "gazebo",
  "level": "L1",
  "seed": 11,
  "sensing": "camera_dr",
  "photometric_dr": true,
  "marker_layout": "real",
  "dt": 0.02,
  "geometry": {
    "target_type": "quadrotor_skids",
    "car_w": 0.172,
    "car_l": 0.203,
    "skid_x": 0.17,
    "gear_len": 0.47,
    "gear_r": 0.015,
    "inflation": 0.015,
    "goal_x": 0.03,
    "goal_y": 0.05,
    "goal_yaw_deg": 5.0,
    "fov_half_deg": 33.5,
    "min_tag_range": 0.40,
    "area": 1.5,
    "tags": [
      {"id": 14, "x": -0.028, "y": 0.0, "z": 0.14},
      {"id": 6,  "x": 0.14,   "y": -0.09, "z": 0.15},
      {"id": 3,  "x": -0.13,  "y": 0.0, "z": 0.16}
    ],
    "aircraft": null
  },
  "episodes": [
    {
      "episode_id": 0,
      "seed": 11,
      "outcome": "timeout",
      "steps": 2250,
      "duration": 45.0,
      "final_error": 612.4,
      "dr_params": { "brightness": 0.62, "contrast": 1.15, "gamma": 1.2, "blur_k": 3 },
      "end_frame_url": "/api/replay/media/gazebo_l6/student_L2_cons_adr_s0_gazebo_L1/ep_000/end.jpg",
      "frames": [
        {
          "t": 0.0,
          "true_pose": [0.2669, 1.3002, -0.6951],
          "vel": [0.0, 0.0, 0.0],
          "delayed_tag": [0.2670, 1.3000, -0.6951],
          "aux_pose": [0.2669, 1.3002, -0.6951],
          "valid": true,
          "blackout": false,
          "outlier": false,
          "tag_age": 0.08,
          "steps_since_valid": 4,
          "imu_w": -0.0004,
          "imu_yaw": -0.6951,
          "action": [-0.078, 0.0604, 0.0137],
          "shield_clear": true,
          "reason": "policy",
          "camera_frame_url": "/api/replay/media/gazebo_l6/student_L2_cons_adr_s0_gazebo_L1/ep_000/step_0000.jpg",
          "tags_px": [{"id": 14, "c": [[612.1, 310.2], [654.0, 309.8], [653.4, 351.7], [611.6, 352.1]]}]
        }
      ]
    }
  ]
}
```

**Size & Speed Budget:**
- A 45 s episode at 50 Hz = `2,250` frames. Compact JSON (rounding floats to 4 decimals, omitting `camera_frame_url` on non-keyframe steps so the UI holds the most recent keyframe or interpolates) is `~380 KB/episode` (`~7.6 MB` for 20 episodes).
- Keyframe JPEGs at `640×360` q=75 every 5 steps (`10 Hz`) = `~22 KB × 450 = ~9.9 MB/episode` (on disk, fetched lazily by the browser only as scrubbed). Encoding `640×360` JPEG in OpenCV takes `< 1.2 ms` per 5th physics step (`+0.24 ms/step` amortized, `< 1%` of Gazebo wall time).

### 2.4 Live Training & Active Gazebo Visibility

To answer *"how its doing while training"* and show live Gazebo runs in the browser:
1. **Incremental Checkpoint Replay (`run_policy_episodes.py`):** Write `<out>/<run_id>.traces.json.tmp` and atomically `os.replace` after every completed episode (with `"in_progress": true` until final episode finishes). The Observatory's `#auto-refresh` (`5 s`) polls `/api/replay/bundle/{id}` when `in_progress` is true and updates the episode list live.
2. **Curves Auto-Refresh (`app.js:767-769`):** Extend `#auto-refresh` callback from `loadOverview(); loadLeaderboard();` to also invoke `loadCurvesRuns();` + `loadSelectedCurves();` whenever `#tab-curves` is active (reusing existing `active: (time.time() - mtime) < 60` badge in `server.py:335`).
3. **Gazebo Rollout Trigger (`server.py:439-481`):** Extend `POST /api/rollout` with optional `"backend": "rl_eval" | "gazebo_l6"`, `"sensing": "emulated" | "camera" | "camera_dr"`, `"target": "quadrotor_skids" | "strl001_belly"`. When `"gazebo_l6"`, spawn `simulation/gazebo/run_policy_episodes.py --bridge gazebo --weights <npz> --save-traces rl/runs/gazebo_l6/<job_id>.traces.json` (accepting `.npz` weights listed by `/api/rollout/options` alongside `.pt` weights).

---

## 3. Observatory Backend & Normalization Changes (`server.py`, `replay.py`)

### 3.1 `tools/observatory/server.py`

1. **`replay_list()` (`L378-391`):** Scan `RL_RUNS / "gazebo_l6"` for `*.traces.json` and `traces/*.json` (and any `*.json` whose top-level keys contain `"format": "ugv_replay_v1"` or non-empty `episodes[0].frames` — explicitly skipping summary-only JSONs like `dbg_cam2.json` and `baseline_gazebo_L1.json`). Return:
   ```python
   {"bundle_id": f"gazebo_l6:{p.stem.removesuffix('.traces')}", "source": "gazebo_l6", "label": f"[gazebo_l6] {p.stem.removesuffix('.traces')}"}
   ```
   Similarly populate `"source"` and `"label"` for `"eval"` (`[eval]`) and `"l7"` (`[l7]`).
2. **`replay_bundle(bundle_id)` (`L394-416`):** Handle `bundle_id.startswith("gazebo_l6:")` using `safe_path(RL_RUNS / "gazebo_l6", ...)`, checking `<stem>.traces.json`, `traces/<stem>.json`, and `<stem>.json`, then passing through `eval_bundle(c)`.
3. **Media Route `GET /api/replay/media/{source}/{run_id}/{rel_path:path}`:** Serves keyframe JPEGs safely via `safe_path()`:
   - `source == "gazebo_l6"` $\rightarrow$ `RL_RUNS / "gazebo_l6" / f"{run_id}_frames" / rel_path`
   - `source == "l7"` $\rightarrow$ `CALIB_RUNS / run_id / rel_path` (supports both `end.jpg` and `frames/00055889.jpg`)
4. **`_counts()` (`L37-59`):** Count `*.traces.json` in `gazebo_l6` separately or include in `traces` pill.

### 3.2 `tools/observatory/replay.py`

1. **Fix Geometry Overwrite (`L222`):** Replace `data["geometry"] = GEOMETRY` with a shallow-merge:
   ```python
   base_geo = dict(GEOMETRY)
   if isinstance(data.get("geometry"), dict):
       base_geo.update(data["geometry"])
   data["geometry"] = base_geo
   ```
2. **Fix `tag_age` Truncation (`L200`):** Keep `tag_age` in float seconds if provided as `tag_age` (`float(...)`), and preserve `steps_since_valid` as `int` (`tag_age_s = frame.get("tag_age")` if float, else `steps * dt`).
3. **Per-Tick Camera Keyframe Alignment in `l7_bundle(run_id)` (`L69-157`):**
   - Parse `d / "frames.jsonl"` if present, filtering entries where `entry.get("jpg")` is non-null (and keeping all `tags` corners).
   - For each tick `r` in `ticks.jsonl` at monotonic time `r["t"]`, binary-search the latest `frames.jsonl` entry with `cap <= r["t"]` (within `<= 0.5 s` tolerance):
     - Attach `"camera_frame_url": f"/api/replay/media/l7/{run_id}/frames/{jpg}"` (or `end.jpg` on final tick if `frames/` absent).
     - Attach `"tags_px": [{"id": t["id"], "how": t.get("how"), "c": t.get("c")} for t in frame_row.get("tags", [])]`.
   - Verified on `calibration/runs/l7/20261113-185039-stage-park-0.7/`: `frames.jsonl` shares the exact `time.monotonic()` clock base (`9732.8..9738.2+`) with `ticks.jsonl` (`t`), and `frames/` stores JPEGs every ~6 camera frames (`~5 Hz`).

---

## 4. STRL001 Fixed-Wing Belly Geometry & 2D/2.5D Canvas Rendering (`app.js`)

### 4.1 Coordinate Convention & `geometry.aircraft` Spec

Preserve the repo's canonical drone/policy frame (`docs/council_v2/06_yolo_keypoint_and_wheel_detection.md:22-23`, `tools/observatory/static/app.js:435`):
- **Origin `(0, 0, 0)`**: Belly docking center on the floor (`z = 0`).
- **Approach corridor**: Car spawns at **`+y`** (top of 2D canvas) and enters toward **`-y`** (`yaw = 0` faces `-y`, downward on canvas).
- **`+x` right, `-x` left, `+z` up** (out of floor).
- Therefore, for STRL001 facing forward (`-y`, away from the approaching car, matching the rear-approach view in `docs/council_v2/strix_plane_realflight.png`):
  - **Inverted V-tail rear fins**: Aft at **`+y`** (`y ≈ +0.28..+0.42 m`), sloping downward from tail boom (`z_root > z_tip`) with dihedral `~42°`, carrying ArUco tags on fin wedges / under-boom vertical faces.
  - **Main left/right landing wheels**: Straddling the entry corridor at **`x = ±main_wheel_x`** (`|x| >= 0.17 m` for `0.172 m` car clearance), `y ≈ 0.00 m`.
  - **Nose wheel**: Forward at **`(0, nose_wheel_y)`** (`y ≈ -0.38 m`), acting as the forward hard stop.
  - **High wing & twin tractor props**: Wing spans **`±wingspan/2`** (`x ≈ ±0.65 m`) at `y ≈ -0.08..+0.08 m`, `z ≈ +0.26 m` (overhead clearance above `0.17 m` car height); twin props at **`(±prop_x, prop_y, prop_z)`** (`x ≈ ±0.24 m`, `y ≈ -0.18 m`, `z ≈ +0.26 m`).

> [!IMPORTANT]
> Exact STRL001 airframe tape-measure values are not yet surveyed in `drone_dimensions.json`. Store defaults under `geometry.aircraft` with `"surveyed": false` so measured dimensions drop in via JSON without editing JS.

```json
"target_type": "strl001_belly",
"aircraft": {
  "surveyed": false,
  "wingspan_m": 1.30,
  "wing_root_y_m": -0.10,
  "wing_chord_root_m": 0.24,
  "wing_chord_tip_m": 0.14,
  "wing_sweep_y_m": 0.08,
  "wing_z_m": 0.26,
  "fuselage_len_m": 0.84,
  "fuselage_w_m": 0.13,
  "belly_z_m": 0.18,
  "prop_x_m": 0.24,
  "prop_y_m": -0.18,
  "prop_r_m": 0.11,
  "prop_z_m": 0.26,
  "main_wheel_x_m": 0.19,
  "main_wheel_y_m": 0.02,
  "main_wheel_r_m": 0.035,
  "main_wheel_w_m": 0.025,
  "nose_wheel_y_m": -0.38,
  "nose_wheel_r_m": 0.028,
  "nose_wheel_w_m": 0.022,
  "vtail_root_y_m": 0.30,
  "vtail_span_x_m": 0.22,
  "vtail_chord_m": 0.13,
  "vtail_dihedral_deg": 42.0,
  "vtail_root_z_m": 0.28,
  "vtail_tip_z_m": 0.11
}
```

### 4.2 2D / 2.5D Canvas Rendering (`drawReplay()` in `app.js`)

Add a view toggle in `#layers` or transport bar: **`View: [2D Top-Down | 2.5D Oblique]`**.

**Projection math (`W2S(x, y, z = 0)`):**
- **2D Top-Down** (`tilt = 0`): Identical to current `app.js:435`: `[cx + x * ppm, cy - y * ppm]`.
- **2.5D Oblique** (`tilt = 1`, lightweight axonometric projection preserving metric scale on floor `Z = 0` while lifting elevated structures vertically):
  ```js
  // Oblique camera looking from +y (behind aircraft) slightly above floor:
  // floor (x, y, 0) stays readable; +z lifts vertically on screen, +y foreshortens slightly
  const W2S3 = (x, y, z = 0) => is25D
    ? [cx + x * ppm, cy - y * ppm * 0.72 - z * ppm * 0.68]
    : [cx + x * ppm, cy - y * ppm];
  ```
  Why this oblique projection works cleanly on `<canvas>` without WebGL:
  - Floor geometry (goal box, blind ring, wheel contact patches, UGV footprint, swept-clear shield) remains convex and directly comparable.
  - Elevated parts (`belly_z_m = 0.18`, `wing_z_m = 0.26`, `prop_z_m = 0.26`, inverted V-tail `vtail_root_z_m = 0.28 -> vtail_tip_z_m = 0.11`) visibly float **above** the car (`car_h ≈ 0.14 m`) with ground-projected drop shadows at `z = 0`, immediately revealing under-belly headroom and V-tail fin slope!

**STRL001 Draw Order (Painter's Order Back-to-Front in 2.5D):**
1. **Floor Grid, Blind Ring (`0.40 m`), Goal Box (`±0.03 × ±0.05 m`)** at `z = 0`.
2. **Landing Gear Floor Contacts & Swept-Clear Shield (`z = 0`):**
   - Left/right main wheels at `(±main_wheel_x_m, main_wheel_y_m, 0)` and nose wheel at `(0, nose_wheel_y_m, 0)` drawn as rounded tire rectangles (`main_wheel_w_m × 2*main_wheel_r_m`) with dark rubber fill + cyan/blue rim.
   - `Swept-Clear Shield` layer (`L['Swept-Clear Shield']`): dashed amber (`#f5b041`) safety inflation circles/capsules (`r + inflation`) around all 3 wheel contacts (plus nose fuselage projection if low).
3. **Floor Drop Shadow** (`rgba(74, 163, 255, 0.08)`) of wings, fuselage, and V-tail projected at `z = 0`.
4. **Path Trail, Camera FOV Cone (`±33.5°`), Tag Ghost (`#f5b041`), Aux Ghost (`#a569bd`), and True UGV Body (`0.172 × 0.203 m`)** at `z = 0` (drawn **below** the elevated wing/belly so the car visibly drives *under* the aircraft belly!).
5. **Elevated Aircraft Belly & Fuselage (`z = belly_z_m`):** Translucent streamlined fuselage hull (`rgba(74, 163, 255, 0.18)` fill, `#4aa3ff` stroke) from nose (`-0.42 m`) to tail boom (`+0.38 m`).
6. **High Main Wing (`z = wing_z_m`) & Twin Props (`(±prop_x_m, prop_y_m, prop_z_m)`):**
   - Swept trapezoidal left/right wings with subtle leading-edge highlight; dashed outline when UGV is directly underneath (`alpha = 0.35`) so the car stays visible.
   - Twin tractor propeller discs/blades ahead of the leading edge (`±prop_x_m`, radius `prop_r_m`).
7. **Inverted V-Tail Rear Fins & ArUco Tags (`z = vtail_root_z_m -> vtail_tip_z_m`):**
   - Two angled fin polygons from `(0, vtail_root_y_m, vtail_root_z_m)` out-and-down to `(±vtail_span_x_m, vtail_root_y_m + vtail_chord_m*0.6, vtail_tip_z_m)`.
   - ArUco tag squares (`L['ArUco Tags']`) anchored on their actual `geometry.tags` `[x, y, z]` coordinates (e.g., left fin wedge `ID_L`, right fin wedge `ID_R`, under-boom `ID_C`) with cyan border, ID label, and a short surface-normal tick pointing toward the `+y` approach corridor.

*(When `target_type === "quadrotor_skids"`, `drawReplay()` keeps the existing two-skid capsule renderer, using `W2S3` in 2.5D mode to lift the central deck/payload to `z = 0.225..0.43 m` and tags to `z = 0.14..0.16 m`.)*

---

## 5. Synchronized Camera POV Panel (`app.js`, `index.html`, `style.css`)

### 5.1 Layout Upgrade (`#tab-replay`)

Replace `#endframe-card` (`tools/observatory/static/index.html:99`) with a side-by-side / split viewport above the 3 telemetry strips:

```text
+----------------------------------+------------------------------+
|  2D / 2.5D Top/Oblique Canvas    |  Camera POV (1280x720 / 16:9)|
|  (#replay-canvas)                |  (#camera-canvas / <img>)    |
|  - STRL001 wings/props/wheels    |  - Real L7 frame OR          |
|  - Inverted V-tail + ArUco tags  |    Gazebo vision_in_loop     |
|  - UGV + shield + FOV + trail    |  - Detected corners overlay  |
|                                  |  - Mode pill + PnP/DR stats  |
+----------------------------------+------------------------------+
|  strip1: valid / blackout / outlier / tag_age                   |
|  strip2: imu_w / imu_yaw                                        |
|  strip3: vx / vy / w + shield_blocked                           |
+-----------------------------------------------------------------+
```

### 5.2 Three Camera POV Data Modes (Automatic by Frame Payload)

1. **Recorded Image Frame (`f.camera_frame_url` or nearest preceding keyframe in `ep.frames`):**
   - **Gazebo `--sensing camera[_dr]`**: Displays the `vision_in_loop.py` synthetic grayscale JPEG keyframe (shows the actual photometric DR — glare, shadow gradient, blur, card tilt).
   - **Real L7 field run**: Displays the matched `calibration/runs/l7/<run_id>/frames/<jpg>` (or `end.jpg` at the final frame).
   - **Overlay canvas (`#camera-overlay`)**: Draws detected `f.tags_px` corner polygons (`#3dd68c` green for `det`, `#4aa3ff` cyan for `klt`), tag ID badge, optical center crosshair `(640, 360)`, and a HUD line:
     `keyframe Δt=0.04s | det: #14 (45px) | tag_age=0.08s | DR: gain=0.62 blur=3`.
2. **Live Pinhole Fallback (`--sensing emulated` or `rl/evaluate.py` `sim_eval` bundles without JPEGs):**
   - If no JPEG is present for the episode, render a **synthetic 3D wireframe Camera POV** directly on `#camera-canvas` in JS using the true pinhole model from `vision_in_loop.py:124-143` (`FOCAL_PX=968, CX=640, CY=360, CAM_FWD_M=0.071, CAM_Z_M=0.06445`):
     - Project the floor horizon (`Z = 0`), blind ring, landing gear / STRL001 wheels, V-tail fin silhouette, and `geometry.tags` squares from `poseOf(f)`.
     - Badge: `SYNTHETIC PINHOLE POV (no camera frames recorded)`.
3. **Final Frame Verification (`ep.end_frame_url`):**
   - Keep a clickable toggle/thumbnail (`Jump to end.jpg`) whenever `ep.end_frame_url` exists so L7 runs preserve instant end-of-run photo inspection.

### 5.3 Replay Dropdown Badge & Filter (`loadReplayList()` in `app.js:312-323`)

- Format `<option>` labels as `[gazebo_l6] <run_id>`, `[l7] <run_id>`, `[eval] <run_id>` (or group via `<optgroup label="Gazebo L6">` / `<optgroup label="L7 Real Car">` / `<optgroup label="Fast Sim Eval">`).
- Add a source badge pill next to `#bundle-select` showing `source`, `level`, `sensing`, `controller`, and overall pass/fail (`e.g. 97/100 PASS` or `0/20 FAIL`) immediately on selection.

---

## 6. Verification & Rollout Checklist (For Implementation Turn)

1. **Trace generation check:** Run `uv run python simulation/gazebo/run_policy_episodes.py --bridge fake --smoke --episodes 3 --sensing camera_dr --save-traces /tmp/test_gz.traces.json --camera-every 5` and verify `ugv_replay_v1` schema, `tags_px`, and JPEG keyframes load cleanly via `eval_bundle()`.
2. **Gazebo L6 check:** Run `run_policy_episodes.py --bridge gazebo --baseline --level L1 --episodes 5 --sensing camera_dr --save-traces rl/runs/gazebo_l6/baseline_L1_camdr_demo.traces.json` and confirm `[gazebo_l6]` appears in `/api/replay/list` and plays back in Observatory with both 2D/2.5D canvas and `vision_in_loop` camera POV.
3. **L7 Camera POV check:** Open `l7:20261113-185039-stage-park-0.7` in Observatory; scrub `0..262` and confirm `frames/*.jpg` updates smoothly along the timeline and lands on `end.jpg` with green `tags_px` overlays.
4. **STRL001 geometry check:** Toggle `target_type: "strl001_belly"` + `2.5D Oblique` view and verify wings, twin props, left/right main wheels, nose wheel, inverted V-tail fins + ArUco tags, and under-belly UGV clearance render with zero overlap ambiguity.
