#!/usr/bin/env python3
"""Offline sim-to-real contract audit for L7 UGV logs (NumPy-only, root .venv).

Compares a recorded L7 run (ticks.jsonl + meta.json) against the L1 simulation
reference (tools/l7/sim_reference_L1.npz from dump_sim_reference.py), a real-car
axis probe run (fit.json / ticks.jsonl vs simulation/gazebo/mecanum_bayesian_profile.json),
and Gazebo L6 gate results (rl/runs/gazebo_l6/*.json).

Prints a human-readable channel/check table, emits JSON, and exits non-zero when
any contract check fails its threshold.

================================================================================
RUNTIME MONITOR DESIGN NOTE (for future integration into l7_run.py; NOT wired)
================================================================================
The following 5 online checks over rolling 1.0 s (50-tick) and 2.0 s (100-tick)
windows would have halted 20261113-183857-policy-ep3-p0 within <= 2.0 s
(thresholds derived directly from Deliverable A's L1 sim reference & ep3 logs):

1. Rolling 1.0 s gyro-rate std (`imu_w` over 50 ticks):
   - Sim L1 ref: full-ep std = 0.091 rad/s; 1.0 s window p99 = 0.055 rad/s (max 0.092).
   - Threshold: brake if `std(imu_w[t-1s:t]) > 0.25 rad/s`.
   - ep3 behavior: `t=[0,1]s` already hits `0.515 rad/s` (5.6x sim max) -> trips at t=1.0 s.

2. Rolling 1.0 s action saturation (`|act_j| > 0.8` fraction over 50 ticks):
   - Sim L1 ref: full-ep `sat(|a2|>0.8) = 0.0014`; 1.0 s window p99 = 0.00, max = 0.04.
   - Threshold: brake if `mean(|act_j[t-1s:t]| > 0.8) > 0.25` for any axis j.
   - ep3 behavior: `a2` saturation is `0.52` in `[0,1]s` and `0.86` in `[1,2]s` -> trips at t=1.0 s.

3. Rolling 1.0 s input z-score vs sim reference (`imu_w` RMS z-score):
   - Sim L1 ref: `imu_w` mean = +0.002, std = 0.091 rad/s.
   - Threshold: brake if `sqrt(mean(((imu_w[t-1s:t] - mu_sim) / sigma_sim)**2)) > 3.0`
     (i.e. > 0.27 rad/s RMS).
   - ep3 behavior: `0.515 / 0.091 = 5.65 sigma` in `[0,1]s` -> trips at t=1.0 s.

4. Rolling 2.0 s translation action sign-flip rate (`|act_j| > 0.1` reversals/s):
   - Sim L1 ref: `|a|>0.1` flip p95 is `0.67 /s` (`a0`) and `0.17 /s` (`a1`).
   - Threshold: brake if `flips(|a0|>0.1)/s > 2.5` or `flips(|a1|>0.1)/s > 1.5` over 2.0 s.
   - ep3 behavior: `a0` flips `3 /s` in `[0,1]s` and `3 /s` in `[1,2]s` -> trips at t=2.0 s.

5. Rolling 2.0 s raw camera-tag rate (`src == "tag"` fraction over 100 ticks):
   - Sim L1 ref: 30 Hz camera = 0.588 real tag fixes/tick (no DR masking).
   - Threshold: brake if `mean(src == "tag" over [t-2s:t]) < 0.25` after acquisition.
   - Behavior on ep1/ep2: catches dead-reckoning masking missing tags (`ep1` 13.5%, `ep2` 3.8%)
     within 2.0 s of losing visual lock.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SIM_REF = Path(__file__).resolve().parent / "sim_reference_L1.npz"
DEFAULT_PROFILE = REPO / "simulation" / "gazebo" / "mecanum_bayesian_profile.json"
DEFAULT_L6_DIR = REPO / "rl" / "runs" / "gazebo_l6"

CHANNEL_NAMES = [
    "tag_x",
    "tag_y",
    "tag_yaw",
    "tag_age",
    "valid",
    "imu_w",
    "imu_yaw",
    "prev_a0",
    "prev_a1",
    "prev_a2",
    "tvel_x",
    "tvel_y",
    "tvel_yaw",
]
ACTION_NAMES = ["act_0", "act_1", "act_2"]
EMA_TAU = 0.25
HALF_BASE = (0.21 + 0.149) / 2.0  # mirrors rl/sim.py and src/ugv/staged_baseline.py
NOMINAL_DT = 0.02

# -----------------------------------------------------------------------------
# Stated thresholds (every threshold justified by measured values or sim L1 ref)
# -----------------------------------------------------------------------------

# 1. Gyro-rate channel (`imu_w`, dim 5):
#    Sim L1 adds 0.002 rad/s white noise and has total std = 0.091 rad/s
#    ([p05, p95] = [-0.057, +0.045], span = 0.102 rad/s). Real ep1/ep2/ep3 hit
#    std = 0.618 / 0.215 / 0.633 rad/s (span 1.85 / 0.69 / 2.05 rad/s); ep3
#    replay saturates |a2|>0.8 on 74% of ticks with real imu_w vs 0% when zeroed.
IMU_W_STD_FACTOR = 2.0
IMU_W_STD_FLOOR = 0.19  # > 2x sim std (0.091), below ep2 (0.215) & ep3 (0.633)
IMU_W_SPAN_FACTOR = 3.0
IMU_W_SPAN_FLOOR = 0.30  # > 3x sim p95-p05 span (0.102)

# 2. Valid-observation & raw camera-tag rate (`valid`, dim 4; `src == "tag"`):
#    Sim L1 30 Hz camera yields valid = 0.588 (100% real tags). Static student
#    probe showed forward command scales with valid rate (-0.86 at 1.0, -0.50 at
#    0.50, -0.10 at 0.20, 0.0 at 0.10). Real ep1 had only 0.135 raw tag ticks
#    (masked by DR to 0.469 valid); ep2 had 0.154 valid / 0.038 tag.
MIN_VALID_RATE = 0.35
MIN_TAG_RATE = 0.30

# 3. Pose age & inter-observation cadence:
#    Sim L1 30 Hz camera has tag_age p95 = 0.020 s, inter_valid_dt p95 = 0.040 s,
#    inter_tag_dt p95 = 0.040 s. Real ep2 reached tag_age p95 = 1.091 s (stale
#    brake); ep1 reached inter_tag_dt p95 = 0.258 s under DR masking.
MAX_TAG_AGE_P95_FACTOR = 4.0
MAX_TAG_AGE_P95_FLOOR = 0.15
MAX_INTER_VALID_DT_P95_FACTOR = 3.0
MAX_INTER_VALID_DT_P95_FLOOR = 0.12
MAX_INTER_TAG_DT_P95_FACTOR = 5.0
MAX_INTER_TAG_DT_P95_FLOOR = 0.20

# 4. Action saturation (`|a_j| > 0.8`) and sign-flip rate (`|a_j| > 0.1` flips/s):
#    Sim L1 saturation is [0.000, 0.000, 0.0014] across 202k ticks; ep1/ep2/ep3
#    saturated turn action a2 on 82.2% / 22.8% / 74.1% of ticks.
MAX_ACT_SAT_ABS = 0.10
MAX_ACT_SAT_ADD = 0.10  # sim_sat + 0.10 (>70x sim max 0.0014, well below ep2 0.228)
#    Sim L1 |a|>0.1 sign-flip p95 is [0.67, 0.17, 5.47] /s; ep1 (3.40, 5.85 /s)
#    and ep3 (3.44, 3.60 /s) chatter violently on translation axes.
MAX_TRANS_FLIP_FACTOR = 2.0
MAX_TRANS_FLIP_FLOOR = 1.5  # /s
MAX_TURN_FLIP_FACTOR = 1.5
MAX_TURN_FLIP_FLOOR = 8.5  # /s

# 5. Control-loop rate & jitter:
#    Student policy and tag_vel EMA are trained at 50 Hz (dt = 0.020 s, 20 ms
#    tick budget). Real ep1 dropped to 45.8 Hz with loop_ms p95 = 31.3 ms and
#    dt p95 = 0.032 s; ep2 had loop_ms p95 = 28.1 ms.
MIN_LOOP_HZ = 45.0  # >= 90% of 50 Hz nominal rate
MAX_LOOP_DT_P95 = 0.030  # <= 1.5x nominal 0.020 s dt
MAX_LOOP_MS_P95 = 25.0  # <= 1.25x 20 ms tick budget

# 6. Remaining frame channel drift (dims 0..2, 6..12):
#    prev_act std in sim L1 is [0.049, 0.089, 0.118]; ep1/ep3 hit 0.57/0.53 on
#    prev_a0 and 0.91/0.87 on prev_a2. Geometric/vel channels use 3.5x sim std
#    with a 0.35 floor for std and max(3.0*sim_std, 0.40) for mean shift.
PREV_ACT_STD_FACTOR = 2.5
PREV_ACT_STD_FLOOR = 0.25
FRAME_MEAN_STD_MULT = 3.0
FRAME_MEAN_FLOOR = 0.40
FRAME_STD_MULT = 3.5
FRAME_STD_FLOOR = 0.35

# 7. Near-goal tag visibility (< 0.45 m):
#    Sim L1 (under_drone_loss=False) keeps the tag visible down to the goal
#    (tag_rate ~0.60 below 0.45 m), whereas the real camera loses the tag below
#    ~0.4 m due to vertical FOV. Checked only when >= 10 ticks occur < 0.45 m.
NEAR_GOAL_DIST_M = 0.45
MIN_NEAR_GOAL_TICKS = 10
MIN_NEAR_GOAL_TAG_RATE = 0.30

# 8. Plant mismatch (probe run vs simulation/gazebo/mecanum_bayesian_profile.json):
#    Sim profile has deadzone = 0.0 and p0 couples full turn into 1.13 m/s
#    translation (vx=-0.98, vy=+0.56). Real car probe has deadzones fwd=0.051,
#    left~0.13 (stalls at 0.15), ccw=0.101 and gains fwd=2.32, left~1.6, ccw=15.66.
MAX_DEADZONE_DIFF = 0.05
MAX_GAIN_REL_ERR = 0.50
MAX_SIM_TURN_COUPLING_MPS = 0.25


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _quantiles(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _sign_flip_rate(a: np.ndarray, duration_s: float, deadband: float = 0.1) -> float:
    """Sign reversals per second across consecutive samples with |a| > deadband."""
    if duration_s <= 0 or a.size < 2:
        return 0.0
    nz = np.abs(a) > deadband
    s = np.sign(a[nz])
    if s.size < 2:
        return 0.0
    return float(np.count_nonzero(s[1:] != s[:-1])) / duration_s


def _mix_np(cmd: np.ndarray) -> np.ndarray:
    vx, vy, w = cmd[..., 0], cmd[..., 1], cmd[..., 2] * HALF_BASE * 4.0
    trans = np.stack([vx - vy, vx + vy, vx + vy, vx - vy], axis=-1)
    turn = np.stack([-w, -w, w, w], axis=-1)
    return np.clip(-trans + turn, -1.0, 1.0)


def _wheel_features_np(wheels: np.ndarray, deadzone: float) -> np.ndarray:
    dz = float(np.clip(deadzone, 0.0, 0.3))
    q = np.sign(wheels) * np.clip(np.abs(wheels) - dz, 0.0, None) / (1.0 - dz + 1e-6)
    mean_abs = np.abs(q).mean(axis=-1, keepdims=True)
    return np.concatenate(
        [np.clip(q, 0.0, None), np.clip(q, None, 0.0), q * np.abs(q), q * mean_abs],
        axis=-1,
    )


def compute_sim_stats(sim_ref_path: Path) -> dict[str, Any]:
    z = np.load(sim_ref_path, allow_pickle=True)
    frames = np.asarray(z["frames"], dtype=np.float64)
    acts = np.asarray(z["acts"], dtype=np.float64)
    ep_id = np.asarray(z["ep_id"], dtype=np.int64)
    pose = np.asarray(z["pose"], dtype=np.float64) if "pose" in z.files else None
    dt = float(z["dt"]) if "dt" in z.files else NOMINAL_DT
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}

    channel_stats = {name: _quantiles(frames[:, i]) for i, name in enumerate(CHANNEL_NAMES)}
    act_stats = {
        name: {
            **_quantiles(acts[:, j]),
            "sat": float(np.mean(np.abs(acts[:, j]) > 0.8)),
        }
        for j, name in enumerate(ACTION_NAMES)
    }

    # Group by episode while preserving chronological order within each episode
    # (MecanumVecEnv dumps ticks interleaved across parallel envs).
    order = np.argsort(ep_id, kind="stable")
    frames_s = frames[order]
    acts_s = acts[order]
    ep_s = ep_id[order]
    iv_dts: list[float] = []
    flips_ep: list[list[float]] = []
    ep_boundaries = np.flatnonzero(np.diff(ep_s, prepend=ep_s[0] - 1, append=ep_s[-1] + 1))
    for k in range(len(ep_boundaries) - 1):
        lo, hi = int(ep_boundaries[k]), int(ep_boundaries[k + 1])
        dur = (hi - lo) * dt
        flips_ep.append([_sign_flip_rate(acts_s[lo:hi, j], dur) for j in range(3)])
        v_idx = np.flatnonzero(frames_s[lo:hi, 4] > 0.5)
        if v_idx.size > 1:
            iv_dts.extend((np.diff(v_idx) * dt).tolist())

    flips_arr = np.asarray(flips_ep, dtype=np.float64)
    for j, name in enumerate(ACTION_NAMES):
        act_stats[name]["flip_mean"] = float(np.mean(flips_arr[:, j])) if flips_arr.size else 0.0
        act_stats[name]["flip_p95"] = (
            float(np.percentile(flips_arr[:, j], 95)) if flips_arr.size else 0.0
        )

    iv_arr = np.asarray(iv_dts, dtype=np.float64)
    near_tag_rate = float("nan")
    if pose is not None and pose.shape[0] == frames.shape[0]:
        near = np.hypot(pose[:, 0], pose[:, 1]) < NEAR_GOAL_DIST_M
        if np.count_nonzero(near) >= MIN_NEAR_GOAL_TICKS:
            near_tag_rate = float(np.mean(frames[near, 4] > 0.5))

    return {
        "meta": meta,
        "dt": dt,
        "channels": channel_stats,
        "actions": act_stats,
        "valid_rate": float(np.mean(frames[:, 4] > 0.5)),
        "tag_rate": float(np.mean(frames[:, 4] > 0.5)),  # all valid fixes in L1 sim are camera tags
        "inter_valid_dt": _quantiles(iv_arr),
        "inter_tag_dt": _quantiles(iv_arr),
        "near_goal_tag_rate": near_tag_rate,
    }


def extract_run_metrics(run_dir: Path) -> dict[str, Any]:
    ticks_path = run_dir / "ticks.jsonl"
    if not ticks_path.is_file():
        raise FileNotFoundError(f"Missing ticks.jsonl in {run_dir}")
    rows = [json.loads(line) for line in ticks_path.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"{ticks_path} has {len(rows)} rows; need >= 2 ticks to audit")

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}

    t = np.array([float(r["t"]) for r in rows], dtype=np.float64)
    dt_arr = np.diff(t)
    duration_s = float(t[-1] - t[0]) if t[-1] > t[0] else len(rows) * NOMINAL_DT
    hz = float(len(rows) - 1) / duration_s if duration_s > 0 else float("nan")

    has_actions = all("act" in r and r["act"] is not None for r in rows)
    acts = np.array([r["act"][:3] for r in rows], dtype=np.float64) if has_actions else None

    # Reconstruct the exact 13-dim policy frame as in src/ugv/policy.py + tools/l7/l7_run.py.
    ema = np.zeros(3, dtype=np.float64)
    prev_tag = np.zeros(3, dtype=np.float64)
    prev_valid = False
    dt_since_valid = 0.0
    last_valid_t = float(t[0])
    last_tag_t = float(t[0])
    prev_act = np.zeros(3, dtype=np.float64)

    frames = np.zeros((len(rows), len(CHANNEL_NAMES)), dtype=np.float64)
    raw_imu_ws = np.zeros(len(rows), dtype=np.float64)
    is_valid = np.zeros(len(rows), dtype=bool)
    is_tag = np.zeros(len(rows), dtype=bool)
    is_dr = np.zeros(len(rows), dtype=bool)
    raw_tag_ages = np.zeros(len(rows), dtype=np.float64)
    valid_ts: list[float] = []
    tag_ts: list[float] = []
    near_total = 0
    near_tag = 0

    for i, r in enumerate(rows):
        src = r.get("src", "none")
        v = src != "none"
        tg = src == "tag"
        is_valid[i] = v
        is_tag[i] = tg
        is_dr[i] = src == "dr"

        if v:
            last_valid_t = float(r["t"])
            valid_ts.append(float(r["t"]))
        if tg:
            last_tag_t = float(r["t"])
            tag_ts.append(float(r["t"]))

        tag_age = max(0.0, float(r["t"]) - last_valid_t)
        raw_tag_ages[i] = max(0.0, float(r["t"]) - last_tag_t)

        est = r.get("est")
        if est is not None and math.hypot(float(est[0]), float(est[1])) < NEAR_GOAL_DIST_M:
            near_total += 1
            if tg:
                near_tag += 1

        pose = r.get("pose")
        if v and pose is not None:
            tx, ty, tyaw = float(pose[0]), float(pose[1]), float(pose[2])
        else:
            tx, ty, tyaw = 0.0, 0.0, 0.0
        tag = np.array([tx, ty, tyaw], dtype=np.float64)

        dt_since_valid += NOMINAL_DT
        if v:
            if prev_valid and dt_since_valid <= 0.5:
                diff = tag - prev_tag
                diff[2] = _wrap(diff[2])
                diff /= max(dt_since_valid, NOMINAL_DT)
                ema = 0.5 * ema + 0.5 * diff
            else:
                ema = np.zeros(3, dtype=np.float64)
            prev_tag = tag
            prev_valid = True
            dt_since_valid = 0.0
        else:
            ema = ema * math.exp(-NOMINAL_DT / EMA_TAU)

        imu = r.get("imu") or [0.0, 0.0]
        imu_yaw = _wrap(float(imu[0])) if imu[0] is not None else 0.0
        # Use imu_w_fed when logged (e.g. gyro-zero / gyro-lp experiments), else raw imu[1].
        raw_imu_w = float(imu[1]) if imu[1] is not None else 0.0
        imu_w = float(r["imu_w_fed"]) if r.get("imu_w_fed") is not None else raw_imu_w
        raw_imu_ws[i] = raw_imu_w

        frames[i, 0:3] = tag
        frames[i, 3] = min(tag_age, 5.0)
        frames[i, 4] = 1.0 if v else 0.0
        frames[i, 5] = imu_w
        frames[i, 6] = imu_yaw
        frames[i, 7:10] = prev_act
        frames[i, 10:13] = ema

        if has_actions and acts is not None:
            prev_act = acts[i]

    channel_stats = {name: _quantiles(frames[:, idx]) for idx, name in enumerate(CHANNEL_NAMES)}

    act_stats: dict[str, dict[str, float]] = {}
    if has_actions and acts is not None:
        for j, name in enumerate(ACTION_NAMES):
            act_stats[name] = {
                **_quantiles(acts[:, j]),
                "sat": float(np.mean(np.abs(acts[:, j]) > 0.8)),
                "flip_rate": _sign_flip_rate(acts[:, j], duration_s),
            }

    iv_arr = np.diff(valid_ts) if len(valid_ts) > 1 else np.array([], dtype=np.float64)
    it_arr = np.diff(tag_ts) if len(tag_ts) > 1 else np.array([], dtype=np.float64)
    loop_ms_vals = [
        float(r["loop_ms"]) for r in rows if "loop_ms" in r and r["loop_ms"] is not None
    ]
    loop_ms_arr = np.array(loop_ms_vals, dtype=np.float64)

    policy_meta = meta.get("policy") or {}
    mode = (
        (meta.get("args") or {}).get("mode")
        or summary.get("mode")
        or ("policy" if has_actions else "unknown")
    )

    return {
        "run_id": meta.get("run_id") or run_dir.name,
        "mode": mode,
        "n_ticks": len(rows),
        "duration_s": duration_s,
        "hz": hz,
        "dt": _quantiles(dt_arr),
        "loop_ms": _quantiles(loop_ms_arr),
        "has_actions": has_actions,
        "policy_sha": policy_meta.get("sha"),
        "channels": channel_stats,
        "raw_imu_w_std": float(np.std(raw_imu_ws)),
        "actions": act_stats,
        "valid_rate": float(np.mean(is_valid)),
        "tag_rate": float(np.mean(is_tag)),
        "dr_rate": float(np.mean(is_dr)),
        "raw_tag_age": _quantiles(raw_tag_ages),
        "inter_valid_dt": _quantiles(iv_arr),
        "inter_tag_dt": _quantiles(it_arr),
        "near_goal_ticks": near_total,
        "near_goal_tag_rate": float(near_tag) / near_total
        if near_total >= MIN_NEAR_GOAL_TICKS
        else float("nan"),
    }


def _sim_profile_metrics(profile_path: Path) -> dict[str, Any]:
    prof = json.loads(profile_path.read_text())
    dz = float(prof["deadzone"])
    mean_coef = np.mean([p["coefficients"] for p in prof["posterior_particles"]], axis=0)
    p0 = prof["posterior_particles"][0]
    p0_coef = np.asarray(p0["coefficients"], dtype=np.float64)
    p0_dz = float(p0["deadzone"])

    def eval_plant(coef: np.ndarray, deadzone: float) -> dict[str, float]:
        # rl/sim.py mapping: act[1]=-1 -> physical fwd, act[0]=+1 -> physical left, act[2]=+1 -> ccw
        v_fwd = _wheel_features_np(_mix_np(np.array([0.0, -1.0, 0.0])), deadzone) @ coef
        v_left = _wheel_features_np(_mix_np(np.array([1.0, 0.0, 0.0])), deadzone) @ coef
        v_ccw = _wheel_features_np(_mix_np(np.array([0.0, 0.0, 1.0])), deadzone) @ coef
        return {
            "deadzone": deadzone,
            "fwd_gain": float(abs(v_fwd[1])),
            "left_gain": float(abs(v_left[0])),
            "ccw_full_w": float(abs(v_ccw[2])),
            "ccw_gain_per_duty": float(abs(v_ccw[2])) / (HALF_BASE * 4.0),
            "ccw_trans_coupling": float(math.hypot(float(v_ccw[0]), float(v_ccw[1]))),
        }

    return {
        "mean": eval_plant(mean_coef, dz),
        "p0": eval_plant(p0_coef, p0_dz),
    }


def _load_probe_metrics(probe_dir: Path) -> dict[str, Any]:
    fit_path = probe_dir / "fit.json"
    if fit_path.is_file():
        data = json.loads(fit_path.read_text())
    else:
        raise FileNotFoundError(f"Missing fit.json in probe dir {probe_dir}")

    pulses = data.get("pulses", [])
    fit = data.get("fit", {})

    # Infer effective deadzone from both polyfit and observed stalled pulses.
    axes_out: dict[str, dict[str, Any]] = {}
    for axis in ("fwd", "left", "ccw"):
        ax_fit = fit.get(axis, {})
        key = "v_ss" if axis in ("fwd", "left") else "yaw_rate_dps_ss"
        move_thresh = 0.05 if axis in ("fwd", "left") else 15.0
        stalled = [
            abs(float(p["duty"]))
            for p in pulses
            if p.get("axis") == axis and p.get(key) is not None and abs(float(p[key])) < move_thresh
        ]
        moved = [
            (abs(float(p["duty"])), abs(float(p[key])))
            for p in pulses
            if p.get("axis") == axis
            and p.get(key) is not None
            and abs(float(p[key])) >= move_thresh
        ]
        max_stall_duty = max(stalled) if stalled else 0.0
        raw_dz = float(ax_fit["deadzone"]) if "deadzone" in ax_fit else float("nan")
        # When sparse/noisy pulses make linear fit intercept negative (e.g. left), fall back to stall bound.
        eff_dz = raw_dz if (math.isfinite(raw_dz) and 0.0 <= raw_dz <= 0.5) else max_stall_duty

        raw_gain = float(ax_fit["gain"]) if "gain" in ax_fit else float("nan")
        if axis == "ccw" and math.isfinite(raw_gain):
            raw_gain_si = math.radians(raw_gain)  # deg/s -> rad/s per duty
        else:
            raw_gain_si = raw_gain

        # Robust median secant gain above effective deadzone if polyfit was ill-conditioned.
        if (not math.isfinite(raw_dz) or raw_dz < 0.0) and moved:
            secants = [
                (math.radians(v) if axis == "ccw" else v) / max(u - eff_dz, 0.05)
                for u, v in moved
                if u > eff_dz + 0.02
            ]
            eff_gain = float(np.median(secants)) if secants else raw_gain_si
        else:
            eff_gain = raw_gain_si

        axes_out[axis] = {
            "raw_gain_si": raw_gain_si,
            "eff_gain_si": eff_gain,
            "raw_deadzone": raw_dz,
            "eff_deadzone": eff_dz,
            "max_stall_duty": max_stall_duty,
            "n_moved": len(moved),
        }
    return {"probe_dir": str(probe_dir), "axes": axes_out}


def _check_l6_gate(l6_dir: Path, weights_sha: str | None) -> dict[str, Any]:
    if not weights_sha:
        return {"status": "SKIP", "reason": "no weights sha for run"}
    if not l6_dir.is_dir():
        return {"status": "FAIL", "sha": weights_sha, "reason": f"missing L6 dir {l6_dir}"}

    matches: list[dict[str, Any]] = []
    for p in sorted(l6_dir.rglob("*.json")):
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if (
            d.get("sha") == weights_sha
            and ("success" in d)
            and (d.get("bridge") == "gazebo" or "gazebo" in p.as_posix())
        ):
            matches.append(
                {
                    "file": str(p),
                    "success": int(d.get("success", 0)),
                    "episodes": int(d.get("episodes", 0)),
                    "level": d.get("level"),
                }
            )
    passing = [m for m in matches if m["success"] > 0]
    if passing:
        return {"status": "PASS", "sha": weights_sha, "matches": matches}
    best = max((m["success"] for m in matches), default=0)
    return {
        "status": "FAIL",
        "sha": weights_sha,
        "matches": matches,
        "reason": f"no Gazebo L6 result with success>0 for sha={weights_sha} (found {len(matches)} files, max success={best})",
    }


def _discover_probe(run_dir: Path) -> Path | None:
    parent = run_dir.parent
    if not parent.is_dir():
        return None
    probes = sorted(
        p for p in parent.glob("*-probe-*") if p.is_dir() and (p / "fit.json").is_file()
    )
    return probes[-1] if probes else None


def evaluate_contract(
    run_dir: Path,
    sim_ref_path: Path = DEFAULT_SIM_REF,
    probe_dir: Path | None = None,
    profile_path: Path = DEFAULT_PROFILE,
    l6_dir: Path | None = DEFAULT_L6_DIR,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    sim = compute_sim_stats(sim_ref_path)
    real = extract_run_metrics(run_dir)

    checks: list[dict[str, Any]] = []

    def add_check(
        name: str, channel: str, value: float | str, limit: str, passed: bool, detail: str = ""
    ):
        checks.append(
            {
                "name": name,
                "channel": channel,
                "value": value,
                "limit": limit,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    def add_skip(name: str, channel: str, detail: str):
        checks.append(
            {
                "name": name,
                "channel": channel,
                "value": None,
                "limit": "N/A",
                "status": "SKIP",
                "detail": detail,
            }
        )

    # ---- 1. Gyro-rate channel (`imu_w`, dim 5) ----
    sim_iw = sim["channels"]["imu_w"]
    real_iw = real["channels"]["imu_w"]
    iw_std_lim = max(IMU_W_STD_FACTOR * sim_iw["std"], IMU_W_STD_FLOOR)
    add_check(
        "gyro_imu_w_std",
        "imu_w",
        round(real_iw["std"], 4),
        f"<= {iw_std_lim:.3f}",
        real_iw["std"] <= iw_std_lim,
        f"sim_std={sim_iw['std']:.4f}",
    )
    sim_iw_span = sim_iw["p95"] - sim_iw["p05"]
    real_iw_span = real_iw["p95"] - real_iw["p05"]
    iw_span_lim = max(IMU_W_SPAN_FACTOR * sim_iw_span, IMU_W_SPAN_FLOOR)
    add_check(
        "gyro_imu_w_span",
        "imu_w",
        round(real_iw_span, 4),
        f"<= {iw_span_lim:.3f}",
        real_iw_span <= iw_span_lim,
        f"sim_span={sim_iw_span:.4f}",
    )
    if real["raw_imu_w_std"] > 0.02 and real_iw["std"] < 0.005:
        add_check(
            "gyro_imu_w_zeroed",
            "imu_w",
            round(real_iw["std"], 4),
            ">= 0.005 when moving",
            False,
            f"raw_imu_w_std={real['raw_imu_w_std']:.4f} was masked/zeroed",
        )

    # ---- 2. Valid observation & raw camera-tag rate ----
    add_check(
        "valid_pose_rate",
        "valid",
        round(real["valid_rate"], 4),
        f">= {MIN_VALID_RATE:.2f}",
        real["valid_rate"] >= MIN_VALID_RATE,
        f"sim={sim['valid_rate']:.3f}",
    )
    add_check(
        "raw_tag_rate",
        "src_tag",
        round(real["tag_rate"], 4),
        f">= {MIN_TAG_RATE:.2f}",
        real["tag_rate"] >= MIN_TAG_RATE,
        f"dr_rate={real['dr_rate']:.3f}, sim={sim['tag_rate']:.3f}",
    )

    # ---- 3. Pose age & inter-valid dt ----
    sim_age_p95 = sim["channels"]["tag_age"]["p95"]
    real_age_p95 = real["channels"]["tag_age"]["p95"]
    age_lim = max(MAX_TAG_AGE_P95_FACTOR * sim_age_p95, MAX_TAG_AGE_P95_FLOOR)
    add_check(
        "tag_age_p95",
        "tag_age",
        round(real_age_p95, 4),
        f"<= {age_lim:.3f}",
        real_age_p95 <= age_lim,
        f"sim_p95={sim_age_p95:.3f}",
    )

    sim_iv_p95 = sim["inter_valid_dt"]["p95"]
    real_iv_p95 = real["inter_valid_dt"]["p95"]
    iv_lim = max(MAX_INTER_VALID_DT_P95_FACTOR * sim_iv_p95, MAX_INTER_VALID_DT_P95_FLOOR)
    add_check(
        "inter_valid_dt_p95",
        "valid_dt",
        round(real_iv_p95, 4) if math.isfinite(real_iv_p95) else float("nan"),
        f"<= {iv_lim:.3f}",
        math.isfinite(real_iv_p95) and real_iv_p95 <= iv_lim,
        f"sim_p95={sim_iv_p95:.3f}",
    )

    sim_it_p95 = sim["inter_tag_dt"]["p95"]
    real_it_p95 = real["inter_tag_dt"]["p95"]
    it_lim = max(MAX_INTER_TAG_DT_P95_FACTOR * sim_it_p95, MAX_INTER_TAG_DT_P95_FLOOR)
    add_check(
        "inter_tag_dt_p95",
        "tag_dt",
        round(real_it_p95, 4) if math.isfinite(real_it_p95) else float("nan"),
        f"<= {it_lim:.3f}",
        math.isfinite(real_it_p95) and real_it_p95 <= it_lim,
        f"sim_p95={sim_it_p95:.3f}",
    )

    # ---- 4. Remaining 13-dim input channel drift checks ----
    for ch in CHANNEL_NAMES:
        if ch in ("tag_age", "valid", "imu_w"):
            continue
        if ch.startswith("prev_a") and not real["has_actions"]:
            add_skip(f"frame_{ch}", ch, f"no act logged (mode={real['mode']})")
            continue
        s_ch = sim["channels"][ch]
        r_ch = real["channels"][ch]
        mean_lim = max(FRAME_MEAN_STD_MULT * s_ch["std"], FRAME_MEAN_FLOOR)
        if ch.startswith("prev_a"):
            std_lim = max(PREV_ACT_STD_FACTOR * s_ch["std"], PREV_ACT_STD_FLOOR)
        else:
            std_lim = max(FRAME_STD_MULT * s_ch["std"], FRAME_STD_FLOOR)
        mean_err = abs(r_ch["mean"] - s_ch["mean"])
        ok = (mean_err <= mean_lim) and (r_ch["std"] <= std_lim)
        add_check(
            f"frame_{ch}",
            ch,
            f"mean={r_ch['mean']:+.3f},std={r_ch['std']:.3f}",
            f"|dm|<={mean_lim:.2f},std<={std_lim:.2f}",
            ok,
            f"sim=({s_ch['mean']:+.3f},{s_ch['std']:.3f})",
        )

    # ---- 5. Action saturation & sign-flip rate ----
    if real["has_actions"]:
        for name in ACTION_NAMES:
            s_act = sim["actions"][name]
            r_act = real["actions"][name]
            sat_lim = max(s_act["sat"] + MAX_ACT_SAT_ADD, MAX_ACT_SAT_ABS)
            add_check(
                f"action_sat_{name}",
                name,
                round(r_act["sat"], 4),
                f"<= {sat_lim:.3f}",
                r_act["sat"] <= sat_lim,
                f"sim_sat={s_act['sat']:.4f}",
            )
            if name in ("act_0", "act_1"):
                flip_lim = max(MAX_TRANS_FLIP_FACTOR * s_act["flip_p95"], MAX_TRANS_FLIP_FLOOR)
            else:
                flip_lim = max(MAX_TURN_FLIP_FACTOR * s_act["flip_p95"], MAX_TURN_FLIP_FLOOR)
            add_check(
                f"action_flips_{name}",
                name,
                round(r_act["flip_rate"], 3),
                f"<= {flip_lim:.2f}/s",
                r_act["flip_rate"] <= flip_lim,
                f"sim_p95={s_act['flip_p95']:.2f}/s",
            )
    else:
        for name in ACTION_NAMES:
            add_skip(f"action_sat_{name}", name, f"no act logged (mode={real['mode']})")
            add_skip(f"action_flips_{name}", name, f"no act logged (mode={real['mode']})")

    # ---- 6. Control-loop rate & jitter ----
    add_check(
        "loop_rate_hz",
        "loop_hz",
        round(real["hz"], 2),
        f">= {MIN_LOOP_HZ:.1f}",
        real["hz"] >= MIN_LOOP_HZ,
        "nominal=50.0",
    )
    add_check(
        "loop_dt_p95",
        "loop_dt",
        round(real["dt"]["p95"], 4),
        f"<= {MAX_LOOP_DT_P95:.3f}",
        real["dt"]["p95"] <= MAX_LOOP_DT_P95,
        f"std={real['dt']['std']:.4f}",
    )
    if math.isfinite(real["loop_ms"]["p95"]):
        add_check(
            "loop_ms_p95",
            "loop_ms",
            round(real["loop_ms"]["p95"], 2),
            f"<= {MAX_LOOP_MS_P95:.1f}",
            real["loop_ms"]["p95"] <= MAX_LOOP_MS_P95,
            f"max={real['loop_ms']['max'] if 'max' in real['loop_ms'] else real['loop_ms']['p95']:.2f}",
        )

    # ---- 7. Near-goal vertical FOV check ----
    if math.isfinite(real["near_goal_tag_rate"]):
        add_check(
            "near_goal_tag_rate",
            "near_tag",
            round(real["near_goal_tag_rate"], 4),
            f">= {MIN_NEAR_GOAL_TAG_RATE:.2f}",
            real["near_goal_tag_rate"] >= MIN_NEAR_GOAL_TAG_RATE,
            f"ticks={real['near_goal_ticks']}, sim={sim['near_goal_tag_rate']:.3f}",
        )
    else:
        add_skip(
            "near_goal_tag_rate",
            "near_tag",
            f"only {real['near_goal_ticks']} ticks < {NEAR_GOAL_DIST_M}m (need >= {MIN_NEAR_GOAL_TICKS})",
        )

    # ---- 8. Plant checks (probe vs sim Bayesian profile) ----
    plant_info: dict[str, Any] = {}
    if probe_dir is not None and profile_path.is_file():
        probe = _load_probe_metrics(probe_dir)
        prof_metrics = _sim_profile_metrics(profile_path)
        plant_info = {"probe": probe, "sim_profile": prof_metrics}
        sim_dz = prof_metrics["mean"]["deadzone"]
        max_dz_diff = max(abs(ax["eff_deadzone"] - sim_dz) for ax in probe["axes"].values())
        add_check(
            "plant_deadzone_mismatch",
            "plant_dz",
            round(max_dz_diff, 4),
            f"<= {MAX_DEADZONE_DIFF:.2f}",
            max_dz_diff <= MAX_DEADZONE_DIFF,
            (
                f"real=({probe['axes']['fwd']['eff_deadzone']:.3f},"
                f"{probe['axes']['left']['eff_deadzone']:.3f},"
                f"{probe['axes']['ccw']['eff_deadzone']:.3f}) vs sim={sim_dz:.3f}"
            ),
        )
        # Gain check (fwd & ccw against sim p0/mean)
        sim_p0 = prof_metrics["p0"]
        fwd_r = probe["axes"]["fwd"]["eff_gain_si"]
        fwd_s = sim_p0["fwd_gain"]
        fwd_rel = abs(fwd_r - fwd_s) / max(fwd_s, 0.1)
        add_check(
            "plant_fwd_gain_mismatch",
            "plant_fwd",
            round(fwd_r, 3),
            f"rel<= {MAX_GAIN_REL_ERR:.2f}",
            fwd_rel <= MAX_GAIN_REL_ERR,
            f"sim_p0={fwd_s:.3f} m/s/duty (rel_err={fwd_rel:.2f})",
        )
        ccw_r = probe["axes"]["ccw"]["eff_gain_si"]
        ccw_s = sim_p0["ccw_gain_per_duty"]
        ccw_rel = abs(ccw_r - ccw_s) / max(ccw_s, 0.1)
        add_check(
            "plant_ccw_gain_mismatch",
            "plant_ccw",
            round(ccw_r, 3),
            f"rel<= {MAX_GAIN_REL_ERR:.2f}",
            ccw_rel <= MAX_GAIN_REL_ERR,
            f"sim_p0={ccw_s:.3f} rad/s/duty (rel_err={ccw_rel:.2f})",
        )
        add_check(
            "plant_sim_p0_turn_coupling",
            "plant_coup",
            round(sim_p0["ccw_trans_coupling"], 3),
            f"<= {MAX_SIM_TURN_COUPLING_MPS:.2f} m/s",
            sim_p0["ccw_trans_coupling"] <= MAX_SIM_TURN_COUPLING_MPS,
            f"p0 full turn={sim_p0['ccw_full_w']:.2f} rad/s couples {sim_p0['ccw_trans_coupling']:.2f} m/s trans",
        )
    else:
        add_skip("plant_checks", "plant", "no probe dir or sim profile provided")

    # ---- 9. Gazebo L6 gate check ----
    weights_sha = real["policy_sha"]
    if weights_path is not None and weights_path.is_file():
        z_w = np.load(weights_path, allow_pickle=True)
        if "meta_sha" in z_w.files:
            weights_sha = str(z_w["meta_sha"])
    if l6_dir is not None:
        l6_res = _check_l6_gate(l6_dir, weights_sha)
        if l6_res["status"] == "SKIP":
            add_skip("gazebo_l6_gate", "l6_gate", l6_res["reason"])
        else:
            add_check(
                "gazebo_l6_gate",
                "l6_gate",
                l6_res["status"],
                "success > 0",
                l6_res["status"] == "PASS",
                l6_res.get("reason") or f"sha={weights_sha} passed",
            )
    else:
        l6_res = {"status": "SKIP", "reason": "no l6_dir provided"}
        add_skip("gazebo_l6_gate", "l6_gate", l6_res["reason"])

    failed = [c for c in checks if c["status"] == "FAIL"]
    return {
        "run_id": real["run_id"],
        "mode": real["mode"],
        "passed": len(failed) == 0,
        "failed_checks": [c["name"] for c in failed],
        "failed_channels": sorted({c["channel"] for c in failed}),
        "sim_meta": sim["meta"],
        "sim": sim,
        "real": real,
        "plant": plant_info,
        "l6_gate": l6_res,
        "checks": checks,
    }


def format_table(report: dict[str, Any]) -> str:
    sim = report["sim"]
    real = report["real"]
    lines: list[str] = []
    lines.append(
        f"=== L7 Contract Audit: {report['run_id']} "
        f"(mode={report['mode']}, ticks={real['n_ticks']}, dur={real['duration_s']:.2f}s, "
        f"hz={real['hz']:.1f}) ==="
    )
    lines.append(
        f"{'channel':<10} | {'real_mean':>9} {'real_std':>8} {'real_p05':>8} {'real_p95':>8} | "
        f"{'sim_mean':>9} {'sim_std':>8} {'sim_p05':>8} {'sim_p95':>8} | extra"
    )
    lines.append("-" * 105)
    for ch in CHANNEL_NAMES:
        r = real["channels"][ch]
        s = sim["channels"][ch]
        extra = ""
        if ch == "valid":
            extra = f"tag={real['tag_rate']:.3f} dr={real['dr_rate']:.3f}"
        elif ch == "tag_age":
            extra = f"raw_tag_p95={real['raw_tag_age']['p95']:.3f}s"
        lines.append(
            f"{ch:<10} | {r['mean']:+9.4f} {r['std']:8.4f} {r['p05']:+8.4f} {r['p95']:+8.4f} | "
            f"{s['mean']:+9.4f} {s['std']:8.4f} {s['p05']:+8.4f} {s['p95']:+8.4f} | {extra}"
        )
    lines.append("-" * 105)
    for name in ACTION_NAMES:
        s = sim["actions"][name]
        if real["has_actions"]:
            r = real["actions"][name]
            extra = (
                f"sat={r['sat']:.3f}(sim {s['sat']:.4f}) "
                f"flips={r['flip_rate']:.2f}/s(sim_p95 {s['flip_p95']:.2f})"
            )
            lines.append(
                f"{name:<10} | {r['mean']:+9.4f} {r['std']:8.4f} {r['p05']:+8.4f} {r['p95']:+8.4f} | "
                f"{s['mean']:+9.4f} {s['std']:8.4f} {s['p05']:+8.4f} {s['p95']:+8.4f} | {extra}"
            )
        else:
            lines.append(
                f"{name:<10} | {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} | "
                f"{s['mean']:+9.4f} {s['std']:8.4f} {s['p05']:+8.4f} {s['p95']:+8.4f} | mode={real['mode']} (no act)"
            )
    lines.append("-" * 105)
    lines.append(
        f"timing     : hz={real['hz']:.2f} dt_p95={real['dt']['p95']:.4f}s "
        f"loop_ms_p95={real['loop_ms']['p95']:.2f}ms "
        f"iv_dt_p95={real['inter_valid_dt']['p95']:.3f}s(sim {sim['inter_valid_dt']['p95']:.3f}s) "
        f"it_dt_p95={real['inter_tag_dt']['p95']:.3f}s"
    )
    lines.append("checks:")
    for c in report["checks"]:
        lines.append(
            f"  [{c['status']:<4}] {c['name']:<26} ({c['channel']:<10}): "
            f"val={c['value']} limit={c['limit']}  {c['detail']}"
        )
    status = "PASS" if report["passed"] else "FAIL"
    lines.append(
        f"OVERALL: {status} ({len(report['failed_checks'])} failed: "
        f"{', '.join(report['failed_checks']) or 'none'})"
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="L7 run directory (contains ticks.jsonl)")
    ap.add_argument("--sim-ref", type=Path, default=DEFAULT_SIM_REF)
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="Probe run dir (auto-discovers *-probe-* if omitted)",
    )
    ap.add_argument("--no-probe", action="store_true", help="Skip plant probe comparison")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--l6-dir", type=Path, default=DEFAULT_L6_DIR)
    ap.add_argument("--no-l6", action="store_true", help="Skip Gazebo L6 gate check")
    ap.add_argument(
        "--weights", type=Path, default=None, help="Optional .npz weights override for L6 SHA check"
    )
    ap.add_argument("--json", type=Path, default=None, help="Write full JSON report to this path")
    ap.add_argument("--json-only", action="store_true", help="Print only JSON to stdout")
    args = ap.parse_args()

    probe_dir = None if args.no_probe else (args.probe or _discover_probe(args.run_dir))
    l6_dir = None if args.no_l6 else args.l6_dir

    report = evaluate_contract(
        run_dir=args.run_dir,
        sim_ref_path=args.sim_ref,
        probe_dir=probe_dir,
        profile_path=args.profile,
        l6_dir=l6_dir,
        weights_path=args.weights,
    )

    payload = json.dumps(report, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")

    if args.json_only:
        print(payload)
    else:
        print(format_table(report))
        print("\n=== JSON ===")
        print(payload)

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
