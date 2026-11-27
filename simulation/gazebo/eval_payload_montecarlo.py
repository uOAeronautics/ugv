"""Headless CPU-only Monte Carlo open-loop evaluation of payload_car prior samples in Gazebo.

Runs N >= 32 prior-sampled worlds (generated via generate.py --car payload --sample-seed N)
through open-loop forward, strafe-left, and yaw-CCW step responses (mirroring check_mecanum_car.py
and the L7 probe-axes protocol) using isolated gz sim -s instances with Sensors stripped (no GPU).

Results are PRIOR SAMPLES, NOT a calibrated car. Saves summary percentiles and per-sample
records with run IDs to simulation/gazebo/payload_montecarlo.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate  # noqa: E402
import lockstep_env as le  # noqa: E402
from payload_prior import PRIOR_STATUS, load_prior  # noqa: E402

DEFAULT_OUT = HERE / "payload_montecarlo.json"

# Open-loop step protocol parameters (DT = 0.02 s per lockstep tick).
SETTLE_S = 0.50
DRIVE_S = 1.50
BRAKE_FWD_S = 1.00
BRAKE_OTHER_S = 0.60
STEADY_WINDOW_S = 0.40
ACCEL_WINDOW_TICKS = 10  # 0.20 s window for smoothed initial acceleration
DRIVE_DUTY = 1.0

# Physical sanity bounds (no tipping, no numerical blow-up).
MAX_TILT_RAD = math.radians(25.0)
MIN_Z_M = -0.015
MAX_Z_M = 0.040
MAX_LIN_SPEED_MPS = 5.0
MAX_YAW_RATE_RADPS = 40.0


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _body_delta(p0: tuple, p1: tuple) -> tuple[float, float, float]:
    dx = p1[1] - p0[1]
    dy = p1[2] - p0[2]
    yaw0 = p0[6]
    dx_b = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    dy_b = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
    dyaw = _wrap(p1[6] - yaw0)
    return dx_b, dy_b, dyaw


def _run_axis_step(
    gz: le.Gazebo,
    duties: list[float],
    drive_s: float,
    brake_s: float,
) -> tuple[list[tuple], list[tuple]]:
    drive_ticks = int(round(drive_s / le.DT))
    brake_ticks = int(round(brake_s / le.DT))
    drive_trace = [gz.full_pose()]
    for _ in range(drive_ticks):
        gz.step(duties)
        drive_trace.append(gz.full_pose())
    brake_trace = [drive_trace[-1]]
    for _ in range(brake_ticks):
        gz.step([0.0, 0.0, 0.0, 0.0])
        brake_trace.append(gz.full_pose())
    return drive_trace, brake_trace


def _body_velocities(trace: list[tuple]) -> tuple[list[float], list[float], list[float]]:
    vx_list, vy_list, wz_list = [], [], []
    for p0, p1 in zip(trace[:-1], trace[1:], strict=True):
        dt = p1[0] - p0[0]
        if dt <= 1e-6:
            dt = le.DT
        dx_b, dy_b, dyaw = _body_delta(p0, p1)
        vx_list.append(dx_b / dt)
        vy_list.append(dy_b / dt)
        wz_list.append(dyaw / dt)
    return vx_list, vy_list, wz_list


def evaluate_seed(seed: int, batch_tag: str) -> dict:
    run_id = f"mc-payload-{batch_tag}-seed{seed:03d}"
    t_wall0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"ugv-mc-s{seed}-") as tmp_dir:
        tmp = Path(tmp_dir)
        sdf_path = tmp / f"payload_s{seed}.sdf"
        headless_path = tmp / f"payload_s{seed}_headless.sdf"
        log_path = tmp / f"payload_s{seed}.log"

        args = type(
            "Args",
            (),
            dict(
                car="payload",
                drone=False,
                room=False,
                marker_layout="real",
                chassis_mass_kg=None,
                wheel_mass_kg=None,
                require_measured_physics=False,
                motor_profile=None,
                nominal_motors=False,
                sample_seed=seed,
                _return_sample=True,
            ),
        )()
        root, _, sample = generate.build_world(args)
        ET.indent(root)
        ET.ElementTree(root).write(sdf_path, encoding="utf-8", xml_declaration=True)
        le.headless_world(sdf_path, headless_path)

        hand_sign = int(sample["contact"]["mecanum_hand_sign"])
        # For top-X CAD rollers (hand_sign == -1): FL=+1, FR=-1, RL=-1, RR=+1 strafes +Y (left).
        # For Freenove bottom-X (hand_sign == +1): FL=-1, FR=+1, RL=+1, RR=-1 strafes +Y (left).
        d_fwd = [DRIVE_DUTY, DRIVE_DUTY, DRIVE_DUTY, DRIVE_DUTY]
        d_left = [
            -hand_sign * DRIVE_DUTY,
            hand_sign * DRIVE_DUTY,
            hand_sign * DRIVE_DUTY,
            -hand_sign * DRIVE_DUTY,
        ]
        d_ccw = [-DRIVE_DUTY, DRIVE_DUTY, -DRIVE_DUTY, DRIVE_DUTY]

        all_poses: list[tuple] = []
        gz = None
        error: str | None = None
        metrics: dict = {}
        try:
            gz = le.Gazebo(headless_path, log_path)
            for _ in range(int(round(SETTLE_S / le.DT))):
                gz.step([0.0, 0.0, 0.0, 0.0])
                all_poses.append(gz.full_pose())

            # 1. Forward step + brake
            fwd_drv, fwd_brk = _run_axis_step(gz, d_fwd, DRIVE_S, BRAKE_FWD_S)
            all_poses.extend(fwd_drv)
            all_poses.extend(fwd_brk)

            # 2. Strafe left step
            gz.reset_pose(0.0, 0.0, 0.0)
            left_drv, left_brk = _run_axis_step(gz, d_left, DRIVE_S, BRAKE_OTHER_S)
            all_poses.extend(left_drv)
            all_poses.extend(left_brk)

            # 3. Yaw CCW step
            gz.reset_pose(0.0, 0.0, 0.0)
            ccw_drv, ccw_brk = _run_axis_step(gz, d_ccw, DRIVE_S, BRAKE_OTHER_S)
            all_poses.extend(ccw_drv)
            all_poses.extend(ccw_brk)

            vx_f, _, _ = _body_velocities(fwd_drv)
            _, vy_l, _ = _body_velocities(left_drv)
            _, _, wz_c = _body_velocities(ccw_drv)

            steady_n = max(1, int(round(STEADY_WINDOW_S / le.DT)))
            top_speed_fwd = float(np.mean(vx_f[-steady_n:]))
            top_speed_strafe = float(np.mean(vy_l[-steady_n:]))
            yaw_rate_ccw = float(np.mean(wz_c[-steady_n:]))

            # Peak smoothed acceleration over 0.20 s window during initial 0.8 s spin-up.
            spin_ticks = min(len(vx_f), int(round(0.80 / le.DT)))
            v_series = [0.0, *vx_f[:spin_ticks]]
            win = min(ACCEL_WINDOW_TICKS, len(v_series) - 1)
            accels = [
                (v_series[i + win] - v_series[i]) / (win * le.DT)
                for i in range(len(v_series) - win)
            ]
            accel_fwd = float(max(accels)) if accels else 0.0

            # Stop distance during forward brake phase (body +X displacement).
            vx_brk, _, _ = _body_velocities(fwd_brk)
            stop_idx = len(fwd_brk) - 1
            for idx, v in enumerate(vx_brk):
                if abs(v) < 0.005:
                    stop_idx = idx + 1
                    break
            stop_dx, _, _ = _body_delta(fwd_brk[0], fwd_brk[stop_idx])
            fwd_total_dx, _, _ = _body_delta(fwd_drv[0], fwd_drv[-1])
            _, left_total_dy, _ = _body_delta(left_drv[0], left_drv[-1])
            ccw_total_dyaw = sum(wz_c) * le.DT

            strafe_fwd_ratio = (
                float(top_speed_strafe / top_speed_fwd) if top_speed_fwd > 1e-6 else float("nan")
            )

            z_vals = [p[3] for p in all_poses]
            roll_abs = [abs(p[4]) for p in all_poses]
            pitch_abs = [abs(p[5]) for p in all_poses]
            all_lin = [abs(v) for v in (*vx_f, *vy_l)]
            all_yaw = [abs(w) for w in wz_c]

            all_finite = all(math.isfinite(val) for p in all_poses for val in p) and all(
                math.isfinite(val)
                for val in (
                    top_speed_fwd,
                    accel_fwd,
                    top_speed_strafe,
                    strafe_fwd_ratio,
                    yaw_rate_ccw,
                    stop_dx,
                )
            )
            max_roll = max(roll_abs) if roll_abs else float("nan")
            max_pitch = max(pitch_abs) if pitch_abs else float("nan")
            min_z = min(z_vals) if z_vals else float("nan")
            max_z = max(z_vals) if z_vals else float("nan")
            max_lin = max(all_lin) if all_lin else float("nan")
            max_wz = max(all_yaw) if all_yaw else float("nan")

            sanity_reasons: list[str] = []
            if not all_finite:
                sanity_reasons.append("non_finite")
            if max_roll >= MAX_TILT_RAD or max_pitch >= MAX_TILT_RAD:
                sanity_reasons.append("tipped")
            if min_z < MIN_Z_M or max_z > MAX_Z_M:
                sanity_reasons.append("z_out_of_bounds")
            if max_lin >= MAX_LIN_SPEED_MPS or max_wz >= MAX_YAW_RATE_RADPS:
                sanity_reasons.append("velocity_blowup")
            if top_speed_fwd <= 0.02 or top_speed_strafe <= 0.01 or yaw_rate_ccw <= 0.1:
                sanity_reasons.append("insufficient_or_wrong_sign_motion")
            if stop_dx < -0.005:
                sanity_reasons.append("negative_stop_distance")

            metrics = {
                "top_speed_fwd_mps": round(top_speed_fwd, 5),
                "accel_fwd_mps2": round(accel_fwd, 5),
                "top_speed_strafe_mps": round(top_speed_strafe, 5),
                "strafe_forward_ratio": round(strafe_fwd_ratio, 5)
                if math.isfinite(strafe_fwd_ratio)
                else None,
                "yaw_rate_radps": round(yaw_rate_ccw, 5),
                "stop_distance_m": round(stop_dx, 5),
                "fwd_displacement_m": round(fwd_total_dx, 5),
                "strafe_displacement_m": round(left_total_dy, 5),
                "ccw_yaw_rad": round(ccw_total_dyaw, 5),
                "max_abs_roll_deg": round(math.degrees(max_roll), 3),
                "max_abs_pitch_deg": round(math.degrees(max_pitch), 3),
                "min_z_m": round(min_z, 5),
                "max_z_m": round(max_z, 5),
                "sane": len(sanity_reasons) == 0,
                "sanity_reasons": sanity_reasons,
            }
        except Exception as exc:
            error = str(exc)
            metrics = {"sane": False, "sanity_reasons": [f"runtime_error: {error}"]}
        finally:
            if gz is not None:
                gz.close()

    wall_s = round(time.monotonic() - t_wall0, 2)
    return {
        "run_id": run_id,
        "seed": seed,
        "wall_s": wall_s,
        "error": error,
        "sampled": {
            "total_mass_kg": sample["mass"]["total_mass_kg"],
            "assembly_com_m": sample["mass"]["assembly_com_m"],
            "motor_variant_id": sample["motor"]["variant_id"],
            "gear_ratio": sample["motor"]["gear_ratio"],
            "rated_voltage_v": sample["motor"]["rated_voltage_v"],
            "effective_voltage_v": sample["motor"]["effective_voltage_v"],
            "no_load_speed_radps": sample["motor"]["no_load_speed_radps"],
            "stall_torque_nm": sample["motor"]["stall_torque_nm"],
            "wheel_deadzone": sample["motor"]["wheel_deadzone"][0],
            "wheel_duty_exponent": sample["motor"]["wheel_duty_exponent"][0],
            "wheel_drive_tau_s": sample["motor"]["wheel_drive_tau_s"][0],
            "wheel_brake_tau_s": sample["motor"]["wheel_brake_tau_s"][0],
            "roller_mu": sample["contact"]["roller_mu"],
            "roller_bearing_damping_nm_s_per_rad": sample["contact"][
                "roller_bearing_damping_nm_s_per_rad"
            ],
        },
        "metrics": metrics,
    }


def _percentiles(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "min": round(float(arr.min()), 5),
        "p05": round(float(np.percentile(arr, 5)), 5),
        "p50": round(float(np.percentile(arr, 50)), 5),
        "p95": round(float(np.percentile(arr, 95)), 5),
        "max": round(float(arr.max()), 5),
        "mean": round(float(arr.mean()), 5),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=36, help="Number of Monte Carlo seeds (>= 32)")
    ap.add_argument("--seed-start", type=int, default=0, help="Starting integer seed")
    ap.add_argument("--workers", type=int, default=6, help="Parallel headless Gazebo CPU workers")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = ap.parse_args(argv)
    if args.samples < 1:
        ap.error("--samples must be >= 1")

    prior = load_prior()
    batch_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    seeds = list(range(args.seed_start, args.seed_start + args.samples))
    print(
        f"PRIOR SAMPLE, not measured: running {len(seeds)} headless CPU Gazebo samples "
        f"(seeds {seeds[0]}..{seeds[-1]}, workers={args.workers})",
        flush=True,
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda s: evaluate_seed(s, batch_tag), seeds))
    total_wall_s = round(time.monotonic() - t0, 2)

    sane_recs = [r for r in records if r["metrics"].get("sane")]
    sanity_fraction = len(sane_recs) / len(records)

    metric_keys = (
        "top_speed_fwd_mps",
        "accel_fwd_mps2",
        "top_speed_strafe_mps",
        "strafe_forward_ratio",
        "yaw_rate_radps",
        "stop_distance_m",
    )
    summary_sane: dict[str, dict] = {}
    if sane_recs:
        summary_sane["total_mass_kg"] = _percentiles(
            [r["sampled"]["total_mass_kg"] for r in sane_recs]
        )
        for key in metric_keys:
            vals = [r["metrics"][key] for r in sane_recs if r["metrics"].get(key) is not None]
            if vals:
                summary_sane[key] = _percentiles(vals)

    report = {
        "status": PRIOR_STATUS,
        "warning": (
            "PRIOR SAMPLE, not measured. Open-loop Monte Carlo evaluation over unmeasured "
            "payload_car priors in headless Gazebo. Do NOT treat as a calibrated car or trust "
            "controllers tuned on these samples on hardware."
        ),
        "batch_run_id": f"mc-payload-{batch_tag}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "prior_file": "payload_prior.json",
        "prior_sha256": prior["_sha256"],
        "protocol": {
            "engine": "gz sim -s (CPU headless, Sensors plugin stripped, DART physics 1 ms step)",
            "dt_s": le.DT,
            "settle_s": SETTLE_S,
            "drive_s": DRIVE_S,
            "brake_fwd_s": BRAKE_FWD_S,
            "brake_other_s": BRAKE_OTHER_S,
            "steady_window_s": STEADY_WINDOW_S,
            "accel_window_s": round(ACCEL_WINDOW_TICKS * le.DT, 4),
            "drive_duty": DRIVE_DUTY,
            "sanity_criteria": {
                "max_abs_roll_deg": round(math.degrees(MAX_TILT_RAD), 2),
                "max_abs_pitch_deg": round(math.degrees(MAX_TILT_RAD), 2),
                "min_z_m": MIN_Z_M,
                "max_z_m": MAX_Z_M,
                "max_linear_speed_mps": MAX_LIN_SPEED_MPS,
                "max_yaw_rate_radps": MAX_YAW_RATE_RADPS,
                "min_fwd_speed_mps": 0.02,
                "min_strafe_speed_mps": 0.01,
                "min_yaw_rate_radps": 0.1,
            },
        },
        "n_samples": len(records),
        "n_sane": len(sane_recs),
        "sanity_fraction": round(sanity_fraction, 6),
        "total_wall_s": total_wall_s,
        "summary_sane": summary_sane,
        "samples": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nSaved {args.output} ({len(records)} samples, sane={len(sane_recs)}/{len(records)} "
        f"= {sanity_fraction:.1%}, wall={total_wall_s:.1f}s)"
    )
    print(json.dumps(summary_sane, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
