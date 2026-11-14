#!/usr/bin/env python3
"""Fit per-axis steady-state response, dynamics, coupling, and latency from an l7_run.py probe run.

Yaw rate comes from the 100 Hz gyro; translation from tag poses inside each pulse.
Prints one row per pulse and a constrained least-squares v = G * (|duty| - dz) fit (G >= 0, dz >= 0).
usage: fit_probe.py <run_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _exp_tau(t: np.ndarray, y: np.ndarray, y_ref: float, approach: bool) -> float | None:
    if len(t) < 3 or abs(y_ref) < 1e-6:
        return None
    u = y / y_ref
    if approach:
        m = (u >= 0.10) & (u <= 0.85)
        if m.sum() >= 3 and np.ptp(t[m]) > 0.01:
            z = -np.log(1.0 - u[m])
            slope = float(np.polyfit(t[m], z, 1)[0])
            if slope > 0.5:
                tau = 1.0 / slope
                if 0.002 <= tau <= 2.0:
                    return float(tau)
        i0 = np.where(u >= 0.10)[0]
        if len(i0):
            i1 = np.where((np.arange(len(u)) >= i0[0]) & (u >= 0.632))[0]
            if len(i1):
                dt = float(t[i1[0]] - t[i0[0]])
                tau = dt / 0.894
                if 0.002 <= tau <= 2.0:
                    return float(tau)
    else:
        m = (u >= 0.10) & (u <= 0.90)
        if m.sum() >= 3 and np.ptp(t[m]) > 0.01:
            z = -np.log(u[m])
            slope = float(np.polyfit(t[m], z, 1)[0])
            if slope > 0.5:
                tau = 1.0 / slope
                if 0.002 <= tau <= 2.0:
                    return float(tau)
        i0 = np.where(u <= 0.90)[0]
        if len(i0):
            i1 = np.where((np.arange(len(u)) >= i0[0]) & (u <= 0.368))[0]
            if len(i1):
                dt = float(t[i1[0]] - t[i0[0]])
                tau = dt / 0.894
                if 0.002 <= tau <= 2.0:
                    return float(tau)
    return None


def _fit_nonneg_deadzone(pts: list[tuple[float, float]], stall_thresh: float) -> dict | None:
    stalled = [(u, v) for u, v in pts if v < stall_thresh]
    mov = [(u, v) for u, v in pts if v >= stall_thresh]
    dz_lb = max((u for u, _ in stalled), default=0.0)
    if not mov:
        return None
    U = np.array([u for u, _ in mov], dtype=float)
    V = np.array([v for _, v in mov], dtype=float)
    if len(mov) >= 2 and np.ptp(U) > 1e-9:
        G, b = np.polyfit(U, V, 1)
        if G >= 0.0 and b <= 0.0:
            dz = float(-b / G) if G > 0.0 else 0.0
        else:
            G = max(0.0, float(np.dot(U, V) / np.dot(U, U)))
            dz = 0.0
    else:
        dz = dz_lb if dz_lb < float(U[0]) else 0.0
        G = max(0.0, float(np.mean(V) / (float(U[0]) - dz))) if float(U[0]) > dz else 0.0
    return {
        "gain": float(G),
        "deadzone": float(dz),
        "deadzone_lb": float(dz_lb),
        "n": len(mov),
        "n_stalled": len(stalled),
    }


def _extract_ws_lag(ws_path: Path, imu: list[dict], it: np.ndarray) -> float | None:
    rows = [json.loads(line) for line in open(ws_path) if line.strip()]
    if not rows:
        return None
    ack_diffs = []
    for r in rows:
        t_send = next(
            (r[k] for k in ("send_t", "sent_t", "t", "ts") if k in r and r[k] is not None), None
        )
        t_ack = next(
            (r[k] for k in ("pwm_t", "ack_t", "recv_t") if k in r and r[k] is not None), None
        )
        if t_send is not None and t_ack is not None:
            ack_diffs.append(float(t_ack) - float(t_send))
    if ack_diffs:
        return float(np.median(ack_diffs))

    # Fallback: cross-correlate WS command magnitude with IMU wheel/PWM vector magnitude.
    ws_t, ws_u = [], []
    for r in rows:
        t_send = next(
            (r[k] for k in ("send_t", "sent_t", "t", "ts") if k in r and r[k] is not None), None
        )
        cmd = next(
            (
                r[k]
                for k in ("pwm", "wh", "wheels", "cmd", "duty", "u")
                if k in r and r[k] is not None
            ),
            None,
        )
        if t_send is not None and cmd is not None:
            arr = np.asarray(cmd, dtype=float).ravel()
            ws_t.append(float(t_send))
            ws_u.append(float(np.linalg.norm(arr)))
    w_arr = np.array([np.linalg.norm(s.get("w") or [0.0]) for s in imu], dtype=float)
    if len(ws_t) >= 5 and np.std(ws_u) > 1e-6 and len(it) >= 5 and np.std(w_arr) > 1e-6:
        wt = np.asarray(ws_t, dtype=float)
        wu = np.asarray(ws_u, dtype=float)
        best_lag, best_r = None, -1.0
        for lag in np.linspace(0.0, 0.30, 121):
            wi = np.interp(wt + lag, it, w_arr, left=np.nan, right=np.nan)
            ok = np.isfinite(wi)
            if ok.sum() >= 5 and np.std(wi[ok]) > 1e-6 and np.std(wu[ok]) > 1e-6:
                r = float(np.corrcoef(wu[ok], wi[ok])[0, 1])
                if r > best_r:
                    best_r, best_lag = r, float(lag)
        return best_lag
    return None


def analyze_run(run: Path) -> dict:
    ticks = [json.loads(line) for line in open(run / "ticks.jsonl")]
    imu = [json.loads(line) for line in open(run / "imu.jsonl")]
    it = np.array([s["t"] for s in imu], dtype=float)
    gz = np.array([s["g"][2] for s in imu], dtype=float)
    st = np.array([bool(s.get("st", False)) for s in imu], dtype=bool)
    if st.any():
        gz = gz - np.median(gz[st])

    pulses, cur = [], None
    coast_rows = []
    all_poses: dict[float, list[float]] = {}
    for r in ticks:
        if r.get("pose") and r.get("pose_t") is not None:
            all_poses[float(r["pose_t"])] = r["pose"]
        phase = r.get("phase")
        pr = r.get("probe")
        is_drive = (
            phase not in ("baseline", "coast")
            and isinstance(pr, (list, tuple))
            and len(pr) >= 2
            and pr[0] in ("fwd", "left", "ccw")
            and abs(float(pr[1])) > 0.0
        )
        if is_drive:
            key = (str(pr[0]), float(pr[1]))
            if cur is None or cur["key"] != key or r["t"] - cur["rows"][-1]["t"] > 0.3:
                cur = {"key": key, "rows": []}
                pulses.append(cur)
            cur["rows"].append(r)
        else:
            cur = None
            if phase != "baseline" and (phase == "coast" or len(pulses) > 0):
                coast_rows.append(r)

    all_pt = np.array(sorted(all_poses), dtype=float)
    all_P = (
        np.array([all_poses[t] for t in all_pt], dtype=float) if len(all_pt) else np.zeros((0, 3))
    )

    out = []
    drive_taus_gz, brake_taus_gz = [], []
    drive_taus_vis, brake_taus_vis = [], []

    for i, p in enumerate(pulses):
        axis, duty = p["key"]
        t0, t1 = float(p["rows"][0]["t"]), float(p["rows"][-1]["t"])
        next_t0 = float(pulses[i + 1]["rows"][0]["t"]) if i + 1 < len(pulses) else float("inf")
        row: dict = {"axis": axis, "duty": duty}

        m = (it > t0 + 0.3) & (it < t1)
        row["yaw_rate_dps_ss"] = float(np.mean(gz[m])) if m.any() else None
        onset = it[(it > t0) & (it <= t1) & (np.abs(gz) > 8.0)]
        row["yaw_onset_s"] = float(onset[0] - t0) if axis == "ccw" and len(onset) else None

        if row["yaw_rate_dps_ss"] is not None and abs(row["yaw_rate_dps_ss"]) >= 5.0:
            gm = (it >= t0) & (it <= t1)
            dtau = _exp_tau(it[gm], gz[gm], row["yaw_rate_dps_ss"], approach=True)
            if dtau is not None:
                row["drive_tau_gz_s"] = dtau
                drive_taus_gz.append(dtau)

        tail_end = min(t1 + 0.8, next_t0 - 0.02)
        if tail_end > t1 + 0.10:
            pre = (it >= t1 - 0.04) & (it <= t1)
            y_end = float(np.mean(gz[pre])) if pre.any() else row["yaw_rate_dps_ss"]
            if y_end is not None and abs(y_end) >= 5.0:
                tm = (it > t1) & (it <= tail_end)
                btau = _exp_tau(it[tm], gz[tm], y_end, approach=False)
                if btau is not None:
                    row["brake_tau_gz_s"] = btau
                    brake_taus_gz.append(btau)

        ps = {}
        for r in p["rows"]:
            if r.get("pose") and r.get("pose_t", 0.0) > t0:
                ps[float(r["pose_t"])] = r["pose"]
        if len(ps) >= 4:
            tt = np.array(sorted(ps), dtype=float)
            P = np.array([ps[t] for t in tt], dtype=float)
            th = P[0, 2] - math.pi / 2.0
            f = (P[:, 0] - P[0, 0]) * math.cos(th) + (P[:, 1] - P[0, 1]) * math.sin(th)
            left_disp = -(P[:, 0] - P[0, 0]) * math.sin(th) + (P[:, 1] - P[0, 1]) * math.cos(th)
            late = tt > t0 + 0.3
            if late.sum() >= 3:
                v_fwd = float(np.polyfit(tt[late], f[late], 1)[0])
                v_left = float(np.polyfit(tt[late], left_disp[late], 1)[0])
                row["v_fwd_ss"] = v_fwd
                row["v_left_ss"] = v_left
                if axis in ("fwd", "left"):
                    row["v_ss"] = v_fwd if axis == "fwd" else v_left
            s = f if axis == "fwd" else left_disp
            moved = tt[np.abs(s) > 0.01]
            row["onset_s"] = float(moved[0] - t0) if len(moved) else None

            if axis in ("fwd", "left") and row.get("v_ss") is not None and abs(row["v_ss"]) >= 0.05:
                dt_v = np.diff(tt)
                ok = dt_v > 0.015
                if ok.sum() >= 3:
                    v_vis = np.diff(s)[ok] / dt_v[ok]
                    t_vis = (0.5 * (tt[:-1] + tt[1:]))[ok]
                    v_dtau = _exp_tau(t_vis, v_vis, row["v_ss"], approach=True)
                    if v_dtau is not None:
                        row["drive_tau_vis_s"] = v_dtau
                        drive_taus_vis.append(v_dtau)

                if tail_end > t1 + 0.10 and len(all_pt):
                    pm = (all_pt >= t1 - 0.05) & (all_pt <= tail_end)
                    if pm.sum() >= 4:
                        pt_tail = all_pt[pm]
                        P_tail = all_P[pm]
                        s_tail = (
                            (P_tail[:, 0] - P[0, 0]) * math.cos(th)
                            + (P_tail[:, 1] - P[0, 1]) * math.sin(th)
                            if axis == "fwd"
                            else -(P_tail[:, 0] - P[0, 0]) * math.sin(th)
                            + (P_tail[:, 1] - P[0, 1]) * math.cos(th)
                        )
                        dt_t = np.diff(pt_tail)
                        ok_t = dt_t > 0.015
                        if ok_t.sum() >= 3:
                            v_tail = np.diff(s_tail)[ok_t] / dt_t[ok_t]
                            t_tail = (0.5 * (pt_tail[:-1] + pt_tail[1:]))[ok_t]
                            v_btau = _exp_tau(t_tail, v_tail, row["v_ss"], approach=False)
                            if v_btau is not None:
                                row["brake_tau_vis_s"] = v_btau
                                brake_taus_vis.append(v_btau)

        out.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    fit = {}
    for axis, key, thresh, unit in (
        ("fwd", "v_ss", 0.02, "m/s per duty"),
        ("left", "v_ss", 0.02, "m/s per duty"),
        ("ccw", "yaw_rate_dps_ss", 3.0, "deg/s per duty"),
    ):
        pts = [
            (abs(r["duty"]), abs(r[key]))
            for r in out
            if r["axis"] == axis and r.get(key) is not None
        ]
        res = _fit_nonneg_deadzone(pts, thresh)
        if res is not None:
            res["unit"] = unit
            fit[axis] = res

    # Cross-coupling ratios (OLS through origin on non-stalled pulses)
    cross_coupling: dict[str, float | None] = {
        "left_per_fwd": None,
        "yaw_dps_per_left_mps": None,
        "trans_per_ccw": None,
    }
    fwd_c = [
        (r["v_fwd_ss"], r["v_left_ss"])
        for r in out
        if r["axis"] == "fwd"
        and r.get("v_fwd_ss") is not None
        and r.get("v_left_ss") is not None
        and abs(r["v_fwd_ss"]) >= 0.02
    ]
    if fwd_c:
        x = np.array([a for a, _ in fwd_c])
        y = np.array([b for _, b in fwd_c])
        cross_coupling["left_per_fwd"] = float(np.dot(x, y) / np.dot(x, x))
    left_c = [
        (r["v_left_ss"], r["yaw_rate_dps_ss"])
        for r in out
        if r["axis"] == "left"
        and r.get("v_left_ss") is not None
        and r.get("yaw_rate_dps_ss") is not None
        and abs(r["v_left_ss"]) >= 0.02
    ]
    if left_c:
        x = np.array([a for a, _ in left_c])
        y = np.array([b for _, b in left_c])
        cross_coupling["yaw_dps_per_left_mps"] = float(np.dot(x, y) / np.dot(x, x))
    ccw_c = [
        (abs(r["yaw_rate_dps_ss"]), math.hypot(r["v_fwd_ss"], r["v_left_ss"]))
        for r in out
        if r["axis"] == "ccw"
        and r.get("yaw_rate_dps_ss") is not None
        and r.get("v_fwd_ss") is not None
        and r.get("v_left_ss") is not None
        and abs(r["yaw_rate_dps_ss"]) >= 3.0
    ]
    if ccw_c:
        x = np.array([a for a, _ in ccw_c])
        y = np.array([b for _, b in ccw_c])
        cross_coupling["trans_per_ccw"] = float(np.dot(x, y) / np.dot(x, x))

    dynamics = {
        "drive_tau_gz_s": float(np.median(drive_taus_gz)) if drive_taus_gz else None,
        "brake_tau_gz_s": float(np.median(brake_taus_gz)) if brake_taus_gz else None,
        "drive_tau_vis_s": float(np.median(drive_taus_vis)) if drive_taus_vis else None,
        "brake_tau_vis_s": float(np.median(brake_taus_vis)) if brake_taus_vis else None,
        "n_coast_ticks": len(coast_rows),
    }

    # Latency: WS send-to-PWM & optical cap-to-gyro cross-correlation on ccw steps
    latency: dict[str, float | None] = {
        "ws_send_to_pwm_lag_s": None,
        "cap_to_gyro_lag_s": None,
        "cap_to_gyro_corr": None,
        "frame_proc_ms": None,
    }
    ws_path = run / "ws_state.jsonl"
    if ws_path.exists():
        latency["ws_send_to_pwm_lag_s"] = _extract_ws_lag(ws_path, imu, it)

    frames_path = run / "frames.jsonl"
    if frames_path.exists():
        proc_ms = []
        for line in open(frames_path):
            if line.strip():
                fr = json.loads(line)
                if fr.get("done") is not None and fr.get("cap") is not None:
                    proc_ms.append((float(fr["done"]) - float(fr["cap"])) * 1000.0)
        if proc_ms:
            latency["frame_proc_ms"] = float(np.median(proc_ms))

    if len(all_pt) >= 5 and (frames_path.exists() or len(all_pt) >= 10):
        yaws = np.unwrap(all_P[:, 2])
        dt_p = np.diff(all_pt)
        ok_p = (dt_p > 0.01) & (dt_p < 0.25)
        if ok_p.sum() >= 5:
            t_opt = (0.5 * (all_pt[:-1] + all_pt[1:]))[ok_p]
            omega_opt = np.degrees(np.diff(yaws)[ok_p] / dt_p[ok_p])
            ccw_wins = [
                (float(p["rows"][0]["t"]), float(p["rows"][-1]["t"]))
                for p in pulses
                if p["key"][0] == "ccw"
            ]
            if ccw_wins:
                in_ccw = np.zeros(len(t_opt), dtype=bool)
                for a, b in ccw_wins:
                    in_ccw |= (t_opt >= a - 0.1) & (t_opt <= b + 0.2)
                if in_ccw.sum() >= 5:
                    t_opt, omega_opt = t_opt[in_ccw], omega_opt[in_ccw]
            if len(t_opt) >= 5 and np.std(omega_opt) > 1.0:
                best_lag, best_r = None, -1.0
                for lag in np.linspace(-0.25, 0.25, 201):
                    gz_i = np.interp(t_opt - lag, it, gz, left=np.nan, right=np.nan)
                    ok = np.isfinite(gz_i)
                    if ok.sum() >= 5 and np.std(gz_i[ok]) > 1e-6 and np.std(omega_opt[ok]) > 1e-6:
                        r = float(np.corrcoef(omega_opt[ok], gz_i[ok])[0, 1])
                        if r > best_r:
                            best_r, best_lag = r, float(lag)
                latency["cap_to_gyro_lag_s"] = best_lag
                latency["cap_to_gyro_corr"] = best_r if best_lag is not None else None

    result = {
        "pulses": out,
        "fit": fit,
        "cross_coupling": cross_coupling,
        "dynamics": dynamics,
        "latency": latency,
    }
    print("FIT", json.dumps(fit, indent=1))
    print("CROSS_COUPLING", json.dumps(cross_coupling, indent=1))
    print("DYNAMICS", json.dumps(dynamics, indent=1))
    print("LATENCY", json.dumps(latency, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    result = analyze_run(args.run_dir)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
