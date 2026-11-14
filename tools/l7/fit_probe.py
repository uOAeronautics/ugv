#!/usr/bin/env python3
"""Fit per-axis steady-state response from an l7_run.py probe run.

Yaw rate comes from the 100 Hz gyro; translation from tag poses inside each pulse.
Prints one row per pulse and a least-squares v = G * (|duty| - dz) fit per axis.
usage: fit_probe.py <run_dir> [--json out.json]
"""
import json, math, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
ticks = [json.loads(l) for l in open(run / "ticks.jsonl")]
imu = [json.loads(l) for l in open(run / "imu.jsonl")]
it = np.array([s["t"] for s in imu]); gz = np.array([s["g"][2] for s in imu])
gz = gz - np.median(gz[np.array([s["st"] for s in imu])])
pulses, cur = [], None
for r in ticks:
    key = tuple(r["probe"])
    if cur is None or cur["key"] != key or r["t"] - cur["rows"][-1]["t"] > 0.3:
        cur = {"key": key, "rows": []}; pulses.append(cur)
    cur["rows"].append(r)
out = []
for p in pulses:
    axis, duty = p["key"]; t0, t1 = p["rows"][0]["t"], p["rows"][-1]["t"]
    row = {"axis": axis, "duty": duty}
    m = (it > t0 + 0.3) & (it < t1)
    row["yaw_rate_dps_ss"] = float(np.mean(gz[m])) if m.any() else None
    onset = it[(it > t0) & (np.abs(gz) > 8)]
    row["yaw_onset_s"] = float(onset[0] - t0) if axis == "ccw" and len(onset) else None
    ps = {}
    for r in p["rows"]:
        if r["pose"] and r["pose_t"] > t0:
            ps[r["pose_t"]] = r["pose"]
    if len(ps) >= 4:
        tt = np.array(sorted(ps)); P = np.array([ps[t] for t in tt])
        th = P[0, 2] - math.pi / 2
        f = (P[:, 0] - P[0, 0]) * math.cos(th) + (P[:, 1] - P[0, 1]) * math.sin(th)
        l = -(P[:, 0] - P[0, 0]) * math.sin(th) + (P[:, 1] - P[0, 1]) * math.cos(th)
        s = f if axis == "fwd" else l
        late = tt > t0 + 0.3
        if late.sum() >= 3:
            row["v_ss"] = float(np.polyfit(tt[late], s[late], 1)[0])
        moved = tt[np.abs(s) > 0.01]
        row["onset_s"] = float(moved[0] - t0) if len(moved) else None
    out.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
fit = {}
for axis, key in (("fwd", "v_ss"), ("left", "v_ss"), ("ccw", "yaw_rate_dps_ss")):
    pts = [(abs(r["duty"]), abs(r[key])) for r in out if r["axis"] == axis and r.get(key) is not None]
    mov = [(u, v) for u, v in pts if v > (0.05 if key == "v_ss" else 15)]
    if len(mov) >= 2:
        G, b = np.polyfit([u for u, _ in mov], [v for _, v in mov], 1)
        fit[axis] = {"gain": float(G), "deadzone": float(-b / G), "n": len(mov),
                     "unit": "m/s per duty" if key == "v_ss" else "deg/s per duty"}
print("FIT", json.dumps(fit, indent=1))
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"pulses": out, "fit": fit}, indent=1))
