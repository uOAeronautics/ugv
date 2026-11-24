#!/usr/bin/env python3
"""Detect the floor/wall junction v-coordinate in Poly Haven 1k panoramas.

Each handheld 360 shot has its true horizon at roughly constant arc direction
(v = asin(dir.y) fixed up to camera roll). We fit v(a) = v0 + b*cos(a) + c*sin(a),
where a is the azimuth around the image (0..2 pi), so roll comes along for free.
The grounded dome in the challenge view samples the photo at
v = asin(dir.y)/pi + 0.5 - (same terms), which lands the junction exactly on the
y=0 plane at every azimuth.

Usage: uv run python tools/hdri_horizons/predict_horizons.py [--only id ...]
Output: src/ugv/lab/static/hdri-horizons.json (tracked, tiny floats only)
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "cache" / "hdri"
CACHE_V2 = ROOT / "runs" / "cache" / "hdri_v2"
OUT = ROOT / "src" / "ugv" / "lab" / "static" / "hdri-horizons.json"
BAND = (
    0.44,
    0.80,
)  # junction is below the photo equator for level/up-pitched cameras; window bands sit above

sys.path.insert(0, str(Path(__file__).parent))
import rgbe  # noqa: E402

load_hdr = rgbe.load_hdr
row_luminance = rgbe.luminance_grid


def smooth(p, k=5):
    out = []
    for i in range(len(p)):
        s = p[max(0, i - k // 2) : i + k // 2 + 1]
        out.append(sum(s) / len(s))
    return out


def column_candidates(grid, w, h):
    """For every column, rows with a strong local gradient (log-lumifference)."""
    cols = []
    lo, hi = int(h * BAND[0]), int(h * BAND[1])
    for x in range(w):
        col = smooth([grid[y][x] for y in range(h)], 5)
        grads = [(abs(col[y + 1] - col[y - 1]), y) for y in range(max(1, lo), min(h - 1, hi))]
        grads.sort(reverse=True)
        best = grads[0][0] if grads else 0.0
        kept = [(g, y) for g, y in grads[:4] if g > 0.35 * best and g > 0.05]
        if kept:
            cols.append((x, kept))
    return cols


def fit_horizon(cands, w, h):
    """Robust v = v0 + b cos a + c sin a fit over per-column candidate rows."""
    pts = []
    for x, kept in cands:
        for strength, y in kept:
            pts.append((x / w * 2 * math.pi, 1.0 - (y + 0.5) / h, strength))
    if len(pts) < 40:
        return None

    def gauss3(A, y):
        M = [row[:] + [yv] for row, yv in zip(A, y, strict=True)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                return None
            M[col], M[piv] = M[piv], M[col]
            for r in range(3):
                if r != col:
                    f = M[r][col] / M[col][col]
                    for k in range(4):
                        M[r][k] -= f * M[col][k]
        return [M[i][3] / M[i][i] for i in range(3)]

    def solve(points):
        m00 = m01 = m02 = m11 = m12 = m22 = m0 = m1 = m2 = 0.0
        for a, v, wt in points:
            ca, sa = math.cos(a), math.sin(a)
            m00 += wt
            m01 += wt * ca
            m02 += wt * sa
            m11 += wt * ca * ca
            m12 += wt * ca * sa
            m22 += wt * sa * sa
            m0 += wt * v
            m1 += wt * v * ca
            m2 += wt * v * sa
        return gauss3([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]], [m0, m1, m2])

    # The junction is the LOWEST edge that stays strong all the way around; window bands
    # above it dominate a naive histogram, so seed from the bottom of the candidate band.
    bins = [0.0] * h
    for _a, v, wt in pts:
        y = int((1.0 - v) * h)
        if 0 <= y < h:
            bins[y] += wt
    peak = max(bins)
    junction_row = max((y for y, c in enumerate(bins) if c >= 0.45 * peak), default=None)
    if junction_row is None:
        return None
    v0_seed = 1.0 - (junction_row + 0.5) / h
    model = [v0_seed, 0.0, 0.0]
    for _ in range(8):
        chosen = []
        for a, v, wt in pts:
            pred = model[0] + model[1] * math.cos(a) + model[2] * math.sin(a)
            if abs(v - pred) < 0.06:
                chosen.append((a, v, wt))
        if len(chosen) < 40:
            return None
        new = solve(chosen)
        if new is None:
            return None
        model = new
    inliers = sum(
        1
        for a, v, wt in pts
        if abs(v - (model[0] + model[1] * math.cos(a) + model[2] * math.sin(a))) < 0.05
    )
    conf = inliers / len(pts)
    return model, conf


def analyze(path):
    w, h, rgb = load_hdr(path)
    grid = row_luminance(rgb, w, h)
    cands = column_candidates(grid, w, h)
    if len(cands) < w * 0.25:
        return None
    out = fit_horizon(cands, w, h)
    if out is None:
        return None
    (v0, b, c), conf = out
    if conf < 0.30:
        return None
    # v0 is the junction position in the photo's own v coordinate; the dome shader subtracts
    # the same terms from the sampled v, so what the sidecar stores is the offset, not v0.
    return {
        "offset": round(0.5 - v0, 4),
        "roll_b": round(b, 4),
        "roll_c": round(c, 4),
        "conf": round(conf, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    ph = [(p.name[:-7], p) for p in sorted(CACHE.glob("*_1k.hdr"))]
    v2 = [(p.name[:-7], p) for p in sorted(CACHE_V2.glob("*_2k.hdr"))]
    entries = ph + v2
    if args.only:
        ids = set(args.only)
        entries = [(i, p) for i, p in entries if i in ids]
    horizons = {}
    for i, (hid, path) in enumerate(entries, 1):
        try:
            r = analyze(path)
        except Exception as e:
            print(f"ERR {hid}: {e}", flush=True)
            continue
        if r:
            horizons[hid] = r
        if i % 100 == 0:
            print(f"{i}/{len(entries)} analyzed, {len(horizons)} detected", flush=True)
    print(f"detected: {len(horizons)}/{len(entries)}")
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    merged = dict(old)
    merged.update({k: v for k, v in horizons.items()})
    OUT.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
