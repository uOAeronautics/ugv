"""Sim-vs-Real observation contract for L7 field runs (wraps tools/l7/audit_contract.py)."""

from pathlib import Path

import numpy as np

from .data import REPO, sanitize


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _pass_of(v: dict, status: str | None):
    """Tri-state pass: True/False when the report says so, None when it does not.

    An unknown result must not be shown as a pass, so a missing flag stays None.
    """
    if "pass" in v and v["pass"] is not None:
        return bool(v["pass"])
    if status in ("PASS", "OK"):
        return True
    if status in ("FAIL", "FAILED"):
        return False
    return None


def norm_checks(raw, real_metrics):
    checks = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            v = v if isinstance(v, dict) else {}
            checks.append(
                {
                    "name": k,
                    "pass": _pass_of(v, str(v.get("status", "")).upper()),
                    "measured": v.get("value", v.get("real", "—")),
                    "limit": v.get("threshold", v.get("target", "—")),
                }
            )
        return checks
    if isinstance(raw, list):
        for v in raw:
            if not isinstance(v, dict):
                continue
            status = str(v.get("status", "")).upper()
            checks.append(
                {
                    "name": v.get("name", "check"),
                    "pass": _pass_of(v, status),
                    "measured": v.get("value", v.get("real", "—")),
                    "limit": v.get("limit", v.get("threshold", v.get("target", "—"))),
                }
            )
        return checks
    # no checks in the report: derive the three gate checks from the real metrics
    rm = real_metrics or {}
    vr = _num(rm.get("valid_rate"))
    if vr is not None:
        checks.append(
            {
                "name": "valid_pose_rate",
                "pass": vr >= 0.5,
                "measured": round(vr, 3),
                "limit": ">= 0.50",
            }
        )
    hz = _num(rm.get("hz"))
    if hz is not None:
        checks.append(
            {
                "name": "loop_rate_hz",
                "pass": hz >= 40.0,
                "measured": round(hz, 2),
                "limit": ">= 40.0",
            }
        )
    loop_ms = rm.get("loop_ms") if isinstance(rm.get("loop_ms"), dict) else {}
    p95 = _num(loop_ms.get("p95"))
    if p95 is not None:
        checks.append(
            {
                "name": "loop_ms_p95",
                "pass": p95 <= 25.0,
                "measured": round(p95, 2),
                "limit": "<= 25.0",
            }
        )
    return checks


def _channel_row(name, real_mean, real_std, sim_mean, sim_std):
    return {
        "name": name,
        "real_mean": _num(real_mean),
        "real_std": _num(real_std),
        "sim_mean": _num(sim_mean),
        "sim_std": _num(sim_std),
    }


def norm_channels(real_metrics, sim):
    rm = real_metrics or {}
    sim = sim or {}
    sim_ch = sim.get("channels") if isinstance(sim.get("channels"), dict) else {}
    real_ch = rm.get("channels") if isinstance(rm.get("channels"), dict) else {}

    def hz_of(src):
        iv = src.get("inter_valid_dt") if isinstance(src.get("inter_valid_dt"), dict) else {}
        m = _num(iv.get("mean"))
        if m and m > 0:
            return 1.0 / m, (_num(iv.get("std")) or 0.0) / (m * m)
        h = _num(src.get("hz"))
        vr = _num(src.get("valid_rate"))
        if h is not None and vr is not None:
            return h * vr, None
        return h, None

    real_hz, real_hz_std = hz_of(rm)
    sim_hz, sim_hz_std = hz_of(sim)
    if sim_hz is None:
        sim_hz, sim_hz_std = 50.0, 0.0  # nominal 50 Hz loop, tag every frame

    fa_r = real_ch.get("tag_age") if isinstance(real_ch.get("tag_age"), dict) else {}
    fa_s = sim_ch.get("tag_age") if isinstance(sim_ch.get("tag_age"), dict) else {}
    fa_real_p50 = _num(fa_r.get("p50"))
    fa_sim_p50 = _num(fa_s.get("p50"))

    vr_r, vr_s = _num(rm.get("valid_rate")), _num(sim.get("valid_rate"))

    iw_r = (
        _num((real_ch.get("imu_w") or {}).get("std"))
        if isinstance(real_ch.get("imu_w"), dict)
        else None
    )
    iw_s = (
        _num((sim_ch.get("imu_w") or {}).get("std"))
        if isinstance(sim_ch.get("imu_w"), dict)
        else None
    )

    loop_r = rm.get("loop_ms") if isinstance(rm.get("loop_ms"), dict) else {}
    lm_p50, lm_p95 = _num(loop_r.get("p50")), _num(loop_r.get("p95"))
    lm_std = _num(loop_r.get("std"))

    return [
        _channel_row("vision_hz", real_hz, real_hz_std, sim_hz, sim_hz_std),
        _channel_row(
            "frame_age_p50_ms",
            fa_real_p50 * 1000 if fa_real_p50 is not None else None,
            (_num(fa_r.get("std")) or 0.0) * 1000 if _num(fa_r.get("std")) is not None else None,
            fa_sim_p50 * 1000 if fa_sim_p50 is not None else 0.0,
            (_num(fa_s.get("std")) or 0.0) * 1000 if _num(fa_s.get("std")) is not None else None,
        ),
        _channel_row(
            "dropout_rate",
            1.0 - vr_r if vr_r is not None else None,
            None,
            1.0 - vr_s if vr_s is not None else 0.0,
            None,
        ),
        _channel_row("imu_w_std", iw_r, None, iw_s if iw_s is not None else 0.05, None),
        _channel_row("loop_ms_p50", lm_p50, lm_std, 20.0, 0.0),
        _channel_row("loop_ms_p95", lm_p95, lm_std, 20.0, 0.0),
    ]


# ---- Joint sim-vs-real overlap (whitened Gaussian-KDE Monte-Carlo) ----

JOINT_OVERLAP_MIN = 0.35
JOINT_MIN_SAMPLES = 20
JOINT_CHANNELS = ["inter_tag_dt_s", "tag_age_s", "imu_w_rad_s", "loop_ms_s"]
JOINT_CHECK_NAME = "joint_whited_kde_overlap"
JOINT_METHOD = "gaussian_kde_mc (Silverman bandwidth, 2000 draws, rng seed 11)"
_JOINT_REF_CAP = 4096  # KDE reference cap per side; keeps the endpoint O(ms) not O(s)
_JOINT_MC_DRAWS = 2000


def _joint_subsample(x: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(x) <= cap:
        return x
    idx = np.random.default_rng(seed).choice(len(x), size=cap, replace=False)
    return x[np.sort(idx)]


def real_joint_samples(run_dir: Path) -> np.ndarray:
    """Matched per-event rows [inter_tag_dt_s, tag_age_s, imu_w_rad_s, loop_ms_s].

    Each inter-tag interval is paired with the tick where the second tag landed
    (imu, loop) and the tick just before it (tag_age, the interval's peak
    staleness — age at the arrival tick itself is always 0).
    """
    import json

    rows = [
        json.loads(line)
        for line in (run_dir / "ticks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    t = np.array([float(r["t"]) for r in rows], dtype=np.float64)
    tag_idx = [i for i, r in enumerate(rows) if r.get("src") == "tag"]
    n = max(0, len(tag_idx) - 1)
    out = np.full((n, len(JOINT_CHANNELS)), np.nan)
    if n == 0:
        return out
    last_valid_t = float(t[0])
    ages = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        if r.get("src", "none") != "none":
            last_valid_t = float(r["t"])
        ages[i] = max(0.0, float(r["t"]) - last_valid_t)
    for j, b in enumerate(tag_idx[1:]):
        r = rows[b]
        out[j, 0] = t[b] - t[tag_idx[j]]
        out[j, 1] = ages[b - 1]
        imu = r.get("imu") or [0.0, 0.0]
        w = r.get("imu_w_fed")
        out[j, 2] = float(w if w is not None else (imu[1] if imu[1] is not None else 0.0))
        lm = r.get("loop_ms")
        out[j, 3] = float(lm) / 1000.0 if lm is not None else np.nan
    return out


def sim_joint_samples(sim_ref_path: Path) -> np.ndarray:
    """Sim joint rows over the same 4 channels, from the per-step reference frames.

    inter_tag_dt uses the audit_contract episode-grouped valid-index diffs; the
    npz carries no loop timing, so loop_ms_s is the nominal 20 ms tick budget
    (the same constant norm_channels uses for the sim loop_ms marginals).
    """
    from tools.l7.audit_contract import CHANNEL_NAMES

    z = np.load(sim_ref_path)
    frames = np.asarray(z["frames"], dtype=np.float64)
    ep_id = np.asarray(z["ep_id"], dtype=np.int64)
    dt = float(z["dt"]) if "dt" in z.files else 0.02
    age_i, valid_i, imu_i = (CHANNEL_NAMES.index(c) for c in ("tag_age", "valid", "imu_w"))
    order = np.argsort(ep_id, kind="stable")
    frames_s, ep_s = frames[order], ep_id[order]
    bounds = np.flatnonzero(np.diff(ep_s, prepend=ep_s[0] - 1, append=ep_s[-1] + 1))
    pairs: list[tuple[int, int]] = []
    for k in range(len(bounds) - 1):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        v = np.flatnonzero(frames_s[lo:hi, valid_i] > 0.5) + lo
        pairs.extend(zip(v[:-1].tolist(), v[1:].tolist(), strict=True))
    out = np.full((len(pairs), len(JOINT_CHANNELS)), np.nan)
    for j, (a, b) in enumerate(pairs):
        out[j, 0] = (b - a) * dt
        out[j, 1] = frames_s[b - 1, age_i]
        out[j, 2] = frames_s[b, imu_i]
        out[j, 3] = dt  # nominal 50 Hz loop budget; sim records no per-step loop_ms
    return out


def whiten_by_sim(real: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center on the sim mean and whiten with the sim covariance (+1e-6*I).

    Falls back to diagonal (per-channel std) whitening when the regularized
    covariance is still numerically singular.
    """
    mu = sim.mean(axis=0)
    cov = np.cov(sim, rowvar=False) + 1e-6 * np.eye(sim.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    if vals.min() <= 1e-10 * max(vals.max(), 1e-12):
        w = 1.0 / np.sqrt(np.maximum(np.var(sim, axis=0), 1e-12))
        return (real - mu) * w, (sim - mu) * w
    transform = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return (real - mu) @ transform, (sim - mu) @ transform


def kde_overlap(real_w: np.ndarray, sim_w: np.ndarray, n_draws: int = _JOINT_MC_DRAWS) -> float:
    """Monte-Carlo overlap coefficient of two Gaussian KDEs in whitened space.

    Draws come from the sim KDE; overlap = E_sim[min(f_sim, f_real) / f_sim].
    Both KDEs share the Silverman bandwidth fitted on the whitened sim sample.
    """
    n, d = sim_w.shape
    h = max(float(n ** (-1.0 / (d + 4))), 1e-6)
    sim_ref = _joint_subsample(sim_w, _JOINT_REF_CAP, seed=11)
    real_ref = _joint_subsample(real_w, _JOINT_REF_CAP, seed=12)

    def kde(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = np.empty(len(x))
        for s in range(0, len(x), 256):
            chunk = x[s : s + 256]
            d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            out[s : s + len(chunk)] = np.exp(-d2 / (2.0 * h * h)).mean(axis=1)
        return out / ((2.0 * np.pi) ** (d / 2.0) * h**d)

    rng = np.random.default_rng(11)
    centers = sim_ref[rng.integers(0, len(sim_ref), size=n_draws)]
    draws = centers + h * rng.standard_normal((n_draws, d))
    f_sim = kde(draws, sim_ref)
    f_real = kde(draws, real_ref)
    ratio = np.minimum(f_sim, f_real) / np.maximum(f_sim, 1e-300)
    return float(np.clip(ratio.mean(), 0.0, 1.0))


def joint_overlap(real: np.ndarray, sim: np.ndarray, channels: list[str] | None = None) -> dict:
    """Joint overlap over matched sample matrices (rows = samples, cols = channels).

    Channels with < JOINT_MIN_SAMPLES finite samples on either side are dropped.
    """
    channels = list(channels or JOINT_CHANNELS)
    real = np.asarray(real, dtype=np.float64).reshape(-1, len(channels))
    sim = np.asarray(sim, dtype=np.float64).reshape(-1, len(channels))
    keep, dropped = [], []
    for j, name in enumerate(channels):
        r = real[:, j][np.isfinite(real[:, j])]
        s = sim[:, j][np.isfinite(sim[:, j])]
        if len(r) >= JOINT_MIN_SAMPLES and len(s) >= JOINT_MIN_SAMPLES:
            keep.append(j)
        else:
            dropped.append(name)
    result = {
        "overlap": None,
        "channels": [channels[j] for j in keep],
        "dropped": dropped,
        "method": JOINT_METHOD,
        "pass": None,
    }
    if len(keep) < 2:
        return result
    # complete-case rows: per-channel filtering would misalign events across
    # channels whenever NaN patterns differ, scrambling the joint structure
    real_k = real[np.isfinite(real[:, keep]).all(axis=1)][:, keep]
    sim_k = sim[np.isfinite(sim[:, keep]).all(axis=1)][:, keep]
    n = min(len(real_k), len(sim_k))
    real_w, sim_w = whiten_by_sim(real_k[:n], sim_k[:n])
    overlap = kde_overlap(real_w, sim_w)
    result["overlap"] = round(overlap, 4)
    result["pass"] = bool(overlap >= JOINT_OVERLAP_MIN)
    return result


def eval_contract(run_id: str, calib_dir: Path) -> dict:
    from tools.l7.audit_contract import evaluate_contract, extract_run_metrics

    run_dir = calib_dir / run_id
    sim_ref = REPO / "tools" / "l7" / "sim_reference_L1.npz"
    note = None
    try:
        report = evaluate_contract(run_dir, sim_ref_path=sim_ref)
    except Exception as e:
        try:
            real = extract_run_metrics(run_dir)
        except Exception as e2:
            real = {"run_id": run_id, "error": str(e2)}
        report = {"run_id": run_id, "real_metrics": real, "note": f"sim_ref unavailable: {e}"}
        note = str(e)
    if note:
        report.setdefault("note", note)
    real_metrics = report.get("real_metrics") or report.get("real") or {}
    sim = report.get("sim")
    sim_ok = isinstance(sim, dict) and bool(sim)
    report["sim_source"] = "sim_reference_L1.npz" if sim_ok else "nominal_contract"
    report["checks"] = norm_checks(report.get("checks"), real_metrics)
    report["channels"] = norm_channels(real_metrics, sim if sim_ok else {})
    try:
        joint = joint_overlap(real_joint_samples(run_dir), sim_joint_samples(sim_ref))
    except Exception as e:
        joint = {
            "overlap": None,
            "channels": [],
            "dropped": list(JOINT_CHANNELS),
            "method": JOINT_METHOD,
            "pass": None,
            "note": f"joint unavailable: {e}",
        }
    report["joint"] = joint
    report["checks"].append(
        {
            "name": JOINT_CHECK_NAME,
            "pass": joint["pass"],
            "measured": joint["overlap"] if joint["overlap"] is not None else "—",
            "limit": f">= {JOINT_OVERLAP_MIN:.2f}",
            "method": joint["method"],
            "dropped": joint["dropped"],
        }
    )
    report["status"] = verdict_of(report)
    return sanitize(report)


def verdict_of(report: dict) -> str:
    """Banner status. Any failing gate makes the verdict FAIL, whatever the report claimed.

    The joint whitened-KDE overlap gate lives in ``checks`` like every other
    gate, so a failing joint check fails the verdict here too.
    """
    checks = report.get("checks") or []
    if any(c.get("pass") is False for c in checks):
        return "FAIL"
    explicit = str(report.get("verdict") or "").upper()
    if explicit in ("PASS", "WARN", "FAIL"):
        return explicit
    if checks and all(c.get("pass") is True for c in checks):
        return "PASS"
    return "PASS" if report.get("passed") and not checks else "INFO"


def contract_runs(calib_dir: Path) -> list[str]:
    if not calib_dir.exists():
        return []
    return [
        d.name for d in sorted(calib_dir.iterdir()) if d.is_dir() and (d / "ticks.jsonl").exists()
    ]


def latest_run(calib_dir: Path) -> str | None:
    """Newest run that exercised the control stack.

    Stationary watch snippets have no validated pose or action channels, so they
    would open the contract panel on an all-FAIL report; prefer the newest
    non-watch run and fall back to the newest overall.
    """
    from tools.l7.audit_contract import extract_run_metrics

    runs = contract_runs(calib_dir)
    if not runs:
        return None
    ordered = sorted(
        runs, key=lambda r: (calib_dir / r / "ticks.jsonl").stat().st_mtime, reverse=True
    )
    for run in ordered:
        if extract_run_metrics(calib_dir / run).get("mode") != "watch":
            return run
    return ordered[0]
