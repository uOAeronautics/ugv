"""Score a parameter set on many episodes, in parallel, with common random numbers.

Episode `i` of a batch with base seed `s` always uses scenario seed `s * 100003 + i`, so two
parameter sets compared on the same `(seed, n)` see the same airframes, cars and cameras: their
difference is the parameters, not luck. Success is reported with a Wilson 95% interval.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ugv.park.camera import NetworkPerception
from ugv.park.controller import ParkParams
from ugv.park.dagger import draw_overlay, save_shard, write_dagger
from ugv.park.episode import PRESETS, Result, Scenario, as_dict, run
from ugv.park.perception import DetectorModel
from ugv.pose.detector import default_runs
from ugv.pose.mesh import keypoints, load_airframe


@dataclass(frozen=True)
class Conditions:
    """What the world throws at the controller. `plant_scale` 0 = the nominal car; `faults` is the
    severity of per-episode car and sensor faults (faults.py), `only` one fault family; `start`
    "ring" (1.3-3 m away), "close" (0.4-1.3 m, any side), "presets" (named Challenge starts),
    or "mixed" (presets + random ring starts); `floor` is traction severity; `perception` is
    "model" (sampled DetectorModel) or "network" (rendered 1280x720 frames + ONNX detector)."""

    plant_scale: float = 1.0
    detector: DetectorModel = DetectorModel()
    faults: float = 0.0
    only: str | None = None
    start: str = "ring"
    floor: float = 0.0
    perception: str = "model"
    preset_repeats: int = 0


_WORKER: dict = {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _scenario(i: int, n: int, seed: int, cond: Conditions) -> tuple[int, str, Scenario]:
    s = seed * 100003 + i
    rng = np.random.default_rng(s)
    names = tuple(PRESETS)
    if cond.start == "presets":
        name = names[i % len(names)]
        return (
            s,
            name,
            Scenario(
                (0.0, 0.0, 0.0),
                PRESETS[name],
                cond.plant_scale,
                cond.detector,
                cond.faults,
                cond.only,
                floor=cond.floor,
            ),
        )
    if cond.start == "mixed":
        repeats = cond.preset_repeats or max(1, round(0.30 * n / len(names)))
        if i < len(names) * repeats:
            name = names[i % len(names)]
            return (
                s,
                name,
                Scenario(
                    (0.0, 0.0, 0.0),
                    PRESETS[name],
                    cond.plant_scale,
                    cond.detector,
                    cond.faults,
                    cond.only,
                    floor=cond.floor,
                ),
            )
        return (
            s,
            "random:ring",
            Scenario.sample(
                rng, cond.plant_scale, cond.detector, cond.faults, cond.only, "ring", cond.floor
            ),
        )
    return (
        s,
        f"random:{cond.start}",
        Scenario.sample(
            rng, cond.plant_scale, cond.detector, cond.faults, cond.only, cond.start, cond.floor
        ),
    )


def _init_worker(perception: str, dump_err_px: float, has_dump: bool, has_overlay: bool) -> None:
    if perception == "network":
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        cv2.setNumThreads(1)
        _WORKER["net"] = NetworkPerception(
            threads=1, dump_frames=has_dump, dump_err_px=dump_err_px, keep_overlay=has_overlay
        )


def _one(args: tuple) -> tuple[dict, dict | None, str | None]:
    i, n, seed, params, cond, shard_dir, overlay_dir = args
    s, start_name, sc = _scenario(i, n, seed, cond)
    net: NetworkPerception | None = _WORKER.get("net")
    t0 = time.perf_counter()
    if net is not None:
        net.reset_episode()
    res = run(sc, params, seed=s, perception=net)
    row = as_dict(res) | {
        "episode": i,
        "seed": s,
        "start": start_name,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    ep_stats, shard = None, None
    if net is not None:
        ep_stats = net.ep.summary()
        row |= {
            "frames": net.ep.frames,
            "missed_views": net.ep.missed_views,
            "mean_render_ms": round(float(np.mean(net.ep.render_ms)), 2)
            if net.ep.render_ms
            else None,
            "mean_detect_ms": round(float(np.mean(net.ep.detect_ms)), 2)
            if net.ep.detect_ms
            else None,
        }
        if shard_dir is not None:
            shard = save_shard(Path(shard_dir), i, not res.success, net.ep.samples)
        if overlay_dir is not None and not res.success and net.ep.worst_overlay is not None:
            _, gray, tuv, tvis, pts, st, ft = net.ep.worst_overlay
            label = f"ep{i} {start_name} {res.outcome} ({st} t={ft:.1f}s)"
            tag = res.outcome.replace(":", "_").replace(" ", "_").replace(";", "")
            cv2.imwrite(
                str(Path(overlay_dir) / f"ep_{i:04d}_{start_name.replace(':', '_')}_{tag}.png"),
                draw_overlay(gray, tuv, tvis, pts, label),
            )
    return row, ep_stats, shard


def _agg_network(rows: list[dict], ep_stats: list[dict], wall_s: float) -> tuple[dict, dict]:
    ren = [x for e in ep_stats for x in e["render_ms"]]
    det = [x for e in ep_stats for x in e["detect_ms"]]
    tot = [r + d for r, d in zip(ren, det, strict=True)]
    ep_walls = [r["wall_s"] for r in rows]
    timing = {
        "total_wall_s": round(wall_s, 2),
        "mean_episode_wall_s": round(float(np.mean(ep_walls)), 2) if ep_walls else None,
        "median_episode_wall_s": round(float(np.median(ep_walls)), 2) if ep_walls else None,
        "render_ms_mean": round(float(np.mean(ren)), 2) if ren else None,
        "render_ms_median": round(float(np.median(ren)), 2) if ren else None,
        "detect_ms_mean": round(float(np.mean(det)), 2) if det else None,
        "detect_ms_median": round(float(np.median(det)), 2) if det else None,
        "tick_ms_mean": round(float(np.mean(tot)), 2) if tot else None,
        "tick_ms_median": round(float(np.median(tot)), 2) if tot else None,
        "tick_ms_p90": round(float(np.percentile(tot, 90)), 2) if tot else None,
    }
    kp_names = list(keypoints(load_airframe(), near=True))
    n_fr = sum(e["frames"] for e in ep_stats)
    in_view = sum(e["in_view_frames"] for e in ep_stats)
    missed = sum(e["missed_views"] for e in ep_stats)
    empty_fr = sum(e["empty_frames"] for e in ep_stats)
    empty_fp = sum(e["empty_false_pts"] for e in ep_stats)
    st_f, st_m = Counter(), Counter()
    for e in ep_stats:
        st_f.update(e["state_frames"])
        st_m.update(e["state_missed"])
    per_kp = {}
    for i, name in enumerate(kp_names):
        vis_n = sum(e["kp_vis"][i] for e in ep_stats)
        det_n = sum(e["kp_det"][i] for e in ep_stats)
        du = [x for e in ep_stats for x in e["kp_du"][i]]
        dv = [x for e in ep_stats for x in e["kp_dv"][i]]
        err = [x for e in ep_stats for x in e["kp_err"][i]]
        per_kp[name] = {
            "visible": vis_n,
            "detected": det_n,
            "recall": round(det_n / vis_n, 4) if vis_n else None,
            "bias_du_px": round(float(np.mean(du)), 2) if du else None,
            "bias_dv_px": round(float(np.mean(dv)), 2) if dv else None,
            "median_err_px": round(float(np.median(err)), 2) if err else None,
            "p90_err_px": round(float(np.percentile(err, 90)), 2) if err else None,
        }
    kp_stats = {
        "frames_rendered": n_fr,
        "frames_in_view": in_view,
        "missed_views": missed,
        "missed_view_rate": round(missed / max(1, in_view), 4),
        "empty_frames": empty_fr,
        "false_pts_per_empty": round(empty_fp / max(1, empty_fr), 3) if empty_fr else None,
        "by_state": {s: {"frames": st_f[s], "missed_views": st_m[s]} for s in sorted(st_f)},
        "per_keypoint": per_kp,
    }
    return timing, kp_stats


def evaluate(
    params: ParkParams,
    n: int = 200,
    seed: int = 11,
    cond: Conditions | None = None,
    workers: int | None = None,
    dump_frames: Path | None = None,
    dump_max: int = 6000,
    dump_err_px: float = 10.0,
    failure_overlays: Path | None = None,
) -> dict:
    cond = cond or Conditions()
    if cond.perception == "network":
        for p in default_runs():
            if not (p / "model.onnx").exists():
                raise FileNotFoundError(
                    f"{p / 'model.onnx'} is missing; cannot run --perception network"
                )
        load_airframe()
    shard_dir = dump_frames / ".shards" if dump_frames is not None else None
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    if failure_overlays is not None:
        failure_overlays.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            i,
            n,
            seed,
            params,
            cond,
            str(shard_dir) if shard_dir else None,
            str(failure_overlays) if failure_overlays else None,
        )
        for i in range(n)
    ]
    workers = workers or min(16, os.cpu_count() or 1)
    t0 = time.perf_counter()
    try:
        if workers == 1:
            _init_worker(
                cond.perception, dump_err_px, dump_frames is not None, failure_overlays is not None
            )
            out = [_one(j) for j in jobs]
        else:
            with ProcessPoolExecutor(
                workers,
                initializer=_init_worker,
                initargs=(
                    cond.perception,
                    dump_err_px,
                    dump_frames is not None,
                    failure_overlays is not None,
                ),
            ) as ex:
                chunk = 1 if cond.perception == "network" else max(1, n // (workers * 4))
                out = list(ex.map(_one, jobs, chunksize=chunk))
        wall_s = time.perf_counter() - t0
        rows = [r for r, _, _ in out]
        ep_stats = [s for _, s, _ in out if s is not None]
        shards = [p for _, _, p in out if p is not None]
        summary = summarize(rows, params, n, seed, cond)
        if ep_stats:
            timing, kp_stats = _agg_network(rows, ep_stats, wall_s)
            summary["timing"] = timing
            summary["keypoint_stats"] = kp_stats
        if dump_frames is not None:
            summary["dagger_dump"] = write_dagger(dump_frames, shards, dump_max)
    finally:
        _WORKER.clear()
        if shard_dir is not None and shard_dir.exists():
            shutil.rmtree(shard_dir, ignore_errors=True)
    return summary


def summarize(rows: list[dict], params: ParkParams, n: int, seed: int, cond: Conditions) -> dict:
    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, n)
    ok = [r for r in rows if r["success"]]
    outcomes = Counter(r["outcome"] for r in rows)
    by_start_rows: dict[str, list[dict]] = defaultdict(list)
    by_end: dict[str, Counter] = defaultdict(Counter)
    by_last: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_start_rows[r.get("start", "unknown")].append(r)
        if not r["success"]:
            by_end[r["outcome"]][r.get("end_state", "")] += 1
            by_last[r["outcome"]][r.get("last_active_state", "")] += 1
    by_start = {}
    for st, grp in by_start_rows.items():
        sk = sum(x["success"] for x in grp)
        slo, shi = wilson(sk, len(grp))
        by_start[st] = {
            "n": len(grp),
            "success": sk,
            "rate": round(sk / len(grp), 4),
            "wilson95": [round(slo, 4), round(shi, 4)],
            "collisions": sum(x["outcome"] == "collision" for x in grp),
            "outcomes": dict(Counter(x["outcome"] for x in grp).most_common()),
        }
    return {
        "n": n,
        "seed": seed,
        "success": k,
        "rate": k / n,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "collisions": outcomes.get("collision", 0),
        "outcomes": dict(outcomes.most_common()),
        "by_start": by_start,
        "by_end_state": {k_: dict(v.most_common()) for k_, v in by_end.items()},
        "by_last_active_state": {k_: dict(v.most_common()) for k_, v in by_last.items()},
        "median_time_s": float(np.median([r["time_s"] for r in ok])) if ok else None,
        "mean_retries": float(np.mean([r["retries"] for r in rows])),
        "params": asdict(params),
        "conditions": {**asdict(cond), "detector": asdict(cond.detector)},
        "episodes": rows,
    }


def failure_examples(rows: list[dict], k: int = 3) -> dict[str, list[int]]:
    """Episode indices per failure outcome, to replay (`run(..., keep_trace=True)`)."""
    out: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if not r["success"]:
            out.setdefault(r["outcome"], [])
            if len(out[r["outcome"]]) < k:
                out[r["outcome"]].append(i)
    return out


__all__ = ["Conditions", "Result", "evaluate", "failure_examples", "wilson"]
