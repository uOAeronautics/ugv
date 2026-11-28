"""DAgger closed-loop frame dump and failure overlay helpers for `ugv.park`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ugv.park.perception import HEIGHT, WIDTH
from ugv.pose.dataset import CROP
from ugv.pose.dataset import H as S1_H
from ugv.pose.dataset import W as S1_W
from ugv.pose.detector import Detection
from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.render import Camera

CROP_MARGIN = 1.35
CROP_MIN, CROP_MAX = 64.0, 1400.0


@dataclass
class EpisodeLog:
    """Per-episode timing, keypoint diagnostics, and optional DAgger/overlay buffers."""

    render_ms: list[float] = field(default_factory=list)
    detect_ms: list[float] = field(default_factory=list)
    frames: int = 0
    in_view_frames: int = 0
    missed_views: int = 0
    empty_frames: int = 0
    empty_false_pts: int = 0
    kp_vis: np.ndarray = field(default_factory=lambda: np.zeros(15, np.int64))
    kp_det: np.ndarray = field(default_factory=lambda: np.zeros(15, np.int64))
    kp_du: list[list[float]] = field(default_factory=lambda: [[] for _ in range(15)])
    kp_dv: list[list[float]] = field(default_factory=lambda: [[] for _ in range(15)])
    kp_err: list[list[float]] = field(default_factory=lambda: [[] for _ in range(15)])
    state_frames: dict[str, int] = field(default_factory=dict)
    state_missed: dict[str, int] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    worst_overlay: tuple[float, np.ndarray, np.ndarray, np.ndarray, list, str, float] | None = None

    def record(
        self,
        gray: np.ndarray,
        cam_model: Camera,
        pose: np.ndarray,
        true_uv: np.ndarray,
        true_vis: np.ndarray,
        det_res: Detection,
        pts: list,
        m: np.ndarray | None,
        render_ms: float,
        detect_ms: float,
        fsm_state: str,
        t: float,
        rng: np.random.Generator,
        *,
        dump_frames: bool = False,
        dump_err_px: float = 10.0,
        keep_overlay: bool = False,
    ) -> None:
        self.frames += 1
        self.render_ms.append(round(render_ms, 2))
        self.detect_ms.append(round(detect_ms, 2))
        det_map = {int(i): (float(u), float(v)) for i, u, v in pts}
        n_vis = int(true_vis.sum())
        if n_vis == 0:
            self.empty_frames += 1
            self.empty_false_pts += len(pts)
        else:
            self.in_view_frames += 1
            if m is None:
                self.missed_views += 1
        if fsm_state:
            self.state_frames[fsm_state] = self.state_frames.get(fsm_state, 0) + 1
            if n_vis >= 2 and m is None:
                self.state_missed[fsm_state] = self.state_missed.get(fsm_state, 0) + 1
        bad_px = missed_vis = false_pos = 0
        max_err = 0.0
        for i in range(min(15, len(true_vis))):
            if true_vis[i]:
                self.kp_vis[i] += 1
                if i in det_map:
                    self.kp_det[i] += 1
                    du = det_map[i][0] - float(true_uv[i, 0])
                    dv = det_map[i][1] - float(true_uv[i, 1])
                    err = float(np.hypot(du, dv))
                    self.kp_du[i].append(round(du, 2))
                    self.kp_dv[i].append(round(dv, 2))
                    self.kp_err[i].append(round(err, 2))
                    max_err = max(max_err, err)
                    bad_px += int(err > dump_err_px)
                else:
                    missed_vis += 1
            elif i in det_map:
                false_pos += 1
        score = 10 * bad_px + 5 * missed_vis + 5 * false_pos + min(max_err, 100.0)
        if keep_overlay and (self.worst_overlay is None or score >= self.worst_overlay[0]):
            self.worst_overlay = (
                score,
                gray.copy(),
                true_uv.copy(),
                true_vis.copy(),
                pts,
                fsm_state,
                t,
            )
        if dump_frames:
            has_err = int(bad_px > 0 or missed_vis > 0 or false_pos > 0)
            self.samples.append(
                make_sample(
                    gray, det_res, true_uv, true_vis, cam_model, pose, rng, has_err, float(score), t
                )
            )

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "in_view_frames": self.in_view_frames,
            "missed_views": self.missed_views,
            "empty_frames": self.empty_frames,
            "empty_false_pts": self.empty_false_pts,
            "render_ms": self.render_ms,
            "detect_ms": self.detect_ms,
            "kp_vis": self.kp_vis.tolist(),
            "kp_det": self.kp_det.tolist(),
            "kp_du": self.kp_du,
            "kp_dv": self.kp_dv,
            "kp_err": self.kp_err,
            "state_frames": self.state_frames,
            "state_missed": self.state_missed,
        }


def make_sample(
    gray: np.ndarray,
    det_res: Detection,
    true_uv: np.ndarray,
    true_vis: np.ndarray,
    cam_model: Camera,
    pose: np.ndarray,
    rng: np.random.Generator,
    has_err: int,
    score: float,
    t: float,
) -> dict:
    """Build matched stage-1 (320x180) and stage-2 (256x256) DAgger records from one frame."""
    img_s1 = cv2.resize(gray, (S1_W, S1_H), interpolation=cv2.INTER_AREA)
    kp_s1 = (true_uv * np.array([S1_W / WIDTH, S1_H / HEIGHT])).astype(np.float32)
    vis_s1 = (
        true_vis
        & (kp_s1[:, 0] >= 0)
        & (kp_s1[:, 0] < S1_W)
        & (kp_s1[:, 1] >= 0)
        & (kp_s1[:, 1] < S1_H)
    )
    K_s1 = (np.diag([S1_W / WIDTH, S1_H / HEIGHT, 1.0]) @ cam_model.K).astype(np.float32)
    if det_res.box is not None:
        x0, y0, size = det_res.box
    elif true_vis.sum() >= 2:
        lo, hi = true_uv[true_vis].min(0), true_uv[true_vis].max(0)
        size = float(np.clip(max(hi - lo) * CROP_MARGIN, CROP_MIN, CROP_MAX))
        x0, y0 = (lo + hi) / 2 - size / 2
    else:
        size = float(rng.uniform(64.0, 700.0))
        x0, y0 = (
            float(rng.uniform(0.0, WIDTH) - size / 2),
            float(rng.uniform(0.0, HEIGHT) - size / 2),
        )
    s = CROP / size
    M = np.array([[s, 0.0, -x0 * s], [0.0, s, -y0 * s]])
    img_s2 = cv2.warpAffine(
        gray, M, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT
    )
    kp_s2 = ((true_uv - np.array([x0, y0])) * s).astype(np.float32)
    vis_s2 = (
        true_vis
        & (kp_s2[:, 0] >= 0)
        & (kp_s2[:, 0] < CROP)
        & (kp_s2[:, 1] >= 0)
        & (kp_s2[:, 1] < CROP)
    )
    K_s2 = cam_model.crop(x0, y0, size, CROP).K.astype(np.float32)
    return {
        "has_err": has_err,
        "score": score,
        "t": round(t, 2),
        "s1": (img_s1, kp_s1, vis_s1, pose, K_s1),
        "s2": (img_s2, kp_s2, vis_s2, pose, K_s2),
    }


def save_shard(shard_dir: Path, ep_idx: int, failed: bool, samples: list[dict]) -> str | None:
    """Persist prioritized closed-loop frames from one episode to a temporary `.npz` shard."""
    kept = [s for s in samples if failed or s["has_err"]]
    if not kept:
        return None
    kept.sort(key=lambda s: (-s["has_err"], -s["score"], s["t"]))
    kept = kept[:150]
    path = shard_dir / f"ep_{ep_idx:05d}.npz"
    np.savez(
        path,
        ep=np.full(len(kept), ep_idx, np.int32),
        failed=np.full(len(kept), int(failed), np.uint8),
        has_err=np.array([s["has_err"] for s in kept], np.uint8),
        score=np.array([s["score"] for s in kept], np.float32),
        s1_img=np.stack([s["s1"][0] for s in kept]),
        s1_kp=np.stack([s["s1"][1] for s in kept]),
        s1_vis=np.stack([s["s1"][2] for s in kept]),
        s1_pose=np.stack([s["s1"][3] for s in kept]),
        s1_K=np.stack([s["s1"][4] for s in kept]),
        s2_img=np.stack([s["s2"][0] for s in kept]),
        s2_kp=np.stack([s["s2"][1] for s in kept]),
        s2_vis=np.stack([s["s2"][2] for s in kept]),
        s2_pose=np.stack([s["s2"][3] for s in kept]),
        s2_K=np.stack([s["s2"][4] for s in kept]),
    )
    return str(path)


def write_dagger(out_dir: Path, shards: list[str], dump_max: int) -> dict:
    """Merge shards sorted by failure/detector-error priority into `pose_kp.py` dataset dirs."""
    if not shards:
        return {"written": 0, "failed_frames": 0, "err_frames": 0}
    items = []
    for si, path in enumerate(shards):
        with np.load(path) as z:
            for j in range(len(z["ep"])):
                items.append(
                    (
                        int(z["failed"][j]),
                        int(z["has_err"][j]),
                        float(z["score"][j]),
                        int(z["ep"][j]),
                        si,
                        j,
                    )
                )
    items.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3], x[5]))
    chosen = items[:dump_max]
    if not chosen:
        return {"written": 0, "failed_frames": 0, "err_frames": 0}
    by_shard: dict[int, list[int]] = {}
    for pos, (*_, si, j) in enumerate(chosen):
        by_shard.setdefault(si, []).append((pos, j))

    kp_names = list(keypoints(load_airframe(), near=True))
    n = len(chosen)
    for sub, prefix, crop in ((out_dir, "s1", False), (out_dir / "crop", "s2", True)):
        sub.mkdir(parents=True, exist_ok=True)
        with np.load(shards[chosen[0][4]]) as z0:
            h, w = z0[f"{prefix}_img"].shape[1:]
        imgs = np.empty((n, h, w), np.uint8)
        kps = np.empty((n, 15, 2), np.float32)
        viss = np.empty((n, 15), bool)
        poses = np.empty((n, 12), np.float32)
        Ks = np.empty((n, 3, 3), np.float32)
        for si, pairs in by_shard.items():
            with np.load(shards[si]) as z:
                pos_idx = [p for p, _ in pairs]
                src_idx = [j for _, j in pairs]
                imgs[pos_idx] = z[f"{prefix}_img"][src_idx]
                kps[pos_idx] = z[f"{prefix}_kp"][src_idx]
                viss[pos_idx] = z[f"{prefix}_vis"][src_idx]
                poses[pos_idx] = z[f"{prefix}_pose"][src_idx]
                Ks[pos_idx] = z[f"{prefix}_K"][src_idx]
        np.save(sub / "images.npy", imgs)
        np.save(sub / "kp.npy", kps)
        np.save(sub / "vis.npy", viss)
        np.save(sub / "pose.npy", poses)
        np.save(sub / "K.npy", Ks)
        meta = {
            "n": n,
            "width": int(w),
            "height": int(h),
            "crop": crop,
            "K": Ks[0].tolist(),
            "keypoints": kp_names,
            "source": "closed_loop_dagger",
            "failed_frames": sum(x[0] for x in chosen),
            "err_frames": sum(x[1] for x in chosen),
        }
        (sub / "meta.json").write_text(json.dumps(meta, indent=1))
        rows_n = min(4, n // 6)
        if rows_n >= 1:
            sheet = np.vstack([np.hstack(list(imgs[r * 6 : r * 6 + 6])) for r in range(rows_n)])
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
            for k_idx in range(rows_n * 6):
                for (u, v_), ok in zip(kps[k_idx], viss[k_idx], strict=True):
                    if ok:
                        cv2.circle(
                            sheet,
                            (int(u) + (k_idx % 6) * w, int(v_) + (k_idx // 6) * h),
                            2,
                            (0, 0, 255),
                            -1,
                        )
            cv2.imwrite(str(sub / "sheet.jpg"), sheet)
    return {
        "written": n,
        "failed_frames": sum(x[0] for x in chosen),
        "err_frames": sum(x[1] for x in chosen),
    }


def draw_overlay(
    gray: np.ndarray, true_uv: np.ndarray, true_vis: np.ndarray, det_pts: list, label: str
) -> np.ndarray:
    """Render green true keypoints, orange detected points, and error lines onto a BGR copy."""
    vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    det_map = {int(i): (int(round(u)), int(round(v))) for i, u, v in det_pts}
    for i in range(len(true_uv)):
        if true_vis[i]:
            p = (int(round(true_uv[i, 0])), int(round(true_uv[i, 1])))
            cv2.circle(vis_img, p, 5, (0, 220, 0), 2)
            if i in det_map:
                cv2.line(vis_img, p, det_map[i], (0, 255, 255), 1)
    for i, (du, dv) in det_map.items():
        cv2.drawMarker(vis_img, (du, dv), (0, 140, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(
            vis_img, str(i), (du + 5, dv - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1
        )
    cv2.putText(vis_img, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis_img
