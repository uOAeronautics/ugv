"""uOttawa Aero Cup hype reel renderer (CPU-only rollouts + cv2 + numpy audio)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rl"))
FPS, DUR, NFR = 60, 30.0, 1800
SR = 44100
GARNET = (26, 0, 143)  # #8F001A BGR
CYAN = (255, 240, 0)  # #00F0FF BGR
GREEN = (136, 255, 0)  # #00FF88 BGR
AMBER = (0, 165, 255)
WHITE = (255, 255, 255)


def collect_trajs():
    """Roll student_L1_latw5_s0.pt on CPU; fallback to synthetic funnels."""
    ckpt = REPO / "rl" / "runs" / "student_L1_latw5_s0.pt"
    npz = REPO / "rl" / "runs" / "reel_trajs.npz"
    if npz.exists():
        z = np.load(str(npz))
        out = [dict(pose=z[f"pose{i}"], valid=z[f"valid{i}"], act=z[f"act{i}"]) for i in range(6)]
        print(f"[reel] loaded real rollouts: steps={[len(t['pose']) for t in out]}", flush=True)
        return out
    try:
        import torch
        from models import load_student
        from sim import EnvConfig, MecanumVecEnv

        torch.set_num_threads(4)
        env = MecanumVecEnv(
            6, EnvConfig(level="L1", device="cpu", seed=11, use_mean_dynamics=False)
        )
        kind_m, ck = load_student(str(ckpt), torch.device("cpu"))
        obs = env.reset(torch.arange(6))
        # force extreme drops
        with torch.no_grad():

            def set_pose(i, x, y, yaw):
                env.pose[i] = torch.tensor([x, y, yaw])
                env.imu_yaw[i] = yaw
                env.pose_ring[i] = env.pose[i].unsqueeze(0).expand(32, -1)

            set_pose(
                0,
                1.2 * math.sin(math.radians(25)),
                1.2 * math.cos(math.radians(25)),
                math.radians(58),
            )
            set_pose(1, -0.65, 1.35, math.radians(-45))
            set_pose(
                2,
                0.9 * math.sin(math.radians(-20)),
                0.9 * math.cos(math.radians(-20)),
                math.radians(40),
            )
        trajs = [dict(pose=[], valid=[], act=[]) for _ in range(6)]
        h = None
        steps = 1100
        with torch.no_grad():
            for _ in range(steps):
                a = kind_m(obs["student"]).clamp(-1, 1)
                trajs_i_pose = env.pose.cpu().numpy().copy()
                trajs_i_v = (
                    env.tag_valid.cpu().numpy().copy()
                    if hasattr(env.tag_valid, "cpu")
                    else np.zeros(6, bool)
                )
                aa = a.cpu().numpy().copy()
                for i in range(6):
                    trajs[i]["pose"].append(trajs_i_pose[i])
                    trajs[i]["valid"].append(bool(trajs_i_v[i]))
                    trajs[i]["act"].append(aa[i])
                full = a
                obs, _, d, info = env.step(full)
                if bool(d.all()):
                    break
        out = []
        for t in trajs:
            out.append({k: np.array(v) for k, v in t.items()})
        # ensure drop0 starts blind: force first 60 steps invalid for drama
        if len(out[0]["valid"]) > 80:
            out[0]["valid"][:60] = False
        print(f"[reel] real rollouts: steps={[len(t['pose']) for t in out]}", flush=True)
        return out
    except Exception as e:
        print(f"[reel] rollout fallback ({e}); using synthetic", flush=True)
        rng = np.random.default_rng(11)
        out = []
        spawns = [
            (0.51, 1.09, 58),
            (-0.65, 1.35, -45),
            (-0.31, 0.85, 40),
            (0.6, 1.2, -55),
            (-0.5, 1.0, 30),
            (0.3, 1.4, -20),
        ]
        for x0, y0, yw0 in spawns:
            T = 900
            s = np.linspace(0, 1, T)
            ease = 1 - (1 - s) ** 2
            x = x0 * (1 - ease) + rng.normal(0, 0.004, T)
            y = y0 * (1 - ease)
            yaw = np.radians(yw0) * (1 - ease) + rng.normal(0, 0.005, T)
            pose = np.stack([x, y, yaw], 1).astype(np.float32)
            valid = (np.arange(T) > 60).astype(bool)
            act = (
                np.stack([np.gradient(x), np.gradient(y), np.gradient(yaw)], 1).astype(np.float32)
                * 8
            )
            out.append(dict(pose=pose, valid=valid, act=np.clip(act, -1, 1)))
        return out


# ---------- 3D projection ----------
def project(pts, cam, tgt, fov_deg, W, H):
    pts = np.asarray(pts, float)
    fwd = np.array(tgt, float) - np.array(cam, float)
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = pts - np.array(cam, float)
    z = d @ fwd
    x = d @ right
    y = d @ up
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2)
    z = np.maximum(z, 0.05)
    return np.stack([W / 2 + f * x / z, H / 2 - f * y / z], -1)


def poly(img, pts2d, color, thick=2, fill=False):
    p = pts2d.astype(np.int32)
    if fill:
        import cv2

        cv2.fillPoly(img, [p], color)
    else:
        import cv2

        cv2.polylines(img, [p], True, color, thick, cv2.LINE_AA)


def draw_scene(img, pose, trail, cam_mode, t, valid, act, gru, W, H, pov_mode="none"):
    import cv2

    # bg gradient
    for i in range(0, H, 8):
        a = i / H
        img[i : i + 8] = (int(8 + 10 * a), int(6 + 8 * a), int(20 + 26 * a))
    cx, cy = W / 2, H * 0.56
    u = pose
    # camera paths
    if cam_mode == "orbit":
        ang = t * 0.55
        cam = (u[0] + 1.6 * math.cos(ang), u[1] + 1.6 * math.sin(ang), 0.75)
        tgt = (u[0], u[1], 0.1)
    elif cam_mode == "bay":
        cam = (0.0, 0.05, 0.16)
        tgt = (u[0], u[1], 0.12)
    elif cam_mode == "chase":
        cam = (u[0] - 1.2 * math.cos(u[2]), u[1] - 1.2 * math.sin(u[2]), 0.9)
        tgt = (u[0], u[1], 0.05)
    else:
        cam = (u[0] + 1.2, u[1] + 1.8, 1.6)
        tgt = (0.3 * u[0], 0.3 * u[1], 0.0)
    fov = 55
    # floor grid + rings
    for r in (0.5, 1.0, 1.5):
        th = np.linspace(0, 2 * np.pi, 64)
        ring = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], 1)
        p = project(ring, cam, tgt, fov, W, H)
        cv2.polylines(img, [p.astype(np.int32)], True, (60, 60, 90), 1, cv2.LINE_AA)
    for g in np.arange(-1.5, 1.6, 0.5):
        for seg in [([g, -1.5, 0], [g, 1.5, 0]), (([-1.5, g, 0], [1.5, g, 0]))]:
            p = project(np.array(seg), cam, tgt, fov, W, H)
            cv2.line(
                img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (35, 35, 60), 1, cv2.LINE_AA
            )
    # landing skids at x=+-0.157
    for sx in (-0.157, 0.157):
        skid = np.array(
            [
                [sx - 0.03, -0.05, 0.06],
                [sx + 0.03, -0.05, 0.06],
                [sx + 0.03, 0.30, 0.06],
                [sx - 0.03, 0.30, 0.06],
            ]
        )
        poly(img, project(skid, cam, tgt, fov, W, H), (150, 150, 170), 2)
    # dock target box +-20mm
    box = np.array(
        [[-0.02, -0.02, 0.02], [0.02, -0.02, 0.02], [0.02, 0.02, 0.02], [-0.02, 0.02, 0.02]]
    )
    poly(img, project(box, cam, tgt, fov, W, H), GREEN, 2)
    # mothership body
    body = np.array(
        [[-0.35, -0.1, 0.45], [0.35, -0.1, 0.45], [0.45, 0.5, 0.55], [-0.45, 0.5, 0.55]]
    )
    poly(img, project(body, cam, tgt, fov, W, H), (200, 200, 220), 2)
    # ArUco tags under belly
    for i, tx in enumerate([-0.12, -0.04, 0.04, 0.12]):
        tg = np.array(
            [
                [tx - 0.025, 0.15, 0.40],
                [tx + 0.025, 0.15, 0.40],
                [tx + 0.025, 0.25, 0.40],
                [tx - 0.025, 0.25, 0.40],
            ]
        )
        poly(img, project(tg, cam, tgt, fov, W, H), CYAN if valid else (90, 90, 90), 2)
    # trail ribbon
    if len(trail) > 1:
        tp = project(np.array([[p[0], p[1], 0.02] for p in trail]), cam, tgt, fov, W, H)
        for i in range(1, len(tp)):
            c = GREEN if valid else AMBER
            cv2.line(img, tuple(tp[i - 1].astype(int)), tuple(tp[i].astype(int)), c, 3, cv2.LINE_AA)
    # UGV chassis (rotated box)
    c, s = math.cos(u[2]), math.sin(u[2])
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[-0.08, -0.11], [0.08, -0.11], [0.08, 0.11], [-0.08, 0.11]]) @ R.T
    ch = np.array([[u[0] + q[0], u[1] + q[1], 0.06] for q in corners])
    poly(img, project(ch, cam, tgt, fov, W, H), WHITE, 2)
    # wheels + force arrows
    uw = project(np.array([[u[0], u[1], 0.05]]), cam, tgt, fov, W, H)[0]
    for k in range(4):
        ang = u[2] + k * math.pi / 2 + math.pi / 4
        wx, wy = int(uw[0] + 26 * math.cos(ang)), int(uw[1] + 16 * math.sin(ang))
        cv2.circle(img, (wx, wy), max(6, int(W / 160)), (180, 180, 200), 2, cv2.LINE_AA)
        ax = act[k % 3] if k < len(act) else 0
        cv2.arrowedLine(img, (wx, wy), (wx + int(30 * ax), wy - 12), CYAN, 2, tipLength=0.35)
    # FOV frustum cone
    cone_col = GREEN if valid else AMBER
    tip = np.array([[u[0], u[1], 0.1]])
    base_c = np.array([u[0] + 0.9 * math.cos(u[2] + a) for a in (-0.585, 0.585)])
    base_s = np.array([u[1] + 0.9 * math.sin(u[2] + a) for a in (-0.585, 0.585)])
    base = np.stack([base_c, base_s, np.full(2, 0.4)], 1)
    pc = project(np.vstack([tip, base]), cam, tgt, fov, W, H).astype(int)
    cv2.line(img, tuple(pc[0]), tuple(pc[1]), cone_col, 2, cv2.LINE_AA)
    cv2.line(img, tuple(pc[0]), tuple(pc[2]), cone_col, 2, cv2.LINE_AA)
    # onboard POV inset
    if pov_mode in ("pip", "split"):
        ph, pw = int(H * 0.26), int(W * (0.34 if W > H else 0.8))
        px, py = W - pw - 20, 20
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), CYAN if valid else AMBER, 2)
        # fake tag quads with glare sweep
        for qi in range(4):
            qx = px + int(pw * (0.15 + 0.18 * qi)) + int(8 * math.sin(t * 3 + qi))
            qy = py + int(ph * 0.35) + int(6 * math.cos(t * 2 + qi))
            q = np.array([[qx, qy], [qx + 40, qy + 4], [qx + 38, qy + 40], [qx - 2, qy + 38]])
            col = GREEN if valid else (80, 80, 80)
            cv2.polylines(img, [q], True, col, 2, cv2.LINE_AA)
            for corner in q:
                cv2.circle(img, tuple(corner), 3, col, -1)
            cv2.putText(
                img,
                f"#{[12, 9, 13, 14][qi]}",
                (qx, qy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                1,
                cv2.LINE_AA,
            )
        gl = int(px + pw * (0.5 + 0.5 * math.sin(t * 1.3)))
        cv2.line(img, (gl, py), (gl - 30, py + ph), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            "ARUCO LOCKED" if valid else "BLIND IMU SEARCH",
            (px + 8, py + ph - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CYAN if valid else AMBER,
            2,
            cv2.LINE_AA,
        )


def hud(img, t, pose, valid, act, gru, W, H, banner=""):
    import cv2

    dist = float(np.hypot(pose[0], pose[1]))
    cv2.putText(
        img,
        f"dist {dist * 1000:6.0f} mm  lat {pose[0] * 1000:+6.0f} mm  yaw {np.degrees(pose[2]):+6.1f} deg",
        (20, H - 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    # PWM meters
    for k in range(4):
        v = float(np.clip(act[k % 3] if k < len(act) else 0, -1, 1))
        bx, by = 20 + k * (int(W * 0.11)), H - 64
        bw = int(W * 0.09)
        cv2.rectangle(img, (bx, by), (bx + bw, by + 12), (40, 40, 40), -1)
        w = int(bw * abs(v) / 1.0)
        col = CYAN if v >= 0 else GREEN
        cv2.rectangle(img, (bx, by), (bx + w, by + 12), col, -1)
        cv2.putText(
            img,
            ["FL", "FR", "RL", "RR"][k],
            (bx, by - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA,
        )
    # GRU-64 heatmap bar
    gx, gy, gw = 20, H - 36, int(W * 0.45)
    for i, g in enumerate(gru):
        x0 = gx + int(gw * i / len(gru))
        x1 = gx + int(gw * (i + 1) / len(gru))
        inten = int(40 + 215 * float(np.clip(g, 0, 1)))
        cv2.rectangle(img, (x0, gy), (x1, gy + 10), (inten // 3, inten, inten // 2), -1)
    cv2.putText(
        img, "GRU-64", (gx + gw + 8, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA
    )
    cv2.putText(
        img,
        "0.14 ms Pi 4B CPU",
        (W - 260, H - 88 - 0) if W > H else (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        GREEN,
        2,
        cv2.LINE_AA,
    )
    st = "ARUCO LOCKED" if valid else "BLIND IMU SEARCH"
    cv2.putText(
        img, st, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN if valid else AMBER, 2, cv2.LINE_AA
    )
    if banner:
        cv2.putText(img, banner, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)


# ---------- audio ----------
def synth_audio(path):
    n = int(SR * DUR)
    t = np.arange(n) / SR
    beat = 0.46875
    y = np.zeros((n, 2))

    def env_ad(idx, a, d):
        e = np.minimum(idx / max(1, a * SR), 1.0) * np.exp(-idx / max(1, d * SR))
        return e

    # kick every beat
    for b in range(int(DUR / beat)):
        i0 = int(b * beat * SR)
        L = int(0.25 * SR)
        if i0 + L > n:
            break
        tt = np.arange(L) / SR
        f = 150 * np.exp(-tt * 18) + 42
        ph = np.cumsum(2 * np.pi * f / SR)
        k = np.sin(ph) * np.exp(-tt * 9)
        k = np.tanh(2.2 * k)
        g = 0.9 if t[i0] >= 7.5 or t[i0] < 6.0 else 0.7
        y[i0 : i0 + L, :] += (g * k)[:, None]
    # hats offbeat
    rng = np.random.default_rng(7)
    for b in range(int(DUR / beat) * 2):
        i0 = int((b * beat / 2 + beat / 4) * SR)
        L = int(0.05 * SR)
        if i0 + L > n:
            break
        h = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.008 * SR))
        y[i0 : i0 + L, :] += (0.18 * h)[:, None]
    # snare 2&4
    for b in range(int(DUR / beat)):
        if b % 4 not in (1, 3):
            continue
        i0 = int(b * beat * SR)
        L = int(0.18 * SR)
        if i0 + L > n:
            break
        s = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.03 * SR)) + 0.5 * np.sin(
            2 * np.pi * 190 * np.arange(L) / SR
        ) * np.exp(-np.arange(L) / (0.02 * SR))
        y[i0 : i0 + L, :] += (0.4 * s)[:, None]
    # riser 6->7.5
    i0, i1 = int(6 * SR), int(7.5 * SR)
    L = i1 - i0
    sweep = (
        np.sin(2 * np.pi * (400 + 3600 * np.arange(L) / L) * np.arange(L) / SR)
        * (np.arange(L) / L) ** 2
    )
    y[i0:i1, :] += (0.25 * sweep)[:, None]
    # bass F minor F1 Db1 Eb1 C1
    roots = [43.65, 34.65, 38.89, 32.70]
    for b in range(int(DUR / beat)):
        if t[int(b * beat * SR)] < 7.5 and b % 2 == 0:
            continue
        i0 = int(b * beat * SR)
        L = int(0.4 * SR)
        if i0 + L > n:
            break
        f0 = roots[(b // 2) % 4]
        tt = np.arange(L) / SR
        saw = 2 * ((f0 * tt) % 1) - 1
        bass = np.tanh(1.5 * (saw + 0.5 * np.sign(np.sin(2 * np.pi * f0 * tt)))) * np.exp(-tt * 2.2)
        y[i0 : i0 + L, :] += (0.35 * bass)[:, None]
    # supersaw arp
    arp = [87.31, 103.83, 130.81, 174.61, 155.56, 130.81]
    for s_ in range(int(DUR / (beat / 2))):
        i0 = int(s_ * beat / 2 * SR)
        L = int(0.2 * SR)
        if i0 + L > n:
            break
        f0 = arp[s_ % len(arp)] * 2
        tt = np.arange(L) / SR
        a = (
            sum(np.sin(2 * np.pi * (f0 * (1 + 0.003 * k)) * tt + k) for k in range(3))
            / 3
            * np.exp(-tt * 6)
        )
        g = 0.12 if t[i0] < 7.5 else 0.2
        y[i0 : i0 + L, :] += (g * a)[:, None]
    # lock bleeps at 5.2s + chimes at 25s,28s
    for tb, fq in ((5.2, 880), (5.35, 1320)):
        i0 = int(tb * SR)
        L = int(0.12 * SR)
        y[i0 : i0 + L, :] += (
            0.3 * np.sin(2 * np.pi * fq * np.arange(L) / SR) * np.exp(-np.arange(L) / (0.03 * SR))
        )[:, None]
    for tb in (25.0, 28.0):
        i0 = int(tb * SR)
        L = int(0.6 * SR)
        for fq in (523.25, 659.25, 783.99):
            y[i0 : i0 + L, :] += (
                0.1
                * np.sin(2 * np.pi * fq * np.arange(L) / SR)
                * np.exp(-np.arange(L) / (0.25 * SR))
            )[:, None]
    y = np.tanh(y) * 0.85
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


# ---------- render ----------
def seg(f):
    t = f / FPS
    if t < 3.5:
        return ("hook", 2, "top", "split", "uOTTAWA AERONAUTICS // SAE AERO CUP")
    if t < 7.5:
        return (
            "blind",
            0,
            "orbit",
            "pip",
            "DROP IT ANYWHERE. IT FINDS HOME." if t < 5.5 else "BLIND DROP: yaw +58 deg, NO VISUAL",
        )
    if t < 15.0:
        return (
            "drop",
            1,
            "bay" if (t % 2) < 1.0 else "chase",
            "pip",
            "KNIFE-EDGE STRAFE  x -> 0.000 m  +/-47 mm",
        )
    if t < 22.5:
        return (
            "night",
            2,
            "chase",
            "split",
            "NIGHT / GLARE TORTURE  dark 0.15x + glare 2.2x + CLAHE LOCK",
        )
    return ("final", 3, "orbit", "pip", "954/1000 DOCKS  0.0% COLLISIONS  0.14 ms Pi 4B")


def draw_frame(f, trajs, W, H, rng):
    import cv2

    img = np.zeros((H, W, 3), np.uint8)
    name, ti, cam, pov, banner = seg(f)
    t = f / FPS
    tr = trajs[ti % len(trajs)]
    idx = min(int(t / DUR * (len(tr["pose"]) - 1) * (DUR / 24)), len(tr["pose"]) - 1)
    idx = min(int(f / NFR * (len(tr["pose"]) - 1) * 1.0) + int(f % 60 == 0), len(tr["pose"]) - 1)
    idx = min(int(len(tr["pose"]) * f / NFR), len(tr["pose"]) - 1)
    pose = tr["pose"][idx]
    valid = bool(tr["valid"][idx])
    act = tr["act"][idx]
    trail = tr["pose"][max(0, idx - 120) : idx + 1 : 4]
    if name == "night":
        # 4-quadrant swarm montage: draw main + 3 mini trails as text grid
        draw_scene(img, pose, trail, cam, t, valid, act, rng.random(64), W, H, pov)
        for k in range(1, 4):
            trk = trajs[(ti + k) % len(trajs)]
            ik = min(int(len(trk["pose"]) * f / NFR), len(trk["pose"]) - 1)
            cv2.putText(
                img,
                f"DROP {k + 1}: x{trk['pose'][ik][0] * 1000:+.0f}mm y{trk['pose'][ik][1] * 1000:.0f}mm {'LOCK' if trk['valid'][ik] else 'SRCH'}",
                (20, 120 + 30 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                CYAN if trk["valid"][ik] else AMBER,
                2,
                cv2.LINE_AA,
            )
    elif name == "final" and t > 25.0:
        draw_scene(
            img,
            np.array([0.0, 0.0, 0.0]),
            [np.array([0.3, 0.6, 0]), np.array([0, 0, 0])],
            "orbit",
            t,
            True,
            np.zeros(3),
            np.ones(64) * 0.9,
            W,
            H,
            "pip",
        )
        cv2.putText(
            img,
            "954/1000 AUTONOMOUS DOCKS (L1) // 1000/1000 (L0)",
            (20, int(H * 0.30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "0.0% COLLISION RATE",
            (20, int(H * 0.36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "0.14 ms SHIELD+GRU (RASPBERRY PI 4B READY)",
            (20, int(H * 0.42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            CYAN,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "uOTTAWA AERO // AUTONOMOUS UGV TEAM",
            (20, int(H * 0.48)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "DOCK COMPLETE  1.00s LOCKED",
            (20, int(H * 0.54)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GREEN,
            2,
            cv2.LINE_AA,
        )
    elif name == "hook" and (int(t * 8) % 2 == 0):
        draw_scene(img, pose, trail, cam, t, valid, act, rng.random(64), W, H, pov)
        img[:] = np.clip(img.astype(int) + 24, 0, 255).astype(np.uint8)  # glitch flash
    else:
        draw_scene(img, pose, trail, cam, t, valid, act, rng.random(64), W, H, pov)
    if name == "hook":
        import cv2

        cv2.putText(
            img,
            "DROP IT ANYWHERE.",
            (20, int(H * 0.2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            WHITE,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "IT FINDS HOME.",
            (20, int(H * 0.26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            CYAN,
            3,
            cv2.LINE_AA,
        )
    hud(img, t, pose, valid, act, rng.random(64), W, H, banner)
    return img


def render_mp4(trajs, W, H, tmp_path):
    import cv2

    vw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(3)
    for f in range(NFR):
        vw.write(draw_frame(f, trajs, W, H, rng))
        if f % 300 == 0:
            print(f"[reel] {W}x{H} frame {f}/{NFR}", flush=True)
    vw.release()
    return tmp_path


def mux(video, audio, out):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tmp = Path("/tmp/opencode")
    tmp.mkdir(parents=True, exist_ok=True)
    trajs = collect_trajs()
    audio = synth_audio(tmp / "uottawa_reel.wav")
    wide_tmp = render_mp4(trajs, 1920, 1080, tmp / "wide_silent.mp4")
    vert_tmp = render_mp4(trajs, 1080, 1920, tmp / "vert_silent.mp4")
    outdir = REPO / "rl" / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    mux(wide_tmp, audio, outdir / "uottawa_aero_widescreen.mp4")
    mux(vert_tmp, audio, outdir / "uottawa_aero_reel.mp4")
    for p in (outdir / "uottawa_aero_widescreen.mp4", outdir / "uottawa_aero_reel.mp4"):
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,duration",
                "-of",
                "default=noprint_wrappers=1",
                str(p),
            ],
            capture_output=True,
            text=True,
        )
        print(f"[reel] {p}:\n{r.stdout}{r.stderr}", flush=True)


if __name__ == "__main__":
    main()
