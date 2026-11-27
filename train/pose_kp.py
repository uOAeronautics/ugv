"""Train the airframe keypoint detector on rendered data (GPU), export ONNX for the Pi.

    cd train && uv run python pose_kp.py --data ../runs/pose_data/v1 --out ../runs/pose_train/kp-v1

Model: MobileNetV3-small (ImageNet weights, first conv folded to one channel: the car camera is
mono), a light upsampling head to stride-4 heatmaps, soft-argmax for sub-pixel keypoints and a
per-keypoint visibility logit. Loss: L1 on visible keypoint coordinates (pixels) plus BCE on
visibility. Reported per epoch: keypoint error in pixels at the 320x180 input (x4 for the
1280x720 camera), visibility accuracy, and the end-to-end number that matters: PnP pose
within 5 deg / 10 cm on held-out views. Writes metrics.jsonl, summary.json, model.pt,
model.onnx, and ONNX Runtime CPU latency (the Pi still has to be measured).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from ugv.pose.mesh import keypoints, load_airframe
from ugv.pose.solve import pose_error, solve_pnp


class KeypointNet(nn.Module):
    def __init__(self, nk: int = 15, pretrained: bool = True) -> None:
        super().__init__()
        self.nk = nk
        weights = (
            torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        conv = m.features[0][0]
        mono = nn.Conv2d(
            1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False
        )
        mono.weight.data = conv.weight.data.sum(dim=1, keepdim=True)
        m.features[0][0] = mono
        self.stem = m.features[:4]  # stride 8, 24 ch
        self.mid = m.features[4:9]  # stride 16, 48 ch
        self.deep = m.features[9:]  # stride 32, 576 ch
        self.lat16 = nn.Conv2d(48, 96, 1)
        self.lat8 = nn.Conv2d(24, 96, 1)
        self.top = nn.Conv2d(576, 96, 1)
        self.head = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.hm_out = nn.Conv2d(64, nk, 1)
        self.glob_vis = nn.Linear(576, 64)
        self.vis_head = nn.Sequential(
            nn.Linear(64 + 64 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 1, H, W) in [0, 1]. Returns keypoints (B, nk, 2) in input pixels, visibility
        logits (B, nk) and heatmaps (B, nk, H/4, W/4)."""
        x = (x - 0.45) / 0.25
        s8 = self.stem(x)
        s16 = self.mid(s8)
        s32 = self.deep(s16)
        p = self.top(s32)
        p = F.interpolate(
            p, size=s16.shape[-2:], mode="bilinear", align_corners=False
        ) + self.lat16(s16)
        p = F.interpolate(p, size=s8.shape[-2:], mode="bilinear", align_corners=False) + self.lat8(
            s8
        )
        feat = self.head(p)
        hm = self.hm_out(feat)
        b, k, h, w = hm.shape
        prob_flat = F.softmax(hm.flatten(2) * 1.0, dim=-1)
        prob = prob_flat.view(b, k, h, w)
        xs = (torch.arange(w, device=x.device, dtype=prob.dtype) + 0.5) * (x.shape[-1] / w)
        ys = (torch.arange(h, device=x.device, dtype=prob.dtype) + 0.5) * (x.shape[-2] / h)
        kx = (prob.sum(2) * xs).sum(-1)
        ky = (prob.sum(3) * ys).sum(-1)
        local = torch.bmm(prob_flat, feat.flatten(2).transpose(1, 2))
        peak = hm.flatten(2).max(dim=-1, keepdim=True).values
        glob = self.glob_vis(s32.mean((2, 3))).unsqueeze(1).expand(-1, k, -1)
        vis = self.vis_head(torch.cat([local, glob, peak], dim=-1)).squeeze(-1)
        return torch.stack([kx, ky], -1), vis, hm


FULL_FX = 968.0  # the car camera at 1280x720

# Camera faults seen in training, each with its own probability per image (see `camera_faults`).
AUG = {
    "gain": 1.0,  # brightness/contrast jitter, always
    "exposure": 0.15,  # badly over- or under-exposed
    "blur": 0.25,  # Gaussian defocus
    "motion": 0.2,  # motion blur along a random direction
    "noise": 0.5,  # sensor noise, up to 6% of full scale
    "occlusion": 0.3,  # 1-3 blocked rectangles (cable, smudge, a hand), keypoints stay labelled
}


def camera_faults(x: torch.Tensor, p: dict) -> torch.Tensor:
    """Photometric and occlusion faults on a batch (B, 1, H, W) in [0, 1], on the GPU. Geometry is
    untouched, so the keypoint labels stay valid (mount shake and knocks are the PnP's problem)."""
    n, _, h, w = x.shape
    dev = x.device
    u = lambda lo, hi: torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)  # noqa: E731
    pick = lambda q: (torch.rand(n, 1, 1, 1, device=dev) < q).float()  # noqa: E731
    x = x * u(0.7, 1.3) + u(-0.1, 0.1)
    e = pick(p["exposure"])
    over = (torch.rand(n, 1, 1, 1, device=dev) < 0.5).float()
    x = e * (over * (x * u(2.0, 4.0)) + (1 - over) * (x * u(0.08, 0.3))) + (1 - e) * x
    x = x.clamp(0, 1)
    if p["blur"]:
        k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=x.dtype)
        k2 = (k[:, None] * k[None, :]) / 256
        b = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k2[None, None])
        b = F.conv2d(F.pad(b, (2, 2, 2, 2), mode="replicate"), k2[None, None])
        x = torch.lerp(x, b, pick(p["blur"]))
    if p["motion"]:
        horiz = torch.full((1, 1, 1, 9), 1 / 9, device=dev, dtype=x.dtype)
        mh = F.conv2d(F.pad(x, (4, 4, 0, 0), mode="replicate"), horiz)
        mv = F.conv2d(F.pad(x, (0, 0, 4, 4), mode="replicate"), horiz.transpose(2, 3))
        which = (torch.rand(n, 1, 1, 1, device=dev) < 0.7).float()  # turning blurs sideways
        x = torch.lerp(x, which * mh + (1 - which) * mv, pick(p["motion"]))
    x = x + torch.randn_like(x) * u(0, 0.06) * pick(p["noise"])
    if p["occlusion"]:
        for _ in range(3):
            on = torch.rand(n, device=dev) < p["occlusion"] / 2
            for i in torch.nonzero(on).flatten().tolist():
                bh, bw = (
                    int(h * np.random.uniform(0.1, 0.35)),
                    int(w * np.random.uniform(0.05, 0.3)),
                )
                y0, x0 = np.random.randint(0, h - bh), np.random.randint(0, w - bw)
                x[i, :, y0 : y0 + bh, x0 : x0 + bw] = float(np.random.uniform(0, 1))
    return x.clamp_(0, 1)


def load(data: Path):
    """Images (memmap), keypoints, visibility, poses, per-sample intrinsics, meta."""
    images = np.load(data / "images.npy", mmap_mode="r")
    meta = json.loads((data / "meta.json").read_text())
    if (data / "K.npy").exists():
        Ks = np.load(data / "K.npy")
    else:
        Ks = np.repeat(np.array(meta["K"], np.float32)[None], len(images), 0)
    kp, vis, pose = (np.load(data / f) for f in ("kp.npy", "vis.npy", "pose.npy"))
    return images, kp, vis, pose, Ks, meta


def load_dagger(paths: list[Path], crop: bool, shape: tuple[int, int], kp_names: list[str]):
    """Load one or more DAgger directories (`tools/park/eval.py --dump-frames`), choosing the
    `crop/` subdir automatically when `--crop` is active."""
    imgs, kps, viss = [], [], []
    for raw in paths:
        p = raw / "crop" if crop and (raw / "crop" / "meta.json").exists() else raw
        im, kp, vis, _, _, dmeta = load(p)
        if (im.shape[1], im.shape[2]) != shape:
            raise ValueError(f"{p} shape {im.shape[1:]} != expected {shape}")
        if dmeta.get("keypoints") != kp_names:
            raise ValueError(f"{p} keypoints do not match training target")
        imgs.append(np.asarray(im))
        kps.append(kp)
        viss.append(vis)
    return np.concatenate(imgs), np.concatenate(kps), np.concatenate(viss)


def batches(idx: np.ndarray, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(idx) if rng is not None else idx
    for i in range(0, len(order), bs):
        yield np.sort(order[i : i + bs])


def to_gpu(images, kp, vis, b, dev, train: bool):
    x = torch.from_numpy(np.ascontiguousarray(images[b])).to(dev).float().div_(255).unsqueeze(1)
    if train:
        x = camera_faults(x, AUG)
    k = torch.from_numpy(kp[b]).to(dev)
    v = torch.from_numpy(vis[b]).to(dev)
    inside = (
        (k[..., 0] >= 0) & (k[..., 0] < x.shape[-1]) & (k[..., 1] >= 0) & (k[..., 1] < x.shape[-2])
    )
    return x, k, v & inside


@torch.no_grad()
def evaluate(model, images, kp, vis, poses, idx, dev, Ks, model_pts, pnp_n: int = 400) -> dict:
    """Keypoint error in input pixels and in full-camera pixels (each sample's own scale)."""
    model.eval()
    errs, full, vis_ok, vis_n, preds, probs = [], [], 0, 0, [], []
    for b in batches(idx, 256, None):
        x, k, v = to_gpu(images, kp, vis, b, dev, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pk, pv, _ = model(x)
        pk, pv = pk.float(), pv.float()
        e = (pk - k).norm(dim=-1)
        errs.append(e[v].cpu().numpy())
        scale = torch.from_numpy(FULL_FX / Ks[b, 0, 0]).to(dev)[:, None].expand_as(e)
        full.append((e * scale)[v].cpu().numpy())
        vis_ok += ((pv > 0) == v).sum().item()
        vis_n += v.numel()
        preds.append(pk.cpu().numpy())
        probs.append(torch.sigmoid(pv).cpu().numpy())
    errs, full = np.concatenate(errs), np.concatenate(full)
    preds, probs = np.concatenate(preds), np.concatenate(probs)
    hits, solved = 0, 0
    for j, i in enumerate(idx[:pnp_n]):
        use = probs[j] > 0.5
        if use.sum() < 6:
            continue
        sol = solve_pnp(
            model_pts[use],
            preds[j][use].astype(np.float64),
            Ks[i].astype(np.float64),
            reproj_px=3.0,
        )
        if sol is None:
            continue
        solved += 1
        R_gt, t_gt = poses[i, :9].reshape(3, 3), poses[i, 9:]
        rot, pos = pose_error(sol[0], sol[1], R_gt, t_gt)
        hits += rot < 5 and pos < 0.10
    model.train()
    n = min(pnp_n, len(idx))
    return {
        "val_px_median": float(np.median(errs)),
        "val_px_p90": float(np.percentile(errs, 90)),
        "val_px_median_at_1280": float(np.median(full)),
        "val_px_p90_at_1280": float(np.percentile(full, 90)),
        "val_vis_acc": vis_ok / max(1, vis_n),
        "val_pnp_solved": solved / n,
        "val_pnp_within_5deg_10cm": hits / n,
    }


def latency(onnx_path: Path, h: int, w: int) -> dict:
    import onnxruntime as ort

    out = {}
    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 1, h, w).astype(np.float32)
        for _ in range(5):
            sess.run(None, {"image": x})
        t0 = time.perf_counter()
        for _ in range(50):
            sess.run(None, {"image": x})
        out[f"desktop_cpu_ms_{threads}thread"] = round((time.perf_counter() - t0) / 50 * 1000, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pool",
        type=int,
        help="train on a rolling pool of this many generated samples (ugv.pose.dataset) and use "
        "--data only as the fixed validation set; --crop/--bay/--workers set the generator",
    )
    ap.add_argument("--crop", action="store_true", help="pool: second-stage crops")
    ap.add_argument("--bay", type=float, default=0.0, help="pool: share of bay-range views")
    ap.add_argument("--workers", type=int, default=12, help="pool: generator processes")
    ap.add_argument(
        "--dagger",
        type=Path,
        nargs="+",
        default=None,
        help="closed-loop DAgger dataset dir(s) to mix into training batches",
    )
    ap.add_argument(
        "--dagger-frac",
        type=float,
        default=0.25,
        help="fraction of each training batch drawn from --dagger",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    images, kp, vis, poses, Ks, meta = load(args.data)
    d_img = d_kp = d_vis = None
    if args.dagger:
        d_img, d_kp, d_vis = load_dagger(
            args.dagger, args.crop, (meta["height"], meta["width"]), meta["keypoints"]
        )
    n = len(images)
    perm = np.random.default_rng(123).permutation(n)
    val, tr = np.sort(perm[: max(256, n // 12)]), np.sort(perm[max(256, n // 12) :])
    pool = None
    if args.pool:  # --data is the fixed validation set only; the training set keeps changing
        from ugv.pose.dataset import RollingPool

        val = np.arange(n)
        pool = RollingPool(
            args.pool,
            crop=args.crop,
            bay=args.bay,
            workers=args.workers,
            seed0=7_000_000_000 + args.seed * 1_000_000_000,
            shape=(meta["height"], meta["width"]),
        )
        pool.start()
        tr = np.arange(args.pool)
    kpd = keypoints(load_airframe(), near=True)
    model_pts = np.array([kpd[k] for k in meta["keypoints"]])
    model = KeypointNet(kp.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "metrics.jsonl").open("w")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        losses = []
        for b in batches(tr, args.bs, rng):
            m_dag = int(round(len(b) * args.dagger_frac)) if d_img is not None and len(b) > 1 else 0
            pb = b[: len(b) - m_dag] if m_dag else b
            if pool is not None:
                with pool.lock:
                    x, k, v = to_gpu(pool.images, pool.kp, pool.vis, pb, dev, True)
            else:
                x, k, v = to_gpu(images, kp, vis, pb, dev, True)
            if m_dag:
                db = np.sort(rng.integers(0, len(d_img), size=m_dag))
                xd, kd, vd = to_gpu(d_img, d_kp, d_vis, db, dev, True)
                x, k, v = torch.cat([x, xd]), torch.cat([k, kd]), torch.cat([v, vd])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pk, pv, _ = model(x)
            l_kp = ((pk.float() - k).abs().sum(-1) * v).sum() / v.sum().clamp(min=1)
            l_vis = F.binary_cross_entropy_with_logits(pv.float(), v.float())
            loss = l_kp + 5.0 * l_vis
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append((l_kp.item(), l_vis.item()))
            step += 1
        m = evaluate(model, images, kp, vis, poses, val, dev, Ks, model_pts)
        row = {
            "epoch": epoch,
            "step": step,
            "train_kp_loss": float(np.mean([a for a, _ in losses])),
            "train_vis_loss": float(np.mean([b for _, b in losses])),
            **m,
            "minutes": round((time.time() - t0) / 60, 2),
            **({"pool_replaced": pool.replaced} if pool is not None else {}),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    model.eval().float()
    h, w = meta["height"], meta["width"]
    onnx_path = args.out / "model.onnx"
    torch.onnx.export(
        model.cpu(),
        torch.zeros(1, 1, h, w),
        str(onnx_path),
        input_names=["image"],
        output_names=["keypoints", "visibility", "heatmaps"],
        opset_version=17,
        dynamo=False,
    )
    final = evaluate(model.to(dev), images, kp, vis, poses, val, dev, Ks, model_pts, pnp_n=len(val))
    summary = {
        "data": str(args.data),
        "train_n": len(tr),
        "pool": args.pool,
        "pool_replaced": pool.replaced if pool is not None else 0,
        "dagger": [str(p) for p in args.dagger] if args.dagger else None,
        "dagger_n": len(d_img) if d_img is not None else 0,
        "dagger_frac": args.dagger_frac if d_img is not None else 0.0,
        "val_n": len(val),
        "epochs": args.epochs,
        "input": [h, w],
        "params": sum(p.numel() for p in model.parameters()),
        **final,
        **latency(onnx_path, h, w),
        "note": "synthetic validation (same generator); not evidence about real frames",
    }
    if pool is not None:
        pool.close()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
