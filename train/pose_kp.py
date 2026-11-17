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

NK = 12


class KeypointNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
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
            nn.Conv2d(64, NK, 1),
        )
        self.vis = nn.Linear(576, NK)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 1, H, W) in [0, 1]. Returns keypoints (B, NK, 2) in input pixels, visibility
        logits (B, NK) and heatmaps (B, NK, H/4, W/4)."""
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
        hm = self.head(p)
        b, k, h, w = hm.shape
        prob = F.softmax(hm.flatten(2) * 1.0, dim=-1).view(b, k, h, w)
        xs = (torch.arange(w, device=x.device, dtype=prob.dtype) + 0.5) * (x.shape[-1] / w)
        ys = (torch.arange(h, device=x.device, dtype=prob.dtype) + 0.5) * (x.shape[-2] / h)
        kx = (prob.sum(2) * xs).sum(-1)
        ky = (prob.sum(3) * ys).sum(-1)
        vis = self.vis(s32.mean((2, 3)))
        return torch.stack([kx, ky], -1), vis, hm


def load(data: Path):
    images = np.load(data / "images.npy", mmap_mode="r")
    return (
        images,
        np.load(data / "kp.npy"),
        np.load(data / "vis.npy"),
        np.load(data / "pose.npy"),
        json.loads((data / "meta.json").read_text()),
    )


def batches(idx: np.ndarray, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(idx) if rng is not None else idx
    for i in range(0, len(order), bs):
        yield np.sort(order[i : i + bs])


def to_gpu(images, kp, vis, b, dev, train: bool):
    x = torch.from_numpy(np.ascontiguousarray(images[b])).to(dev).float().div_(255).unsqueeze(1)
    if train:  # photometric jitter on the GPU
        n = x.shape[0]
        x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + torch.empty(
            n, 1, 1, 1, device=dev
        ).uniform_(-0.1, 0.1)
        x = (
            x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 0.03)
        ).clamp_(0, 1)
    k = torch.from_numpy(kp[b]).to(dev)
    v = torch.from_numpy(vis[b]).to(dev)
    inside = (
        (k[..., 0] >= 0) & (k[..., 0] < x.shape[-1]) & (k[..., 1] >= 0) & (k[..., 1] < x.shape[-2])
    )
    return x, k, v & inside


@torch.no_grad()
def evaluate(model, images, kp, vis, poses, idx, dev, K, model_pts, pnp_n: int = 400) -> dict:
    model.eval()
    errs, vis_ok, vis_n, preds, probs = [], 0, 0, [], []
    for b in batches(idx, 256, None):
        x, k, v = to_gpu(images, kp, vis, b, dev, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pk, pv, _ = model(x)
        pk, pv = pk.float(), pv.float()
        e = (pk - k).norm(dim=-1)
        errs.append(e[v].cpu().numpy())
        vis_ok += ((pv > 0) == v).sum().item()
        vis_n += v.numel()
        preds.append(pk.cpu().numpy())
        probs.append(torch.sigmoid(pv).cpu().numpy())
    errs = np.concatenate(errs)
    preds, probs = np.concatenate(preds), np.concatenate(probs)
    hits, solved = 0, 0
    for j, i in enumerate(idx[:pnp_n]):
        use = probs[j] > 0.5
        if use.sum() < 6:
            continue
        sol = solve_pnp(model_pts[use], preds[j][use].astype(np.float64), K, reproj_px=3.0)
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
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    images, kp, vis, poses, meta = load(args.data)
    n = len(images)
    perm = np.random.default_rng(123).permutation(n)
    val, tr = np.sort(perm[: max(256, n // 12)]), np.sort(perm[max(256, n // 12) :])
    K = np.array(meta["K"])
    kpd = keypoints(load_airframe())
    model_pts = np.array([kpd[k] for k in meta["keypoints"]])
    model = KeypointNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "metrics.jsonl").open("w")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        losses = []
        for b in batches(tr, args.bs, rng):
            x, k, v = to_gpu(images, kp, vis, b, dev, True)
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
        m = evaluate(model, images, kp, vis, poses, val, dev, K, model_pts)
        row = {
            "epoch": epoch,
            "step": step,
            "train_kp_loss": float(np.mean([a for a, _ in losses])),
            "train_vis_loss": float(np.mean([b for _, b in losses])),
            **m,
            "minutes": round((time.time() - t0) / 60, 2),
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
    final = evaluate(model.to(dev), images, kp, vis, poses, val, dev, K, model_pts, pnp_n=len(val))
    summary = {
        "data": str(args.data),
        "train_n": len(tr),
        "val_n": len(val),
        "epochs": args.epochs,
        "input": [h, w],
        "params": sum(p.numel() for p in model.parameters()),
        **final,
        "val_px_median_at_1280": final["val_px_median"] * 1280 / w,
        **latency(onnx_path, h, w),
        "note": "synthetic validation (same generator); not evidence about real frames",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
