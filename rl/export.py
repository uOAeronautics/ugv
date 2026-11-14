"""Export student torch weights -> NumPy .npz for Pi (+ optional .onnx).

Parity is checked against the deployed NumPy forward (src/ugv/policy.py),
not a copy, so a drift between training and deploy code fails the export.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from models import Student
from sim import HIST, STUDENT_DIM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ugv.policy import StudentPolicy  # noqa: E402  (numpy-only deploy module)


def parity_error(model: Student, arrays: dict, batch: int = 4) -> float:
    hist = np.random.default_rng(0).normal(size=(batch, HIST, STUDENT_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(hist)).numpy().astype(np.float64)
    deployed = StudentPolicy(arrays)
    got = np.stack([deployed.forward(h.astype(np.float64)) for h in hist])
    return float(np.abs(ref - got).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = Student(
        act_dim=ckpt.get("act_dim", 3),
        h=ckpt.get("hidden", 64) or 64,
        head_w=ckpt.get("head_w", 64) or 64,
    )
    model.load_state_dict(ckpt["student"] if "student" in ckpt else ckpt, strict=False)
    model.eval()
    arrays = {
        k: v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
        if k.startswith("gru.") or k.startswith("head.")
    }
    err = parity_error(model, arrays)
    print(f"parity max-abs-err {err:.2e}")
    assert err < 1e-5, f"numpy deploy forward diverged: {err}"
    blob = b"".join(v.tobytes() for k, v in sorted(arrays.items()))
    digest = hashlib.sha256(blob).hexdigest()[:16]
    np.savez(
        args.out,
        **arrays,
        meta_level=np.array(ckpt.get("level", "?")),
        meta_sha=np.array(digest),
        meta_stateful=np.array(bool(ckpt.get("stateful", False))),
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.1f} kB) sha {digest}")
    if args.onnx:
        torch.onnx.export(
            model,
            (torch.zeros(1, HIST, STUDENT_DIM),),
            args.onnx,
            input_names=["hist"],
            output_names=["act"],
            dynamo=False,
        )
        print(f"wrote {args.onnx}")


if __name__ == "__main__":
    main()
