"""Deploy-time parking student: NumPy forward pass + shield. No torch."""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path

from .parking import Pose2, gear_obstacles, swept_clear

HIST = 50
DIM = 13  # tag(3) + age + valid + imu_w + imu_yaw + prev_act(3) + tag_vel(3); mirrors rl/sim.py


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class StudentPolicy:
    """Tiny GRU-64 + 2x64 MLP. Drop-in alternative to ContinuousDocking.step."""

    def __init__(self, arrays: dict, sha: str = "?", vehicle=None, gear=None, inflation=0.015):
        self.a = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        self.sha = sha
        self.vehicle = vehicle or {"width_m": 0.16, "length_m": 0.22}
        self.gear = gear or {"clear_width_m": 0.254, "bar_length_m": 0.3048,
                             "bar_radius_m": 0.03, "inward_slope": 0.0}
        self.inflation = inflation
        self.obstacles = gear_obstacles(self.gear)
        self.reset()

    @classmethod
    def load(cls, weights: str | Path, **kw) -> "StudentPolicy":
        z = np.load(weights, allow_pickle=True)
        arrays = {k: z[k] for k in z.files if not k.startswith("meta")}
        return cls(arrays, str(z["meta_sha"]) if "meta_sha" in z.files else "?", **kw)

    def reset(self):
        """Clear per-episode state (history + tag-velocity estimator)."""
        self.hist = np.zeros((HIST, DIM))
        self._ema = np.zeros(3)
        self._prev_tag = np.zeros(3)
        self._prev_valid = False

    def forward(self, hist: np.ndarray) -> np.ndarray:
        Wir, Wiz, Win = np.split(self.a["gru.weight_ih_l0"], 3)
        Whr, Whz, Whn = np.split(self.a["gru.weight_hh_l0"], 3)
        bir, biz, bin_ = np.split(self.a["gru.bias_ih_l0"], 3)
        bhr, bhz, bhn = np.split(self.a["gru.bias_hh_l0"], 3)
        h = np.zeros(Whr.shape[1])
        for t in range(hist.shape[0]):
            x = hist[t]
            r = _sigmoid(x @ Wir.T + h @ Whr.T + bir + bhr)
            zz = _sigmoid(x @ Wiz.T + h @ Whz.T + biz + bhz)
            nn = np.tanh(x @ Win.T + r * (h @ Whn.T + bhn) + bin_)
            h = (1 - zz) * nn + zz * h
        y = np.tanh(h @ self.a["head.0.weight"].T + self.a["head.0.bias"])
        y = np.tanh(y @ self.a["head.2.weight"].T + self.a["head.2.bias"])
        return np.tanh(y @ self.a["head.4.weight"].T + self.a["head.4.bias"])

    def step(self, tag_xyyaw: tuple | None, tag_age: float, tag_valid: bool,
             imu_w: float, imu_yaw: float, prev_act: np.ndarray,
             estimate: Pose2) -> tuple[np.ndarray, str]:
        """Returns (action[3] in [-1,1], reason). Brakes on bad input or blocked sweep."""
        # One frame spec with sim.py: an invalid tag is zeroed fields + valid=0,
        # never the last-seen values.
        if not np.isfinite([imu_w, imu_yaw]).all() or (not tag_valid and tag_age > 1.5):
            self._push(np.zeros(DIM))
            return np.zeros(3), "brake: stale/invalid observation"
        try:
            if tag_valid and tag_xyyaw is not None:
                tx, ty, tyaw = (float(v) for v in tag_xyyaw)
            else:
                tx, ty, tyaw = 0.0, 0.0, 0.0
            tag = np.array([tx, ty, tyaw])
            if tag_valid and self._prev_valid:
                diff = (tag - self._prev_tag) / 0.02
                diff[2] = (diff[2] + math.pi) % (2 * math.pi) - math.pi
                self._ema = 0.5 * self._ema + 0.5 * diff
            else:
                self._ema = np.zeros(3)
            self._prev_tag = tag if tag_valid else self._prev_tag
            self._prev_valid = bool(tag_valid)
            frame = np.array([tx, ty, tyaw, min(tag_age, 5.0), 1.0 if tag_valid else 0.0,
                              float(imu_w), float(imu_yaw),
                              * (float(v) for v in prev_act),
                              * (float(v) for v in self._ema)], dtype=float)
            if not np.isfinite(frame).all():
                raise ValueError("non-finite frame")
        except (TypeError, ValueError):
            self._push(np.zeros(DIM))
            return np.zeros(3), "brake: non-finite observation"
        self._push(frame)
        act = self.forward(self.hist).clip(-1, 1)
        # shield: predicted 1-step sweep must be clear
        nxt = Pose2(estimate.x + float(act[0]) * 0.6 * 0.02,
                    estimate.z + float(act[1]) * 0.6 * 0.02,
                    estimate.yaw + float(act[2]) * 2.5 * 0.02)
        if not swept_clear(estimate, nxt, self.vehicle["width_m"], self.vehicle["length_m"],
                           self.obstacles, self.inflation):
            return np.zeros(3), "brake: swept footprint blocked"
        return act, "policy"

    def _push(self, frame):
        self.hist = np.concatenate([self.hist[1:], frame[None, :]])

    def benchmark(self, iters=1000):
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            self.forward(self.hist)
        dt = (time.perf_counter() - t0) / iters * 1000
        return {"ms_per_step": dt, "sha": self.sha}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    p = StudentPolicy.load(a.weights)
    print(p.benchmark() if a.benchmark else p.forward(p.hist))
