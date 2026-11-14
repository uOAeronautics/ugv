"""Deploy-time parking student: NumPy forward pass + shield. No torch."""

from __future__ import annotations
import math
import numpy as np
from pathlib import Path

from .parking import Pose2, gear_obstacles, swept_clear

HIST = 50
DIM = 13  # tag(3) + age + valid + imu_w + imu_yaw + prev_act(3) + tag_vel(3); mirrors rl/sim.py
V_X_MAX = 1.27
V_Y_MAX = 0.63
W_MAX = 2.20
STOP_HORIZON = 0.35  # > 200ms camera latency + 120ms brake_tau + margin
EMA_TAU = 0.25
# Pre-acquisition FOV search: with the tag outside the +/-33.5 deg camera FOV
# for this long (sim hist_age; the tag frame reports age clamped at 5.0), the
# bearing error is dominated by yaw, so rotate in place toward imu_yaw = 0
# (zero blind translation). Yaw 0 puts bearing = spawn bearing (<=30 deg on
# L1) inside the FOV. Mirrors the rl/distill.py --fov-search collection reflex.
SEARCH_AGE = 1.0
SEARCH_P = 1.8
SEARCH_D = 0.3
SEARCH_MAX = 0.6


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class StudentPolicy:
    """Tiny GRU-64 + 2x64 MLP. Drop-in alternative to ContinuousDocking.step."""

    def __init__(
        self,
        arrays: dict,
        sha: str = "?",
        vehicle=None,
        gear=None,
        inflation=0.015,
        stateful: bool = False,
    ):
        self.a = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        self.sha = sha
        self.stateful = stateful
        self.vehicle = vehicle or {"width_m": 0.16, "length_m": 0.22}
        self.gear = gear or {
            "clear_width_m": 0.254,
            "bar_length_m": 0.3048,
            "bar_radius_m": 0.03,
            "inward_slope": 0.0,
        }
        self.inflation = inflation
        self.obstacles = gear_obstacles(self.gear)
        self._h_dim = 64
        self._act_dim = 3
        if "gru.weight_ih_l0" in self.a:
            self._Wir, self._Wiz, self._Win = (
                np.ascontiguousarray(w.T) for w in np.split(self.a["gru.weight_ih_l0"], 3)
            )
            self._Whr, self._Whz, self._Whn = (
                np.ascontiguousarray(w.T) for w in np.split(self.a["gru.weight_hh_l0"], 3)
            )
            bir, biz, self._bin = np.split(self.a["gru.bias_ih_l0"], 3)
            bhr, bhz, self._bhn = np.split(self.a["gru.bias_hh_l0"], 3)
            self._br = bir + bhr
            self._bz = biz + bhz
            self._h_dim = self._Whr.shape[0]
            self._W0 = np.ascontiguousarray(self.a["head.0.weight"].T)
            self._b0 = self.a["head.0.bias"]
            self._W2 = np.ascontiguousarray(self.a["head.2.weight"].T)
            self._b2 = self.a["head.2.bias"]
            self._W4 = np.ascontiguousarray(self.a["head.4.weight"].T)
            self._b4 = self.a["head.4.bias"]
            self._act_dim = self._b4.shape[0]
        self.reset()

    @classmethod
    def load(cls, weights: str | Path, **kw) -> "StudentPolicy":
        z = np.load(weights, allow_pickle=True)
        arrays = {k: z[k] for k in z.files if not k.startswith("meta")}
        if "stateful" not in kw and "meta_stateful" in z.files:
            kw["stateful"] = bool(z["meta_stateful"])
        return cls(arrays, str(z["meta_sha"]) if "meta_sha" in z.files else "?", **kw)

    def reset(self):
        """Clear per-episode state (history + tag-velocity estimator)."""
        self.hist = np.zeros((HIST, DIM))
        self._ema = np.zeros(3)
        self._prev_tag = np.zeros(3)
        self._prev_valid = False
        self._dt_since_valid = 0.0
        self._h = np.zeros(self._h_dim)

    def _step_gru(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        r = _sigmoid(x @ self._Wir + h @ self._Whr + self._br)
        zz = _sigmoid(x @ self._Wiz + h @ self._Whz + self._bz)
        nn = np.tanh(x @ self._Win + r * (h @ self._Whn + self._bhn) + self._bin)
        return (1.0 - zz) * nn + zz * h

    def _head(self, h: np.ndarray) -> np.ndarray:
        y = np.tanh(h @ self._W0 + self._b0)
        y = np.tanh(y @ self._W2 + self._b2)
        return np.tanh(y @ self._W4 + self._b4)

    def forward(self, hist: np.ndarray) -> np.ndarray:
        h = np.zeros(self._h_dim)
        Wir, Wiz, Win = self._Wir, self._Wiz, self._Win
        Whr, Whz, Whn = self._Whr, self._Whz, self._Whn
        br, bz, bin_, bhn = self._br, self._bz, self._bin, self._bhn
        for t in range(hist.shape[0]):
            x = hist[t]
            r = _sigmoid(x @ Wir + h @ Whr + br)
            zz = _sigmoid(x @ Wiz + h @ Whz + bz)
            nn = np.tanh(x @ Win + r * (h @ Whn + bhn) + bin_)
            h = (1.0 - zz) * nn + zz * h
        return self._head(h)

    def step(
        self,
        tag_xyyaw: tuple | None,
        tag_age: float,
        tag_valid: bool,
        imu_w: float,
        imu_yaw: float,
        prev_act: np.ndarray,
        estimate: Pose2,
        dt: float = 0.02,
        dt_stale: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        """Returns (action[3|4] in [-1,1], reason). Brakes on bad input or blocked sweep."""
        # One frame spec with sim.py: an invalid tag is zeroed fields + valid=0,
        # never the last-seen values.
        self._dt_since_valid += dt
        if not np.isfinite([imu_w, imu_yaw]).all() or (not tag_valid and tag_age > 1.5):
            self._push(np.zeros(DIM))
            return np.zeros(self._act_dim), "brake: stale/invalid observation"
        try:
            if tag_valid and tag_xyyaw is not None:
                tx, ty, tyaw = (float(v) for v in tag_xyyaw)
            else:
                tx, ty, tyaw = 0.0, 0.0, 0.0
            tag = np.array([tx, ty, tyaw])
            if tag_valid:
                if self._prev_valid and self._dt_since_valid <= 0.5:
                    diff = tag - self._prev_tag
                    diff[2] = (diff[2] + math.pi) % (2 * math.pi) - math.pi
                    diff /= max(self._dt_since_valid, dt)
                    self._ema = 0.5 * self._ema + 0.5 * diff
                else:
                    self._ema = np.zeros(3)
                self._prev_tag = tag
                self._prev_valid = True
                self._dt_since_valid = 0.0
            else:
                self._ema = self._ema * math.exp(-dt / EMA_TAU)
            frame = np.array(
                [
                    tx,
                    ty,
                    tyaw,
                    min(tag_age, 5.0),
                    1.0 if tag_valid else 0.0,
                    float(imu_w),
                    float(imu_yaw),
                    *(float(v) for v in prev_act[:3]),
                    *(float(v) for v in self._ema),
                ],
                dtype=float,
            )
            if not np.isfinite(frame).all():
                raise ValueError("non-finite frame")
        except (TypeError, ValueError):
            self._push(np.zeros(DIM))
            return np.zeros(self._act_dim), "brake: non-finite observation"
        self._push(frame)
        # Blind-search reflex: prolonged total tag loss (e.g. blind spawn with
        # the tag outside the camera FOV) -> in-place yaw servo toward
        # imu_yaw = 0 with rate damping, zero translation. Applied through the
        # swept-clear shield below, so a blocked spin still brakes.
        search = (not tag_valid) and self._dt_since_valid >= SEARCH_AGE
        if search:
            yaw_w = (float(imu_yaw) + math.pi) % (2 * math.pi) - math.pi
            u_turn = max(-SEARCH_MAX, min(SEARCH_MAX, -SEARCH_P * yaw_w - SEARCH_D * float(imu_w)))
            act = np.zeros(self._act_dim)
            act[2] = u_turn
            search_reason = "search: blind, rotating to acquire tag"
        elif self.stateful:
            self._h = self._step_gru(frame, self._h)
            act = self._head(self._h).clip(-1, 1)
            search_reason = ""
        else:
            act = self.forward(self.hist).clip(-1, 1)
            search_reason = ""
        # shield: rotate body velocity by estimate.yaw, forward-predict past dt_stale,
        # then sweep the full stopping horizon (covers latency + brake_tau)
        c, s = math.cos(estimate.yaw), math.sin(estimate.yaw)
        vx_b = float(act[0]) * V_X_MAX
        vy_b = float(act[1]) * V_Y_MAX
        vx_w = c * vx_b - s * vy_b
        vz_w = s * vx_b + c * vy_b
        w = float(act[2]) * W_MAX
        nxt = Pose2(
            estimate.x + vx_w * (dt_stale + STOP_HORIZON),
            estimate.z + vz_w * (dt_stale + STOP_HORIZON),
            estimate.yaw + w * (dt_stale + STOP_HORIZON),
        )
        start = (
            estimate
            if not dt_stale
            else Pose2(
                estimate.x + vx_w * dt_stale,
                estimate.z + vz_w * dt_stale,
                estimate.yaw + w * dt_stale,
            )
        )
        # parking.py uses CW+ yaw from +z toward +x; policy.py uses CCW+ yaw.
        if not swept_clear(
            Pose2(start.x, start.z, -start.yaw),
            Pose2(nxt.x, nxt.z, -nxt.yaw),
            self.vehicle["width_m"],
            self.vehicle["length_m"],
            self.obstacles,
            self.inflation,
        ):
            return np.zeros_like(act), "brake: swept footprint blocked"
        if search_reason:
            return act, search_reason
        return act, "policy"

    def _push(self, frame):
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = frame

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
