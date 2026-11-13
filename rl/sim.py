"""Batched mecanum SE(2) parking env with grounded DR + sensor emulator.

Dynamics: torch port of src/ugv/mecanum_dynamics features/step. Each episode
samples one joint posterior draw (60 particles) plus extra DR (wheel gain, yaw
coupling, battery sag). Sensor emulator: tag FOV/range noise, latency,
drops/outliers/blackout, under-drone loss; IMU yaw-rate noise + bias.
Collision: car rect vs two skid capsules + room bounds.
Runs on CUDA (3090) or CPU. Gazebo is the L6 gate only.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "simulation" / "gazebo" / "mecanum_bayesian_profile.json"

LEVELS = {
    "L0": dict(dist=0.75, bearing=0.0, yaw=0.0, noise=0.0, latency=0.0, drops=0.0,
               outliers=0.0, blackout=0.0, under_drone_loss=False,
               full_posterior=False, area=1.0, cam_hz=50.0),
    "L1": dict(dist=(0.5, 1.5), bearing=30.0, yaw=60.0, noise=1.0, latency=0.08,
               drops=0.0, outliers=0.0, blackout=0.0, under_drone_loss=False,
               full_posterior=False, area=1.5, cam_hz=30.0),
    "L2": dict(dist=(0.5, 1.5), bearing=30.0, yaw=60.0, noise=1.0, latency=0.08,
               drops=0.0, outliers=0.0, blackout=0.0, under_drone_loss=False,
               full_posterior=True, area=1.5, cam_hz=30.0),
    "L3": dict(dist=(0.5, 1.5), bearing=30.0, yaw=60.0, noise=1.0, latency=(0.08, 0.20),
               drops=(0.25, 0.45), outliers=0.02, blackout=3.0, under_drone_loss=True,
               full_posterior=True, area=1.5, cam_hz=30.0),
    "L4": dict(dist=(0.5, 4.0), bearing=180.0, yaw=180.0, noise=1.0, latency=(0.08, 0.20),
               drops=(0.25, 0.45), outliers=0.02, blackout=3.0, under_drone_loss=True,
               full_posterior=True, area=4.0, cam_hz=30.0),
    "L5": dict(dist=(0.5, 8.0), bearing=180.0, yaw=180.0, noise=1.0, latency=(0.08, 0.20),
               drops=(0.25, 0.45), outliers=0.02, blackout=3.0, under_drone_loss=True,
               full_posterior=True, area=12.0, cam_hz=30.0),
}

# Provisional geometry (config/parking.json unverified): car 0.16 x 0.22 m,
# gear clear width 0.254 m, bar length 0.3048 m, radius 0.03 m.
CAR_W, CAR_L = 0.16, 0.22
GEAR_GAP, GEAR_LEN, GEAR_R = 0.254, 0.3048, 0.03
SKID_X = GEAR_GAP / 2 + GEAR_R  # 0.157
HALF_BASE = (0.21 + 0.149) / 2  # wheelbase+track mixer arm
HIST = 50  # 1 s at 50 Hz
_RING_K = 32  # latency ring depth; covers 0.64 s at 50 Hz
# Frame: tag x,y,yaw + age + valid + imu_w + imu_yaw + prev_act(3) + tag vel est(3).
# The smoothed tag velocity is explicit because the GRU could not extract
# transit-critical velocity from raw poses (fit error 100x worse at the slot
# mouth); it is valid-gated finite differencing, computable onboard.
STUDENT_DIM = 13
FOV_HALF = math.radians(33.5)


def _wheel_features(u: torch.Tensor, deadzone: torch.Tensor) -> torch.Tensor:
    dz = deadzone[:, None].clamp(0, 0.3)
    q = torch.sign(u) * torch.clamp(u.abs() - dz, min=0) / (1 - dz + 1e-6)
    p = torch.clamp(q, min=0)
    n = torch.clamp(q, max=0)
    mean_abs = q.abs().mean(dim=1, keepdim=True)
    return torch.cat([p, n, q * q.abs(), q * mean_abs.expand_as(q)], dim=1)  # [N,16]


@dataclass
class EnvConfig:
    level: str = "L1"
    dt: float = 0.02
    max_time: float = 45.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    spawn_dist: tuple | float | None = None  # override LEVELS dist (curriculum)
    bearing_max: float | None = None  # override LEVELS bearing deg (curriculum)
    yaw_max: float | None = None  # override LEVELS yaw deg (curriculum)
    latency_override: float | tuple | None = None  # override LEVELS latency (curriculum)
    drops_override: float | tuple | None = None  # override LEVELS drops (curriculum)
    rate_w: float = 0.0  # action-rate penalty weight; L0-gate teacher used 0.0
    lat_w: float = 2.0  # extra lateral-error weight in progress potential
    yaw_w: float = 1.0  # yaw-error weight in progress potential
    cam_hz: float | None = None  # override LEVELS cam_hz (50 Hz L0, 30 Hz L1+)
    use_mean_dynamics: bool = True  # L0/L1 use posterior mean unless False (particles[0])
    full_posterior: bool | None = None  # override LEVELS full_posterior if set
    declare_parked: bool = False  # enable 3-dim still-outside-goal false_park detection


class MecanumVecEnv:
    ACT_DIM = 3

    def __init__(self, n: int, cfg: EnvConfig):
        self.n = n
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = cfg.dt
        prof = json.loads(PROFILE.read_text())
        self.mean_coef = torch.tensor(prof["coefficients"], dtype=torch.float32, device=self.device)
        self.mean_dz = float(prof["deadzone"])
        self.mean_drive = torch.tensor(prof["drive_tau_s"], dtype=torch.float32, device=self.device)
        self.mean_brake = torch.tensor(prof["brake_tau_s"], dtype=torch.float32, device=self.device)
        self.mean_cov = torch.tensor(prof["innovation_covariance"], dtype=torch.float32, device=self.device)
        self.particles = [{
            "coef": torch.tensor(p["coefficients"], dtype=torch.float32, device=self.device),
            "dz": float(p["deadzone"]),
            "drive": torch.tensor(p["drive_tau_s"], dtype=torch.float32, device=self.device),
            "brake": torch.tensor(p["brake_tau_s"], dtype=torch.float32, device=self.device),
            "cov": torch.tensor(p["innovation_covariance"], dtype=torch.float32, device=self.device),
        } for p in prof["posterior_particles"]]
        self.level = LEVELS[cfg.level]
        self.cam_hz = float(cfg.cam_hz if cfg.cam_hz is not None else self.level.get("cam_hz", 50.0))
        # Persistent generators: every stochastic draw in reset/step comes from
        # these streams, so a fixed seed + fixed call pattern is reproducible.
        # Device draws use gen, CPU index draws use gen_cpu.
        self.gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        self.gen_cpu = torch.Generator(device="cpu").manual_seed(cfg.seed + 1_000_000)
        dev = self.device
        self.pose = torch.zeros(n, 3, device=dev)
        self.vel = torch.zeros(n, 3, device=dev)
        self.time = torch.zeros(n, device=dev)
        self.still = torch.zeros(n, device=dev)
        self.vel_still = torch.zeros(n, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latched_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latched_coll = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latched_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latched_false_park = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_act = torch.zeros(n, 3, device=dev)
        self.hist = torch.zeros(n, HIST, STUDENT_DIM, device=dev)
        self.hist_age = torch.full((n,), 1e9, device=dev)
        self.tag_ema = torch.zeros(n, 3, device=dev)
        self.prev_tag = torch.zeros(n, 3, device=dev)
        self.prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dt_since_valid = torch.zeros(n, device=dev)
        self.cam_accum = torch.zeros(n, device=dev)
        self.imu_bias = torch.zeros(n, device=dev)
        self.imu_yaw = torch.zeros(n, device=dev)
        self.tag_raw = torch.zeros(n, 3, device=dev)
        self.tag_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.imu_w = torch.zeros(n, device=dev)
        self.coef = torch.zeros(n, 16, 3, device=dev)
        self.dz = torch.zeros(n, device=dev)
        self.drive_tau = torch.ones(n, 3, device=dev)
        self.brake_tau = torch.ones(n, 3, device=dev)
        self.cov = torch.eye(3, device=dev).expand(n, 3, 3).clone()
        self.wheel_gain = torch.ones(n, 4, device=dev)
        self.yaw_scale = torch.ones(n, device=dev)
        self.sag = torch.ones(n, device=dev)
        self.latency = torch.zeros(n, device=dev)
        self.drop_p = torch.zeros(n, device=dev)
        self.blackout_start = torch.full((n,), 1e9, device=dev)
        self.pose_ring = torch.zeros(n, _RING_K, 3, device=dev)
        self.ring_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.ring_head = 0
        self.reset(torch.arange(n))

    # ---- reset ----
    def _sample_dynamics(self, idx: torch.Tensor):
        dev = self.device
        idx_d = idx.to(dev)
        m = len(idx)
        full = self.level["full_posterior"] if self.cfg.full_posterior is None else self.cfg.full_posterior
        if full:
            pick = torch.randint(0, len(self.particles), (m,), generator=self.gen_cpu)
            coef = torch.stack([self.particles[int(i)]["coef"] for i in pick]).to(dev)
            dz = torch.tensor([self.particles[int(i)]["dz"] for i in pick],
                              dtype=torch.float32, device=dev)
            drive = torch.stack([self.particles[int(i)]["drive"] for i in pick]).to(dev)
            brake = torch.stack([self.particles[int(i)]["brake"] for i in pick]).to(dev)
            cov = torch.stack([self.particles[int(i)]["cov"] for i in pick]).to(dev)
            # extra DR grounded in observed run-to-run variation
            self.wheel_gain[idx_d] = 0.85 + 0.30 * torch.rand(m, 4, device=dev, generator=self.gen)
            self.yaw_scale[idx_d] = 0.5 + 1.0 * torch.rand(m, device=dev, generator=self.gen)
            self.sag[idx_d] = 0.90 + 0.10 * torch.rand(m, device=dev, generator=self.gen)
        elif self.cfg.use_mean_dynamics:
            coef = self.mean_coef.unsqueeze(0).expand(m, -1, -1)
            dz = torch.full((m,), self.mean_dz, dtype=torch.float32, device=dev)
            drive = self.mean_drive.unsqueeze(0).expand(m, -1)
            brake = self.mean_brake.unsqueeze(0).expand(m, -1)
            cov = self.mean_cov.unsqueeze(0).expand(m, -1, -1)
            self.wheel_gain[idx_d] = 1.0
            self.yaw_scale[idx_d] = 1.0
            self.sag[idx_d] = 1.0
        else:
            p0 = self.particles[0]
            coef = p0["coef"].unsqueeze(0).expand(m, -1, -1)
            dz = torch.full((m,), p0["dz"], dtype=torch.float32, device=dev)
            drive = p0["drive"].unsqueeze(0).expand(m, -1)
            brake = p0["brake"].unsqueeze(0).expand(m, -1)
            cov = p0["cov"].unsqueeze(0).expand(m, -1, -1)
            self.wheel_gain[idx_d] = 1.0
            self.yaw_scale[idx_d] = 1.0
            self.sag[idx_d] = 1.0
        self.coef[idx_d] = coef
        self.dz[idx_d] = dz
        self.drive_tau[idx_d] = drive
        self.brake_tau[idx_d] = brake
        self.cov[idx_d] = cov

    def reset(self, idx: torch.Tensor | None = None):
        if idx is None:
            idx = torch.arange(self.n)
        idx = idx.cpu()
        m = len(idx)
        area = self.level["area"]
        g = self.gen_cpu
        span = self.cfg.spawn_dist if self.cfg.spawn_dist is not None else self.level["dist"]
        if isinstance(span, float):
            d = torch.full((m,), span)
        else:
            d = span[0] + (span[1] - span[0]) * torch.rand(m, generator=g)
        bmax = math.radians(self.cfg.bearing_max if self.cfg.bearing_max is not None
                              else self.level["bearing"])
        b = (torch.rand(m, generator=g) * 2 - 1) * bmax
        ymax = math.radians(self.cfg.yaw_max if self.cfg.yaw_max is not None
                            else self.level["yaw"])
        yaw = (torch.rand(m, generator=g) * 2 - 1) * ymax
        dev = self.device
        self.pose[idx.to(dev)] = torch.stack(
            [(d * torch.sin(b)).clamp(-area, area),
             (d * torch.cos(b)).clamp(-area, area), yaw], dim=1).to(dev)
        self.vel[idx.to(dev)] = 0
        self.time[idx.to(dev)] = 0
        self.still[idx.to(dev)] = 0
        self.vel_still[idx.to(dev)] = 0
        for name in ("done", "latched_success", "latched_coll", "latched_timeout", "latched_false_park"):
            getattr(self, name)[idx.to(dev)] = False
        self._sample_dynamics(idx)
        # sensor state
        lat = self.cfg.latency_override if self.cfg.latency_override is not None else self.level["latency"]
        di_lat = idx.to(dev)
        if isinstance(lat, tuple):
            self.latency[di_lat] = lat[0] + (lat[1] - lat[0]) * torch.rand(m, device=dev, generator=self.gen)
        else:
            self.latency[di_lat] = float(lat)
        dr = self.cfg.drops_override if self.cfg.drops_override is not None else self.level["drops"]
        if isinstance(dr, tuple):
            self.drop_p[di_lat] = dr[0] + (dr[1] - dr[0]) * torch.rand(m, device=dev, generator=self.gen)
        else:
            self.drop_p[di_lat] = float(dr)
        self.blackout_total = float(self.level["blackout"])
        if self.blackout_total > 0:
            # one random blackout window per episode
            self.blackout_start[di_lat] = torch.rand(m, device=dev, generator=self.gen) * (self.cfg.max_time - self.blackout_total)
        else:
            self.blackout_start[di_lat] = 1e9
        di = idx.to(dev)
        self.prev_act[di] = 0
        self.hist[di] = 0
        self.hist_age[di] = 1e9
        self.tag_ema[di] = 0
        self.prev_tag[di] = 0
        self.prev_valid[di] = False
        self.dt_since_valid[di] = 0
        if self.cam_hz < 1.0 / self.dt - 1e-6:
            self.cam_accum[di] = (1.0 / self.cam_hz) - self.dt
        else:
            self.cam_accum[di] = 0.0
        self.imu_bias[di] = 0.002 * torch.randn(len(di), device=dev, generator=self.gen)
        self.imu_yaw[di] = self.pose[di, 2].clone()
        # latency ring: prefill with the spawn pose so delayed reads are valid
        self.pose_ring[di] = self.pose[di].unsqueeze(1).expand(-1, _RING_K, -1)
        self.ring_count[di] = 1
        return self._privileged()

    def _push_buffer(self):
        self.pose_ring[:, self.ring_head % _RING_K] = self.pose
        self.ring_head += 1
        self.ring_count += 1

    def _delayed_pose(self):
        # per-env delayed read; clamped to available history
        sb = (self.latency / self.dt).round().long().clamp_min(0)
        avail = (self.ring_count - 1).clamp_min(0)
        use = torch.minimum(sb, avail)
        pos = (self.ring_head - 1 - use) % _RING_K
        return self.pose_ring[torch.arange(self.n, device=self.device), pos]

    # ---- step ----
    def step(self, act: torch.Tensor):
        from reward import step_reward
        dev = self.device
        act_t = act.to(dev)
        cmd = act_t[:, :3].clamp(-1, 1)
        prev_done = self.done.clone()
        saved = {"pose": self.pose.clone(), "vel": self.vel.clone(), "time": self.time.clone(),
                 "still": self.still.clone(), "vel_still": self.vel_still.clone(),
                 "hist": self.hist.clone(), "age": self.hist_age.clone(),
                 "imu_yaw": self.imu_yaw.clone(), "prev_act": self.prev_act.clone(),
                 "ema": self.tag_ema.clone(), "ptag": self.prev_tag.clone(),
                 "pvalid": self.prev_valid.clone(), "dt_sv": self.dt_since_valid.clone(),
                 "cam_acc": self.cam_accum.clone()}
        # mecanum inverse mixer -> normalized wheel PWM [-1,1]. NOTE: the fitted
        # coefficients retain hardware PWM polarity (mecanum_dynamics.py):
        # translation needs negation (+cmd -> +body vel) while the turn
        # channel matches the ideal convention un-negated (verified empirically
        # per-channel on posterior-mean dynamics).
        vx, vy, w = cmd[:, 0], cmd[:, 1], cmd[:, 2] * HALF_BASE * 4.0
        trans = torch.stack([vx - vy, vx + vy, vx + vy, vx - vy], dim=1)
        turn = torch.stack([-w, -w, w, w], dim=1).expand_as(trans)
        wheels = (-trans + turn)
        wheels = (wheels * self.wheel_gain * self.sag[:, None]).clamp(-1, 1)
        feat = _wheel_features(wheels, self.dz)
        target = torch.bmm(feat[:, None, :], self.coef).squeeze(1)  # [N,3]
        target[:, 2] *= self.yaw_scale
        braking = wheels.abs().max(dim=1).values < 1e-8
        tau = torch.where(braking[:, None].expand_as(self.drive_tau), self.brake_tau, self.drive_tau)
        decay = torch.exp(-self.dt / tau.clamp_min(0.02))
        self.vel = decay * self.vel + (1 - decay) * target
        # OU-like transition noise from posterior covariance. Gated by activity:
        # the covariance was fitted on driving transitions (roller ripple, load
        # transfer, coupling variation); the 60 s rest recording shows a still
        # chassis, and a braked mecanum at rest is held by static friction, so
        # process noise vanishes at rest (observation noise remains in the
        # sensor emulator). Coasting keeps disturbances via the velocity term.
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(0.02)
        scale = torch.sqrt(1 - torch.exp(-2 * self.dt / tau_mean)).squeeze(1)
        cmd_lvl = wheels.abs().mean(dim=1)
        spd_lvl = (self.vel[:, :2].norm(dim=1) / 0.6 + self.vel[:, 2].abs() / 2.5).clamp_max(1.0)
        activity = torch.maximum(cmd_lvl, spd_lvl).clamp(0.03, 1.0)
        try:
            L = torch.linalg.cholesky((self.cov + 1e-12 * torch.eye(3, device=dev).expand_as(self.cov)))
            z = torch.randn(self.n, 3, device=dev, generator=self.gen)
            self.vel += (activity * scale)[:, None] * torch.bmm(L, z[:, :, None]).squeeze(2)
        except torch.linalg.LinAlgError:
            pass
        prev_pose = self.pose.clone()
        th = self.pose[:, 2] + self.vel[:, 2] * self.dt / 2
        c, s = torch.cos(th), torch.sin(th)
        self.pose[:, 0] += self.dt * (c * self.vel[:, 0] - s * self.vel[:, 1])
        self.pose[:, 1] += self.dt * (s * self.vel[:, 0] + c * self.vel[:, 1])
        self.pose[:, 2] += self.dt * self.vel[:, 2]
        self.pose[:, 2] = torch.atan2(torch.sin(self.pose[:, 2]), torch.cos(self.pose[:, 2]))
        self.time += self.dt
        self._push_buffer()

        # ---- sensors ----
        noise = float(self.level["noise"])
        delayed = self._delayed_pose()
        dist = delayed[:, :2].norm(dim=1)
        # Bearing: sim yaw=0 faces -y (SDF front camera + Q/left convention);
        # facing rotates +yaw from the tag direction, so relative bearing ADDS
        # yaw at the delayed capture timestamp.
        bearing = torch.atan2(delayed[:, 0], delayed[:, 1]) + delayed[:, 2]
        bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
        # Near/under the deck (< 0.30 m) the point-bearing model is meaningless
        # (y -> 0 or negative), so the per-level switch alone decides there:
        # L0-L2 keep the tag (spec: perfect state / tags visible), L3+ lose it.
        # The camera minimum range (~0.10 m) lies inside this zone, so it is
        # covered by under_drone_loss rather than applied at every level.
        near_deck = dist < 0.30
        in_fov = (bearing.abs() < FOV_HALF) & (dist < 5.0)
        if self.level["under_drone_loss"]:
            seen = in_fov & ~near_deck
        else:
            seen = in_fov | near_deck
        if self.cam_hz >= 1.0 / self.dt - 1e-6:
            cam_tick = torch.ones(self.n, dtype=torch.bool, device=dev)
        else:
            cam_period = 1.0 / self.cam_hz
            self.cam_accum += self.dt
            cam_tick = self.cam_accum >= (cam_period - 1e-6)
            self.cam_accum = torch.where(cam_tick, self.cam_accum - cam_period, self.cam_accum)
        in_blackout = (self.time > self.blackout_start) & (self.time < self.blackout_start + self.blackout_total)
        dropped = torch.rand(self.n, device=dev, generator=self.gen) < self.drop_p
        valid = seen & cam_tick & (~in_blackout) & (~dropped)
        pos_std = noise * (0.008 + 0.020 * dist)
        yaw_std = noise * (math.radians(2) + math.radians(2) * dist)
        tag = delayed + torch.randn(delayed.shape, device=dev, generator=self.gen) * torch.stack([pos_std, pos_std, yaw_std], dim=1)
        out_p = float(self.level.get("outliers", 0.0))
        if out_p > 0:
            spike = torch.rand(self.n, device=dev, generator=self.gen) < out_p
            tag[spike, :2] += 0.15 * torch.randn(int(spike.sum()), 2, device=dev, generator=self.gen)
        tag_raw = tag.clone()  # pre-zeroing estimator values, for the golden trace
        tag[~valid] = 0
        self.tag_raw = tag_raw
        self.tag_valid = valid
        self.hist_age = torch.where(valid, torch.zeros_like(self.hist_age), self.hist_age + self.dt)
        # valid-gated smoothed tag velocity (finite difference over dt_since_valid
        # + EMA): holds/decays across inter-frame invalid ticks (e.g. 30 Hz camera).
        self.dt_since_valid += self.dt
        pair = valid & self.prev_valid & (self.dt_since_valid <= 0.5)
        raw_diff = tag - self.prev_tag
        raw_diff[:, 2] = torch.atan2(torch.sin(raw_diff[:, 2]), torch.cos(raw_diff[:, 2]))
        diff = raw_diff / self.dt_since_valid.clamp_min(self.dt)[:, None]
        ema_valid = torch.where(pair[:, None].expand_as(tag),
                                0.5 * self.tag_ema + 0.5 * diff,
                                torch.zeros_like(tag))
        ema_decay = self.tag_ema * math.exp(-self.dt / 0.25)
        self.tag_ema = torch.where(valid[:, None].expand_as(tag), ema_valid, ema_decay)
        self.prev_tag = torch.where(valid[:, None].expand_as(tag), tag, self.prev_tag)
        self.prev_valid = self.prev_valid | valid
        self.dt_since_valid = torch.where(valid, torch.zeros_like(self.dt_since_valid), self.dt_since_valid)
        # IMU: true yaw rate + white + bias, yaw integrated with drift
        gyro_white = noise * 0.002 * torch.randn(self.n, device=dev, generator=self.gen)
        imu_w = self.vel[:, 2] + gyro_white + self.imu_bias
        self.imu_w = imu_w
        self.imu_bias *= (1 - self.dt / 0.5)
        self.imu_bias += math.sqrt(2 * self.dt / 0.5) * 0.001 * torch.randn(self.n, device=dev, generator=self.gen)
        self.imu_yaw += imu_w * self.dt

        # ---- termination ----
        slow = (self.vel.abs() < torch.tensor([0.01, 0.01, math.radians(2)], device=dev)).all(dim=1)
        near = ((self.pose[:, 0].abs() < 0.03) & (self.pose[:, 1].abs() < 0.05)
                & (self.pose[:, 2].abs() < math.radians(5)))
        self.still = torch.where(slow & near, self.still + self.dt, torch.zeros_like(self.still))
        self.vel_still = torch.where(slow, self.vel_still + self.dt, torch.zeros_like(self.vel_still))
        success = self.still >= 1.0
        if act_t.shape[1] >= 4:
            declared = act_t[:, 3] > 0.5
            false_park = declared & (~(near & slow))
        elif self.cfg.declare_parked:
            false_park = (self.vel_still >= 1.0) & (~near)
        else:
            false_park = torch.zeros(self.n, dtype=torch.bool, device=dev)
        # collision: car rect vs skid capsules (conservative AABB approx)
        hw, hl = CAR_W / 2, CAR_L / 2
        touch_x = (self.pose[:, 0].abs() + hw > SKID_X - GEAR_R) & (self.pose[:, 0].abs() - hw < SKID_X + GEAR_R)
        touch_y = self.pose[:, 1].abs() < GEAR_LEN / 2 + hl
        collision = touch_x & touch_y
        area = self.level["area"]
        out = (self.pose[:, 0].abs() > area) | (self.pose[:, 1].abs() > area)
        timeout = (self.time >= self.cfg.max_time) | out
        done = success | collision | timeout | false_park

        # student history frame (angles wrapped: the raw integrated IMU yaw
        # drifts unboundedly over long episodes, saturating recurrent gates)
        imu_yaw_w = torch.atan2(torch.sin(self.imu_yaw), torch.cos(self.imu_yaw))
        frame = torch.cat([tag,
                           self.hist_age.clamp_max(5.0)[:, None], valid.float()[:, None],
                           imu_w[:, None], imu_yaw_w[:, None],
                           self.prev_act, self.tag_ema], dim=1)
        self.hist = torch.cat([self.hist[:, 1:], frame[:, None, :]], dim=1)
        rate_prev = self.prev_act.detach()
        self.prev_act = cmd.detach()

        reward = step_reward(prev_pose, self.pose, cmd, rate_prev, self.vel, success,
                             near, collision, false_park, self.dt,
                             rate_w=self.cfg.rate_w, lat_w=self.cfg.lat_w,
                             yaw_w=self.cfg.yaw_w)
        # out-of-bounds penalty
        reward = reward - out.float() * 2.0
        # latch terminal outcomes; freeze previously-done envs so flags can't be overwritten
        newly = ~prev_done & done
        self.latched_success |= newly & success
        self.latched_coll |= newly & collision
        self.latched_false_park |= newly & false_park & ~success & ~collision
        self.latched_timeout |= newly & timeout & ~success & ~collision & ~false_park
        self.done |= done
        if bool(prev_done.any()):
            self.pose[prev_done] = saved["pose"][prev_done]
            self.vel[prev_done] = saved["vel"][prev_done]
            self.time[prev_done] = saved["time"][prev_done]
            self.still[prev_done] = saved["still"][prev_done]
            self.vel_still[prev_done] = saved["vel_still"][prev_done]
            self.hist[prev_done] = saved["hist"][prev_done]
            self.hist_age[prev_done] = saved["age"][prev_done]
            self.imu_yaw[prev_done] = saved["imu_yaw"][prev_done]
            self.prev_act[prev_done] = saved["prev_act"][prev_done]
            self.tag_ema[prev_done] = saved["ema"][prev_done]
            self.prev_tag[prev_done] = saved["ptag"][prev_done]
            self.prev_valid[prev_done] = saved["pvalid"][prev_done]
            self.dt_since_valid[prev_done] = saved["dt_sv"][prev_done]
            self.cam_accum[prev_done] = saved["cam_acc"][prev_done]
            reward = torch.where(prev_done, 0.0, reward)
            success, collision, timeout, false_park, done = (
                self.latched_success, self.latched_coll,
                self.latched_timeout, self.latched_false_park, self.done)
        obs = {"pose": self.pose.clone(), "vel": self.vel.clone(),
               "student": self.hist.clone(), "tag": tag.clone(), "valid": valid.clone()}
        info = {"success": success, "collision": collision, "near": near,
                "timeout": timeout, "false_park": false_park, "done": done, "out": out}
        return obs, reward, done, info

    def _privileged(self):
        return {"pose": self.pose.clone(), "vel": self.vel.clone(),
                "student": self.hist.clone()}
