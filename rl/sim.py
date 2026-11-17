"""Batched mecanum SE(2) parking env with grounded DR + sensor emulator.

Frame convention: drone at origin, skids along ``y``, car pose ``(x, y, yaw)``
with ``yaw = 0`` facing ``-y`` toward the drone and **CCW-positive** yaw
(opposite to ``ugv.parking.Pose2`` CW-positive yaw).

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
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "simulation" / "gazebo" / "mecanum_bayesian_profile.json"

LEVELS = {
    "L0": dict(
        dist=0.75,
        bearing=0.0,
        yaw=0.0,
        noise=0.0,
        latency=0.0,
        drops=0.0,
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        full_posterior=False,
        area=1.0,
        cam_hz=50.0,
    ),
    "L1": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=0.08,
        drops=0.0,
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        full_posterior=False,
        area=1.5,
        cam_hz=30.0,
    ),
    "L2": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=0.08,
        drops=0.0,
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=False,
        full_posterior=True,
        area=1.5,
        cam_hz=30.0,
    ),
    "L3": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.08, 0.20),
        drops=(0.25, 0.45),
        outliers=0.02,
        blackout=3.0,
        under_drone_loss=True,
        full_posterior=True,
        area=1.5,
        cam_hz=30.0,
    ),
    "L4": dict(
        dist=(0.5, 4.0),
        bearing=180.0,
        yaw=180.0,
        noise=1.0,
        latency=(0.08, 0.20),
        drops=(0.25, 0.45),
        outliers=0.02,
        blackout=3.0,
        under_drone_loss=True,
        full_posterior=True,
        area=4.0,
        cam_hz=30.0,
    ),
    "L5": dict(
        dist=(0.5, 8.0),
        bearing=180.0,
        yaw=180.0,
        noise=1.0,
        latency=(0.08, 0.20),
        drops=(0.25, 0.45),
        outliers=0.02,
        blackout=3.0,
        under_drone_loss=True,
        full_posterior=True,
        area=12.0,
        cam_hz=30.0,
    ),
    "L_chaos": dict(
        # Chaos level: wide spawn, full posterior with turn-column-consistent
        # particles only, bursty vision drops, blind-approach blackouts, and
        # heavy-but-bounded DR (see EnvConfig dr_scale and set_dr_scale()).
        dist=(0.5, 2.5),
        bearing_rad=(-1.2, 1.2),
        yaw_rad=(-math.pi, math.pi),
        pos_noise_xy=0.015,
        pos_noise_z=0.03,  # SE(2): reserved, no z axis on the tag; kept for schema parity
        yaw_noise=0.06,
        latency_steps=(1, 6),
        drops=(0.10, 0.30),
        burst_drops=True,
        outliers=0.02,
        blackout_prob=0.008,
        blackout_len=(15, 50),  # camera frames; converted to seconds via dt
        under_drone_loss=True,
        area_scale=2.5,
        full_posterior=True,
        randomize_dynamics=True,
        consistent_particles=True,
        wheel_fault_prob=0.10,
        floor_slip_prob=0.015,
        action_delay_prob=0.5,
        real_plant=True,
        duty_cap=0.44,
        min_tag_range=0.40,
        cam_hz_range=(10.0, 30.0),
        gyro_noise_std=(0.05, 0.70),
        disturbance_std=0.03,
        reanchor_imu_yaw=True,
    ),
    "L_field": dict(
        # L_real plus what the field adds: a miscalibrated plant (gain 0.5-1.5x, deadzone
        # 0.4-2x, vs L_real 0.7-1.3x / 0.5-1.5x) and folded or oblique tags, modelled as
        # correlated measurement bursts (fold_rate events/s, fold_len seconds each).
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.05, 0.10),
        drops=0.0,
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=True,
        full_posterior=False,
        area=1.5,
        cam_hz=20.0,
        cam_hz_range=(10.0, 30.0),
        min_tag_range=0.40,
        gyro_noise_std=(0.05, 0.7),
        duty_cap=0.44,
        real_plant=True,
        calib_gain=(0.5, 1.5),
        calib_dz=(0.4, 2.0),
        fold_rate=0.15,
        fold_len=(0.3, 1.5),
    ),
    "L_real": dict(
        dist=(0.5, 1.5),
        bearing=30.0,
        yaw=60.0,
        noise=1.0,
        latency=(0.05, 0.10),
        drops=0.0,
        outliers=0.0,
        blackout=0.0,
        under_drone_loss=True,
        full_posterior=False,
        area=1.5,
        cam_hz=20.0,
        cam_hz_range=(10.0, 30.0),
        min_tag_range=0.40,
        gyro_noise_std=(0.05, 0.7),
        duty_cap=0.44,
        real_plant=True,
    ),
}

# H-0001 plant-calibration: L1 with the real-car plant held at FIXED measured values (no DR on
# the plant). Gains [strafe, fwd, yaw] per duty and deadzones [strafe, fwd, yaw] as used by
# the real_plant branch; gyro std 0.63 rad/s is the field value. Forward and yaw gains come
# from calibration/runs/l7/20261113-181040-probe-axes/fit.json. The strafe gain and deadzone
# are NOT measured here: that fit's strafe entry is degenerate (n=3, deadzone -3.69), so the
# nominal 1.6 / 0.13 are kept.
LEVELS["L1_fitted"] = dict(
    LEVELS["L1"],
    real_plant=True,
    randomize_dynamics=False,
    plant_gain=(1.6, 2.3186, 15.66),
    plant_dz=(0.13, 0.05, 0.10),
    gyro_noise_std=0.63,
)

# Provisional geometry (config/parking.json unverified): car 0.16 x 0.22 m,
# gear clear width 0.254 m, bar length 0.3048 m, radius 0.03 m.
CAR_W, CAR_L = 0.16, 0.22
GEAR_GAP, GEAR_LEN, GEAR_R = 0.254, 0.3048, 0.03
SKID_X = GEAR_GAP / 2 + GEAR_R  # 0.157
HALF_BASE = (0.21 + 0.149) / 2  # wheelbase+track mixer arm
HIST = 50  # 1 s at 50 Hz
_RING_K = 32  # latency ring depth; covers 0.64 s at 50 Hz
_CMD_K = 24  # command onset delay ring depth; covers 0.48 s at 50 Hz
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
    noise_override: float | None = None  # override LEVELS noise scale (curriculum)
    rate_w: float = 0.0  # action-rate penalty weight; L0-gate teacher used 0.0
    lat_w: float = 2.0  # extra lateral-error weight in progress potential
    yaw_w: float = 1.0  # yaw-error weight in progress potential
    cam_hz: float | None = None  # override LEVELS cam_hz (50 Hz L0, 30 Hz L1+)
    use_mean_dynamics: bool = True  # L0/L1 use posterior mean unless False (particles[0])
    full_posterior: bool | None = None  # override LEVELS full_posterior if set
    declare_parked: bool = False  # enable 3-dim still-outside-goal false_park detection
    real_plant: bool | None = None  # override LEVELS real_plant (real-car physical axes + fit)
    duty_cap: float | None = None  # wheel-level duty cap (e.g. 0.44 on real car)
    gyro_noise_std: float | tuple[float, float] | None = None  # gyro white noise std (rad/s)
    cam_hz_range: tuple[float, float] | None = None  # per-env valid-pose rate range (Hz)
    min_tag_range: float | None = None  # tag loss range threshold when under_drone_loss is True
    calib_gain: tuple[float, float] | None = None  # real-plant gain multiplier range
    calib_dz: tuple[float, float] | None = None  # real-plant deadzone multiplier range
    fold_rate: float | None = None  # folded-tag measurement bursts per second
    # --- Domain-randomization scaling (Deliverable A) ---
    dr_scale: float = 1.0  # 0 = nominal/easy anchor, 1 = full distribution
    consistent_particles: bool = False  # sample only turn-consistent posterior particles
    burst_drops: bool = False  # Markov-sticky vision dropouts (0.70*s retention)
    disturbance_std: float = 0.0  # per-step body-frame impulse std (m/s, rad/s), x s
    reanchor_imu_yaw: bool = False  # pull imu_yaw toward the tag on valid ticks
    spawn_bearing: tuple | None = None  # rad, overrides level bearing (radians if set)
    spawn_yaw: tuple | None = None  # rad, overrides level yaw
    pos_noise_xy: float | None = None  # absolute XY tag noise std; overrides noise model
    yaw_noise: float | None = None  # absolute yaw tag noise std; overrides noise model
    pos_noise_z: float | None = None  # reserved (SE(2): no z); kept for schema parity


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
        self.mean_cov = torch.tensor(
            prof["innovation_covariance"], dtype=torch.float32, device=self.device
        )
        self.particles = [
            {
                "coef": torch.tensor(p["coefficients"], dtype=torch.float32, device=self.device),
                "dz": float(p["deadzone"]),
                "drive": torch.tensor(p["drive_tau_s"], dtype=torch.float32, device=self.device),
                "brake": torch.tensor(p["brake_tau_s"], dtype=torch.float32, device=self.device),
                "cov": torch.tensor(
                    p["innovation_covariance"], dtype=torch.float32, device=self.device
                ),
            }
            for p in prof["posterior_particles"]
        ]
        self.level = LEVELS[cfg.level]
        self.real_plant = bool(
            self.level.get("real_plant", False) if cfg.real_plant is None else cfg.real_plant
        )
        self.cam_hz = float(
            cfg.cam_hz if cfg.cam_hz is not None else self.level.get("cam_hz", 50.0)
        )
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
        self.env_cam_hz = torch.full((n,), self.cam_hz, device=dev)
        self.gyro_noise_std = torch.full((n,), 0.002, device=dev)
        self.rf_gain = torch.zeros(n, 3, device=dev)
        self.rf_dz = torch.zeros(n, 3, device=dev)
        self.rf_onset_steps = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self.rf_strafe_yaw = torch.zeros(n, device=dev)
        self.cmd_ring = torch.zeros(n, _CMD_K, 3, device=dev)
        self.cmd_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.cmd_head = 0
        # ---- effective config: cfg overrides level dict (existing levels carry
        # none of the new keys, so they all resolve to the declared defaults) ----
        self.eff_consistent = bool(
            cfg.consistent_particles or self.level.get("consistent_particles", False)
        )
        self.eff_burst = bool(cfg.burst_drops or self.level.get("burst_drops", False))
        self.eff_reanchor = bool(cfg.reanchor_imu_yaw or self.level.get("reanchor_imu_yaw", False))
        self.eff_disturbance = float(
            cfg.disturbance_std
            if cfg.disturbance_std > 0
            else self.level.get("disturbance_std", 0.0)
        )
        self.eff_calib_gain = tuple(cfg.calib_gain or self.level.get("calib_gain", (0.7, 1.3)))
        self.eff_calib_dz = tuple(cfg.calib_dz or self.level.get("calib_dz", (0.5, 1.5)))
        self.eff_fold_rate = float(
            cfg.fold_rate if cfg.fold_rate is not None else self.level.get("fold_rate", 0.0)
        )
        self.eff_fold_len = tuple(self.level.get("fold_len", (0.3, 1.5)))
        self.eff_floor_slip = float(self.level.get("floor_slip_prob", 0.0))
        self.eff_wheel_fault = float(self.level.get("wheel_fault_prob", 0.0))
        self.eff_action_delay = float(self.level.get("action_delay_prob", 0.0))
        self.eff_randomize = bool(self.level.get("randomize_dynamics", True))
        self.eff_latency_steps = self.level.get("latency_steps")
        self.eff_blackout_prob = float(self.level.get("blackout_prob", 0.0))
        self.eff_blackout_len = self.level.get("blackout_len_steps") or self.level.get(
            "blackout_len"
        )
        self.area = (
            float(self.level["area"])
            if "area" in self.level
            else (float(self.level.get("area_scale", 1.0)) * 1.5)
        )
        # Blackout is a per-env tensor (steps x dt); legacy levels hold the level
        # constant in every slot (bit-identical values to the old float).
        self.blackout_total = torch.zeros(n, device=dev)
        # Integer latency in sim steps; _delayed_pose reads this.
        self.latency_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # Camera period in sim steps (rounded 1/hz/dt); latent_dynamics col 11.
        self.cam_period_steps = torch.ones(n, dtype=torch.long, device=dev)
        # Per-env effective min tag range (0.15 m .. cfg.min_tag_range).
        self.env_min_tag_range = torch.full((n,), 0.30, device=dev)
        # ADR frontier bookkeeping: envs [0, n_frontier) train at frontier_scale,
        # the rest anchor at low dr_scale (no RNG involved).
        self.frontier_scale = float(cfg.dr_scale)
        self.anchor_frac = 0.0
        self.anchor_max = 0.2
        self.n_frontier = n
        self.dr_scale = torch.full((n,), float(cfg.dr_scale), device=dev)
        # Per-episode action delay (posterior-plant path only).
        self.action_delay = torch.zeros(n, dtype=torch.long, device=dev)
        # Markov-sticky vision dropout state (burst_drops).
        self.drop_streak = torch.zeros(n, dtype=torch.long, device=dev)
        # Folded-tag burst: seconds left and the (x, y, yaw) measurement error it adds.
        self.fold_left = torch.zeros(n, device=dev)
        self.fold_vec = torch.zeros(n, 3, device=dev)
        self.wheel_fault = torch.zeros(n, dtype=torch.bool, device=dev)
        self.particle_id = torch.full((n,), -1, dtype=torch.long, device=dev)
        # Posterior particles whose turn channel actually works at cmd=[0,0,±1].
        self.valid_particle_ids = self._compute_valid_particle_ids()
        self.reset(torch.arange(n))

    # ---- ADR frontier / DR scaling ----
    def _compute_valid_particle_ids(self):
        # The 16-feature wheel map and the [P,16,3] posterior coefficients are
        # what this sim actually deploys, so the turn-authority check is done
        # with a steady-state pass through the same map (no separate feature
        # definition to drift). Checks both pure cardinal commands and combined
        # forward+turn commands [0, -0.6, +-0.8] plus bounded forward coupling
        # so combined maneuvers never invert turn/strafe signs.
        dev = self.device
        coef = torch.stack([p["coef"] for p in self.particles]).to(dev)  # [P,16,3]
        dz = torch.tensor([p["dz"] for p in self.particles], dtype=torch.float32, device=dev)
        ub = HALF_BASE * 4.0
        cmds = torch.tensor(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
                [0, -0.6, 0.8],
                [0, -0.6, -0.8],
            ],
            dtype=torch.float32,
            device=dev,
        )

        def wheels_for(c):
            # Same inverse mixer as step() with gain=1 and zero sag.
            vx, vy, w = float(c[0]), float(c[1]), float(c[2]) * ub
            trans = torch.tensor([vx - vy, vx + vy, vx + vy, vx - vy], device=dev)
            turn = torch.tensor([-w, -w, w, w], device=dev)
            return -trans + turn

        tgt = []
        for c in cmds:
            u = wheels_for(c).unsqueeze(0).expand(coef.shape[0], 4).contiguous()
            feat = _wheel_features(u, dz)
            tgt.append(torch.bmm(feat.unsqueeze(1), coef).squeeze(1))  # [P,3]
        dxp, dxm, dyp, dym, wp, wm, fwp, fwm = tgt
        ok = (
            (dxp[:, 0] > 0.25)
            & (dxm[:, 0] < -0.25)
            & (dyp[:, 1] > 0.15)
            & (dym[:, 1] < -0.15)
            & (wp[:, 2] > 0.40)
            & (wm[:, 2] < -0.40)
            & (fwp[:, 2] > 0.15)
            & (fwm[:, 2] < -0.15)
            & (dym[:, 0].abs() < 0.55)
            & (dym[:, 2].abs() < 1.5)
        )
        ids = torch.nonzero(ok).squeeze(1)
        if ids.numel() == 0:
            ids = torch.arange(coef.shape[0], device=dev)
        return ids

    def set_dr_scale(
        self, frontier_scale: float, anchor_frac: float = 0.0, anchor_max: float = 0.2
    ) -> None:
        """ADR frontier step. Deterministic: no RNG draws. Envs [0, n_frontier)
        train at `frontier_scale`; the remaining anchor envs are evenly spaced in
        [0, min(anchor_max, frontier_scale)]."""
        frontier_scale = min(max(float(frontier_scale), 0.0), 1.0)
        anchor_frac = float(anchor_frac)
        anchor_max = float(anchor_max)
        self.frontier_scale = frontier_scale
        self.anchor_frac = anchor_frac
        self.anchor_max = anchor_max
        self.n_frontier = int(round((1.0 - anchor_frac) * self.n))
        self.dr_scale[: self.n_frontier] = frontier_scale
        n_anchor = self.n - self.n_frontier
        if n_anchor > 0:
            lo = min(anchor_max, frontier_scale)
            self.dr_scale[self.n_frontier :] = torch.linspace(0.0, lo, n_anchor, device=self.device)
        self._apply_dr_env_fields()

    def _apply_dr_env_fields(self):
        """Recompute deterministic per-env DR fields (min tag range)."""
        base = (
            self.cfg.min_tag_range
            if self.cfg.min_tag_range is not None
            else self.level.get("min_tag_range")
        )
        if base is not None and float(base) > 0.0:
            self.env_min_tag_range[:] = torch.clamp(
                (self.dr_scale - 0.35) / 0.65, 0.0, 1.0
            ) * float(base)
        else:
            self.env_min_tag_range[:] = torch.clamp((self.dr_scale - 0.35) / 0.65, 0.0, 1.0) * 0.30
        steps = (1.0 / self.env_cam_hz.clamp_min(1e-3) / self.dt).round().long().clamp_min(1)
        self.cam_period_steps[:] = steps

    # ---- snapshot API (bit-identical branching) ----
    _SNAPSHOT_TENSORS = (
        "pose",
        "vel",
        "time",
        "still",
        "vel_still",
        "done",
        "latched_success",
        "latched_coll",
        "latched_timeout",
        "latched_false_park",
        "prev_act",
        "hist",
        "hist_age",
        "tag_ema",
        "prev_tag",
        "prev_valid",
        "dt_since_valid",
        "cam_accum",
        "imu_bias",
        "imu_yaw",
        "tag_raw",
        "tag_valid",
        "imu_w",
        "coef",
        "dz",
        "drive_tau",
        "brake_tau",
        "cov",
        "wheel_gain",
        "yaw_scale",
        "sag",
        "latency",
        "latency_steps",
        "drop_p",
        "blackout_start",
        "blackout_total",
        "pose_ring",
        "ring_count",
        "env_cam_hz",
        "cam_period_steps",
        "gyro_noise_std",
        "rf_gain",
        "rf_dz",
        "rf_onset_steps",
        "rf_strafe_yaw",
        "cmd_ring",
        "cmd_count",
        "action_delay",
        "drop_streak",
        "fold_left",
        "fold_vec",
        "wheel_fault",
        "particle_id",
        "env_min_tag_range",
        "dr_scale",
    )
    _SNAPSHOT_SCALARS = (
        "ring_head",
        "cmd_head",
        "frontier_scale",
        "anchor_frac",
        "anchor_max",
        "n_frontier",
    )

    def get_state(self) -> dict:
        """Deep copy of every tensor and scalar that influences future steps,
        plus both generator states. Restoring it and replaying the same actions
        reproduces (obs, rew, done, info) bit-identically."""
        state = {name: getattr(self, name).clone() for name in self._SNAPSHOT_TENSORS}
        state.update({name: getattr(self, name) for name in self._SNAPSHOT_SCALARS})
        state["gen_state"] = self.gen.get_state().clone()
        state["gen_cpu_state"] = self.gen_cpu.get_state().clone()
        return state

    def set_state(self, state: dict) -> None:
        for name in self._SNAPSHOT_TENSORS:
            getattr(self, name).copy_(state[name])
        for name in self._SNAPSHOT_SCALARS:
            setattr(self, name, state[name])
        self.gen.set_state(state["gen_state"].clone())
        self.gen_cpu.set_state(state["gen_cpu_state"].clone())

    # ---- privileged latent dynamics for the extrinsics head (RMA) ----
    def _plant_nominal(self):
        """Nominal real-plant (gains, deadzones) per [strafe, fwd, yaw] axis: the level's
        plant_gain / plant_dz when set (L1_fitted), else the 2026-11-13 defaults."""
        return (
            tuple(self.level.get("plant_gain", (1.6, 2.3, 15.7))),
            tuple(self.level.get("plant_dz", (0.13, 0.05, 0.10))),
        )

    def latent_dynamics(self) -> torch.Tensor:
        """Normalized ground-truth dynamics parameters per env, [N, 12] float32,
        clamped to [-5, 5]. Column layout:
          0..2  strafe/fwd/yaw gains (real plant) | wheel gains (posterior)
          3..5  deadzones                          | wheel gain 4..6 replicated
          6..8  onset delay steps / 6              | action delay / 2
          9     mean drive tau (0.08 centre)       | mean drive tau (0.055 centre)
          10    strafe-yaw coupling / 1.0          | yaw-rate scale deviation
          11    camera period steps vs 2.5 (=20 Hz)
        """
        dev = self.device
        if self.real_plant:
            nom = torch.tensor(self._plant_nominal()[0], device=dev)
            scale = torch.tensor([0.5, 0.7, 5.0], device=dev)
            c0 = (self.rf_gain - nom) / scale
            c3 = (self.rf_dz - 0.12) / 0.08
            c6 = self.rf_onset_steps.float() / 6.0
            c9 = ((self.drive_tau.mean(dim=1) - 0.08) / 0.05).unsqueeze(-1)
            c10 = (self.rf_strafe_yaw / 1.0).unsqueeze(-1)
        else:
            c0 = (self.wheel_gain[:, :3] - 1.0) / 0.25
            c3 = torch.stack(
                [
                    (self.wheel_gain[:, 3] - 1.0) / 0.25,
                    (self.dz - 0.025) / 0.025,
                    (self.sag - 0.95) / 0.05,
                ],
                dim=1,
            )
            c6 = self.coef[:, 0, :] + self.action_delay.float().unsqueeze(-1) / 2.0
            c9 = ((self.drive_tau.mean(dim=1) - 0.30) / 0.10).unsqueeze(-1)
            c10 = ((self.yaw_scale - 1.0) / 0.2).unsqueeze(-1)
        c11 = ((self.cam_period_steps.float() - 2.5) / 2.5).unsqueeze(-1)
        out = torch.cat([c0, c3, c6, c9, c10, c11], dim=1)
        return out.clamp(-5.0, 5.0).float()

    # ---- reset ----
    def _sample_dynamics(self, idx: torch.Tensor):
        dev = self.device
        idx_d = idx.to(dev)
        m = len(idx)
        full = (
            self.level["full_posterior"]
            if self.cfg.full_posterior is None
            else self.cfg.full_posterior
        )
        s_ = self.dr_scale[idx_d]  # [m] per-env DR scale
        all_s_one = bool((s_ == 1.0).all())
        self.particle_id[idx_d] = -1
        if self.real_plant:
            # Real-car fit (2026-11-13 calibration): axes [left (+x), fwd (-y), ccw (+yaw)].
            # Nominal gains: [1.6 m/s, 2.3 m/s, 15.7 rad/s] per duty; +-30% DR.
            # Nominal deadzones: [0.13, 0.05, 0.10]; 0.5x-1.5x DR.
            g_nom, z_nom = self._plant_nominal()
            nom_gain = torch.tensor(g_nom, device=dev)
            nom_dz = torch.tensor(z_nom, device=dev)
            if self.eff_randomize:
                g_lo, g_hi = self.eff_calib_gain
                z_lo, z_hi = self.eff_calib_dz
                g_scale = g_lo + (g_hi - g_lo) * torch.rand(m, 3, device=dev, generator=self.gen)
                dz_scale = z_lo + (z_hi - z_lo) * torch.rand(m, 3, device=dev, generator=self.gen)
                # Onset lag = pure delay + first-order tau matching measured onset times:
                # strafe 0.25-0.37 s, fwd ~0.18 s, yaw ~0.07 s.
                d_left = 0.16 + 0.10 * torch.rand(m, device=dev, generator=self.gen)
                d_fwd = 0.08 + 0.06 * torch.rand(m, device=dev, generator=self.gen)
                d_ccw = 0.02 + 0.04 * torch.rand(m, device=dev, generator=self.gen)
                delay_s = torch.stack([d_left, d_fwd, d_ccw], dim=1)
                tau_left = 0.08 + 0.06 * torch.rand(m, device=dev, generator=self.gen)
                tau_fwd = 0.05 + 0.05 * torch.rand(m, device=dev, generator=self.gen)
                tau_ccw = 0.03 + 0.03 * torch.rand(m, device=dev, generator=self.gen)
                strafe_yaw = math.radians(5.0) + math.radians(10.0) * torch.rand(
                    m, device=dev, generator=self.gen
                )
            else:
                g_scale = torch.ones(m, 3, device=dev)
                dz_scale = torch.ones(m, 3, device=dev)
                delay_s = torch.tensor([[0.21, 0.11, 0.04]], device=dev).expand(m, 3).clone()
                tau_left = torch.full((m,), 0.11, device=dev)
                tau_fwd = torch.full((m,), 0.075, device=dev)
                tau_ccw = torch.full((m,), 0.045, device=dev)
                strafe_yaw = torch.full((m,), math.radians(10.0), device=dev)
            drive = torch.stack([tau_left, tau_fwd, tau_ccw], dim=1)
            rf_gain = nom_gain * g_scale
            rf_dz = nom_dz * dz_scale
            rf_onset = (delay_s / self.dt).round().long().clamp(0, _CMD_K - 1)
            brake = torch.tensor([0.08, 0.08, 0.05], device=dev).unsqueeze(0).expand(m, -1)
            cov = (
                torch.diag(torch.tensor([0.015**2, 0.015**2, 0.04**2], device=dev))
                .unsqueeze(0)
                .expand(m, -1, -1)
            )
            if not all_s_one:
                s2 = s_.unsqueeze(-1)
                dz_anc = 0.5 * nom_dz
                rf_gain = nom_gain + s2 * (rf_gain - nom_gain)
                rf_dz = dz_anc + s2 * (rf_dz - dz_anc)
                rf_onset = (s2 * rf_onset.float()).round().long()
                drive = 0.05 + s2 * (drive - 0.05)
                strafe_yaw = s_ * strafe_yaw
                cov = (s_[:, None, None] ** 2) * cov
            self.rf_gain[idx_d] = rf_gain
            self.rf_dz[idx_d] = rf_dz
            self.rf_onset_steps[idx_d] = rf_onset
            self.rf_strafe_yaw[idx_d] = strafe_yaw
            self.wheel_gain[idx_d] = 1.0
            self.yaw_scale[idx_d] = 1.0
            self.sag[idx_d] = 1.0
            self.action_delay[idx_d] = 0
            coef = torch.zeros(m, 16, 3, device=dev)
            coef[:, 0, :] = rf_gain / nom_gain
            coef[:, 1, :] = rf_dz / nom_dz
            coef[:, 2, :] = rf_onset.float() * self.dt * 5.0
            coef[:, 3, :] = drive * 5.0
            coef[:, 4, 0] = strafe_yaw
            dz = self.rf_dz[idx_d, 1]
        elif full:
            if self.eff_consistent:
                sel = torch.randint(
                    0, self.valid_particle_ids.numel(), (m,), generator=self.gen_cpu
                )
                pick_list = [int(self.valid_particle_ids[int(i)]) for i in sel]
            else:
                pick_list = [
                    int(i)
                    for i in torch.randint(0, len(self.particles), (m,), generator=self.gen_cpu)
                ]
            pick = torch.tensor(pick_list, dtype=torch.long)
            coef = torch.stack([self.particles[i]["coef"] for i in pick_list]).to(dev)
            dz = torch.tensor(
                [self.particles[i]["dz"] for i in pick_list], dtype=torch.float32, device=dev
            )
            drive = torch.stack([self.particles[i]["drive"] for i in pick_list]).to(dev)
            brake = torch.stack([self.particles[i]["brake"] for i in pick_list]).to(dev)
            cov = torch.stack([self.particles[i]["cov"] for i in pick_list]).to(dev)
            self.particle_id[idx_d] = pick.to(dev)
            # extra DR grounded in observed run-to-run variation
            if self.eff_randomize:
                wheel_gain_s = 0.85 + 0.30 * torch.rand(m, 4, device=dev, generator=self.gen)
                yaw_scale_s = 0.5 + 1.0 * torch.rand(m, device=dev, generator=self.gen)
                sag_s = 0.90 + 0.10 * torch.rand(m, device=dev, generator=self.gen)
            else:
                wheel_gain_s = torch.ones(m, 4, device=dev)
                yaw_scale_s = torch.ones(m, device=dev)
                sag_s = torch.ones(m, device=dev)
            if self.eff_action_delay > 0.0:
                has = torch.rand(m, device=dev, generator=self.gen) < self.eff_action_delay
                self.action_delay[idx_d] = torch.where(
                    has,
                    torch.randint(1, 3, (m,), device=dev, generator=self.gen),
                    torch.zeros(m, dtype=torch.long, device=dev),
                )
            else:
                self.action_delay[idx_d] = 0
            if all_s_one:
                self.wheel_gain[idx_d] = wheel_gain_s
                self.yaw_scale[idx_d] = yaw_scale_s
                self.sag[idx_d] = sag_s
            else:
                s2 = s_.unsqueeze(-1)
                s3 = s_[:, None, None]
                if self.cfg.use_mean_dynamics and not self.eff_consistent:
                    anc_coef = self.mean_coef.unsqueeze(0)
                    anc_dz = self.mean_dz
                    anc_drive = self.mean_drive.unsqueeze(0)
                    anc_brake = self.mean_brake.unsqueeze(0)
                else:
                    p0 = self.particles[0]
                    anc_coef = p0["coef"].unsqueeze(0)
                    anc_dz = p0["dz"]
                    anc_drive = p0["drive"].unsqueeze(0)
                    anc_brake = p0["brake"].unsqueeze(0)
                coef = anc_coef + s3 * (coef - anc_coef)
                dz = anc_dz + s_ * (dz - anc_dz)
                drive = anc_drive + s2 * (drive - anc_drive)
                brake = anc_brake + s2 * (brake - anc_brake)
                cov = (s3**2) * cov
                self.wheel_gain[idx_d] = 1.0 + s2 * (wheel_gain_s - 1.0)
                self.yaw_scale[idx_d] = 1.0 + s_ * (yaw_scale_s - 1.0)
                self.sag[idx_d] = 1.0 + s_ * (sag_s - 1.0)
                self.action_delay[idx_d] = (s_ * self.action_delay[idx_d].float()).round().long()
        elif self.cfg.use_mean_dynamics:
            coef = self.mean_coef.unsqueeze(0).expand(m, -1, -1)
            dz = torch.full((m,), self.mean_dz, dtype=torch.float32, device=dev)
            drive = self.mean_drive.unsqueeze(0).expand(m, -1)
            brake = self.mean_brake.unsqueeze(0).expand(m, -1)
            cov = self.mean_cov.unsqueeze(0).expand(m, -1, -1)
            self.wheel_gain[idx_d] = 1.0
            self.yaw_scale[idx_d] = 1.0
            self.sag[idx_d] = 1.0
            self.action_delay[idx_d] = 0
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
            self.action_delay[idx_d] = 0
        # per-episode wheel fault: one axis/wheel loses gain (U(0.3, 0.7)).
        # No draws at all unless the level configures a fault probability.
        self.wheel_fault[idx_d] = False
        if self.eff_wheel_fault > 0.0:
            has = torch.rand(m, device=dev, generator=self.gen) < self.eff_wheel_fault
            self.wheel_fault[idx_d] = has
            if bool(has.any()):
                mag = 0.3 + 0.4 * torch.rand(m, device=dev, generator=self.gen)
                if self.real_plant:
                    sel = torch.randint(0, 3, (m,), device=dev, generator=self.gen)
                    g = self.rf_gain[idx_d].clone()
                    for a in range(3):
                        msk = has & (sel == a)
                        if bool(msk.any()):
                            g[msk, a] *= mag[msk]
                    self.rf_gain[idx_d] = g
                    coef = coef.clone()
                    coef[:, 0, :] = g / torch.tensor(self._plant_nominal()[0], device=dev)
                else:
                    sel = torch.randint(0, 4, (m,), device=dev, generator=self.gen)
                    w = self.wheel_gain[idx_d].clone()
                    for wi in range(4):
                        msk = has & (sel == wi)
                        if bool(msk.any()):
                            w[msk, wi] *= mag[msk]
                    self.wheel_gain[idx_d] = w
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
        dev = self.device
        di = idx.to(dev)
        m = len(idx)
        area = self.area
        g = self.gen_cpu
        s_ = self.dr_scale[di]
        all_s_one = bool((s_ == 1.0).all())
        s_cpu = None if all_s_one else s_.cpu()
        span = self.cfg.spawn_dist if self.cfg.spawn_dist is not None else self.level["dist"]
        if isinstance(span, float):
            d = torch.full((m,), span)
        else:
            d = span[0] + (span[1] - span[0]) * torch.rand(m, generator=g)
        # Radians-range overrides (e.g. L_chaos); otherwise degree bearing/yaw.
        b_spec = (
            self.cfg.spawn_bearing
            if self.cfg.spawn_bearing is not None
            else self.level.get("bearing_rad")
        )
        if b_spec is not None:
            b = b_spec[0] + (b_spec[1] - b_spec[0]) * torch.rand(m, generator=g)
        else:
            bmax = math.radians(
                self.cfg.bearing_max if self.cfg.bearing_max is not None else self.level["bearing"]
            )
            b = (torch.rand(m, generator=g) * 2 - 1) * bmax
        y_spec = self.cfg.spawn_yaw if self.cfg.spawn_yaw is not None else self.level.get("yaw_rad")
        if y_spec is not None:
            yaw = y_spec[0] + (y_spec[1] - y_spec[0]) * torch.rand(m, generator=g)
        else:
            ymax = math.radians(
                self.cfg.yaw_max if self.cfg.yaw_max is not None else self.level["yaw"]
            )
            yaw = (torch.rand(m, generator=g) * 2 - 1) * ymax
        if not all_s_one:
            # dr_scale=0 anchors spawns at 0.75 m dead-ahead (curriculum entry)
            d = 0.75 + s_cpu * (d - 0.75)
            b = s_cpu * b
            yaw = s_cpu * yaw
        # Spawn strictly inside the out-of-bounds termination (|pose| > area):
        # spawning exactly ON the kill boundary outs to sub-mm plant noise /
        # blind pre-acquisition sweep drag before the first tag. No policy can
        # survive that; it is a spawn-support artifact, not task difficulty.
        lim = area - 0.20
        self.pose[di] = torch.stack(
            [(d * torch.sin(b)).clamp(-lim, lim), (d * torch.cos(b)).clamp(-lim, lim), yaw], dim=1
        ).to(dev)
        self.vel[di] = 0
        self.time[di] = 0
        self.still[di] = 0
        self.vel_still[di] = 0
        for name in (
            "done",
            "latched_success",
            "latched_coll",
            "latched_timeout",
            "latched_false_park",
        ):
            getattr(self, name)[di] = False
        self._sample_dynamics(idx)
        # sensor state
        if self.eff_latency_steps is not None:
            ls = self.eff_latency_steps
            if isinstance(ls, (tuple, list)):
                lo_s, hi_s = float(ls[0]), float(ls[1])
            else:
                lo_s = hi_s = float(ls)
            sampled = lo_s + (hi_s - lo_s) * torch.rand(m, device=dev, generator=self.gen)
            if all_s_one:
                steps = sampled.round().long()
            else:
                steps = (s_ * sampled).round().long()
            self.latency_steps[di] = steps.clamp_min(0)
            self.latency[di] = steps.float() * self.dt
        else:
            lat = (
                self.cfg.latency_override
                if self.cfg.latency_override is not None
                else self.level.get("latency", 0.0)
            )
            if isinstance(lat, tuple):
                sampled = lat[0] + (lat[1] - lat[0]) * torch.rand(m, device=dev, generator=self.gen)
                if all_s_one:
                    self.latency[di] = sampled
                else:
                    self.latency[di] = s_ * sampled
            else:
                if all_s_one:
                    self.latency[di] = float(lat)
                else:
                    self.latency[di] = s_ * float(lat)
            self.latency_steps[di] = (self.latency[di] / self.dt).round().long().clamp_min(0)
        dr = (
            self.cfg.drops_override
            if self.cfg.drops_override is not None
            else self.level.get("drops", self.level.get("drop_prob", 0.0))
        )
        if isinstance(dr, tuple):
            sampled_dr = dr[0] + (dr[1] - dr[0]) * torch.rand(m, device=dev, generator=self.gen)
        else:
            sampled_dr = torch.full((m,), float(dr), device=dev)
        if all_s_one:
            self.drop_p[di] = sampled_dr
        else:
            self.drop_p[di] = s_ * sampled_dr
        if self.eff_blackout_prob > 0.0 or self.eff_blackout_len is not None:
            # Bernoulli blackout episodes with a length in camera frames, e.g. L_chaos
            bl = self.eff_blackout_len
            if bl is not None:
                lo_b, hi_b = float(bl[0]), float(bl[1])
            else:
                lo_b, hi_b = 15.0, 50.0
            maybe = torch.rand(m, device=dev, generator=self.gen) < self.eff_blackout_prob
            len_samp = lo_b + (hi_b - lo_b) * torch.rand(m, device=dev, generator=self.gen)
            if all_s_one:
                total = maybe.float() * len_samp.round() * self.dt
            else:
                total = maybe.float() * (s_ * len_samp).round() * self.dt
            self.blackout_total[di] = total
            start = torch.rand(m, device=dev, generator=self.gen) * (self.cfg.max_time - total)
            self.blackout_start[di] = torch.where(maybe, start, torch.full_like(start, 1e9))
        else:
            total = float(self.level.get("blackout", 0.0))
            if all_s_one:
                self.blackout_total[di] = total
                if total > 0:
                    # one random blackout window per episode
                    self.blackout_start[di] = torch.rand(m, device=dev, generator=self.gen) * (
                        self.cfg.max_time - total
                    )
                else:
                    self.blackout_start[di] = 1e9
            else:
                total_i = s_ * total
                self.blackout_total[di] = total_i
                if total > 0:
                    self.blackout_start[di] = torch.rand(m, device=dev, generator=self.gen) * (
                        self.cfg.max_time - total_i
                    )
                else:
                    self.blackout_start[di] = 1e9
        self.drop_streak[di] = 0
        self.fold_left[di] = 0
        self.fold_vec[di] = 0
        self.prev_act[di] = 0
        self.hist[di] = 0
        self.hist_age[di] = 1e9
        self.tag_ema[di] = 0
        self.prev_tag[di] = 0
        self.prev_valid[di] = False
        self.dt_since_valid[di] = 0
        hz_rng = (
            self.cfg.cam_hz_range
            if self.cfg.cam_hz_range is not None
            else self.level.get("cam_hz_range")
        )
        if self.cfg.cam_hz is not None or hz_rng is None:
            hz_val = float(self.cfg.cam_hz if self.cfg.cam_hz is not None else self.cam_hz)
            self.env_cam_hz[di] = hz_val
        else:
            self.env_cam_hz[di] = hz_rng[0] + (hz_rng[1] - hz_rng[0]) * torch.rand(
                m, device=dev, generator=self.gen
            )
            if not all_s_one:
                # quantize the camera period to whole sim steps and ramp it with s:
                # s=0 -> every step, s=1 -> the sampled rate
                steps_sampled = (1.0 / self.env_cam_hz[di] / self.dt).round()
                eff = 1 + (s_ * (steps_sampled - 1)).round()
                self.env_cam_hz[di] = 1.0 / (eff.clamp_min(1.0) * self.dt)
        fast_cam = self.env_cam_hz[di] >= (1.0 / self.dt - 1e-6)
        self.cam_accum[di] = torch.where(
            fast_cam, torch.zeros(m, device=dev), (1.0 / self.env_cam_hz[di]) - self.dt
        )
        g_std = (
            self.cfg.gyro_noise_std
            if self.cfg.gyro_noise_std is not None
            else self.level.get("gyro_noise_std", 0.002)
        )
        if isinstance(g_std, tuple):
            self.gyro_noise_std[di] = g_std[0] + (g_std[1] - g_std[0]) * torch.rand(
                m, device=dev, generator=self.gen
            )
        else:
            self.gyro_noise_std[di] = float(g_std)
        if not all_s_one:
            self.gyro_noise_std[di] = 0.002 + s_ * (self.gyro_noise_std[di] - 0.002)
        self._apply_dr_env_fields()
        if self.real_plant:
            self.coef[di, 4, 1] = self.gyro_noise_std[di]
            self.coef[di, 4, 2] = self.env_cam_hz[di] / 30.0
        self.imu_bias[di] = 0.002 * torch.randn(len(di), device=dev, generator=self.gen)
        self.imu_yaw[di] = self.pose[di, 2].clone()
        # latency ring: prefill with the spawn pose so delayed reads are valid
        self.pose_ring[di] = self.pose[di].unsqueeze(1).expand(-1, _RING_K, -1)
        self.ring_count[di] = 1
        self.cmd_ring[di] = 0.0
        self.cmd_count[di] = 1
        return self._privileged()

    def _fold_error(self, cam_tick: torch.Tensor, cam_period: torch.Tensor) -> torch.Tensor:
        """Folded or oblique tag: a burst whose error keeps one sign (a curled card leans
        one way), not white noise. Starts at `fold_rate * dr_scale` events/s."""
        dev, n = self.device, self.n
        rand = lambda *shape: torch.rand(*shape, device=dev, generator=self.gen)  # noqa: E731
        start = (
            (self.fold_left <= 0)
            & cam_tick
            & (rand(n) < self.eff_fold_rate * self.dr_scale * cam_period)
        )
        lo, hi = self.eff_fold_len
        sign = torch.where(rand(n) < 0.5, -1.0, 1.0)
        new = (
            torch.stack(
                [sign * (0.01 + 0.03 * rand(n)), 0.05 * rand(n), sign * (0.15 + 0.25 * rand(n))],
                dim=1,
            )
            * self.dr_scale[:, None]
        )
        self.fold_vec = torch.where(start[:, None], new, self.fold_vec)
        self.fold_left = torch.where(start, lo + (hi - lo) * rand(n), self.fold_left)
        err = torch.where((self.fold_left > 0)[:, None], self.fold_vec, torch.zeros_like(new))
        self.fold_left = (self.fold_left - self.dt).clamp_min(0.0)
        return err

    def _push_buffer(self):
        self.pose_ring[:, self.ring_head % _RING_K] = self.pose
        self.ring_head += 1
        self.ring_count += 1

    def _delayed_pose(self):
        # per-env delayed read; clamped to available history
        sb = self.latency_steps.clamp_min(0)
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
        any_s_neq = bool((self.dr_scale != 1.0).any())
        if not self.real_plant and bool((self.action_delay > 0).any()):
            # Posterior-plant onset delay: hold the command in the shared ring
            # and read it back delayed by the per-env action_delay steps.
            self.cmd_ring[:, self.cmd_head % _CMD_K] = cmd
            self.cmd_head += 1
            self.cmd_count += 1
            avail_c = (self.cmd_count - 1).clamp_min(0)[:, None]
            use_c = torch.minimum(self.action_delay[:, None], avail_c)
            pos_c = (self.cmd_head - 1 - use_c) % _CMD_K
            env_idx = torch.arange(self.n, device=dev)[:, None]
            ax_idx = torch.arange(3, device=dev)[None, :]
            cmd = self.cmd_ring[env_idx, pos_c, ax_idx]
        prev_done = self.done.clone()
        saved = {
            "pose": self.pose.clone(),
            "vel": self.vel.clone(),
            "time": self.time.clone(),
            "still": self.still.clone(),
            "vel_still": self.vel_still.clone(),
            "hist": self.hist.clone(),
            "age": self.hist_age.clone(),
            "imu_yaw": self.imu_yaw.clone(),
            "prev_act": self.prev_act.clone(),
            "ema": self.tag_ema.clone(),
            "ptag": self.prev_tag.clone(),
            "pvalid": self.prev_valid.clone(),
            "dt_sv": self.dt_since_valid.clone(),
            "cam_acc": self.cam_accum.clone(),
        }
        # mecanum inverse mixer -> normalized wheel PWM [-1,1]. NOTE: the fitted
        # coefficients retain hardware PWM polarity (mecanum_dynamics.py):
        # translation needs negation (+cmd -> +body vel) while the turn
        # channel matches the ideal convention un-negated (verified empirically
        # per-channel on posterior-mean dynamics).
        if self.real_plant:
            dcap = float(
                self.cfg.duty_cap
                if self.cfg.duty_cap is not None
                else self.level.get("duty_cap", 0.44)
            )
            # Physical axes: cmd[:,0] -> left (+x), cmd[:,1] -> -fwd (+y), cmd[:,2] -> ccw (+yaw)
            u_raw = torch.stack([cmd[:, 0], -cmd[:, 1], cmd[:, 2]], dim=1) * dcap
            w_scale = (u_raw.abs().sum(dim=1) / max(dcap, 1e-6)).clamp_min(1.0)
            u_capped = u_raw / w_scale[:, None]
            self.cmd_ring[:, self.cmd_head % _CMD_K] = u_capped
            self.cmd_head += 1
            self.cmd_count += 1
            avail_c = (self.cmd_count - 1).clamp_min(0)[:, None]
            use_c = torch.minimum(self.rf_onset_steps, avail_c)
            pos_c = (self.cmd_head - 1 - use_c) % _CMD_K
            env_idx = torch.arange(self.n, device=dev)[:, None]
            ax_idx = torch.arange(3, device=dev)[None, :]
            u_del = self.cmd_ring[env_idx, pos_c, ax_idx]
            eff = torch.sign(u_del) * torch.clamp(u_del.abs() - self.rf_dz, min=0.0)
            vx_tgt = self.rf_gain[:, 0] * eff[:, 0]
            vy_tgt = -self.rf_gain[:, 1] * eff[:, 1]
            w_tgt = self.rf_gain[:, 2] * eff[:, 2]
            w_tgt = (
                w_tgt
                + torch.sign(u_del[:, 0])
                * (u_del[:, 0].abs() / 0.15).clamp_max(1.0)
                * self.rf_strafe_yaw
            )
            target = torch.stack([vx_tgt, vy_tgt, w_tgt], dim=1)
            braking = u_del.abs().max(dim=1).values < 1e-8
            cmd_lvl = (u_del.abs().mean(dim=1) / max(dcap, 1e-6)).clamp_max(1.0)
        else:
            vx, vy, w = cmd[:, 0], cmd[:, 1], cmd[:, 2] * HALF_BASE * 4.0
            trans = torch.stack([vx - vy, vx + vy, vx + vy, vx - vy], dim=1)
            turn = torch.stack([-w, -w, w, w], dim=1).expand_as(trans)
            wheels = -trans + turn
            wheels = (wheels * self.wheel_gain * self.sag[:, None]).clamp(-1, 1)
            feat = _wheel_features(wheels, self.dz)
            target = torch.bmm(feat[:, None, :], self.coef).squeeze(1)  # [N,3]
            target[:, 2] *= self.yaw_scale
            braking = wheels.abs().max(dim=1).values < 1e-8
            cmd_lvl = wheels.abs().mean(dim=1)
        tau = torch.where(
            braking[:, None].expand_as(self.drive_tau), self.brake_tau, self.drive_tau
        )
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
        spd_lvl = (self.vel[:, :2].norm(dim=1) / 0.6 + self.vel[:, 2].abs() / 2.5).clamp_max(1.0)
        activity = torch.maximum(cmd_lvl, spd_lvl).clamp(0.03, 1.0)
        try:
            L = torch.linalg.cholesky(
                self.cov + 1e-12 * torch.eye(3, device=dev).expand_as(self.cov)
            )
            z = torch.randn(self.n, 3, device=dev, generator=self.gen)
            self.vel += (activity * scale)[:, None] * torch.bmm(L, z[:, :, None]).squeeze(2)
        except torch.linalg.LinAlgError:
            pass
        # Body-frame impulse disturbance (m/s, rad/s), scaled by dr_scale.
        if self.eff_disturbance > 0.0:
            dscale = self.eff_disturbance * (self.dr_scale if any_s_neq else 1.0)
            self.vel += torch.randn(self.n, 3, device=dev, generator=self.gen) * (
                dscale[:, None] if torch.is_tensor(dscale) else dscale
            )
        # Floor slip: transient horizontal grip loss.
        if self.eff_floor_slip > 0.0:
            rate = self.eff_floor_slip * (self.dr_scale if any_s_neq else 1.0)
            slip = torch.rand(self.n, device=dev, generator=self.gen) < rate
            if bool(slip.any()):
                mag = 0.6 + 0.3 * torch.rand(self.n, device=dev, generator=self.gen)
                self.vel[slip, :2] = self.vel[slip, :2] * mag[slip].unsqueeze(1)
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
        noise = float(
            self.cfg.noise_override
            if self.cfg.noise_override is not None
            else self.level.get("noise", 1.0)
        )
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
        min_range = self.env_min_tag_range
        near_deck = dist < min_range
        in_fov = (bearing.abs() < FOV_HALF) & (dist < 5.0)
        if self.level["under_drone_loss"]:
            seen = in_fov & ~near_deck
        else:
            seen = in_fov | near_deck
        fast_cam = self.env_cam_hz >= (1.0 / self.dt - 1e-6)
        cam_period = 1.0 / self.env_cam_hz.clamp_min(1e-3)
        self.cam_accum += self.dt
        cam_tick = fast_cam | (self.cam_accum >= (cam_period - 1e-6))
        self.cam_accum = torch.where(
            fast_cam,
            torch.zeros_like(self.cam_accum),
            torch.where(cam_tick, self.cam_accum - cam_period, self.cam_accum),
        )
        in_blackout = (self.time > self.blackout_start) & (
            self.time < self.blackout_start + self.blackout_total
        )
        dropped = torch.rand(self.n, device=dev, generator=self.gen) < self.drop_p
        if self.eff_burst:
            # Markov-sticky vision dropouts: if the previous camera frame was
            # dropped, this frame stays dropped with probability 0.70 * s.
            stick = torch.rand(self.n, device=dev, generator=self.gen) < (0.70 * self.dr_scale)
            stay = (self.drop_streak > 0) & stick & cam_tick
            dropped = dropped | stay
            self.drop_streak = torch.where(
                cam_tick,
                torch.where(dropped, self.drop_streak + 1, torch.zeros_like(self.drop_streak)),
                self.drop_streak,
            )
        valid = seen & cam_tick & (~in_blackout) & (~dropped)
        pos_xy = (
            self.cfg.pos_noise_xy
            if self.cfg.pos_noise_xy is not None
            else self.level.get("pos_noise_xy")
        )
        yn = self.cfg.yaw_noise if self.cfg.yaw_noise is not None else self.level.get("yaw_noise")
        if pos_xy is not None:
            pos_std = torch.full((self.n,), float(pos_xy), device=dev)
        else:
            pos_std = noise * (0.008 + 0.020 * dist)
        if yn is not None:
            yaw_std = torch.full((self.n,), float(yn), device=dev)
        else:
            yaw_std = noise * (math.radians(2) + math.radians(2) * dist)
        if any_s_neq:
            pos_std = pos_std * self.dr_scale
            yaw_std = yaw_std * self.dr_scale
        tag = delayed + torch.randn(delayed.shape, device=dev, generator=self.gen) * torch.stack(
            [pos_std, pos_std, yaw_std], dim=1
        )
        out_p = float(self.level.get("outliers", self.level.get("outlier_prob", 0.0)))
        spike = torch.zeros(self.n, dtype=torch.bool, device=dev)
        if out_p > 0:
            if any_s_neq:
                spike = torch.rand(self.n, device=dev, generator=self.gen) < out_p * self.dr_scale
            else:
                spike = torch.rand(self.n, device=dev, generator=self.gen) < out_p
            tag[spike, :2] += 0.15 * torch.randn(
                int(spike.sum()), 2, device=dev, generator=self.gen
            )
        if self.eff_fold_rate > 0:
            tag = tag + self._fold_error(cam_tick, cam_period)
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
        ema_valid = torch.where(
            pair[:, None].expand_as(tag), 0.5 * self.tag_ema + 0.5 * diff, torch.zeros_like(tag)
        )
        ema_decay = self.tag_ema * math.exp(-self.dt / 0.25)
        self.tag_ema = torch.where(valid[:, None].expand_as(tag), ema_valid, ema_decay)
        self.prev_tag = torch.where(valid[:, None].expand_as(tag), tag, self.prev_tag)
        self.prev_valid = self.prev_valid | valid
        self.dt_since_valid = torch.where(
            valid, torch.zeros_like(self.dt_since_valid), self.dt_since_valid
        )
        # IMU: true yaw rate + white + bias, yaw integrated with drift
        gyro_white = (
            noise * self.gyro_noise_std * torch.randn(self.n, device=dev, generator=self.gen)
        )
        imu_w = self.vel[:, 2] + gyro_white + self.imu_bias
        self.imu_w = imu_w
        self.imu_bias *= 1 - self.dt / 0.5
        self.imu_bias += (
            math.sqrt(2 * self.dt / 0.5)
            * 0.001
            * torch.randn(self.n, device=dev, generator=self.gen)
        )
        self.imu_yaw += imu_w * self.dt
        if self.eff_reanchor:
            # Re-anchor the integrated yaw toward the tag yaw on valid,
            # non-outlier camera ticks (50% pull) so imu_yaw cannot drift
            # unboundedly before the blind final approach.
            tick = valid & ~spike
            if bool(tick.any()):
                delta = tag[:, 2] - torch.atan2(torch.sin(self.imu_yaw), torch.cos(self.imu_yaw))
                delta = torch.atan2(torch.sin(delta), torch.cos(delta))
                self.imu_yaw = torch.where(tick, self.imu_yaw + 0.5 * delta, self.imu_yaw)

        # ---- termination ----
        slow = (self.vel.abs() < torch.tensor([0.01, 0.01, math.radians(2)], device=dev)).all(dim=1)
        near = (
            (self.pose[:, 0].abs() < 0.03)
            & (self.pose[:, 1].abs() < 0.05)
            & (self.pose[:, 2].abs() < math.radians(5))
        )
        self.still = torch.where(slow & near, self.still + self.dt, torch.zeros_like(self.still))
        self.vel_still = torch.where(
            slow, self.vel_still + self.dt, torch.zeros_like(self.vel_still)
        )
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
        touch_x = (self.pose[:, 0].abs() + hw > SKID_X - GEAR_R) & (
            self.pose[:, 0].abs() - hw < SKID_X + GEAR_R
        )
        touch_y = self.pose[:, 1].abs() < GEAR_LEN / 2 + hl
        collision = touch_x & touch_y
        area = self.area
        out = (self.pose[:, 0].abs() > area) | (self.pose[:, 1].abs() > area)
        timeout = (self.time >= self.cfg.max_time) | out
        done = success | collision | timeout | false_park

        # student history frame (angles wrapped: the raw integrated IMU yaw
        # drifts unboundedly over long episodes, saturating recurrent gates)
        imu_yaw_w = torch.atan2(torch.sin(self.imu_yaw), torch.cos(self.imu_yaw))
        frame = torch.cat(
            [
                tag,
                self.hist_age.clamp_max(5.0)[:, None],
                valid.float()[:, None],
                imu_w[:, None],
                imu_yaw_w[:, None],
                self.prev_act,
                self.tag_ema,
            ],
            dim=1,
        )
        self.hist = torch.cat([self.hist[:, 1:], frame[:, None, :]], dim=1)
        rate_prev = self.prev_act.detach()
        self.prev_act = cmd.detach()

        reward = step_reward(
            prev_pose,
            self.pose,
            cmd,
            rate_prev,
            self.vel,
            success,
            near,
            collision,
            false_park,
            self.dt,
            rate_w=self.cfg.rate_w,
            lat_w=self.cfg.lat_w,
            yaw_w=self.cfg.yaw_w,
        )
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
                self.latched_success,
                self.latched_coll,
                self.latched_timeout,
                self.latched_false_park,
                self.done,
            )
        obs = {
            "pose": self.pose.clone(),
            "vel": self.vel.clone(),
            "student": self.hist.clone(),
            "tag": tag.clone(),
            "valid": valid.clone(),
        }
        info = {
            "success": success,
            "collision": collision,
            "near": near,
            "timeout": timeout,
            "false_park": false_park,
            "done": done,
            "out": out,
        }
        return obs, reward, done, info

    def _privileged(self):
        return {"pose": self.pose.clone(), "vel": self.vel.clone(), "student": self.hist.clone()}
