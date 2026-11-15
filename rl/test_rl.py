import json

import torch
from export import parity_error
from models import DYN_DIM, Student
from sim import LEVELS, EnvConfig, MecanumVecEnv, _wheel_features


def test_env_steps():
    env = MecanumVecEnv(8, EnvConfig(level="L1", device="cpu"))
    obs = env.reset()
    a = torch.zeros(8, 3)
    obs, rew, done, info = env.step(a)
    assert obs["pose"].shape == (8, 3)
    assert rew.shape == (8,)
    assert set(info) >= {"success", "collision", "timeout", "done"}


def test_env_posterior_and_sensor():
    env = MecanumVecEnv(4, EnvConfig(level="L3", device="cpu"))
    env.reset()
    assert env.coef.shape == (4, 16, 3)
    obs, _, _, _ = env.step(torch.zeros(4, 3))
    assert obs["student"].shape == (4, 50, 13)
    assert obs["valid"].dtype == torch.bool


def test_student_params_small():
    s = Student()
    assert s.n_params() < 60000
    out = s(torch.zeros(2, 50, 13))
    assert out.shape == (2, 3)


def test_export_parity_against_deployed_forward():
    s = Student()
    s.eval()
    arrays = {k: v.detach().cpu().numpy() for k, v in s.state_dict().items()}
    assert parity_error(s, arrays, batch=2) < 1e-5


def test_returned_history_matches_internal_state():
    # train_student.py stores obs["student"] per step; it must equal env.hist.
    env = MecanumVecEnv(4, EnvConfig(level="L1", device="cpu"))
    env.reset()
    for _ in range(5):
        a = torch.randn(4, 3).clamp(-1, 1)
        obs, _, _, _ = env.step(a)
        assert torch.equal(obs["student"], env.hist)


def test_partial_reset_preserves_other_envs():
    # reset(idx) must not resample physics/sensor draws of running episodes.
    env = MecanumVecEnv(4, EnvConfig(level="L3", device="cpu"))
    env.reset()
    coef1 = env.coef[1].clone()
    lat1 = env.latency[1].clone()
    drop1 = env.drop_p[1].clone()
    gain1 = env.wheel_gain[1].clone()
    env.reset(idx=torch.tensor([0]))
    assert torch.equal(env.coef[1], coef1)
    assert torch.equal(env.latency[1], lat1)
    assert torch.equal(env.drop_p[1], drop1)
    assert torch.equal(env.wheel_gain[1], gain1)


def test_same_seed_same_results():
    # one persistent generator per env: same seed + same call pattern repeats.
    from evaluate import run

    a = run(32, "L0", "brake", "cpu", seed=7)
    b = run(32, "L0", "brake", "cpu", seed=7)
    assert a == b


def test_action_rate_penalty_nonzero_on_step_change():
    # 0.4: the rate term must see the previous action, not the current one twice.
    from reward import step_reward

    n = 4
    kw = dict(
        prev_pose=torch.zeros(n, 3),
        pose=torch.zeros(n, 3),
        vel=torch.zeros(n, 3),
        success=torch.zeros(n, dtype=torch.bool),
        near=torch.zeros(n, dtype=torch.bool),
        collision=torch.zeros(n, dtype=torch.bool),
        false_park=torch.zeros(n, dtype=torch.bool),
    )
    flat = step_reward(act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3), **kw)
    step = step_reward(act=torch.ones(n, 3), prev_act=torch.zeros(n, 3), **kw)
    # rate -0.01*3 = -0.03 (near false, so no act_pen)
    assert torch.allclose(flat - step, torch.full((n,), 0.03), atol=1e-4)
    off = step_reward(act=torch.ones(n, 3), prev_act=torch.zeros(n, 3), rate_w=0.0, **kw)
    assert torch.allclose(flat - off, torch.zeros(n), atol=1e-6)


def test_terminal_bonus_only_on_success():
    # 0.5: a collision or timeout near the goal must not pay the +10 bonus.
    from reward import step_reward

    n = 3
    kw = dict(
        prev_pose=torch.zeros(n, 3),
        pose=torch.zeros(n, 3),
        act=torch.zeros(n, 3),
        prev_act=torch.zeros(n, 3),
        vel=torch.zeros(n, 3),
        near=torch.ones(n, dtype=torch.bool),
        collision=torch.tensor([True, False, False]),
        false_park=torch.zeros(n, dtype=torch.bool),
    )
    r = step_reward(success=torch.tensor([False, True, False]), **kw)
    assert r[0] < r[1] and abs(r[1] - r[0] - 15.0) < 1e-5  # 10 bonus + 5 collision avoided


def test_gae_resets_at_terminal():
    # 0.6: R=ones, V=0, terminal at t=1, gamma=lam=1 -> adv=[2,1,2,1].
    # The old code carried gae through D[t+1], leaking adv 2.0 back into t=1 (3.0).
    from models import gae_advantages

    R = torch.ones(4, 2)
    V = torch.zeros(4, 2)
    D = torch.tensor([[0, 0], [1, 1], [0, 0], [0, 0]])
    adv, ret = gae_advantages(R, V, D, torch.zeros(2), 1.0, 1.0)
    assert torch.allclose(adv, torch.tensor([[2, 2], [1, 1], [2, 2], [1, 1]], dtype=torch.float32))
    assert torch.allclose(ret, adv)


def test_yaw_priced_in_potential():
    # Failures cluster at ~5.8 deg while the limit is 5 deg; the potential must
    # price yaw materially (old weight 0.3 made 5 deg worth ~0.026).
    import math

    from reward import progress_potential

    flat = progress_potential(torch.zeros(1, 3))
    tilted = progress_potential(torch.tensor([[0.0, 0.0, math.radians(5)]]))
    assert abs((flat - tilted).item() - math.radians(5)) < 1e-6


def test_lateral_weight_prices_offset():
    # s1 failures held +52 mm mean lateral offset; lat_w prices it explicitly.
    from reward import progress_potential

    centered = progress_potential(torch.zeros(1, 3))
    off = progress_potential(torch.tensor([[0.08, 0.0, 0.0]]))
    gap_default = (centered - off).item()
    gap_zero = (
        progress_potential(torch.zeros(1, 3), lat_w=0.0)
        - progress_potential(torch.tensor([[0.08, 0.0, 0.0]]), lat_w=0.0)
    ).item()
    assert abs(gap_default - (gap_zero + 2.0 * 0.08)) < 1e-6


def test_fov_facing_and_sign():
    # 1.5: sim yaw=0 faces -y (front camera). Car at +y facing the drone sees
    # it; facing away it doesn't; a tag on the +x side comes into view on a
    # right turn (yaw -), not a left turn (old sign was flipped).
    env = MecanumVecEnv(1, EnvConfig(level="L0", device="cpu"))
    env.reset()
    env.pose[0] = torch.tensor([0.0, 0.75, 0.0])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert bool(obs["valid"][0])
    env.pose[0] = torch.tensor([0.0, 0.75, 3.14159])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert not bool(obs["valid"][0])
    env.pose[0] = torch.tensor([0.5, 0.75, 0.0])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert not bool(obs["valid"][0])  # bearing +0.588 just outside 0.585 half-FOV
    env.pose[0] = torch.tensor([0.5, 0.75, 0.1])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert not bool(obs["valid"][0])
    env.pose[0] = torch.tensor([0.5, 0.75, -0.1])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert bool(obs["valid"][0])


def test_under_drone_loss_only_from_l3():
    # 1.1: L0-L2 keep the tag everywhere inside 0.30 m, including at the goal,
    # inside the camera minimum range and slightly past the goal (y < 0);
    # L3 loses it. Regression: a dist > 0.10 FOV clause blinded every level
    # for the whole 1 s hold.
    for level, expect in (("L0", True), ("L1", True), ("L3", False)):
        for pose in ([0.0, 0.20, 0.0], [0.0, 0.0, 0.0], [0.02, -0.03, 0.05]):
            env = MecanumVecEnv(1, EnvConfig(level=level, device="cpu", seed=5))
            env.reset()
            env.drop_p.zero_()
            env.blackout_start.fill_(1e9)
            obs = None
            for _ in range(15):  # flush the latency ring past L3's 80-200 ms
                env.pose[0] = torch.tensor(pose)
                env.vel.zero_()
                obs, _, _, _ = env.step(torch.zeros(1, 3))
            assert bool(obs["valid"][0]) == expect, (level, pose)


def test_fov_behind_drone_y_negative():
    # Car behind the drone (y < 0) facing +y (yaw=pi) sees the drone; facing -y (yaw=0) doesn't.
    import math

    env = MecanumVecEnv(1, EnvConfig(level="L0", device="cpu"))
    env.reset()
    env.pose[0] = torch.tensor([0.0, -1.0, math.pi])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert bool(obs["valid"][0])
    env.pose[0] = torch.tensor([0.0, -1.0, 0.0])
    obs, _, _, _ = env.step(torch.zeros(1, 3))
    assert not bool(obs["valid"][0])


def test_cam_hz_and_tag_ema_hold_decay():
    # L1 uses 30 Hz camera: inter-frame valid=False ticks decay tag_ema instead of zeroing.
    env = MecanumVecEnv(1, EnvConfig(level="L1", device="cpu", seed=0))
    env.reset()
    env.drop_p.zero_()
    env.blackout_start.fill_(1e9)
    valids, emas = [], []
    for _ in range(6):
        obs, _, _, _ = env.step(torch.tensor([[0.2, 0.0, 0.0]]))
        valids.append(bool(obs["valid"][0]))
        emas.append(obs["student"][0, -1, 10:].clone())
    assert valids[0] and not valids[1] and valids[2]
    # Step 2 (3rd step, valid=True) updates tag_ema over dt_since_valid=0.04;
    # Step 3 (4th step, valid=False) holds/decays instead of zeroing.
    assert emas[2].abs().sum() > 0
    assert emas[3].abs().sum() > 0
    assert torch.allclose(emas[3], emas[2] * torch.exp(torch.tensor(-0.02 / 0.25)), atol=1e-5)


def test_use_mean_dynamics_vs_legacy_and_full_posterior():
    env_mean = MecanumVecEnv(2, EnvConfig(level="L0", device="cpu", use_mean_dynamics=True))
    assert torch.allclose(env_mean.coef[0], env_mean.mean_coef)
    env_p0 = MecanumVecEnv(2, EnvConfig(level="L0", device="cpu", use_mean_dynamics=False))
    assert torch.allclose(env_p0.coef[0], env_p0.particles[0]["coef"])
    env_l2 = MecanumVecEnv(8, EnvConfig(level="L2", device="cpu", seed=1))
    assert any(not torch.allclose(env_l2.coef[i], env_l2.mean_coef) for i in range(8))


def test_declare_parked_and_false_park():
    # 4th action dim > 0.5 outside near/slow triggers false_park
    env = MecanumVecEnv(2, EnvConfig(level="L3", device="cpu", seed=0))
    env.reset()
    env.pose[0] = torch.tensor([0.0, 0.8, 0.0])  # outside goal
    env.pose[1] = torch.tensor([0.0, 0.01, 0.0])  # inside goal, still
    env.vel.zero_()
    act = torch.tensor([[0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.9]])
    _, rew, done, info = env.step(act)
    assert bool(info["false_park"][0]) and bool(done[0])
    assert not bool(info["false_park"][1]) and rew[0] < -4.0

    # 3-dim declare_parked flag triggers false_park after 1.0s still outside goal
    env3 = MecanumVecEnv(1, EnvConfig(level="L3", device="cpu", seed=0, declare_parked=True))
    env3.reset()
    for _ in range(55):
        env3.pose[0] = torch.tensor([0.0, 0.8, 0.0])
        env3.vel.zero_()
        _, _, done, info = env3.step(torch.zeros(1, 3))
        if bool(done[0]):
            break
    assert bool(info["false_park"][0]) and bool(done[0])


def test_baselines_no_mean_dynamics_flag():
    # baselines.py mirrors train_teacher/train_student: default mean_coef,
    # --no-mean-dynamics selects the particle-0 plant.
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "baselines.py", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--no-mean-dynamics" in r.stdout


def test_trainers_and_archive_smoke(tmp_path):
    import subprocess
    import sys

    from experiment_tree import load_archive, pareto_frontier, record_run

    # Teacher 1-iter smoke + --init vs --resume
    t_ckpt = tmp_path / "t.pt"
    subprocess.run(
        [
            sys.executable,
            "train_teacher.py",
            "--level",
            "L0",
            "--envs",
            "8",
            "--iters",
            "1",
            "--steps",
            "8",
            "--minibatches",
            "2",
            "--hidden",
            "32",
            "--out",
            str(t_ckpt),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "train_teacher.py",
            "--level",
            "L0",
            "--envs",
            "8",
            "--iters",
            "1",
            "--steps",
            "8",
            "--minibatches",
            "2",
            "--hidden",
            "32",
            "--init",
            str(t_ckpt),
            "--out",
            str(t_ckpt),
        ],
        check=True,
    )
    # Student 1-iter stateful smoke + --init + --resume
    s_ckpt = tmp_path / "s.pt"
    subprocess.run(
        [
            sys.executable,
            "train_student.py",
            "--level",
            "L0",
            "--envs",
            "8",
            "--iters",
            "1",
            "--steps",
            "8",
            "--minibatches",
            "2",
            "--stateful",
            "--ent",
            "0.002",
            "--curric",
            "1",
            "--spawn-dist",
            "0.3,0.6",
            "--out",
            str(s_ckpt),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "train_student.py",
            "--level",
            "L0",
            "--envs",
            "8",
            "--iters",
            "1",
            "--steps",
            "8",
            "--minibatches",
            "2",
            "--stateful",
            "--resume",
            str(s_ckpt),
            "--out",
            str(s_ckpt),
        ],
        check=True,
    )
    # Experiment archive
    arch = tmp_path / "archive.jsonl"
    record_run(
        "r1",
        None,
        "student",
        "L0",
        {},
        1.0,
        {
            "success": 90,
            "episodes": 100,
            "success_lb": 0.82,
            "collision_rate": 0.0,
            "time_median": 5.0,
        },
        path=arch,
    )
    record_run(
        "r2",
        "r1",
        "student",
        "L0",
        {"lr": 1e-4},
        1.0,
        {
            "success": 99,
            "episodes": 100,
            "success_lb": 0.94,
            "collision_rate": 0.0,
            "time_median": 4.0,
        },
        path=arch,
    )
    front = pareto_frontier(load_archive(arch), level="L0")
    assert [e["run_id"] for e in front] == ["r2"]


def test_yaw_weight_plumbing_and_default():
    # yaw_w param defaults to 1.0 (current behavior unchanged); scales the yaw
    # term in the potential; --yaw-weight flag defaults to 1.0.
    import math
    import subprocess
    import sys

    from reward import progress_potential, step_reward

    tilted = torch.tensor([[0.0, 0.0, math.radians(5)]])
    flat = torch.zeros(1, 3)
    assert (
        abs((progress_potential(flat) - progress_potential(tilted)).item() - math.radians(5)) < 1e-6
    )  # default yaw_w=1.0
    gap_03 = (progress_potential(flat, yaw_w=0.3) - progress_potential(tilted, yaw_w=0.3)).item()
    assert abs(gap_03 - 0.3 * math.radians(5)) < 1e-6
    n = 2
    kw = dict(
        prev_pose=torch.zeros(n, 3),
        pose=torch.zeros(n, 3),
        act=torch.zeros(n, 3),
        prev_act=torch.zeros(n, 3),
        vel=torch.zeros(n, 3),
        success=torch.zeros(n, dtype=torch.bool),
        near=torch.zeros(n, dtype=torch.bool),
        collision=torch.zeros(n, dtype=torch.bool),
        false_park=torch.zeros(n, dtype=torch.bool),
    )
    assert torch.equal(step_reward(**kw), step_reward(yaw_w=1.0, **kw))
    from sim import EnvConfig

    assert EnvConfig().yaw_w == 1.0 and EnvConfig(yaw_w=0.3).yaw_w == 0.3
    r = subprocess.run(
        [sys.executable, "train_teacher.py", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0 and "--yaw-weight" in r.stdout


def test_real_fit_plant():
    # Defaults unchanged; L_real models physical axes, duty cap, deadzone, onset lag,
    # strafe-induced yaw, gyro noise [0.05, 0.7], cam_hz [10, 30], and tag loss < 0.4 m.
    cfg_def = EnvConfig()
    assert cfg_def.real_plant is None and cfg_def.min_tag_range is None and cfg_def.duty_cap is None

    env = MecanumVecEnv(16, EnvConfig(level="L_real", device="cpu", seed=3))
    assert env.real_plant
    assert (env.gyro_noise_std >= 0.05).all() and (env.gyro_noise_std <= 0.7).all()
    assert (env.env_cam_hz >= 10.0).all() and (env.env_cam_hz <= 30.0).all()
    assert env.gyro_noise_std.std() > 0.05 and env.env_cam_hz.std() > 1.0

    # Partial reset(idx=[0]) leaves env 1 parameters untouched
    g1 = env.rf_gain[1].clone()
    hz1 = env.env_cam_hz[1].clone()
    env.reset(torch.tensor([0]))
    assert torch.equal(env.rf_gain[1], g1) and torch.equal(env.env_cam_hz[1], hz1)

    # Physical axes & onset lag: forward (-y) responds faster and faster peak than strafe (+x);
    # strafing induces yaw coupling.
    env_f = MecanumVecEnv(4, EnvConfig(level="L_real", device="cpu", seed=7))
    env_f.reset()
    for _ in range(25):  # 0.5 s
        env_f.step(torch.tensor([[0.0, -1.0, 0.0]] * 4))
    assert (env_f.vel[:, 1] < -0.5).all() and env_f.vel[:, 0].abs().max() < 0.1

    env_s = MecanumVecEnv(4, EnvConfig(level="L_real", device="cpu", seed=7))
    env_s.reset()
    for _ in range(3):  # 0.06 s: before strafe onset lag
        env_s.step(torch.tensor([[1.0, 0.0, 0.0]] * 4))
    assert env_s.vel[:, 0].abs().max() < 0.05
    for _ in range(25):  # 0.56 s: after onset, strafe moves +x and induces +yaw
        env_s.step(torch.tensor([[1.0, 0.0, 0.0]] * 4))
    assert (env_s.vel[:, 0] > 0.2).all() and (env_s.vel[:, 2] > 0.05).all()

    # Tag loss below 0.4 m range
    env_t = MecanumVecEnv(2, EnvConfig(level="L_real", device="cpu", seed=5))
    env_t.reset()
    env_t.pose[0] = torch.tensor([0.0, 0.35, 0.0])  # < 0.40 m -> blind
    env_t.pose[1] = torch.tensor([0.0, 0.65, 0.0])  # >= 0.40 m -> visible on cam ticks
    env_t.pose_ring.copy_(env_t.pose.unsqueeze(1).expand_as(env_t.pose_ring))
    saw_1 = False
    for _ in range(6):
        obs, _, _, _ = env_t.step(torch.zeros(2, 3))
        assert not bool(obs["valid"][0])
        saw_1 = saw_1 or bool(obs["valid"][1])
    assert saw_1


def test_dr_scale_equivalence_and_bounds():
    # dr_scale=1.0 (the default) is bit-identical to an explicit dr_scale=1.0;
    # dr_scale=0.0 collapses spawn variance/DR to the curriculum anchor.
    acts = [
        torch.randn(8, 3, generator=torch.Generator().manual_seed(i)).clamp(-1, 1)
        for i in range(12)
    ]
    a = MecanumVecEnv(8, EnvConfig(level="L3", device="cpu", seed=11))
    b = MecanumVecEnv(8, EnvConfig(level="L3", device="cpu", seed=11, dr_scale=1.0))
    for act in acts:
        obs_a, rew_a, done_a, _ = a.step(act)
        obs_b, rew_b, done_b, _ = b.step(act)
        assert torch.equal(obs_a["pose"], obs_b["pose"])
        assert torch.equal(obs_a["student"], obs_b["student"])
        assert torch.equal(rew_a, rew_b) and torch.equal(done_a, done_b)
    z = MecanumVecEnv(16, EnvConfig(level="L3", device="cpu", seed=13, dr_scale=0.0))
    # zero spawn bearing/yaw: every env dead-ahead at 0.75 m
    assert torch.allclose(z.pose[:, 0], torch.zeros(16))
    assert torch.allclose(z.pose[:, 1], torch.full((16,), 0.75))
    assert torch.allclose(z.pose[:, 2], torch.zeros(16))
    # zero drops/blackouts/latency
    assert torch.equal(z.drop_p, torch.zeros(16))
    assert torch.equal(z.blackout_total, torch.zeros(16))
    assert torch.equal(z.latency, torch.zeros(16))
    assert torch.equal(z.latency_steps, torch.zeros(16, dtype=torch.long))
    # dynamics collapse to nominal (smaller spread than the full DR)
    assert torch.equal(z.wheel_gain, torch.ones(16, 4))
    assert torch.equal(z.yaw_scale, torch.ones(16))
    assert torch.equal(z.sag, torch.ones(16))
    full = MecanumVecEnv(16, EnvConfig(level="L3", device="cpu", seed=13, dr_scale=1.0))
    assert full.wheel_gain.std() > 0.02


def test_env_get_set_state_exact_branching():
    env = MecanumVecEnv(6, EnvConfig(level="L3", device="cpu", seed=7))
    env.reset()
    gen = torch.Generator().manual_seed(123)
    acts = [torch.randn(6, 3, generator=gen).clamp(-1, 1) for _ in range(15)]
    st = env.get_state()
    out1 = []
    for act in acts:
        out1.append(env.step(act))
    pose1 = env.pose.clone()
    env.set_state(st)
    out2 = []
    for act in acts:
        out2.append(env.step(act))
    for (o1, r1, d1, i1), (o2, r2, d2, i2) in zip(out1, out2, strict=True):
        for key in ("pose", "vel", "student", "tag", "valid"):
            assert torch.equal(o1[key], o2[key]), key
        assert torch.equal(r1, r2)
        assert torch.equal(d1, d2)
        for key in ("success", "collision", "timeout", "false_park", "done", "out"):
            assert torch.equal(i1[key], i2[key]), key
    assert torch.equal(pose1, env.pose)


def test_consistent_particles_and_l_chaos():
    env = MecanumVecEnv(8, EnvConfig(level="L_chaos", device="cpu", seed=3))
    ids = env.valid_particle_ids
    assert ids.numel() > 0
    # turn-authority check on the retained particles (same steady-state map)
    coef = torch.stack([env.particles[int(i)]["coef"] for i in ids.cpu()]).to(env.device)
    dz = torch.tensor(
        [env.particles[int(i)]["dz"] for i in ids.cpu()], dtype=torch.float32, device=env.device
    )
    from sim import HALF_BASE

    ub = HALF_BASE * 4.0
    for sign in (1.0, -1.0):
        u = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=env.device) * sign * ub
        feat = _wheel_features(u.unsqueeze(0).expand(coef.shape[0], 4).contiguous(), dz)
        tgt = torch.bmm(feat.unsqueeze(1), coef).squeeze(1)
        if sign > 0:
            assert (tgt[:, 2] > 0.40).all()
        else:
            assert (tgt[:, 2] < -0.40).all()
    assert env.eff_burst
    assert env.eff_disturbance > 0.0
    obs, rew, done, info = env.step(torch.zeros(8, 3))
    assert torch.isfinite(obs["pose"]).all()
    env.reset(torch.tensor([0, 2]))
    obs, rew, done, info = env.step(torch.zeros(8, 3))
    assert torch.isfinite(obs["student"]).all()
    assert LEVELS["L_chaos"]["burst_drops"]


def test_student_dyn_head_and_export_parity():
    s = Student()
    s.eval()
    dyn = s.predict_dynamics(torch.randn(4, s.gru.hidden_size))
    assert dyn.shape == (4, DYN_DIM) and DYN_DIM == 12
    env = MecanumVecEnv(4, EnvConfig(level="L3", device="cpu", seed=2))
    lat = env.latent_dynamics()
    assert lat.shape == (4, 12)
    assert torch.isfinite(lat).all()
    assert lat.abs().max() <= 5.0
    # export ships gru.* and head.* only; parity against deployed NumPy forward
    arrays = {k: v.detach().cpu().numpy() for k, v in s.state_dict().items()}
    deploy = {k: v for k, v in arrays.items() if k.startswith("gru.") or k.startswith("head.")}
    assert not any(k.startswith("dyn_head") for k in deploy)
    assert not any(k.startswith("aux_head") for k in deploy)
    assert parity_error(s, deploy, batch=2) < 1e-5


def test_evaluate_save_traces(tmp_path):
    from evaluate import run

    out = tmp_path / "traces.json"
    run(4, "L0", "random", "cpu", seed=3, save_traces=str(out), max_trace_episodes=2)
    data = json.loads(out.read_text())
    assert data["schema"] == "ugv_replay_v1"
    assert data["source"] == "sim_eval"
    assert data["level"] == "L0"
    assert data["episodes"], "no episodes recorded"
    assert len(data["episodes"]) <= 2
    for ep in data["episodes"]:
        assert ep["outcome"] in ("success", "collision", "timeout", "false_park")
        assert ep["steps"] == len(ep["frames"])
        for k, frame in enumerate(ep["frames"]):
            assert frame["t"] == round(k * data["dt"], 3)
        for frame in ep["frames"]:
            for key in (
                "t",
                "true_pose",
                "true_pose_ccw",
                "delayed_tag",
                "aux_pose",
                "valid",
                "blackout",
                "steps_since_valid",
                "imu_w",
                "imu_yaw",
                "action",
                "shield_clear",
            ):
                assert key in frame, key


def test_evaluate_records_provenance():
    from evaluate import run

    base = run(4, "L2", "brake", "cpu", seed=11, dr_scale=0.45, consistent_particles=True)
    assert base["seed"] == 11
    assert base["device"] == "cpu"
    assert base["dr_scale"] == 0.45
    assert base["consistent_particles"] is True
    assert base["level"] == "L2"
    assert base["level_label"] == "L2_cons_dr0.45"

    plain = run(4, "L1", "brake", "cpu", seed=0)
    assert plain["level_label"] == "L1"
    assert plain["dr_scale"] == 1.0
    assert plain["consistent_particles"] is False


def test_fov_search_sweep_mirrors_deploy_reflex():
    import evaluate

    from ugv.policy import search_yaw_target

    blind_s = torch.linspace(0.0, 20.0, 401)
    vec = evaluate.search_yaw_target_vec(blind_s)
    ref = torch.tensor([search_yaw_target(float(b)) for b in blind_s])
    assert torch.allclose(vec, ref, atol=1e-6)


def test_blind_mouth_frac_zero_bit_identical(tmp_path):
    # --blind-mouth-frac 0.0 must not draw RNG or touch tensors on the reset
    # path: two same-seed 1-iteration CPU runs give identical first-iter losses.
    # train_student does not seed the global torch RNG (model init, minibatch
    # permutations), so the harness pins it externally; the trainer code path
    # itself runs unmodified in both arms.
    import os
    import subprocess
    import sys

    wrapper = "import torch; torch.manual_seed(3); from train_student import main; main()"
    base = [
        sys.executable,
        "-c",
        wrapper,
        "--level",
        "L0",
        "--envs",
        "16",
        "--iters",
        "1",
        "--steps",
        "8",
        "--epochs",
        "1",
        "--minibatches",
        "2",
        "--hidden",
        "32",
        "--seed",
        "3",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    runs = []
    for i, extra in enumerate(([], ["--blind-mouth-frac", "0.0"])):
        out = tmp_path / f"bm{i}.pt"
        subprocess.run(
            base + extra + ["--out", str(out)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        first = json.loads((tmp_path / f"bm{i}.metrics.jsonl").read_text().splitlines()[0])
        runs.append({k: v for k, v in first.items() if k not in ("wall_s", "sps")})
    assert runs[0]["pol"] == runs[1]["pol"]
    assert runs[0] == runs[1]


def test_blind_mouth_frac_placement():
    from train_student import blind_mouth_replace

    # spawn_dist far outside the regime box so natural spawns cannot contaminate
    # the in-regime count.
    env = MecanumVecEnv(64, EnvConfig(level="L0", device="cpu", seed=5, spawn_dist=(1.2, 1.4)))
    env.reset()
    before = env.get_state()
    gen = torch.Generator().manual_seed(5 + 777)
    sel = blind_mouth_replace(env, torch.arange(64), 0.5, gen)
    assert len(sel) == 32
    p = env.pose[sel]
    assert (p[:, 1] >= 0.25).all() and (p[:, 1] <= 0.60).all()
    assert (p[:, 0].abs() <= 0.12).all() and (p[:, 2].abs() <= 0.15).all()
    assert (env.vel[sel] == 0).all()
    assert torch.equal(env.imu_yaw[sel], p[:, 2])
    assert torch.equal(env.pose_ring[sel], p.unsqueeze(1).expand_as(env.pose_ring[sel]))
    in_regime = (
        (env.pose[:, 1] >= 0.25)
        & (env.pose[:, 1] <= 0.60)
        & (env.pose[:, 0].abs() <= 0.12)
        & (env.pose[:, 2].abs() <= 0.15)
    )
    assert abs(int(in_regime.sum()) - 32) <= 1
    # untouched envs and both RNG streams stay bit-identical
    after = env.get_state()
    keep = torch.tensor([i for i in range(64) if i not in set(sel.tolist())])
    for key in ("pose", "vel", "coef", "hist", "drop_p", "latency", "wheel_gain"):
        assert torch.equal(before[key][keep], after[key][keep]), key
    assert torch.equal(before["gen_state"], after["gen_state"])
    assert torch.equal(before["gen_cpu_state"], after["gen_cpu_state"])
    # at least 1 replaced when F>0 and batch>=8, even if round(F*m)==0
    assert len(blind_mouth_replace(env, torch.arange(64), 0.001, gen)) == 1


def test_action_sensitivity_smoke(tmp_path):
    from action_sensitivity import audit
    from models import Student

    s = Student(act_dim=3, h=16, head_w=16)
    ck = tmp_path / "tiny.pt"
    torch.save(
        {"student": s.state_dict(), "act_dim": 3, "hidden": 16, "head_w": 16, "run_id": "smoke"},
        ck,
    )
    res = audit(str(ck), level="L0", episodes=4, seed=11, horizon=0.2, delta=0.3)
    assert set(res["regimes"]) == {"mouth", "blind"}
    for reg in res["regimes"].values():
        m = torch.tensor(reg["matrix"])
        assert m.shape == (3, 3) and torch.isfinite(m).all()
        for b in reg["branches"].values():
            assert all(torch.isfinite(torch.tensor(v)).all() for v in b.values())
    assert all(b["valid_frac"] == 0.0 for b in res["regimes"]["blind"]["branches"].values())
    mouth = res["regimes"]["mouth"]["branches"]
    assert mouth["+dx"]["valid_frac"] > 0.0
    assert mouth["-dy"]["progress"] > 0.0  # -dy approaches the drone: y decreases
    out = tmp_path / "sens.json"
    out.write_text(json.dumps(res))
    back = json.loads(out.read_text())
    assert set(back["regimes"]) == {"mouth", "blind"}
    assert back["config"]["steps"] == 10
