import torch
from sim import MecanumVecEnv, EnvConfig
from models import Student
from export import parity_error


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
    kw = dict(prev_pose=torch.zeros(n, 3), pose=torch.zeros(n, 3),
              vel=torch.zeros(n, 3), success=torch.zeros(n, dtype=torch.bool),
              near=torch.zeros(n, dtype=torch.bool),
              collision=torch.zeros(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    flat = step_reward(act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3), **kw)
    step = step_reward(act=torch.ones(n, 3), prev_act=torch.zeros(n, 3), **kw)
    # rate -0.01*3 = -0.03 (near false, so no act_pen)
    assert torch.allclose(flat - step, torch.full((n,), 0.03), atol=1e-4)
    off = step_reward(act=torch.ones(n, 3), prev_act=torch.zeros(n, 3),
                      rate_w=0.0, **kw)
    assert torch.allclose(flat - off, torch.zeros(n), atol=1e-6)


def test_terminal_bonus_only_on_success():
    # 0.5: a collision or timeout near the goal must not pay the +10 bonus.
    from reward import step_reward
    n = 3
    kw = dict(prev_pose=torch.zeros(n, 3), pose=torch.zeros(n, 3),
              act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3),
              vel=torch.zeros(n, 3), near=torch.ones(n, dtype=torch.bool),
              collision=torch.tensor([True, False, False]),
              false_park=torch.zeros(n, dtype=torch.bool))
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
    gap_zero = (progress_potential(torch.zeros(1, 3), lat_w=0.0)
                - progress_potential(torch.tensor([[0.08, 0.0, 0.0]]), lat_w=0.0)).item()
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
    import subprocess, sys
    r = subprocess.run([sys.executable, "baselines.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--no-mean-dynamics" in r.stdout


def test_trainers_and_archive_smoke(tmp_path):
    import subprocess, sys
    from experiment_tree import record_run, load_archive, pareto_frontier
    # Teacher 1-iter smoke + --init vs --resume
    t_ckpt = tmp_path / "t.pt"
    subprocess.run([sys.executable, "train_teacher.py", "--level", "L0", "--envs", "8",
                    "--iters", "1", "--steps", "8", "--minibatches", "2", "--hidden", "32",
                    "--out", str(t_ckpt)], check=True)
    subprocess.run([sys.executable, "train_teacher.py", "--level", "L0", "--envs", "8",
                    "--iters", "1", "--steps", "8", "--minibatches", "2", "--hidden", "32",
                    "--init", str(t_ckpt), "--out", str(t_ckpt)], check=True)
    # Student 1-iter stateful smoke + --init + --resume
    s_ckpt = tmp_path / "s.pt"
    subprocess.run([sys.executable, "train_student.py", "--level", "L0", "--envs", "8",
                    "--iters", "1", "--steps", "8", "--minibatches", "2", "--stateful",
                    "--ent", "0.002", "--curric", "1", "--spawn-dist", "0.3,0.6",
                    "--out", str(s_ckpt)], check=True)
    subprocess.run([sys.executable, "train_student.py", "--level", "L0", "--envs", "8",
                    "--iters", "1", "--steps", "8", "--minibatches", "2", "--stateful",
                    "--resume", str(s_ckpt), "--out", str(s_ckpt)], check=True)
    # Distill 1-iter smoke
    d_ckpt = tmp_path / "d.pt"
    subprocess.run([sys.executable, "distill.py", "--teacher", str(t_ckpt), "--level", "L0",
                    "--iters", "1", "--episodes", "4", "--envs", "4", "--epochs", "1",
                    "--hidden", "32", "--head-w", "32", "--out", str(d_ckpt)], check=True)
    ck = torch.load(d_ckpt, map_location="cpu")
    assert ck["hidden"] == 32 and ck["head_w"] == 32
    # Experiment archive
    arch = tmp_path / "archive.jsonl"
    record_run("r1", None, "student", "L0", {}, 1.0,
               {"success": 90, "episodes": 100, "success_lb": 0.82, "collision_rate": 0.0, "time_median": 5.0},
               path=arch)
    record_run("r2", "r1", "student", "L0", {"lr": 1e-4}, 1.0,
               {"success": 99, "episodes": 100, "success_lb": 0.94, "collision_rate": 0.0, "time_median": 4.0},
               path=arch)
    front = pareto_frontier(load_archive(arch), level="L0")
    assert [e["run_id"] for e in front] == ["r2"]
