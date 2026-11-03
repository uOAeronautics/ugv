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
              first_slow=torch.zeros(n, dtype=torch.bool),
              timeout=torch.zeros(n, dtype=torch.bool),
              collision=torch.zeros(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    flat = step_reward(act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3), **kw)
    step = step_reward(act=torch.ones(n, 3), prev_act=torch.zeros(n, 3), **kw)
    # rate -0.01*3 = -0.03 plus act_pen -0.02*prox(=1 at origin)*1 = -0.02
    assert torch.allclose(flat - step, torch.full((n,), 0.05), atol=1e-4)


def test_terminal_bonus_only_on_success():
    # 0.5: a collision or timeout near the goal must not pay the +10 bonus.
    from reward import step_reward
    n = 3
    kw = dict(prev_pose=torch.zeros(n, 3), pose=torch.zeros(n, 3),
              act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3),
              vel=torch.zeros(n, 3),
              first_slow=torch.zeros(n, dtype=torch.bool),
              timeout=torch.zeros(n, dtype=torch.bool),
              collision=torch.tensor([True, False, False]),
              false_park=torch.zeros(n, dtype=torch.bool))
    r = step_reward(success=torch.tensor([False, True, False]), **kw)
    assert r[0] < r[1] and abs(r[1] - r[0] - 15.0) < 1e-5  # 10 bonus + 5 collision avoided


def test_first_slow_bonus_paid_once():
    # The +2 seeding bonus fires on the first slow&near step, not after.
    from reward import step_reward
    n = 2
    kw = dict(prev_pose=torch.zeros(n, 3), pose=torch.zeros(n, 3),
              act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3),
              vel=torch.zeros(n, 3), success=torch.zeros(n, dtype=torch.bool),
              collision=torch.zeros(n, dtype=torch.bool),
              timeout=torch.zeros(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    r = step_reward(first_slow=torch.tensor([True, False]), **kw)
    assert abs(r[0] - r[1] - 2.0) < 1e-5


def test_terminal_proximity_pays_survival_not_crash():
    # Surviving 45 s at the goal pays closeness; crashing there pays nothing.
    from reward import step_reward
    n = 2
    kw = dict(prev_pose=torch.zeros(n, 3), pose=torch.zeros(n, 3),
              act=torch.zeros(n, 3), prev_act=torch.zeros(n, 3),
              vel=torch.zeros(n, 3), success=torch.zeros(n, dtype=torch.bool),
              first_slow=torch.zeros(n, dtype=torch.bool),
              timeout=torch.ones(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    r = step_reward(collision=torch.tensor([False, True]), **kw)
    assert r[0] > 4.9 and r[1] < 0  # +5 closeness vs -5 crash, same pose


def test_mouth_creep_pays_slow_not_fast():
    # Creeping (0.05 m/s) through the mouth band pays; rushing (0.5) doesn't.
    from reward import step_reward
    n = 2
    pose = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.2, 0.0]])
    vel = torch.tensor([[0.05, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kw = dict(prev_pose=pose, pose=pose, act=torch.zeros(n, 3),
              prev_act=torch.zeros(n, 3), success=torch.zeros(n, dtype=torch.bool),
              first_slow=torch.zeros(n, dtype=torch.bool),
              timeout=torch.zeros(n, dtype=torch.bool),
              collision=torch.zeros(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    r = step_reward(vel=vel, **kw)
    assert r[0] - r[1] > 0.03


def test_mouth_rush_penalized():
    # 0.5 m/s in the mouth band costs >= 0.1/step vs creeping; outside band no effect.
    from reward import step_reward
    n = 3
    pose = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 2.0, 0.0]])
    vel = torch.tensor([[0.05, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kw = dict(prev_pose=pose, pose=pose, act=torch.zeros(n, 3),
              prev_act=torch.zeros(n, 3), success=torch.zeros(n, dtype=torch.bool),
              first_slow=torch.zeros(n, dtype=torch.bool),
              timeout=torch.zeros(n, dtype=torch.bool),
              collision=torch.zeros(n, dtype=torch.bool),
              false_park=torch.zeros(n, dtype=torch.bool))
    r = step_reward(vel=vel, **kw)
    assert r[0] - r[1] > 0.1 and r[2] - r[1] > 0.1  # rush penalized only in band


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
