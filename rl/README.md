# UGV RL training (RTX 3090)

Pi stays torch-free. Train here, export NumPy `.npz` for `src/ugv/policy.py`.
Spec, ladder and commands: `../docs/RL_PARKING.md`. Status + work queue: `PLAN.md`. Agent rules: `AGENTS.md`.

```text
sim.py            batched mecanum env + sensor emulator (torch, CUDA)
reward.py         step reward (Phase-0 baseline + yaw weight 1.0)
models.py         Actor/Critic (privileged teacher), Student (deploy GRU)
train_teacher.py  PPO teacher, asymmetric critic
train_student.py  PPO directly on the Student, privileged critic
distill.py        DAgger teacher->student (optional; direct PPO works at L0)
evaluate.py       seeded benchmark, Wilson bounds, JSON
baselines.py      EKF + staged baseline benchmark (controller: ../src/ugv/staged_baseline.py)
visualize.py      top-down episode PNG/mp4 (matches evaluate.py)
export.py         Student -> .npz, parity vs deployed ugv.policy
runs/             checkpoints + logs (gitignored)
```

L0 recipes (one seed; the 1000-episode eval is the judge):

```sh
# teacher (gate teacher: 512 envs, 120 iters, defaults; it was trained with reward yaw
# weight 0.3, current reward.py uses 1.0, so a rerun is not bit-identical)
uv run python train_teacher.py --level L0 --envs 512 --iters 120 --seed <s> --out runs/teacher_L0_s<s>.pt
# student (best: 991/1000, GRU-64, from scratch)
uv run python train_student.py --level L0 --iters 300 --seed <s> --out runs/student_L0_s<s>.pt
uv run python evaluate.py --level L0 --episodes 1000 --seed 11 --policy runs/<ckpt>.pt --json runs/<ckpt>.json
```
