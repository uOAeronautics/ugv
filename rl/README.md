# UGV RL training (RTX 3090)

Pi stays torch-free. Train here, export NumPy `.npz` for `src/ugv/policy.py`.
Spec, ladder, results and commands: `../docs/RL_PARKING.md`. Follow-up plan: `PLAN.md`.

```text
sim.py            batched mecanum env + sensor emulator (torch, CUDA)
reward.py         step reward
models.py         Actor/Critic (privileged teacher), Student (deploy GRU)
train_teacher.py  PPO teacher, asymmetric critic
train_student.py  PPO directly on the Student, privileged critic
distill.py        DAgger teacher->student (measured: plateaus, 0% L0)
evaluate.py       seeded benchmark, Wilson bounds, JSON
export.py         Student -> .npz, parity vs deployed ugv.policy
runs/             checkpoints + logs (gitignored)
```
