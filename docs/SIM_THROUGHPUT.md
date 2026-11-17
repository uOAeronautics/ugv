# SIM Throughput (measured 2026-11-13, scout)

Scope: end-to-end steps/sec per pipeline stage, bottleneck per stage, and
concrete batch/parallel settings to saturate 16 CPUs + RTX 3090.
GPU probe kept tiny (<=256 envs, <1 s); larger sizes measured on CPU.
`evaluate.py` batch is capped at 512 envs (`evaluate.py:69`).

## (a) rl/sim.py vec env — 100 warm steps, zero-action

| envs | device | env-steps/s | wall/100 steps | VRAM (env only) |
|---|---|---|---|---|
| 256 | cuda | 37,567 | 0.68 s | 0.01 GB |
| 256 | cpu | ~52,000 (cold) | 0.49 s | — |
| 512 | cpu | ~290,000–300,000 (warm; L0 289k, L1 300k) | ~0.17 s | — |
| 1024 | cpu | 263,353 | 0.39 s | — |
| 2048 | cpu | 302,591 | 0.68 s | — |

Notes:

- GPU is **slower** than CPU at <=256 envs: per-env work is tiny matmuls and
  the step is kernel-launch bound. GPU only wins if the policy/PPO update
  shares the device and batches are much larger.
- Cold-start effect is large: first ~10 steps run ~10x slower
  (torch op autotune/allocation; 15.3 ms/step cold vs 1.8 ms/step warm at
  n=512). Always warm up before timing or training.
- Env tensors alone use ~0.01 GB at 256 envs. The documented ~10 GB at
  1024 envs (`docs/PIPELINE.md` 1-GPU rule) is PPO buffers + grads, not
  the env. Env memory scales ~linearly and is negligible (<0.1 GB at 2048).
- **Bottleneck (a): kernel-launch overhead on GPU / cold-start on CPU.**
  Neither is the real constraint (see b).

## (b) evaluate.py — wall time + breakdown

Measured: 200 ep brake/L0/cpu = **5.4 s** (all timeouts, 2250 steps/ep =
450k env-steps, ~83k ep-steps/s incl. loop overhead).
Measured: 20 ep student/L0/cpu = **5.9 s** -> extrapolates to
**~5 min per 1000 ep** for short-episode policies (~18 s median here;
gate-config 6 s median eps extrapolate to ~2–3 min).

Per-step split at batch 512 on CPU (L0, warm):

| component | ms/step (batch 512) | share |
|---|---|---|
| `env.step` (sim+sensors) | 1.8 | ~2% |
| torch Student GRU-64 full-history forward | 91.8 | ~98% |
| Python loop overhead (`run()` slicing, bookkeeping) | small | <5% |

**Bottleneck (b): policy inference, not sim.** `evaluate.py:62-64`
re-forwards the full 50-frame history every step for non-stateful
checkpoints (92 ms vs 1.8 ms sim). The stateful path (`forward_seq`,
1 frame) is ~35x cheaper — numpy equivalents: full-history 1.47 ms/step
vs single-frame 0.04 ms/step (see d).

## (c) Gazebo lockstep roundtrip — 100 D-steps, zero-action

| metric | measured |
|---|---|
| `gz.step` throughput | **50.0 D-steps/s = 1.0x realtime** (20 ms sim per D-step, 20 ms wall) |
| IPC + 1 physics step (n=1) | 1.2 ms/step |
| physics share of a D-step | ~94% (20 × 1 ms phys steps + 1.2 ms IPC) |
| `reset_pose` per episode | **0.80 s** (25 brake + 15 settle D-steps) |
| server startup | ~2 s |

**Bottleneck (c): physics + per-episode reset, not IPC.**
IPC overhead (1.2 ms) is 6% of a D-step. At L3 (median 8.16 s sim,
superseded agent board in git history, smoke test) one episode costs ~8.2 s sim + 0.8 s reset
≈ 9 s wall → **~2.5 h per 1000 ep per world**, fixed at 1x realtime.
Only parallelism helps.

## (d) StudentPolicy numpy ms/step (GRU-64, random weights, same shapes)

| mode | ms/step |
|---|---|
| `forward` full 50-frame history | **1.47** |
| `_step_gru` single frame + `_head` (stateful) | **0.04** (~35x cheaper) |

Deploy loop at 50 Hz has a 20 ms budget: full-history forward fits
(1.5 ms) but leaves little headroom on loaded Pi hardware alongside the
shield sweep; stateful single-frame (0.04 ms) is the safe choice and
matches the `stateful` checkpoint path in `evaluate.py:62`.

## Recommendations (16 CPUs + RTX 3090)

1. **Rollouts: stay on CPU, n = 1024–2048.** Throughput plateaus at
   ~300k env-steps/s from 512 up; 1024 is the sweet spot (headroom for
   the PPO dataloader threads). GPU env adds nothing at these sizes —
   reserve the 3090 for the PPO update + teacher/student forward.
2. **Eval: batch 512 (current cap is fine), run 4–8 eval workers in
   parallel processes.** Sim is 2% of eval cost, so env-count tuning is
   pointless; parallelize episodes across CPUs instead. 8 workers × ~5 min
   → 1000-ep student eval in under a minute wall.
3. **Eval inference: use stateful checkpoints (`forward_seq`).**
   Removes the 98% inference bottleneck (~50x per-step cut on CPU).
   Non-stateful ckpts should be converted, not re-run at full history.
4. **Gazebo L6: N parallel worlds (one `gz sim -s` + bridge per 2 CPUs).**
   1 world = 1x realtime = ~2.5 h/1000 L3 eps. 8 worlds on 16 CPUs →
   ~20 min/1000 eps. Worlds are independent processes; partition with
   `GZ_PARTITION` as `lockstep_env.py:179` already does.
5. **Amortize `reset_pose` (0.8 s/ep):** batch K episodes per world spawn
   where the protocol allows, or overlap reset of world i+1 with rollout
   of world i. At L0 (short eps) reset is ~10% of wall; at L3+ it is
   negligible next to sim time.
6. **GPU budget:** keep the 1-GPU-job rule. Env on CPU + PPO on GPU
   avoids the ~10 GB rollout-buffer duplication that OOMs at 3 parallel
   jobs; two jobs (train + shadow eval) fit if rollouts stay on CPU.

## Repro

- (a): `cd rl && uv run python -c` snippet in scout session (100 warm
  steps, `MecanumVecEnv(n, EnvConfig(level=...))`, zero action).
- (b): `timeout 100 uv run python -c "from evaluate import run; ..."`
  200-ep brake + 20-ep student runs above; per-component means from
  50-step `env.step` timing and 20-rep `Student(...)` forward timing.
- (c): `simulation/gazebo` headless `freenove_world.sdf` + `Gazebo`
  (`lockstep_env.py`), 100× `step([0.1,0,0,0])`, 50× `step(..., n=1)`,
  1× `reset_pose`.
- (d): root `python3` + `src/ugv/policy.py StudentPolicy` with random
  GRU-64 weights, 200-rep `forward` vs `_step_gru+_head` timing.

## Engine decision (2026-11-16, scout; unverified items marked ?)

Keep `rl/sim.py` (custom batched torch, planar 3-DoF + fitted slip/deadzone plant) for training.
Skild's Steel (Nov 9 2026, `skild-ai/steel`, Apache-2.0, a Newton solver) is built for arm and hand
manipulation; neither its repo nor its blog mentions wheeled vehicles or friction models, it needs a
Newton dev fork, and its 3.6 M steps/s headline is on a Blackwell workstation GPU (3090 fit ?). The sim
already steps ~3e5 env-steps/s on CPU at 512 envs (about 6000x real time for a 50 Hz plant) versus 1x
for Gazebo lockstep, so physics is not the bottleneck; policy inference and PPO are. Newton's
`robot_omniwheel` example is a possible offline cross-check of a few trajectories, not a trainer.
H-JEPA (`kevinghst/H-JEPA`, MIT, research code for visual planning) is not adoptable; the
action-conditioned forward-model loss in `train_student.py --fwd-weight` takes its useful idea.
