# rl/ agent rules

Applies to everything under `rl/`. The shared rules in `/home/fer/Documents/AGENTS.md` also apply.

## Read first

```text
spec + gates   := ../docs/RL_PARKING.md   (do not change gates or parked thresholds without approval;
                 D1/D2 compaction approved 2026-11-14: p0 plant, L0 canary, L1 >=90% floor, L2_cons subset)
work queue     := PLAN.md                 (State = current truth; update it when a result changes)
code map       := README.md
```

## Goal of each step

Progress up the ladder (L0 -> L5 in `sim.py`, then Gazebo L6, then the real robot L7).
A step is done when its gate passes on the 1000-episode eval. Prefer the fastest path to the
next gate over extra confirmation.

## Experiments

- Judge = `evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics --json runs/<name>.json`
  (p0 gate plant). Training logs
  (`succ` per iteration) are not a success measure: the stochastic policy rarely completes
  the 1 s hold even when the deterministic policy parks.
- One seed per experiment. Pass the gate -> keep the checkpoint and move on. Fail -> one more
  seed before concluding an idea does not work. Seed variance on identical code spans
  0-256/256.
- Change one thing per experiment. Before changing the reward, look at the failure episodes:
  `visualize.py --policy runs/<ckpt>.pt --level <L> --episodes 6 --seed 11`.
- Before blaming the policy, check what the policy observes (e.g. fraction of steps with a
  valid tag near the goal). Two "policy can't learn X" conclusions here were sim bugs.
- Name runs descriptively and keep them: checkpoints, logs and JSON go to `runs/`
  (gitignored). Never leave results only in `/tmp`. Never overwrite a gate-passing checkpoint.

## Shared resources

- One agent drives `rl/` at a time. If another session is running jobs or editing files
  here, stop and ask the user who owns it.
- One GPU job at a time on the 3090 (student PPO at 1024 envs uses ~10 GB; three in parallel
  ran out of memory).
- Do not edit `sim.py`, `reward.py`, `models.py` or a trainer while a job that imports it is
  starting; a job that loads a half-edited file crashes or silently trains the wrong thing.

## Code rules

- Torch only in `rl/` (its own venv: `cd rl && uv run ...`). The root package stays
  torch-free; the deployed student is NumPy (`src/ugv/policy.py`).
- A change to `sim.py` or `reward.py` invalidates earlier numbers; re-measure the baseline
  being compared against.
- New behavior toggles go in CLI flags or `LEVELS`, not in-place edits between runs.
- Tests: `cd rl && uv run pytest test_rl.py` and from the repo root `uv run pytest`.
- `export.py` must keep parity with the deployed `ugv.policy` forward (< 1e-5).

## Safety

No real-robot motion (L7) without explicit operator authorisation, measured gear clearance in
`config/parking.json`, and the `docs/SAFETY.md` flags. Agents prepare scripts and checklists;
they do not start motion.
