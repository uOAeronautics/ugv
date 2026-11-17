# UGV Research Pipeline

How the RL parking ladder (L0 -> L5 in `rl/sim.py`, L6 Gazebo, L7 real robot) is worked. The
hypothesis cycle itself is in [`explore/LOOP.md`](../explore/LOOP.md); this page covers roles,
GPU use, gating and the two ledgers. Do not restate the cycle here.

## Roles

- **Orchestrator** (the main session): owns `rl/PLAN.md` State, gates and commit order. Splits
  work into disjoint scopes, delegates, judges the reports it gets back.
- **Subagents** (scout, experiment, adversary): briefed and gated per
  [`.agents/skills/mastermind/SKILL.md`](../.agents/skills/mastermind/SKILL.md). Model tiers are
  set there, not here.
- **Scout**: read-only, or edits one scope. Never merges, never pushes, never commits.
- **Adversary**: reviews a claim or diff before it gates, citing file:line evidence. A gate pass
  without adversary review is not a pass.

## One GPU job at a time

One GPU job at a time on the 3090 (student PPO at 1024 envs uses ~10 GB; three in parallel
run out of memory). Check `nvidia-smi` and `ps` first. Do not edit `rl/sim.py`, `reward.py`,
`models.py` or a trainer while a job that imports it is starting.

## The judge

`cd rl && uv run python evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics --json runs/<name>.json`
(p0 gate plant). Training `succ` logs are not success. One seed per experiment; a failure gets one
more seed before the idea is called dead. Gates and thresholds live in `docs/RL_PARKING.md`.

## Monotonic gating

Order: teacher L0 -> student L0 -> L1 -> ... An L(n+1) claim needs L(n) still passing on the same
tree. A regression means revert. Gate artifacts are the checkpoint, `runs/<name>.json` (gitignored)
and a gate row in the explore ledger's evidence (see below). Never overwrite a gate-passing
checkpoint. Never leave results only in `/tmp`.

## Two ledgers, two jobs

These are different files with different writers. Do not mix their rows.

| | `rl/runs/archive.jsonl` | `explore/HYPOTHESES.jsonl` |
|---|---|---|
| Unit | one 1000-episode eval run | one hypothesis, over its life |
| Written by | the eval, one line per run | `tools/explore.py` only, one event per line |
| Holds | `run_id, parent_id, track, level, config_delta, wall_s, eval_metrics` | events `propose`, `predict`, `start`, `close`, `retro`; a hypothesis is the events sharing one `id` |
| Key fields | run id, metrics | `family`, `statement`, `prediction`, `falsifier`, outcome, and `evidence` (a run id or an existing path) |
| Rule | every eval appends one line; tracked by Git | a claim is not evidence; close only with a run id or path |

The run ledger records what was measured. The hypothesis ledger records what was predicted,
what was tried, and what it showed. A closed hypothesis points at run ids in `archive.jsonl`; it
does not copy their numbers. Cycle, refusals and retro rule: [`explore/LOOP.md`](../explore/LOOP.md).

`rl/PLAN.md` State is replaced, not appended, when a measured result changes.

## Commits

Scouts do not commit. The orchestrator commits per completed scope as
`<scope>: <what changed> + <eval result>`. Push only green gates or work marked in progress.
Before push: `cd rl && uv run pytest -q test_rl.py` and `uv run pytest -q` from the root.

## Safety

L0-L6 are simulation only. No L7 actuation while the `docs/SAFETY.md` flags read false. Motion
commands carry a finite duration and an explicit stop. Hardware runs follow `docs/STATE.md`.
