---
name: ugv-research
description: Run the UGV RL-parking research pipeline as orchestrator or scout. Use when spawning parallel Muse Spark low scouts, gating experiments, or updating the agent board.
---

# UGV Research Skill

## Pipeline (summary of `docs/PIPELINE.md`)

Ladder L0->L5 (`rl/sim.py`), L6 Gazebo, L7 robot. Judge is
`cd rl && uv run python evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics
--json runs/<name>.json` (p0 gate plant). One GPU job at a time. Monotonic gating per
`docs/RL_PARKING.md`. Scouts never commit; orchestrator commits per scope.
Orchestrator is Gemini 4 Argon (Medium): zero direct file/log reading;
delegates ALL file reads, log checks, diagnostics, implementations,
experiments, and git status/commits to subagents, consuming only their clean
synthesized reports. All subagents (Scout / EXP / Adversary / Ops) run on
`opencode-go/muse-spark-1.3-contributor#medium` (medium reasoning).

## Spawning parallel low scouts

1. Read `rl/PLAN.md` State + `docs/AGENT_BOARD.md` running rows first.
2. Give each scout a disjoint scope (file set + level + run_id prefix).
   Example scopes: `reward-shaping/L2`, `student-capacity/L1`,
   `gazebo-lockstep/L6`, `adversary/review-<run_id>`.
3. Instruct: model `opencode-go/muse-spark-1.3-contributor#medium`
   (medium reasoning), read-only except own scope files, one seed
   per experiment, append `rl/runs/archive.jsonl`, never touch `rl/runs/`
   of another scout, stop if another session owns `rl/`.
4. Cap at 3 scouts (1 GPU + 1 CPU/sim + 1 adversary/reviewer).

## Scout instructions (paste into subagent prompt)

```text
Scope: <scope>. Files: <files>. Do not touch anything else.
Read rl/AGENTS.md, docs/PIPELINE.md, rl/PLAN.md State first.
Eval: cd rl && uv run python evaluate.py --episodes 1000 --seed 11 --no-mean-dynamics --json runs/<run_id>.json.
Append one line to rl/runs/archive.jsonl. Post board row (format below).
Return: changed files, eval numbers, next step. Do not commit.
```

## Board posting format

```text
[SCOUT] <run_id> | <scope> | running|done|blocked | <evidence: json path or finding>
[EXP] <run_id> | L<n> <succ/coll/med> | pass|fail | <runs/<name>.json + ckpt>
[ADVERSARY] <run_id> | <verdict> | <file:line citations>
[GATE] L<n> <track> | pass|fail | <ckpt + json + adversary sign-off>
```

## Gating checklist (orchestrator)

- `runs/<name>.json` exists from `evaluate.py` (1000 eps, seed 11).
- `archive.jsonl` line appended. Lower levels still pass (monotonic).
- Adversary reviewed with file:line evidence.
- `rl/PLAN.md` State updated; commit `<scope>: <change> + <eval>`.
- Tests: `cd rl && uv run pytest test_rl.py`, root `uv run pytest`.

## Safety reminder

Sim only (L0-L6). No L7 while safety config flags are false. See
`docs/SAFETY.md`. Torch in `rl/` venv only; deployed policy is NumPy.
