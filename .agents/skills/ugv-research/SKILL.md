---
name: ugv-research
description: Run the UGV explore loop (explore/ ledger, tools/explore.py) that drives the car toward a parked car under the drone. Use when proposing, predicting, starting or closing a hypothesis, holding a retro, or deciding the next research step.
compatibility: UGV repo (explore/ + tools/explore.py); needs uv and Python >= 3.13 for the root venv. Safety S0 by default; motion is out of scope.
---

# UGV explore loop

Source of truth: `explore/LOOP.md` (cycle, refusals). Runner: `tools/explore.py`. Ledger:
`explore/HYPOTHESES.jsonl`. History: `explore/HISTORY.md`. Status: `docs/STATE.md`.

## Use the tool, not the ledger

Append to the ledger only through `tools/explore.py`; it refuses illegal steps and exits 2
without writing. Start with:

```sh
uv run python tools/explore.py next     # the one legal next step
uv run python tools/explore.py list     # ledger as a table
```

Cycle: `propose` -> `predict` -> `start` -> `close --outcome confirmed|refuted|inconclusive
--evidence <run id or existing path>`; `retro` every 5 closed hypotheses.

## Rules that bind

- One hypothesis runs at a time. A claim is not evidence: close with a run id or a path.
- Write the prediction and falsifier before any run.
- Three refutations in a family force a family change unless `--escalate` is given.
- Anything that moves the car is outside the loop. Motion follows `docs/SAFETY.md` and
  `docs/STATE.md`; the loop records the decision and its evidence only.
- The RL judge is `rl/evaluate.py`, 1000 episodes, seed 11. Run it only with the GPU free
  (one job at a time, see `rl/AGENTS.md`).

For delegating reads, runs or reviews to subagents, use `.agents/skills/mastermind`.
For log-first diagnosis and release gates, use `.agents/skills/orchestrate-and-gate`.
