# UGV Research Pipeline

High-Orchestrator / Low-Scout + Adversary workflow for the RL parking ladder
(L0 -> L5 in `rl/sim.py`, L6 Gazebo, L7 real robot).

## Roles

- **High-Orchestrator** (this session): owns `rl/PLAN.md` State, gates, merge order.
  Splits work into disjoint scopes, spawns scouts, judges results, commits.
- **Low-Scout** (subagent, Muse Spark): read-only explore or single-scope edit.
  Never touches another scout's scope. Never merges, never pushes.
- **Adversary**: reviews a scout claim or diff before it gates. Must cite file:line
  evidence. A gate pass without adversary review is not a pass.

## 1-GPU-job rule

One GPU job at a time on the 3090 (student PPO at 1024 envs uses ~10 GB;
three in parallel OOM). Before training, check `nvidia-smi` and the board:
if a `[SCOUT]` row is `running`, wait. Never edit `sim.py`, `reward.py`,
`models.py`, or a trainer while a job importing it is starting.

## evaluate.py is the sole judge

- Judge: `cd rl && uv run python evaluate.py --episodes 1000 --seed 11
  --json runs/<name>.json`. Training `succ` logs are not success.
- One seed per experiment. Pass -> keep checkpoint, move on. Fail -> one more
  seed before declaring the idea dead (seed variance spans 0-256/256).
- Change one thing per experiment. Before changing reward, watch failures:
  `visualize.py --policy runs/<ckpt>.pt --level <L> --episodes 6 --seed 11`.
- Check observations before blaming the policy (two past "can't learn X"
  conclusions were sim bugs). A `sim.py`/`reward.py` change invalidates
  earlier numbers: re-measure the compared baseline.

## Monotonic gating

- Gates live in `docs/RL_PARKING.md` (do not change thresholds to pass).
- Order: teacher L0 -> student L0 -> L1 -> ... An L(n+1) claim requires L(n)
  still passing on the same tree. Regression = revert, no exceptions.
- Gate artifacts: checkpoint + `runs/<name>.json` (gitignored) + board `[GATE]`
  row. Never overwrite a gate-passing checkpoint. Never leave results only
  in `/tmp`.

## archive.jsonl + AGENT_BOARD.md protocol

- Every 1000-ep eval appends one JSON line to `rl/runs/archive.jsonl`:
  `run_id, parent_id, track, level, config_delta, wall_s, eval_metrics`.
- Every spawn, result, adversary verdict, and gate gets one row in
  `docs/AGENT_BOARD.md` with format `[TYPE] id | scope | status | evidence`.
- Orchestrator updates `rl/PLAN.md` State when a measured result changes
  (replace the line, do not append history).

## Commit/push discipline

- Scouts do not commit. Orchestrator commits per completed scope with message
  `<scope>: <what changed> + <eval result>`.
- Push only green gates or explicitly marked work-in-progress. `rl/runs/`
  artifacts stay gitignored; `archive.jsonl` is the exception and is tracked.
- Before push: `cd rl && uv run pytest test_rl.py` and root `uv run pytest`.

## Safety

- L0-L6 are sim only. No L7 actuation while config flags read false
  (see `docs/SAFETY.md`: S0 default, S3/S4 require wheels-off-ground +
  clear-zone + human-present + timeout).
- Motion commands always carry finite duration + explicit stop/finally.
  Credentials stay runtime-only; never in logs/docs/git.
- `src/ugv/policy.py` deployed student stays NumPy-only; torch lives in `rl/`
  venv (`cd rl && uv run ...`). Root package stays torch-free.
