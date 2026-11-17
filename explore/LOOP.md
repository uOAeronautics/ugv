# Explore loop

The cycle that drives the car toward a parked car under the plane (STRL-001, tags 14/6/3). The
runner is `tools/explore.py`; it is the state machine and refuses an illegal step. Ledger:
`explore/HYPOTHESES.jsonl` (append-only, one event per line). Design rules: `.steward/GOAL.md`.

## Cycle

1. **propose**: `explore.py propose --family F --statement "..." --cost "..."`.
   Families: classical-control, rl, perception, plant-calibration, task-geometry,
   evaluation-validity, hardware.
2. **predict**: `explore.py predict --id H-NNNN --prediction "..." --falsifier "..."`.
   The prediction and the observation that would refute it are written before any run.
3. **start**: `explore.py start --id H-NNNN`. One hypothesis runs at a time.
4. **close**: `explore.py close --id H-NNNN --outcome confirmed|refuted|inconclusive
   --evidence <run id or existing path>`. A claim is not evidence.
5. **retro**: due after every 5 closed hypotheses. `explore.py retro --text "what did we
   assume that no run tested?"`. Propose, predict and start are refused until it is recorded.
6. **next**: prints the one legal next step.

`list` prints the ledger as a table.

## Refusals

- Family outside the list, or a missing required field.
- Predict, start or close out of order; a second hypothesis started while one runs.
- Evidence that is neither a run id (`YYYYMMDD-HHMMSS...`) nor an existing path.
- Retry of a dead end without `--retry-of H-NNNN --changed "what is different now"`. A bare
  re-proposal of a refuted statement is refused.
- After 3 refuted hypotheses in a family, the next proposal must be another family. To stay,
  pass `--escalate "reason"`; the reason is recorded on the proposal.

## Not in the loop

Anything that moves the car is outside this tool. Running a hypothesis on the Pi or the GPU
follows `docs/SAFETY.md` and `docs/STATE.md`; the loop only records the decision and its evidence.
