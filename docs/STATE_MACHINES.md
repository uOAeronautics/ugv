# STATE MACHINES

Audit of implicit state machines (2026-11-16, haiku scout plus manual verification). Rule: a
mode that gates motion gets a declared transition table (`src/ugv/fsm.py`), so an undeclared
move raises in tests instead of surprising the car. UX gets one stage vocabulary, DX gets
one table to read, AX gets `fsm.history` (who moved, when, why) in logs.

## Done

| Machine | Where | Change |
|---|---|---|
| AutoPark stage | `src/ugv/autopark.py` `STAGE_TABLE`, `TERMINAL_STAGES` | Explicit `Machine`; `runner.stage` is a checked property; `web.py` imports `TERMINAL_STAGES` instead of three copies of the tuple. |
| L7 policy stale latch | `tools/l7/l7_run.py` | `"brake: sensors stale"` never matched `startswith("brake: stale")`, so a dead sensor stream braked forever instead of ending the run as `lost` after 4 s. Both strings match now. |

## Verified not a defect

`web.py` autopark `finally` does clear `state.auto_park`, owner and task (scout flagged it).

## Open, ranked by risk to the car (not changed: each alters motion, so each needs a hardware gate)

1. `web.py` controller/ownership/arming: `controller_stage` is written only on select and fallback, so it never advances (cosmetic). Input-watchdog brakes without releasing arm/owner. Candidate: one `Machine` over `manual -> armed -> following|autopark -> braking`.
2. `web.py` parking loop: guard flips `parking_active` without braking; correctness relies on the loop noticing. `parking_status` has no terminal value.
3. `aruco.py` follower: `enabled = enabled or follow` cannot go False while following; stale `enabled` can survive `disable_follow`.
4. `follow.py` `CoastSearch` (TRACK/COAST/SEARCH/STOP): a loss right after "target reached" coasts along the previous vector; STOP exits only on visibility.
5. `staged_baseline.py` phases: HOLD has no loss timeout (can latch braked); `_short_n` resets on APPROACH so APPROACH/ENTER can cycle unbounded; `_retreat` lags one step.
6. AutoPark: `entering` does not re-check calibration after `staging -> entering`; `acquire_started_at` never resets.
7. Three IMU freshness limits in web autopark (0.25 s start, 0.15 s ready, 0.25 s guard): pick one constant.
8. `rl/sim.py` episode flags: `latched_timeout` latches only timeout; not traced through reset.

Observatory JS has no real mode machine (`LB.stage` is a filter toggle).
