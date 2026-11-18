---
name: orchestrate-and-gate
description: How to work when a system misbehaves - the temperament (fingerprint before refactor, decompose before tuning, delete what lost, close each phase at a commit), T3 orchestration and subagent gating, log-first diagnosis, and the sim-to-real release gates. Use for multi-step debugging, any control, perception or tuning failure, parameter searches, long unattended runs, and before claiming a result or deploying to the Pi.
compatibility: UGV repo; hardware gates need SSH to the Pi (docs/STATE.md) and the S4 gate (a human present, wheels off the ground).
---

# Orchestrate and gate

Long sessions fail in three ways: the agent tunes a parameter when the structure is wrong, it
believes a difference smaller than its noise, and it leaves the work in a state nobody can pick
up. The stance below is the counterweight. Worked examples with numbers:
`references/decision-profiles.md`.

## 1. Temperament

Ordered by how often the move is right, not by ceremony.

**Fingerprint the number before you move the code.** Hash a fixed, small evaluation (here: 24
episodes) before a structural change, and require the same hash after. A refactor that quietly
moves the metric invalidates every comparison you have; the hash costs seconds, the alternative
costs a night. When the hash does move, name the cause before accepting it: a new summary field
is not a regression, an intentional default flip is, and a changed episode row is a bug.

**Decompose before you tune.** An aggregate number is a symptom, not a diagnosis. A failure has
a state, a place, an error at the moment of failure, and an observability count at that place.
Measure those four before touching a gain. The night this rule was learned, "41% parked" became
"111 of 125 collisions happen in ENTER, the estimate is 3-4 cm behind, and ENTER sees 1.2
keypoints per frame" - which is a perception bug, not a tuning problem.

**Treat a plateau as evidence about the model.** If every candidate lands within noise of its
root, the parameters are not the bottleneck. Stop the search and go read code. A search that
returns noise is telling you the failure is structural.

**Prefer converting a bad failure into a good one.** The order is parked > abort > off-target >
collision (an abort costs a retry; a collision costs a person walking over). Rules that turn the
first into the second earn their lost successes. This is also a search gate: no candidate may
collide more than its parent, however many successes it buys.

**Say the noise floor before you believe a delta.** Binomial, not vibes: at n=200 the 95%
interval is about ±7 points near p=0.5 and ±4 near p=0.9, so a 5-point difference at n=100 is
inside the interval. Quote it, then decide whether the effect is bigger than it.

**If it lost, delete it.** A mechanism that loses to its parent is a liability even behind a
flag: it will be switched on by someone who reads the flag's name and not the attempt. Revert
the code, record the attempt, keep the lesson. A flag is for a mechanism that won somewhere
and is not yet default, not for one that lost.

**Own the invalid run in writing.** A mis-set magnitude or a contaminated host makes a run
invalid; it does not make it private. Say so in the attempt, mark the row invalid, re-run, and
keep the old numbers visible with the reason. Silent re-runs destroy trust in every other
number you produced.

**Close a phase at a commit and one promotable artifact.** The candidate that beat its parent
belongs in `config/` or it does not exist. A green tree with an unvalidated `runs/.../best.json`
and a stale config is the most expensive state to resume from. Commit, promote or explicitly
park, then start the next phase - especially when a context or usage cap is near.

## 2. Orchestration

Split early; briefs and routing live in `mastermind`. The parts that decide whether a fan-out
was worth it:

- **One deliverable per child, checkable without reading its transcript.** A file, a JSON
  answer, a passing command.
- **One owner per exclusive resource**: a GPU, the car, a deploy target, a file another job
  re-imports. Concurrent edits to a module a running search imports invalidate its results.
- **Brief in a file**, with paths and line ranges, output format, self-check and a report cap.
- **Launch async and do not poll.** No sleep loops, no `gh --watch`. One watcher per job; three
  watchers on one search produce messages, not information.
- **Gate the deliverable yourself.** A child's "tests pass" counts its own tests.
- **Bounded reads.** `limit`, `maxCharsPerItem`, `afterPosition`; never dump a whole transcript
  into context to answer one question.

## 3. Diagnosis before you retrain or retune

Never blame "sensor noise" or "field of view" from a replay. Pass these first:

1. **Stationary floor.** Measure noise at rest, not in closed loop. A 0.63 rad/s "signal" once
   turned out to be a 1.4 Hz limit cycle over a 0.0008 rad/s sensor floor.
2. **Lag and phase.** Command-to-motion lag, loop period, and whether a filter costs phase
   margin. A 20-60 ms command delay is enough to turn a careful creep into a limit cycle.
3. **Observability per phase.** Count the measurements the controller actually gets in the
   phase where it fails. Blind where precision is needed is a perception bug, and no gain fixes
   it; five points that observe yaw beat twelve that do not.
4. **Plant and frame parity.** Deadzones, gains, axis signs, units (mm vs m, clockwise vs
   counter-clockwise yaw), and the exact camera model: focal, principal point, mount pitch.
   Cross a frame boundary only through the named conversion.

Then change one thing and A/B on matched seeds. Same n, same seeds, same candidates; scale up
only after the effect is bigger than the interval. Test a targeted fix on the failure it was
built for, not only on the aggregate: the pitch refit was judged on the bump and lens families
(+12 and +12 out of 150) with a healthy-car control that had to stay flat.

**Check the check.** A new test that passes on the old code proves nothing. Run it against the
behaviour you replaced, or assert a difference between two paths (with and without the
mechanism). One pitch-refit test originally ended in an assertion that was true by construction;
the real version asserts that without the refit the solve fails or comes out worse.

## 4. Release gates

Simulation success is not hardware success, and a synthetic number is not evidence about a
method that must run on real frames. Before any claim or deploy:

- **Evidence class named.** Synthetic, simulated-camera, real-frame, real-car. A dashboard or a
  doc that mixes them without labels is lying by layout.
- **Artifact identity.** The thing you validated is the thing you will deploy: commit and model
  file, not "the latest run". An unvalidated `best.json` is not a candidate.
- **Detector on real frames.** Keypoint error on the real evaluation set, not the synthetic
  validation split; check that every keypoint the controller relies on exists in the real
  detector's output.
- **Live-frame parity.** Tag IDs, intrinsics, camera drop-in intact (`vision-camera.conf`),
  fresh timestamped state rather than cached values.
- **Rest and end frame.** Never trust an estimate-based "arrived": confirm near-zero velocity
  and inspect the end frame. PnP flips are silent.
- **S4 only with the gate.** `docs/SAFETY.md`: wheels off the ground or a clear zone, a human
  present, a timeout. A skill never raises the level on its own.

## 5. Traps that cost real time

- `pkill -f <pattern>` matches its own shell when the pattern appears in the command line,
  killing the command that would have reported the result. Use the port (`fuser -k 8768/tcp`)
  or the task id.
- `sed -i` on a symlink replaces the link with a file. Edit the target; check `git status` for
  type changes before committing.
- Editing a module while a background job re-imports it changes the experiment mid-run. Stop
  the job, or work on other files.
- `sleep`-based polling is blocked in several harnesses and wastes a turn when it is not. Use
  the task's completion notification.
- A reviewer's claim is a hypothesis. A team of agents agreeing is still a hypothesis. The
  screenshot, the command output or the hash is the arbiter.
