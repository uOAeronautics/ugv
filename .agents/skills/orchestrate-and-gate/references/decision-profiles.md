# Decision profiles

Worked examples from the 2026-11-18 park/fault session (3.5 h, four commits, nine attempts).
Each profile is a situation, the move that worked, and the tell that you are about to get it
wrong. Numbers are from that session's attempts and logs; they are examples, not targets.

## 1. Structural refactor under a metric

**Situation.** `measure()` had to be split into `observe()` + `solve()` so the Lab could show
keypoints without changing the simulation.

**Move.** Hash the episode rows of a fixed 24-episode run before the edit (`fp.py`), split,
hash again. Same hash: proceed. Later the hash moved, and three candidate causes were separated
by comparing one at a time: a new key in the summary (harmless), a new condition entry
(harmless), the episode rows (would be a bug). The rows matched.

**Rule.** The fingerprint is the contract. A refactor is allowed to change structure, never
numbers.

**Tell.** "It should be equivalent" - that is the moment to run the hash, not after the next
1000-episode eval.

## 2. Aggregate failure, decomposed

**Situation.** With all faults on, the controller parked 412/1000 with 241 collisions, against
909/1000 with 7 on a healthy car.

**Move.** Three layers, cheapest first: per-family sweep (300 each) to rank the families; a
diagnosis tool recording state at failure, estimate error, and active faults; then one scratch
script per hypothesis. Result: 111 of 125 collisions in ENTER, contact estimate 3-4 cm off
against a 3.5 cm clearance, and ENTER averaging 1.2 visible keypoints per frame.

**Rule.** Rank families, then locate state, then measure the error at the failure. Only then
open the controller.

**Tell.** Reaching for a gain because it is named after the symptom (`k_lat` for a lateral
error).

## 3. The precision is in a blind spot

**Situation.** The finale needed better lateral accuracy than the estimate could give. The
skeleton had been designed for global pose from 1-2 m away.

**Move.** Compute the geometry: from the parked camera the belly sits at 37-63 deg elevation
against a 20.4 deg half-FOV; only the nose wheel's upper half is visible. Add three near-field
points on that tyre from the mesh. Healthy cars 175/200 with 21 off-target became 200/200 with
none; faulty collisions 99→72 at 300 episodes.

**Rule.** Count observability where the precision is required. If the sensor cannot see the
feature, the fix is keypoints or geometry, not control.

**Tell.** Tuning a controller to be accurate about something it cannot measure.

## 4. Bad failure into good failure

**Situation.** Near-field points cut collisions but exposure spells still let the car drive the
whole remaining distance blind, straight into the gear.

**Move.** `blind_max_m`: farther than 4 cm from the target, an empty measure window means
measure again; after three empty windows, back off. Collisions 72→33 at 300 episodes, successes
flat at ~107. Kept at 4 cm rather than 2 cm because 2 cm traded successes for the same
collisions.

**Rule.** parked > abort > off-target > collision. Prefer the rule that converts a collision
into an abort, then win the successes back.

**Tell.** Judging a safety rule by success rate alone; it will always look like a regression.

## 5. Plateau in a search

**Situation.** Search s4, 8 candidates per generation, 450 episodes each. Generation 1: every
candidate within noise of the root.

**Move.** Stop the search; diagnose. Later s5 produced a candidate with +21 successes but 28
collisions against the parent's 25, and it was rejected without an exception. s6 gained +10 and
was stopped after generation 4.

**Rule.** A search that returns noise says the model is wrong. A gate on the metric you refuse
to trade is worth more than a generation of candidates.

**Tell.** Increasing generations, or episodes per candidate, to "get a cleaner signal" from a
structural problem.

## 6. A mechanism that lost

**Situation.** Camera-yaw self-calibration looked obviously right: the mount yaw is observable
from motion, so estimate it online.

**Move.** A/B: 135→97 on healthy cars. A debug script showed the estimate swinging ±8 deg
around a sub-degree truth, driven by odometry drift and detector noise. `git checkout` the
estimator, write the attempt as `negative`, leave no flag.

**Rule.** Diagnose bug versus physics, then delete. A losing mechanism behind a flag is a trap
for the next reader.

**Tell.** Keeping the code "in case it helps later" with the flag defaulted off.

## 7. Noise accounting

**Situation.** A calibration change lifted all-fault success 60→74/200 but collisions 57→70.

**Move.** Compute the interval before interpreting: at n=200 that is ±4-7 points depending on
rate, so success is a real gain and the collision rise is not. Keep it, let the search weigh it,
validate at scale.

**Rule.** State the interval with the delta. Small effects at small n are not results, and a
scary-looking rise inside the interval is not a regression.

**Tell.** Two runs at n=48 and a conclusion.

## 8. Reviewers as hypotheses

**Situation.** Six UI reviewers produced ~10k words, much of it contradictory.

**Move.** Synthesize rather than average: accept the consensus (merge Overview into Park, six
tabs to four), overrule claims contradicted by the evidence (a coloured left border called an
AI tell, a backdrop blur called slop). Fix, re-shoot, then run a second round of *fresh*
reviewers with opposite priors on the new screenshots.

**Rule.** A reviewer's claim is a hypothesis; the screenshot is the arbiter; a fresh reviewer is
a fresh prior, while a re-prompted reviewer mostly defends its first answer.

**Tell.** Treating unanimity among agents as evidence, or sending findings back to the same
reviewer for approval.

## 9. The phase that did not close

**Situation.** The last 20 minutes produced camera-view parity, a patchy-traction floor model
and 998 CC0 panoramas: 297 lines across 12 files plus two new files. Tests and lint were green.
Then the usage cap hit at 04:03.

**Move (the one that was missed).** Commit the green set, or revert it; validate and promote the
controller candidate that had beaten its parent; write the fault results into the doc. None of
that happened, so the next session inherits an unvalidated `runs/park_search/final.json`, a
config still on the old parameters, and a simulator whose default detector has a capability the
real one lacks.

**Rule.** End every phase at a commit and one promotable artifact. A green tree is not a
checkpoint.

**Tell.** "I'll clean it up after the next experiment."
