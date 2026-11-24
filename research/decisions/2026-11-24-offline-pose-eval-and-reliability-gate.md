---
id: 2026-11-24-offline-pose-eval-and-reliability-gate
date: 2026-11-24
author: fer-claude
status: active
supersedes: []
superseded_by: []
refs: [docs/POSE.md, docs/PARK_SIM.md, research/attempts/2026-11-24-pose-detector-v3c-autopilot-gate.md, .agents/skills/orchestrate-and-gate/SKILL.md]
tags: [pose, detector, evaluation, reliability, workflow]
---

## Decision

1. **No live browser for routine testing or iteration.** Evaluate every detector and solver change offline in Python:
   - Held-out Challenge frame sets (`runs/challenge_frames/{diag1,v3h,v3a}`, never trained on) via `tools/pose/eval_frames.py`.
   - A fast offline closed-loop evaluator (`n >= 50` matched seeds) driving `ugv.park` through `ugv.pose.render` and the real ONNX `KeypointDetector` (zero fabricated stand-ins), saving PNG overlays only for failed or colliding frames.
   - Use a real browser only to record a new held-out set or for a final milestone verification.
2. **Reliability gate against noise and local minima.** Follow `.agents/skills/orchestrate-and-gate/SKILL.md`:
   - Quote the 95% binomial interval before claiming an improvement (`n = 6` browser rounds is a smoke check, not proof of reliability).
   - Strict collision gate: no promoted candidate may collide more than its parent; decompose every collision frame (state, true vs estimated pose, per-keypoint pixel error) before retraining or adding heuristics.
   - Add a regression test ("check the check") for every solver or detector conditioning rule.
3. **Keep only the current best model on disk.** Retain only the active default pair (`runs/pose_train/{kp-v3c,kp-crop-v3c}`), fixed validation sets (`runs/pose_data/{val-v3,val-crop-v3}`), and frozen held-out frame sets (`runs/challenge_frames/{diag1,v3a,v3h}`). Delete superseded checkpoints and fixed training sets once their numbers and failure modes are recorded in `research/attempts/` (the Dream-RSI tree).
4. **Pause training** until the offline closed-loop evaluator, failure-frame decomposition, and `solve_planar` regression test exist (`research/tasks/2026-11-24-pose-detector-v4-reliability.md`).

## Why

- v3c reached `4/6` parked healthy (`1` collision, `1` timeout) and `1/6` parked under bad sensors (`2` collisions, `3` timeouts) at ~10 Hz in live Chromium, taking ~10 minutes for only `n = 12` episodes. That sample size has a ±35-point confidence interval and is too slow to iterate on collisions, bad-sensor recall (`0.375` on `v3a`), or architectural alternatives (e.g. PVNet keypoint voting vs heatmap + nose-crop heuristic).
- Leaving 8 older checkpoints and 11 GB of superseded fixed 24k datasets in `runs/` clutters state when the Dream-RSI attempt records already preserve every failed experiment and its lessons.

## Revisit if

A headless GPU WebGL runner makes Three.js closed-loop evaluation fast enough for `n >= 50` in under a minute, or once real car evaluation frames (`2026-11-17-real-pose-eval-set`) supersede rendered simulator benchmarks.
