---
id: 2026-11-17-remove-student-policy-runtime
date: 2026-11-17
author: fer-claude
owner: fernando
state: done
updated: 2026-11-17
due: none
after: []
check: none
refs: []
tags: [cleanup, runtime]
result: [a72bbdd]
---

## Goal

Remove the RL `StudentPolicy` path from the Pi runtime (`src/ugv/policy.py`, its dashboard
selector and tests) and the Gazebo runners that need `rl/`. It changes motion code, so it needs
the hardware gate in `docs/SAFETY.md` before deploy.

## Done when

Root tests pass, the dashboard still drives manually, and the deploy is checked on the car.

## Log
- 2026-11-17 fernando: claim
- 2026-11-17 fernando: done: code removed; not deployed: the next deploy is a hardware change (docs/STATE.md)
