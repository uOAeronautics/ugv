---
id: 2026-11-17-park-v4-online-motor-identification
date: 2026-11-17
author: fer-claude
question: Does identifying gain and deadzone per axis online (RLS) fix the stuck-in-deadzone and overshoot failures?
parent: 2026-11-17-park-v3-self-calibrated-pitch
outcome: negative
failure_mode: plant-mismatch
refs: [src/ugv/park/actuator.py]
tags: [park]
---

## Tried

Ablation batch: 96 episodes, seed 1, conditions ideal / plant spread only / detector noise only / calibration error only / full (`ugv.park.evaluate`).

## Result

Full fell to 24/96; ideal 89/96. A trace showed the car moving at 0.14 m/s while it believed 0.06: false stall detections from noisy, varying-duty windows had inflated the strafe deadzone to 0.22 (true 0.13).

## Lesson

Online identification needs clean experiments: steady duty, direct measurements. Continuous dead reckoning in the finale stays fragile whatever the model.

## Re-run only if

Never; v6 keeps the identification with the stricter sampling rules.
