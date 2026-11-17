---
id: 2026-11-17-pose-smoke-sift-bank
date: 2026-11-17
author: fer-claude
question: Does SIFT matching against a bank of 48 synthetic reference renders (known pose, triangle-id map) give a pose without training?
parent: none
outcome: negative
failure_mode: underpowered
refs: [tools/pose/smoke.py, src/ugv/pose/solve.py]
tags: [pose, smoke, sift]
---

## Tried

Same 40 queries as `2026-11-17-pose-smoke-keypoint-pnp`. 48 reference renders at random poses;
SIFT (800 features) restricted to the airframe mask in references; Lowe ratio 0.8; all matches
from all references pooled into one RANSAC PnP.

## Result

Generic views: 32/40 solved, 0/40 within 5 deg and 10 cm (median rotation error 132 deg). Car
views: 22/40 solved, 0/40 within (133 deg).

## Lesson

As set up it cannot work: the procedural texture is a smooth sum of 4 cm sine waves (few
corners), queries match the noise background, and pooling matches across references mixes
symmetric views. It says nothing yet about SIFT on the real airframe.

## Re-run only if

With a corner-rich texture (or real photos of the airframe), references on a regular viewing
sphere, per-reference matching with the best reference kept, and a background without
self-similar noise as a control.
