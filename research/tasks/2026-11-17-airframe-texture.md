---
id: 2026-11-17-airframe-texture
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: []
check: none
refs: []
tags: [pose, texture]
---

## Goal

Decide what the airframe looks like to the camera. Options, cheapest first: (a) photograph the
real airframe and texture the mesh from the photos; (b) print a procedural, corner-rich
pattern (high-frequency, non-repeating, e.g. a seeded noise field) on stickers placed at known
model coordinates; (c) optimise a texture for detectability in simulation (differentiable
render, or a symbolic search such as PySR over a small family of procedural functions scored by
SIFT repeatability across views). PySR only makes sense in (c), and only once a score exists.

## Done when

A decision record picks one, with the SIFT repeatability score of each candidate on synthetic
views as the evidence.
