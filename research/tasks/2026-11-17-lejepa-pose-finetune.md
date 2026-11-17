---
id: 2026-11-17-lejepa-pose-finetune
date: 2026-11-17
author: fer-claude
owner: fernando
state: open
updated: 2026-11-17
due: none
after: [2026-11-17-keypoint-detector]
check: none
refs: []
tags: [pose, learning, lejepa]
---

## Goal

Test whether a LeJEPA-pretrained encoder, fine-tuned (not frozen) jointly with the keypoint
head, beats the ImageNet baseline on the real eval set. Arms: frozen, fine-tuned, joint
SSL + keypoint loss; same data and budget.

## Done when

An attempt with all three arms on the real eval set, with seeds, against the
`keypoint-detector` baseline.
