---
id: 2026-11-18-pose-backgrounds-distractors
date: 2026-11-18
author: fer-claude
owner: fernando
state: open
updated: 2026-11-18
due: none
after: [2026-11-17-real-pose-eval-set]
check: none
refs: []
tags: [pose, data]
---

## Goal

Harden the detector against unseen rooms and things that look like an airframe. Field research
of 2026-11-18 (sources read, licences checked where stated):

- Lighting: Poly Haven HDRIs and textures, CC0 (polyhaven.com/license).
- Distractors and occluders: Google Scanned Objects (CC BY 4.0, 1030 objects, SDF for Gazebo);
  Objaverse filtered to CC0/CC-BY per object (dataset ODC-By, objects vary).
- Rooms to render at the car's 6.4 cm camera height: ReplicaCAD (CC BY 4.0); HM3D (academic
  only); Hypersim (CC BY-SA 3.0, 461 scenes, 1.9 TB); ScanNet++ (non-commercial).
- Drone hard negatives (another UAV in the room, like the lab's own): KI4Flight, Roboflow UAV
  sets (CC BY 4.0); all far-field/outdoor, none floor-level. Note that the 9,045 real car frames
  already used as backgrounds include the lab's other drone: a useful hard negative.
- No open benchmark of 6-DoF object pose from a ground robot was found.

## Plan

1. Hold out 3-5 unseen rooms (ReplicaCAD or HM3D at 6.4 cm) before training on anything new.
2. Poly Haven lighting; 3. GSO/Objaverse occluders in the scene and at the frame border;
4. ablate each source on the holdout and the real eval set; keep only what helps both.
Never commit renders, frames or weights; log each object's licence in the manifest.

## Done when

An attempt with the holdout and real-eval numbers per source (kept or dropped).
