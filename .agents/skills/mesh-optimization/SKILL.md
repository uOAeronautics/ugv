---
name: mesh-optimization
description: Turn a heavy CAD or scan mesh into a game-ready one with subagents, without changing what a camera sees. Use when asked to simplify, decimate, optimise or make "game-ready" any STL/OBJ/GLB, to cut triangle count for a simulator or web viewer, or to check that a simplified mesh still matches its original.
compatibility: UGV repo; needs tools/mesh_opt/ (Python, numpy, scipy, trimesh with ray_pyembree, Pillow). Offline gate; no hardware.
---

# Mesh optimisation

The orchestrator owns the gate and the verdict. Workers own one pipeline script each. An
auditor tries to break the result. Read `mastermind` first; this adds what is specific to meshes.

## Loop

1. **Measure the original yourself** (minutes, not delegated): triangles per part, connected
   components, watertight or not, boundary edges, edge-length percentiles, which part holds the
   triangles. These numbers go in the brief labelled *measured*.
2. **Build and calibrate the gate before any worker starts.** `tools/mesh_opt/meshgate.py`
   compares original and candidate by ray casting. It must pass the original against itself,
   pass one known-good candidate and fail one known-bad candidate. Use an inside-out copy and a
   pinhole as the known-bad pair: both passed a version of this gate that lacked back-face and
   pierced-area checks. Write down the noise floor.
3. **Launch two workers on different toolchains**, each with its own script and output
   directory, same brief otherwise. Different methods stalling at the same count is the
   evidence that the floor is real.
4. **Read their traces after 15–20 minutes.** Look for a worker fighting the gate or the brief
   instead of the mesh. Fix the gate or brief, then steer with a short note in a file.
5. **Re-gate what they deliver** with seeds and a resolution they never saw. Take the best
   passing version of each part; a part that passes for nobody ships unchanged.
6. **Look at the sheets.** Open the side-by-side images for the axis views. Numbers miss a
   missing wheel that is 0.1 % of the pixels.
7. **Audit** with a third worker told the result is broken and to prove it (winding, lost
   components, cracks, slivers, thin parts, reproducibility, gate defects).
8. **Export** with `tools/mesh_opt/export_glb.py` and report counts, the binding limit, and
   what was not checked.

## What the gate measures, and why

| Check | Catches | Blind to |
|---|---|---|
| Silhouette IoU per view | lost thin parts, shrunk outline | interior, small parts on a big model |
| Depth on camera-facing pixels | dents, holes, see-through | grazing surfaces (by design) |
| Flat-normal change | faceting, wrong shading | rotation within the 10 % tail |
| Visible original → candidate distance | deleted visible surface | hidden surface (by design) |
| Candidate → original distance | invented or drifted surface | — |
| Pierced area (hole to background) | through-holes, missing faces | shallow dents |
| Dent area (>1 mm depth patch) | local collapse, missing face on a closed part | errors under 1 mm |
| Back-face patch | inverted winding, missing face on a closed part | open surfaces |
| Bounding box | scale, axis or origin changes | — |

Hidden geometry is free to delete because only ray-visible points are compared.

## Lessons (measured on STRL-001, 2026-11-15)

- **Depth along the ray explodes at grazing angles.** A 0.01 mm sideways slip on a knife edge
  read as millimetres and failed a part that was 0.15 mm from the original. Judge depth only
  where both surfaces face the camera and forgive a physical slip, not a pixel.
- **Sampled distance has a noise floor.** Nearest-sample distance on 3 M samples gave 0.78 mm
  p99 for a mesh against itself. Refine to exact point-to-triangle on the nearest few triangles.
- **Scale the gate to the part.** A fixed pixel count on a 7 cm part made the slip window
  31×31 pixels and one gate run take over 25 minutes. Cap resolution by the forgiven slip.
- **"Mostly hidden" was a wrong guess.** A pixel census said 31 % of triangles were visible;
  whole hidden components held 0.4–3 %. Sub-pixel triangles are missed by a pixel census.
- **Never split a surface into visible and hidden faces.** The seam was ~20 k edges and
  opened cracks (depth 12 mm). Delete whole hidden components or nothing.
- **One global quadric collapse beat cleverer schemes** on this mesh: per-component error
  bounds, gate-driven reallocation and vertex pinning all cost more triangles for the same
  fidelity. Try the plain thing first, step down a ladder of targets, stop at the last PASS.
- **An error bound is not the gate.** meshoptimizer at 0.3 mm reported error still gave 5 mm
  depth error: a quadric cannot see wall thickness, so thin shells collapse through themselves.
- **Tool traps.** pymeshlab's quadric filter silently did nothing on a non-manifold mesh when
  given `preserveboundary`, `preservenormal` or `targetperc`; an older build segfaulted. Check
  the triangle count after every filter call. Blender's `BVHTree.ray_cast` is ~50 k rays/s
  against ~2 M for embree. Blender's STL importer drops duplicate and degenerate faces, so
  counts differ from trimesh before anything is simplified. The Blender flatpak cannot see
  `/tmp` and resolves relative paths against its own directory: use absolute paths under home.
- **Small detailed parts are the fragile ones.** Both workers' nacelles passed the seed they
  tuned on and failed a held-out seed. A count that sits on the pass/fail edge is not a pass.
- **`pkill -f name` kills the shell that runs it** when the pattern appears in its own command
  line. Use `pgrep -f "[n]ame"`.

## Equation fits and symbolic regression (measured negative, STRL-001)

Tried as an alternative to triangle removal: fit a height field to one skin panel, then re-tessellate.
Keep the instruments, not the route.

- **Collapse wins when the input is already accurate.** The panel's own tessellation is ~0.01 mm from
  its surface; a degree-12 height field adds 0.52 mm p99, so it can only lose. The equation route is
  a denoiser: it pays on a scan whose triangles are worse than 0.3 mm from the true shape.
- **Adaptive placement is the transferable result.** Insert the centroid of any triangle whose chord
  error exceeds a tolerance, re-Delaunay, repeat: 1 846 triangles reached p99 0.32 mm where a uniform
  3 mm grid needed 23 491, a 12.7× saving. Use it only when you re-tessellate a fitted surface.
- **Do not identify outliers by area-weighted sampling.** 60 k samples over 28 000 mm² found 93 of
  the 256 sliver triangles that hang 3–10 mm below the skin; 164 survived, and they set the max.
  A 0.16 mm² sliver is hit once in ~3 000 draws. Select by per-face centroid residual against the
  local fit instead.
- **A dense basis beats a sparse equation on a smooth surface.** Best PySR (maxsize 100, 16
  populations, `sin` nested-constrained, parsimony 0.02) reached p99 0.906; plain 2-D degree-12
  least squares reached 0.521. Check effective dimensionality first: a 1-D fit was 1.72, so the
  spanwise term is real, but PySR's sparse expressions cannot express the 91-term basis.
- **The default `maxsize=30` is a ceiling, not a budget.** Two first-pass runs peaked at complexity
  29 and 30. Raising the cap to 100 was worth 2.2× on p99. L4 loss, a wing template and 50×
  boundary weights were each measured and rejected (they fit tails or seams at the expense of the
  bulk).

## Brief rules for mesh workers

- Label every fact *measured* or *guess*. Workers obey guesses as if they were measurements.
- Give a hard requirement (gate PASS) and a stretch target (triangle count), and say which is
  which. Ask for the smallest count that passes and what stopped it going lower.
- Originals are read-only; outputs keep file names, frame, units and origin. Require a no-op
  import/export round trip with matching bounds before any simplification.
- Workers may run the gate and may not edit it. Tell them the final gate uses other seeds.
- Require a "what did not work" section with numbers. It is where the reusable lessons are.
- Set a time box. Without one a worker keeps exploring and never writes the deliverable.

## Format

Keep per-link STL when the consumer (an SDF, an existing viewer) already reads it. Ship one
GLB as the game-ready master: indexed vertices, normals split at a crease angle, one named
node per link. STL stores three loose vertices per triangle and no normals, so it is about
four times larger for the same mesh. FBX and OpenUSD add nothing for Gazebo or three.js.

## Commands

See `docs/COMMANDS.md` for the gate, the two pipelines and the exporter.
