#!/usr/bin/env python3
"""Game-ready simplification of the STRL-001 aircraft STLs, run inside Blender.

Pipeline per file (all in the shared model frame, no recentre/rotate/scale):

1. import, weld vertices (``--merge``), drop degenerate faces;
2. split into connected parts and delete a part only when none of its sampled
   points is reachable from outside along any of several hundred directions
   (hidden interior structure);
3. per part: limited dissolve of flat regions, then Decimate COLLAPSE with the
   ratio bisected so the part stays within ``--tol-mm`` of the *original* part
   surface, measured both ways with BVHTree.find_nearest;
4. triangulate and export binary STL with the same name.

    flatpak run org.blender.Blender -b --factory-startup \
        --python tools/mesh_opt/blender_optimize.py -- --out work/mesh_opt/cand_blender

Deterministic: every random choice comes from ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

SRC_DEFAULT = Path("/home/fer/Documents/ugv/simulation/gazebo/strl_001/meshes")
OUT_DEFAULT = Path("/home/fer/Documents/ugv/work/mesh_opt/cand_blender")


# ---------------------------------------------------------------- geometry io


def stl_tri_count(path: Path) -> int:
    with open(path, "rb") as fh:
        fh.read(80)
        return int(np.frombuffer(fh.read(4), "<u4")[0])


def import_stl(path: Path) -> bmesh.types.BMesh:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=str(path))
    obj = bpy.context.selected_objects[0]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bpy.data.objects.remove(obj, do_unlink=True)
    return bm


def bm_arrays(bm: bmesh.types.BMesh) -> tuple[np.ndarray, np.ndarray]:
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    tris = np.array([[v.index for v in f.verts] for f in bm.faces if len(f.verts) == 3])
    return verts, tris


def arrays_bm(verts: np.ndarray, tris: np.ndarray) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bvs = [bm.verts.new(tuple(p)) for p in verts]
    bm.verts.index_update()
    for tri in tris:
        try:
            bm.faces.new([bvs[i] for i in tri])
        except ValueError:  # duplicate face, already present
            pass
    bm.faces.index_update()
    bm.normal_update()
    return bm


# ------------------------------------------------------------- connected parts


def part_of_face(tris: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components over shared vertices; returns (face -> part, n_parts)."""
    n_verts = int(tris.max()) + 1
    parent = np.arange(n_verts)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b, c in tris:
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
        if ra != rc:
            parent[max(ra, rc)] = min(ra, rc)
    roots = np.array([find(int(i)) for i in range(n_verts)])
    face_part = roots[tris[:, 0]]
    _, face_part = np.unique(face_part, return_inverse=True)
    return face_part, int(face_part.max()) + 1


# ------------------------------------------------------------------- sampling


def sample_surface(verts, tris, count, rng) -> np.ndarray:
    tri = verts[tris]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return tri[:, 0].copy()
    pick = rng.choice(len(tris), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return (
        tri[pick, 0]
        + u[:, None] * (tri[pick, 1] - tri[pick, 0])
        + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    )


# ------------------------------------------------------------ hidden geometry


def hidden_faces(verts, tris, bvh, reach, rng, n_dirs, n_rots=16) -> np.ndarray:
    """Faces no outside ray can ever reach.

    A face is dropped only when every one of ``n_dirs`` rays leaving it is blocked
    within ``reach``: the point is then enclosed by other geometry and no orthographic
    view from outside can see it. Rays are rotated per face so the direction set is
    not aligned with the model. A hit closer than 0.1 mm is ignored (self hit or a
    coincident surface), which can only keep a face.
    """
    i = np.arange(n_dirs) + 0.5
    polar, azim = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5**0.5) * i
    base = np.c_[np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)]
    table = []
    for _ in range(n_rots):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        table.append([tuple(d) for d in base @ rot.T])

    eps, ignore = 2e-5, 1e-4
    hidden = np.zeros(len(tris), bool)
    for index, tri in enumerate(tris):
        c = (verts[tri[0]] + verts[tri[1]] + verts[tri[2]]) / 3.0
        free = False
        for d in table[index % n_rots]:
            hit = bvh.ray_cast((c[0] + d[0] * eps, c[1] + d[1] * eps, c[2] + d[2] * eps), d, reach)
            if hit[0] is None or hit[3] < ignore:
                free = True
                break
        hidden[index] = not free
    return hidden


# ---------------------------------------------------------------- simplification


def dissolve(verts, tris, angle_deg) -> tuple[np.ndarray, np.ndarray]:
    bm = arrays_bm(verts, tris)
    bmesh.ops.dissolve_limit(
        bm,
        angle_limit=np.radians(angle_deg),
        verts=bm.verts,
        edges=bm.edges,
        delimit={"NORMAL"},
    )
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    out_v = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    out_t = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    bm.free()
    return out_v, out_t


def collapse_weights(verts, tris, crease_deg, crease_weight) -> np.ndarray:
    """Decimate cost per vertex: 0 pins it, 1 lets it collapse.

    Open boundaries are pinned so neighbouring parts keep sharing their seam, and
    sharp creases are pinned (or heavily weighted) so silhouettes and rims stay put.
    """
    normals = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    face_of_edge = np.tile(np.arange(len(tris)), 3)
    uniq, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    weights = np.ones(len(verts))
    weights[np.unique(uniq[counts == 1])] = 0.0  # open boundary
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(uniq)))
    faces = face_of_edge[order]
    two = counts == 2
    a, b = faces[starts[two]], faces[starts[two] + 1]
    cos = np.abs(np.einsum("ij,ij->i", normals[a], normals[b]))
    sharp = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))) > crease_deg
    creases = [uniq[two][sharp], uniq[counts > 2]]  # sharp folds and non-manifold edges
    crease_verts = np.unique(np.vstack(creases))
    weights[crease_verts] = np.minimum(weights[crease_verts], crease_weight)
    return weights


def decimate(verts, tris, ratio, weights=None) -> tuple[np.ndarray, np.ndarray]:
    me = bpy.data.meshes.new("part")
    me.from_pydata(verts.tolist(), [], tris.tolist())
    me.update()
    ob = bpy.data.objects.new("part", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    if weights is not None:
        group = ob.vertex_groups.new(name="cost")
        for value in np.unique(weights):
            group.add(np.where(weights == value)[0].tolist(), float(value), "REPLACE")
        mod.vertex_group = group.name
        mod.vertex_group_factor = 1.0
    bpy.ops.object.modifier_apply(modifier="dec")
    data = ob.data
    out_v = np.array([v.co[:] for v in data.vertices], dtype=np.float64)
    out_t = np.array([list(p.vertices) for p in data.polygons], dtype=np.int64)
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(data)
    return out_v, out_t


def deviation(verts, tris, ref_bvh, ref_pts) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Distances both ways between this part and the original part surface.

    Returns (p99, max, per candidate vertex, per reference point).
    """
    forward = np.empty(len(verts))
    for i, p in enumerate(verts):
        hit = ref_bvh.find_nearest(p)
        forward[i] = hit[3] if hit[3] is not None else 0.0
    backward = np.empty(len(ref_pts))
    bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)
    for i, p in enumerate(ref_pts):
        hit = bvh.find_nearest(p)
        backward[i] = hit[3] if hit[3] is not None else 0.0
    both = np.concatenate([forward, backward])
    return float(np.percentile(both, 99)), float(both.max()), forward, backward


def simplify_part(verts, tris, args, rng, file_diag_mm, iters=7):
    """Dissolve flat regions, then collapse as far as the tolerance allows.

    A single ratio per part is too coarse: one bad spot would hold the whole part at
    a high triangle count. So after the safe bisection the ratio is pushed further and
    the spots that break the worst-case limit are pinned, which concentrates the
    remaining triangles exactly there.
    """
    tol_p99, tol_max = args.tol_p99_mm / 1000.0, args.tol_max_mm / 1000.0
    # A fixed millimetre budget is wrong for small parts: 1 mm on a 60 mm nacelle is a
    # visible change. Scale the budget with the part so it stays a fraction of its size.
    scale = min(1.0, max(0.15, 0.002 * file_diag_mm) / args.tol_max_mm)
    tol_p99, tol_max = tol_p99 * scale, tol_max * scale
    ref_bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)
    ref_pts = sample_surface(verts, tris, int(np.clip(4 * len(tris), 200, 4000)), rng)

    def measure(v, t):
        return deviation(v, t, ref_bvh, ref_pts)

    def within(result):
        return result[0] <= tol_p99 and result[1] <= tol_max

    base_v, base_t = verts, tris
    for angle in args.dissolve_deg:
        dv, dt = dissolve(verts, tris, angle)
        if within(measure(dv, dt)):
            base_v, base_t = dv, dt
            break
    if len(base_t) <= args.min_tris:
        return base_v, base_t, 1.0, "dissolve"

    weights = collapse_weights(base_v, base_t, args.crease_deg, args.crease_weight)
    base_bvh = BVHTree.FromPolygons(base_v.tolist(), base_t.tolist(), all_triangles=True)

    def collapse(ratio):
        return decimate(base_v, base_t, ratio, weights)

    def pin(points):
        added = False
        for p in points:
            hit = base_bvh.find_nearest(p)
            if hit[2] is None:
                continue
            for index in base_t[hit[2]]:
                if weights[index] != 0.0:
                    weights[index] = 0.0
                    added = True
        return added

    lo, hi, ratio = 0.02, 1.0, 1.0
    v, t = collapse(lo)
    if not within(measure(v, t)):
        ratio, v, t = 1.0, base_v, base_t
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))
            cv, ct = collapse(mid)
            if within(measure(cv, ct)):
                hi, ratio, v, t = mid, mid, cv, ct
            else:
                lo = mid

    for _ in range(args.refine):
        trial = max(0.02, ratio * 0.7)
        cv, ct = collapse(trial)
        result = measure(cv, ct)
        if not within(result):
            if not pin(np.vstack([cv[result[2] > tol_max], ref_pts[result[3] > tol_max]])):
                break
            cv, ct = collapse(trial)
            result = measure(cv, ct)
            if not within(result):
                break
        ratio, v, t = trial, cv, ct
    return v, t, ratio, "collapse"


# ------------------------------------------------------------------ per file


def process(path: Path, out: Path, args) -> dict:
    t0 = time.time()
    before = stl_tri_count(path)
    bm = import_stl(path)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=args.merge)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    verts, tris = bm_arrays(bm)
    bm.free()
    welded = len(tris)

    rng = np.random.default_rng(args.seed)
    file_diag_mm = float(np.linalg.norm(verts.max(0) - verts.min(0))) * 1000.0
    face_part, n_parts = part_of_face(tris)
    if args.keep_hidden:
        hidden = np.zeros(len(tris), bool)
    else:
        diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)
        hidden = hidden_faces(verts, tris, bvh, 2.0 * diag, rng, args.hidden_dirs)
    hidden_tris = int(hidden.sum())
    # Only whole components are removed: a face-level split would open cracks along
    # the ~20k edges where visible and hidden faces meet.
    counts = np.bincount(face_part, minlength=n_parts)
    hidden_counts = np.bincount(face_part[hidden], minlength=n_parts)
    fully_hidden = hidden_counts == counts
    hidden = fully_hidden[face_part]
    hidden_tris = int(hidden.sum())
    hidden_parts = int(fully_hidden.sum())

    verts, tris = verts, tris[~hidden]
    face_part, n_parts = part_of_face(tris)

    out_v, out_t, ratios, kinds, detail = [], [], [], [], []
    for part in range(n_parts):
        faces = np.where(face_part == part)[0]
        sub = tris[faces]
        used, remap = np.unique(sub, return_inverse=True)
        pv, pt = verts[used], remap.reshape(-1, 3)
        if len(pt) < 4:
            out_v.append(pv)
            out_t.append(pt)
            ratios.append(1.0)
            kinds.append("copy")
            detail.append((len(pt), len(pt), 1.0, "copy"))
            continue
        v, t, ratio, kind = simplify_part(pv, pt, args, rng, file_diag_mm)
        out_v.append(v)
        out_t.append(t)
        ratios.append(ratio)
        kinds.append(kind)
        detail.append((len(pt), len(t), ratio, kind))

    offset, merged_v, merged_t = 0, [], []
    for v, t in zip(out_v, out_t, strict=True):
        merged_v.append(v)
        merged_t.append(t + offset)
        offset += len(v)
    final_v, final_t = np.vstack(merged_v), np.vstack(merged_t)

    me = bpy.data.meshes.new(path.stem)
    me.from_pydata(final_v.tolist(), [], final_t.tolist())
    me.update()
    ob = bpy.data.objects.new(path.stem, me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.stl_export(filepath=str(out / path.name), ascii_format=False)

    stats = {
        "file": path.name,
        "tris_before": before,
        "tris_welded": welded,
        "tris_after": int(len(final_t)),
        "parts": n_parts,
        "parts_hidden": hidden_parts,
        "tris_hidden": hidden_tris,
        "parts_collapsed": int(sum(k == "collapse" for k in kinds)),
        "parts_dissolve_only": int(sum(k == "dissolve" for k in kinds)),
        "parts_kept": int(sum(k == "keep" for k in kinds)),
        "parts_copied": int(sum(k == "copy" for k in kinds)),
        "min_ratio": round(min(ratios), 4),
        "seconds": round(time.time() - t0, 1),
    }
    print(
        f"{path.name:22s} {before:7d} -> {stats['tris_after']:6d}  parts {n_parts:4d} "
        f"hidden {hidden_parts:4d}/{hidden_tris:6d} tris  min ratio {stats['min_ratio']:.3f}  "
        f"{stats['seconds']:.0f}s",
        flush=True,
    )
    if args.debug_parts:
        top = sorted(detail, key=lambda d: -d[1])[: args.debug_parts]
        for before_t, after_t, ratio, kind in top:
            print(f"    part {before_t:6d} -> {after_t:6d}  ratio {ratio:.3f}  {kind}")
    return stats


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--files", default="", help="comma separated subset of file names")
    ap.add_argument("--merge", type=float, default=1e-5, help="weld distance in metres")
    ap.add_argument(
        "--dissolve-deg",
        default="6,4,3,2,1",
        help="comma separated dissolve angles, tried in order until the tolerance holds",
    )
    ap.add_argument("--tol-p99-mm", type=float, default=0.25)
    ap.add_argument("--tol-max-mm", type=float, default=0.8)
    ap.add_argument("--refine", type=int, default=3, help="local detail re-add passes")
    ap.add_argument("--min-tris", type=int, default=40, help="parts this small are only dissolved")
    ap.add_argument("--crease-deg", type=float, default=30.0)
    ap.add_argument(
        "--crease-weight", type=float, default=1.0, help="decimate cost at creases; 0 pins them"
    )
    ap.add_argument("--hidden-dirs", type=int, default=42, help="escape rays per face")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-hidden", action="store_true", help="skip hidden face removal")
    ap.add_argument("--stats", type=Path, default=None, help="write per-file stats as JSON")
    ap.add_argument("--debug-parts", type=int, default=0, help="print the N largest parts")
    args = ap.parse_args(argv)
    args.dissolve_deg = [float(x) for x in str(args.dissolve_deg).split(",") if x.strip()]

    names = [n.strip() for n in args.files.split(",") if n.strip()]
    paths = [args.src / n for n in names] if names else sorted(args.src.glob("*.stl"))
    stats = [process(p, args.out, args) for p in paths]
    total_before = sum(s["tris_before"] for s in stats)
    total_after = sum(s["tris_after"] for s in stats)
    print(f"TOTAL {total_before} -> {total_after} tris")
    if args.stats:
        args.stats.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
