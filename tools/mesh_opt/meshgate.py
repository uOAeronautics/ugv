"""Gate a simplified mesh against its original: what a camera sees must not change.

Casts the same orthographic rays at both meshes from many directions and compares
silhouette, depth and shading, then measures surface distance both ways. Hidden
geometry may be deleted freely; anything a ray can reach may not move.

    uv run --no-project --python 3.12 --with trimesh --with scipy --with embreex \
        --with pillow python tools/mesh_opt/meshgate.py ORIG CAND --out work/mesh_opt/gate

ORIG and CAND are each an STL file or a directory of STL files in one shared frame.
Exit status 0 means every threshold held. Units are metres in, millimetres out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

# Thresholds. A candidate fails if any is exceeded. mm unless noted.
LIMITS = {
    "bbox_mm": 1.0,  # largest change of any bounding-box face
    "iou_min": 0.990,  # worst silhouette IoU over all views
    "depth_p99_mm": 1.0,  # worst per-view 99th percentile depth error
    "seethrough_pct": 0.05,  # worst per-view share of pixels >5 mm deeper (a hole)
    "normal_p90_deg": 25.0,  # worst per-view 90th percentile shading-normal change
    "vis_to_cand_p99_mm": 1.0,  # visible original surface -> candidate
    "vis_to_cand_max_mm": 4.0,
    "cand_to_orig_p99_mm": 1.0,  # candidate surface -> original (no invented surface)
    "cand_to_orig_max_mm": 4.0,
    "pierced_mm2": 3.0,  # largest hole to the background, any view (found by audit: IoU hid it)
    "dent_mm2": 12.0,  # largest connected patch more than DENT_MM off in depth, any view
    "backface_mm2": 3.0,  # largest patch showing a back face where the original shows a front
}
HOLE_MM = 5.0
DENT_MM = 1.0  # a patch this far off in depth is a dent or a missing face, not noise
FACING_COS = 0.35  # depth is judged only on faces within ~70 deg of facing the camera
SLIP_MM = 0.75  # sideways slip forgiven before a depth difference counts
AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def load(path: Path) -> trimesh.Trimesh:
    files = sorted(path.glob("*.stl")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no STL under {path}")
    parts = [trimesh.load(f, process=False, force="mesh") for f in files]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    keep = mesh.area_faces > 1e-14  # embree and sampling both choke on zero-area faces
    return trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)


def directions(count: int, seed: int) -> np.ndarray:
    """Six axis views plus a rotated Fibonacci sphere; the seed moves the sphere."""
    i = np.arange(count) + 0.5
    polar, azim = np.arccos(1 - 2 * i / count), np.pi * (1 + 5**0.5) * i
    pts = np.c_[np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)]
    rot = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    return np.vstack([AXES, pts @ rot[:3, :3].T])


class Scene:
    """One mesh plus the shared camera framing taken from the original."""

    def __init__(self, mesh: trimesh.Trimesh, centre: np.ndarray, radius: float, res: int):
        self.mesh, self.centre, self.radius, self.res = mesh, centre, radius, res
        self.rays = RayMeshIntersector(mesh)
        self.normals = mesh.face_normals
        self.p0 = mesh.vertices[mesh.faces[:, 0]]

    def render(self, view: np.ndarray):
        """Return depth (nan = miss), unit normals and hit triangle index per pixel."""
        up = np.array([0, 0, 1.0]) if abs(view[2]) < 0.9 else np.array([1.0, 0, 0])
        right = np.cross(up, view)
        right /= np.linalg.norm(right)
        up = np.cross(view, right)
        span = np.linspace(-self.radius, self.radius, self.res)
        u, v = np.meshgrid(span, -span)
        origin = (
            self.centre + view * 2 * self.radius + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up
        )
        ray = np.tile(-view, (len(origin), 1))
        tri = self.rays.intersects_first(origin, ray)
        hit = tri >= 0
        depth = np.full(len(origin), np.nan)
        normal = np.zeros((len(origin), 3))
        n = self.normals[tri[hit]]
        denom = np.einsum("ij,ij->i", n, ray[hit])
        denom[np.abs(denom) < 1e-12] = 1e-12
        depth[hit] = np.einsum("ij,ij->i", self.p0[tri[hit]] - origin[hit], n) / denom
        normal[hit] = n
        shape = (self.res, self.res)
        return depth.reshape(shape), normal.reshape(*shape, 3), tri.reshape(shape), origin, ray


def neighbour_min_err(a: np.ndarray, b: np.ndarray, reach: int) -> np.ndarray:
    """|a - b| allowing b to shift up to `reach` pixels, so a sideways slip is not read as depth."""
    best = np.full(a.shape, np.inf)
    pad = np.pad(b, reach, constant_values=np.nan)
    for dy in range(2 * reach + 1):
        for dx in range(2 * reach + 1):
            err = np.abs(a - pad[dy : dy + a.shape[0], dx : dx + a.shape[1]])
            best = np.fmin(best, np.where(np.isnan(err), np.inf, err))
    return best


def shade(depth: np.ndarray, normal: np.ndarray, view: np.ndarray) -> np.ndarray:
    side = np.cross(view, [0.3, 0.5, 0.8])
    light = view + 0.9 * side / np.linalg.norm(side)  # off-axis so flat faces do not wash out
    lit = 0.12 + 0.88 * np.abs(normal @ (light / np.linalg.norm(light))) ** 1.5  # two-sided
    return np.where(np.isnan(depth), 1.0, lit)


def compare_view(orig: Scene, cand: Scene, view: np.ndarray):
    do, no, to, origin, ray = orig.render(view)
    dc, nc, _, _, _ = cand.render(view)
    ho, hc = ~np.isnan(do), ~np.isnan(dc)
    both = ho & hc
    # Depth along the ray is only meaningful where the surface faces the camera: on a face seen
    # at angle t a sideways slip e reads as e / tan(t), so knife edges explode. Grazing pixels are
    # left to the silhouette and surface-distance checks, which do not have that amplification.
    facing = (np.abs(no @ view) >= FACING_COS) & (np.abs(nc @ view) >= FACING_COS)
    reach = max(1, int(np.ceil(SLIP_MM / (2000 * orig.radius / orig.res))))
    judged = both & facing
    err = np.where(judged, neighbour_min_err(dc, do, reach), 0.0) * 1000
    err = np.where(np.isfinite(err), err, 0.0)
    deeper = judged & ((dc - do) * 1000 > HOLE_MM) & (err > HOLE_MM)
    # A hole straight through to the background: the original is hit well inside its outline
    # and the candidate is not. Silhouette IoU spends a 1 % budget on this and never notices.
    inside = ndimage.binary_erosion(ho, iterations=reach + 1)
    blobs, count = ndimage.label(inside & ~hc)
    largest = int(np.bincount(blobs.ravel())[1:].max()) if count else 0
    pixel_mm = 2000 * orig.radius / orig.res
    # A missing face on a closed thin part shows the inside wall: too shallow for see-through,
    # too small for a percentile, but one connected patch.
    # What back-face culling would delete: the candidate's first hit is a back face where the
    # original's is a front face. Catches inverted winding and a missing face on a closed part
    # (the ray then lands on the inside of the far wall).
    wrong_side = judged & ((no @ view) > 0) & ((nc @ view) < 0)
    backs, back_count = ndimage.label(wrong_side)
    back = int(np.bincount(backs.ravel())[1:].max()) if back_count else 0
    dents, dent_count = ndimage.label(err > DENT_MM)
    dent = int(np.bincount(dents.ravel())[1:].max()) if dent_count else 0
    cos = np.clip(np.abs(np.einsum("ijk,ijk->ij", no, nc)), 0, 1)
    angle = np.degrees(np.arccos(cos))[both]
    stats = {
        "iou": float(both.sum() / max((ho | hc).sum(), 1)),
        "xor_px": int((ho ^ hc).sum()),
        "depth_p99_mm": float(np.percentile(err[judged], 99)) if judged.any() else 0.0,
        "depth_max_mm": float(err[judged].max()) if judged.any() else 0.0,
        "seethrough_pct": float(100 * deeper.sum() / max(judged.sum(), 1)),
        "normal_p90_deg": float(np.percentile(angle, 90)) if both.any() else 0.0,
        "pierced_mm2": largest * pixel_mm**2,
        "dent_mm2": dent * pixel_mm**2,
        "backface_mm2": back * pixel_mm**2,
    }
    so, sc = shade(do, no, view), shade(dc, nc, view)
    heat = np.clip(err / 3.0, 0, 1)  # full red at 3 mm
    miss = (ho ^ hc).astype(float)
    diff = np.dstack(
        [1 - miss * 0, 1 - heat, 1 - np.maximum(heat, miss)]
    )  # red = depth, yellow = edge
    panel = np.hstack([np.dstack([so] * 3), np.dstack([sc] * 3), diff])
    hits = (origin + ray * do.reshape(-1, 1))[ho.ravel()]
    return stats, (panel * 255).astype(np.uint8), hits, np.unique(to[ho])


def surface_distance(points: np.ndarray, target: trimesh.Trimesh, samples: int, seed: int):
    """Exact distance to the nearest triangle among those owning the 6 nearest samples."""
    cloud, face = trimesh.sample.sample_surface(target, samples, seed=seed)
    _, near = cKDTree(cloud).query(points, k=6, workers=-1)
    best = np.full(len(points), np.inf)
    for column in near.T:
        on_tri = trimesh.triangles.closest_point(target.triangles[face[column]], points)
        best = np.minimum(best, np.linalg.norm(on_tri - points, axis=1))
    return float(np.percentile(best, 99) * 1000), float(best.max() * 1000)


def topology(mesh: trimesh.Trimesh) -> dict:
    merged = mesh.copy()
    merged.merge_vertices()
    edges = np.sort(merged.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "tris": int(len(mesh.faces)),
        "verts": int(len(merged.vertices)),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "components": int(
            len(trimesh.graph.connected_components(merged.face_adjacency, min_len=1))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("orig", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="directory for report.json and sheets")
    ap.add_argument("--views", type=int, default=34, help="sphere views in addition to the 6 axes")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0, help="rotates the view set and the sampling")
    ap.add_argument("--samples", type=int, default=3_000_000)
    ap.add_argument("--max-tris", type=int, default=0, help="fail above this triangle count")
    ap.add_argument(
        "--limit", action="append", default=[], metavar="KEY=VALUE", help="override a threshold"
    )
    args = ap.parse_args()
    limits = dict(LIMITS)
    for item in args.limit:
        key, value = item.split("=")
        if key not in limits:
            raise SystemExit(f"unknown limit {key}; known: {sorted(limits)}")
        limits[key] = float(value)

    orig_mesh, cand_mesh = load(args.orig), load(args.cand)
    centre = orig_mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(orig_mesh.extents) / 2 * 1.02)
    # A pixel finer than a third of the forgiven slip buys nothing and makes the slip window
    # (2 * reach + 1)^2 explode on small parts, so cap the resolution there.
    res = min(args.res, max(256, int(2000 * radius / (SLIP_MM / 3))))
    orig = Scene(orig_mesh, centre, radius, res)
    cand = Scene(cand_mesh, centre, radius, res)
    args.out.mkdir(parents=True, exist_ok=True)

    views, panels, visible_pts, visible_tris = [], [], [], set()
    for index, view in enumerate(directions(args.views, args.seed)):
        stats, panel, hits, tris = compare_view(orig, cand, view)
        stats["view"] = [round(float(x), 3) for x in view]
        views.append(stats)
        visible_pts.append(hits[:: max(1, len(hits) // 20000)])
        visible_tris.update(tris.tolist())
        if (
            index < 6
            or stats["iou"] < limits["iou_min"]
            or stats["depth_p99_mm"] > limits["depth_p99_mm"]
            or stats["pierced_mm2"] > limits["pierced_mm2"]
            or stats["dent_mm2"] > limits["dent_mm2"]
            or stats["backface_mm2"] > limits["backface_mm2"]
        ):
            panels.append((index, panel))
    for index, panel in panels[:16]:
        Image.fromarray(panel).save(args.out / f"view_{index:02d}.png")

    vis = np.vstack(visible_pts)
    cand_pts, _ = trimesh.sample.sample_surface(cand_mesh, 300_000, seed=args.seed)
    v2c = surface_distance(vis, cand_mesh, args.samples, args.seed)
    c2o = surface_distance(cand_pts, orig_mesh, args.samples, args.seed + 1)
    measured = {
        "bbox_mm": float(np.abs(orig_mesh.bounds - cand_mesh.bounds).max() * 1000),
        "iou_min": min(v["iou"] for v in views),
        "depth_p99_mm": max(v["depth_p99_mm"] for v in views),
        "seethrough_pct": max(v["seethrough_pct"] for v in views),
        "normal_p90_deg": max(v["normal_p90_deg"] for v in views),
        "vis_to_cand_p99_mm": v2c[0],
        "vis_to_cand_max_mm": v2c[1],
        "cand_to_orig_p99_mm": c2o[0],
        "cand_to_orig_max_mm": c2o[1],
        "pierced_mm2": max(v["pierced_mm2"] for v in views),
        "dent_mm2": max(v["dent_mm2"] for v in views),
        "backface_mm2": max(v["backface_mm2"] for v in views),
    }
    failed = [
        k for k, v in measured.items() if (v < limits[k] if k == "iou_min" else v > limits[k])
    ]
    topo = {"orig": topology(orig_mesh), "cand": topology(cand_mesh)}
    if args.max_tris and topo["cand"]["tris"] > args.max_tris:
        failed.append("max_tris")
    report = {
        "orig": str(args.orig),
        "cand": str(args.cand),
        "seed": args.seed,
        "pixel_mm": 2000 * radius / res,
        "topology": topo,
        "reduction": 1 - topo["cand"]["tris"] / topo["orig"]["tris"],
        "orig_visible_tri_share": len(visible_tris) / topo["orig"]["tris"],
        "measured": measured,
        "limits": limits,
        "failed": failed,
        "worst_views": sorted(views, key=lambda v: v["iou"])[:3],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"tris {topo['orig']['tris']} -> {topo['cand']['tris']}  ({100 * report['reduction']:.1f}% removed)"
    )
    for key, value in measured.items():
        flag = "FAIL" if key in failed else "ok  "
        print(f"  {flag} {key:22s} {value:9.4f}   limit {limits[key]}")
    print(
        f"  boundary edges {topo['orig']['boundary_edges']} -> {topo['cand']['boundary_edges']}, "
        f"components {topo['orig']['components']} -> {topo['cand']['components']}"
    )
    print("GATE", "FAIL " + ",".join(failed) if failed else "PASS", "->", args.out / "report.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
