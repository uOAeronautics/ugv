"""Pack a directory of per-link STL files into one game-ready GLB.

STL stores three loose vertices per triangle and no normals. This welds them into an
indexed mesh, splits vertices only across creases sharper than --crease degrees so curved
skin shades smooth and hard edges stay hard, and writes one named node per link. The STLs
share the model frame, so every node keeps an identity transform. glTF is +Y up; the model
frame is +Z up, so the root node carries that rotation and the vertex data stays untouched.

    uv run --no-project --python 3.12 --with trimesh --with scipy --with networkx \
        python tools/mesh_opt/export_glb.py IN_DIR OUT.glb [--crease 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def smooth_part(path: Path, crease_deg: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    smooth = trimesh.graph.smooth_shade(mesh, angle=np.radians(crease_deg))
    smooth.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="airframe",
            baseColorFactor=[225, 225, 220, 255],
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )
    )
    _ = smooth.vertex_normals  # computed now so the exporter writes them
    return smooth


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("in_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--crease", type=float, default=30.0, help="edges sharper than this stay hard")
    args = ap.parse_args()
    files = sorted(args.in_dir.glob("*.stl"))
    if not files:
        raise SystemExit(f"no STL under {args.in_dir}")
    scene = trimesh.Scene()
    scene.graph.update(frame_to="model", frame_from=scene.graph.base_frame, matrix=Z_UP_TO_Y_UP)
    tris = verts = 0
    for path in files:
        part = smooth_part(path, args.crease)
        scene.add_geometry(part, node_name=path.stem, geom_name=path.stem, parent_node_name="model")
        tris, verts = tris + len(part.faces), verts + len(part.vertices)
        print(f"{path.stem:22s} tris {len(part.faces):6d}  verts {len(part.vertices):6d}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(scene.export(file_type="glb"))
    print(
        f"wrote {args.out}  {args.out.stat().st_size / 1e6:.2f} MB  {tris} tris  {verts} verts  {len(files)} nodes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
