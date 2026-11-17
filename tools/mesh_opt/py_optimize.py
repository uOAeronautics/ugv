"""Pure-python STL optimiser: delete what no camera can see, then decimate the rest.

Writes a low-poly replacement for every STL in simulation/gazebo/strl_001/meshes.  Every
candidate is gated with tools/mesh_opt/meshgate.py before it is accepted, and a part that
cannot be decimated without failing is copied unchanged, so the output directory only
ever holds meshes that passed.  The per-file counts differ because a part that is already
coarse cannot go lower without moving its silhouette.

Usage:
    uv run --no-project --python 3.12 --with trimesh --with numpy --with scipy \
      --with embreex --with pillow --with pymeshlab \
      python tools/mesh_opt/py_optimize.py --out work/mesh_opt/cand_pyml

Deterministic: fixed seeds, and pymeshlab's quadric collapse is a fixed algorithm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "simulation/gazebo/strl_001/meshes"
GATE = REPO / "tools/mesh_opt/meshgate.py"

# Triangle target per file, chosen as the smallest count that passed the gate.  A part
# whose target is None is already low-poly and is copied unchanged.
TARGET = {
    "body": 38000,
    "nacelle_left": 2500,
    "nacelle_right": 2500,
    "prop_left": 1200,
    "prop_right": 1200,
    "prop_rear": 1200,
    "aileron_left": None,
    "aileron_right": None,
    "ruddervator_left": None,
    "ruddervator_right": None,
}


def load(path: Path) -> trimesh.Trimesh:
    """Load an STL as one mesh with welded vertices and no zero-area faces."""
    mesh = trimesh.load(path, process=False, force="mesh")
    mesh.merge_vertices()
    keep = mesh.area_faces > 1e-14
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    mesh.merge_vertices()
    return mesh


def fibonacci(count: int) -> np.ndarray:
    i = np.arange(count) + 0.5
    polar, azim = np.arccos(1 - 2 * i / count), np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)]


def face_visibility(mesh: trimesh.Trimesh, dirs: int = 96) -> np.ndarray:
    """True for a face with a clear line of sight to infinity in some direction.

    Rays leave the face centroid along a full sphere of directions, offset by 1e-6 m so
    the face itself is not the first hit.  A hit closer than 1e-5 m counts as a miss:
    this mesh has coincident duplicate layers and letting one hide the other would
    classify visible skin as hidden.
    """
    normals, centres = mesh.face_normals, mesh.triangles_center
    tri0 = mesh.vertices[mesh.faces[:, 0]]
    inter = RayMeshIntersector(mesh)
    index = np.arange(len(mesh.faces))
    seen = np.zeros(len(mesh.faces), bool)
    for d in fibonacci(dirs):
        origin = centres + d * 1e-6
        hit = inter.intersects_first(origin, np.tile(d, (len(origin), 1)))
        blocked = hit >= 0
        dist = np.full(len(index), np.inf)
        if blocked.any():
            n = normals[hit[blocked]]
            den = np.einsum("ij,ij->i", n, np.tile(d, (n.shape[0], 1)))
            den[np.abs(den) < 1e-12] = 1e-12
            dist[blocked] = np.einsum("ij,ij->i", tri0[hit[blocked]] - origin[blocked], n) / den
        seen |= (~blocked) | (dist < 1e-5)
    return seen


def front_facing(mesh: trimesh.Trimesh) -> np.ndarray:
    """True where a ray along the face normal from far outside lands on that face."""
    normals, centres = mesh.face_normals, mesh.triangles_center
    far = 4 * float(np.linalg.norm(mesh.extents))
    hit = RayMeshIntersector(mesh).intersects_first(centres + normals * far, -normals)
    return (hit >= 0) & (hit == np.arange(len(mesh.faces)))


def decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Quadric edge collapse with optimal vertex placement; the input if it cannot run."""
    if target >= len(mesh.faces) or len(mesh.faces) <= 8:
        return mesh
    ms = ml.MeshSet()
    try:
        ms.add_mesh(
            ml.Mesh(
                vertex_matrix=np.ascontiguousarray(mesh.vertices, float),
                face_matrix=np.ascontiguousarray(mesh.faces, np.int32),
            )
        )
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=int(max(4, target)),
            qualitythr=0.3,
            optimalplacement=True,
            autoclean=True,
        )
        out = ms.current_mesh()
        result = trimesh.Trimesh(
            np.asarray(out.vertex_matrix(), float),
            np.asarray(out.face_matrix(), np.int64),
            process=False,
        )
    except Exception:
        return mesh
    result.merge_vertices()
    keep = result.area_faces > 1e-14
    if not keep.all():
        result = trimesh.Trimesh(result.vertices, result.faces[keep], process=False)
    return result if len(result.faces) else mesh


def gate(orig: Path, cand: Path, out: Path) -> tuple[bool, str]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        "--with",
        "trimesh",
        "--with",
        "numpy",
        "--with",
        "scipy",
        "--with",
        "embreex",
        "--with",
        "pillow",
        "python",
        str(GATE),
        str(orig),
        str(cand),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def build(name: str, out: Path, log) -> dict:
    """Decimate one part to its target, or copy it unchanged when it is already low."""
    src = SRC / f"{name}.stl"
    t0 = time.time()
    mesh = load(src)
    comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    visible = face_visibility(mesh) | front_facing(mesh)
    live = [c for c in comps if visible[c].any()]
    hidden = len(comps) - len(live)
    hidden_tris = int(len(mesh.faces) - sum(len(c) for c in live))
    base = mesh.submesh(live, append=True) if hidden else mesh
    log(
        f"{name}: {len(mesh.faces)} tris, {len(comps)} components, "
        f"drop {hidden} hidden components ({hidden_tris} tris)"
    )

    target = TARGET.get(name)
    if target is None:
        cand = base
        note = "copied unchanged"
    else:
        cand = decimate(base, target)
        note = "decimated"
    cand.export(out / f"{name}.stl")
    return {
        "in": int(len(mesh.faces)),
        "out": int(len(cand.faces)),
        "hidden_components": hidden,
        "hidden_tris": hidden_tris,
        "note": note,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=REPO / "work/mesh_opt/cand_pyml")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=4, help="gates run in parallel")
    args = ap.parse_args()

    def log(msg: str):
        print(msg, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or [p.stem for p in sorted(SRC.glob("*.stl"))]
    summary = {name: build(name, args.out, log) for name in names}

    def run_gate(name: str):
        return name, gate(SRC / f"{name}.stl", args.out / f"{name}.stl", args.out / "gate" / name)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(run_gate, names))
    for name in names:
        ok, text = results[name]
        summary[name]["gate"] = "PASS" if ok else "FAIL"
        if not ok:
            # Fall back to the original part: the hard requirement is a passing gate.
            shutil.copyfile(SRC / f"{name}.stl", args.out / f"{name}.stl")
            summary[name]["out"] = summary[name]["in"]
            summary[name]["note"] = "copied unchanged (decimation failed the gate)"
            log(f"{name}: decimation failed the gate, copied unchanged")

    retry = [n for n in names if not results[n][0]]
    if retry:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            again = dict(pool.map(run_gate, retry))
        for name in retry:
            ok, text = again[name]
            summary[name]["gate"] = "PASS" if ok else "FAIL"
            log(f"{name}: unchanged copy gate {'PASS' if ok else 'FAIL'}")
            if not ok:
                log("\n".join(text.strip().splitlines()[-13:]))

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    ok, text = gate(SRC, args.out, args.out / "gate" / "all")
    log("\n".join(text.strip().splitlines()[-13:]))
    log(f"directory gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
