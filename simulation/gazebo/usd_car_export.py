"""Visibility culling, per-link quadric decimation, and OBJ+MTL / GLB export for USD car meshes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])


def visible_part_indices(
    full_mesh: trimesh.Trimesh, face_part_id: np.ndarray, views_n: int = 90
) -> set[int]:
    """Return indices of whole USD parts struck by at least one exterior orthographic ray."""
    inter = RayMeshIntersector(full_mesh)
    i = np.arange(views_n) + 0.5
    polar, azim = np.arccos(1.0 - 2.0 * i / views_n), np.pi * (1.0 + 5.0**0.5) * i
    sphere = np.c_[np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)]
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    centre = full_mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(full_mesh.extents) / 2.0 * 1.05)
    span = np.linspace(-radius, radius, 512)
    u, v = np.meshgrid(span, -span)
    uv_u, uv_v = u.reshape(-1, 1), v.reshape(-1, 1)
    seen: set[int] = set()
    for view in np.vstack([axes, sphere]):
        up = np.array([0.0, 0.0, 1.0]) if abs(view[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        right = np.cross(up, view)
        right /= np.linalg.norm(right)
        up = np.cross(view, right)
        orig = centre + view * 2.0 * radius + uv_u * right + uv_v * up
        tri = inter.intersects_first(orig, np.tile(-view, (len(orig), 1)))
        hits = tri[tri >= 0]
        if len(hits):
            seen.update(np.unique(face_part_id[hits]).tolist())
    return seen


def decimate_link(parts: list[dict], target_tris: int) -> tuple[trimesh.Trimesh, list[str]]:
    """Run one quadric edge collapse on a link and transfer each face's nearest USD material."""
    vs, fs, mat_ids = [], [], []
    v_off = 0
    for p in parts:
        vs.append(p["verts_gz"])
        fs.append(p["faces"] + v_off)
        mat_ids.extend([p["mat"]] * len(p["faces"]))
        v_off += len(p["verts_gz"])
    src = trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False)
    keep = src.area_faces > 1e-14
    src_faces = src.faces[keep]
    src_mats = [m for m, k in zip(mat_ids, keep, strict=True) if k]
    src = trimesh.Trimesh(src.vertices, src_faces, process=False)
    src_centers = src.triangles_center
    src.merge_vertices()

    if target_tris < len(src.faces) and len(src.faces) > 12:
        ms = ml.MeshSet()
        ms.add_mesh(
            ml.Mesh(
                vertex_matrix=np.ascontiguousarray(src.vertices, float),
                face_matrix=np.ascontiguousarray(src.faces, np.int32),
            )
        )
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=int(max(8, target_tris)),
            qualitythr=0.3,
            optimalplacement=True,
            autoclean=True,
        )
        out = ms.current_mesh()
        dec = trimesh.Trimesh(
            np.asarray(out.vertex_matrix(), float),
            np.asarray(out.face_matrix(), np.int64),
            process=False,
        )
        dec.merge_vertices()
        dkeep = dec.area_faces > 1e-14
        if not dkeep.all():
            dec = trimesh.Trimesh(dec.vertices, dec.faces[dkeep], process=False)
    else:
        dec = src

    _, nearest = cKDTree(src_centers).query(dec.triangles_center, k=1, workers=-1)
    dec_mats = [src_mats[int(idx)] for idx in nearest]
    return dec, dec_mats


def write_mtl(path: Path, mats: dict[str, dict], used_keys: set[str]) -> None:
    lines = ["# Payload car materials extracted from USD\n"]
    for key in sorted(used_keys):
        m = mats[key]
        r, g, b = m["color"]
        ns = max(4.0, (1.0 - m["roughness"]) * 200.0)
        ks = 0.4 + 0.5 * m["metallic"]
        lines.append(
            f"newmtl {m['name']}\n"
            f"Ka {r * 0.25:.5f} {g * 0.25:.5f} {b * 0.25:.5f}\n"
            f"Kd {r:.5f} {g:.5f} {b:.5f}\n"
            f"Ks {ks:.5f} {ks:.5f} {ks:.5f}\n"
            f"Ns {ns:.2f}\n"
            f"d 1.0\n\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_link_obj(
    path: Path,
    mtl_name: str,
    mesh: trimesh.Trimesh,
    face_mats: list[str],
    mats: dict[str, dict],
    offset: np.ndarray,
) -> None:
    verts = mesh.vertices - offset
    lines = [
        f"# Payload car link mesh ({len(mesh.faces)} tris, metres, Gazebo Z-up)\n",
        f"mtllib {mtl_name}\n",
    ]
    lines.extend(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
    by_mat: dict[str, list[np.ndarray]] = defaultdict(list)
    for face, mkey in zip(mesh.faces, face_mats, strict=True):
        by_mat[mkey].append(face)
    for mkey in sorted(by_mat):
        lines.append(f"usemtl {mats[mkey]['name']}\n")
        for f in by_mat[mkey]:
            lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")
    path.write_text("".join(lines), encoding="utf-8")


def export_scene_glb(
    out_path: Path,
    link_meshes: dict[str, tuple[trimesh.Trimesh, list[str]]],
    mats: dict[str, dict],
) -> None:
    scene = trimesh.Scene()
    scene.graph.update(frame_to="model", frame_from=scene.graph.base_frame, matrix=Z_UP_TO_Y_UP)
    for link_name, (mesh, face_mats) in sorted(link_meshes.items()):
        by_mat: dict[str, list[int]] = defaultdict(list)
        for fi, mkey in enumerate(face_mats):
            by_mat[mkey].append(fi)
        for mkey, fidx in sorted(by_mat.items()):
            sub = mesh.submesh([fidx], append=True)
            sub.merge_vertices()
            m = mats[mkey]
            rgba = [int(round(c * 255)) for c in m["color"]] + [255]
            sub.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    name=m["name"],
                    baseColorFactor=rgba,
                    metallicFactor=float(m["metallic"]),
                    roughnessFactor=float(m["roughness"]),
                )
            )
            node_name = f"{link_name}__{m['name']}"
            scene.add_geometry(
                sub, node_name=node_name, geom_name=node_name, parent_node_name="model"
            )
    out_path.write_bytes(scene.export(file_type="glb"))
