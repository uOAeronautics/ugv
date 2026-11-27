"""Convert the payload car USDZ CAD into Gazebo-ready meshes and measured dimensions.

Reads a USD/USDZ assembly, applies metersPerUnit and the geometric Z-up / +Y-forward ->
Gazebo +X-forward, +Y-left, +Z-up frame transform centred on the four wheel hubs at ground
contact (z = 0 at the lowest roller vertex), exports per-material coloured meshes (per-link
OBJ+MTL and multi-node GLB), decimates for simulation using camera-visibility culling plus
per-link quadric edge collapse, and writes `simulation/gazebo/payload_dimensions.json`.

Usage (isolated deps, no torch):
    uv run --no-project --python 3.12 --with usd-core --with trimesh --with numpy \
        --with scipy --with embreex --with pymeshlab --with pillow \
        python -I simulation/gazebo/convert_payload_usd.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from usd_car_export import (  # noqa: E402
    decimate_link,
    export_scene_glb,
    visible_part_indices,
    write_link_obj,
    write_mtl,
)
from usd_car_geometry import (  # noqa: E402
    WHEEL_LINKS,
    classify_part_link,
    compute_frame_from_hubs,
    measure_car_dimensions,
    validate_wheel_symmetry,
)

DEFAULT_USDZ = HERE / "assets/car/payload_assembly/payload_assembly.usdz"
DEFAULT_OUT_DIR = HERE / "assets/car/payload_assembly"
DEFAULT_DIMS = HERE / "payload_dimensions.json"
NAMED_PARTS = (
    "M12_lens__1_",
    "camera_PCB__1_",
    "BNO055__1_",
    "fairing",
    "_2S_LiPo_72x35x17__envelope_",
)


def _read_materials(stage) -> dict[str, dict]:
    mats: dict[str, dict] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        surf = UsdShade.Material(prim).ComputeSurfaceSource()
        color, metallic, roughness = [0.7, 0.7, 0.7], 0.0, 0.5
        if surf and surf[0]:
            sh = surf[0]
            for name in ("diffuseColor", "baseColor", "displayColor"):
                inp = sh.GetInput(name)
                if inp and inp.Get() is not None:
                    v = inp.Get()
                    color = [float(v[0]), float(v[1]), float(v[2])]
                    break
            mi = sh.GetInput("metallic")
            if mi and mi.Get() is not None:
                metallic = float(mi.Get())
            ri = sh.GetInput("roughness")
            if ri and ri.Get() is not None:
                roughness = float(ri.Get())
        mats[str(prim.GetPath())] = {
            "name": prim.GetName(),
            "color": color,
            "metallic": metallic,
            "roughness": roughness,
        }
    mats.setdefault(
        "default", {"name": "default", "color": [0.7, 0.7, 0.7], "metallic": 0.0, "roughness": 0.5}
    )
    return mats


def _triangulate(fvc: np.ndarray, fvi: np.ndarray) -> np.ndarray:
    if np.all(fvc == 3):
        return fvi.reshape(-1, 3)
    tris = []
    idx = 0
    for c in fvc:
        for k in range(1, int(c) - 1):
            tris.append((int(fvi[idx]), int(fvi[idx + k]), int(fvi[idx + k + 1])))
        idx += int(c)
    return np.asarray(tris, dtype=np.int64)


def load_usd_parts(usdz_path: Path) -> tuple[list[dict], dict[str, dict], dict]:
    """Read all USD mesh prims in world stage coordinates plus materials and stage metadata."""
    stage = Usd.Stage.Open(str(usdz_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usdz_path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    declared_up = str(UsdGeom.GetStageUpAxis(stage))
    mats = _read_materials(stage)
    xcache = UsdGeom.XformCache()
    parts: list[dict] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        um = UsdGeom.Mesh(prim)
        pts, fvc, fvi = (
            um.GetPointsAttr().Get(),
            um.GetFaceVertexCountsAttr().Get(),
            um.GetFaceVertexIndicesAttr().Get(),
        )
        if not pts or not fvc or not fvi:
            continue
        m = xcache.GetLocalToWorldTransform(prim)
        mat4 = np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
        wpts = np.asarray(pts, dtype=np.float64) @ mat4[:3, :3] + mat4[3, :3]
        faces = _triangulate(np.asarray(fvc, dtype=np.int64), np.asarray(fvi, dtype=np.int64))
        if len(faces) == 0:
            continue
        if np.linalg.det(mat4[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]
        if um.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, [0, 2, 1]]
        bound_mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        mat_path = str(bound_mat.GetPath()) if bound_mat else "default"
        parts.append(
            {
                "name": prim.GetName(),
                "verts_stage": wpts,
                "faces": faces,
                "mat": mat_path if mat_path in mats else "default",
            }
        )
    meta = {
        "meters_per_unit": mpu,
        "declared_up_axis": declared_up,
        "sha256": hashlib.sha256(usdz_path.read_bytes()).hexdigest(),
        "raw_meshes": len(parts),
        "raw_triangles": int(sum(len(p["faces"]) for p in parts)),
    }
    return parts, mats, meta


def _transform_array(pts: np.ndarray, frame) -> np.ndarray:
    cx, cy = frame.origin_xy_stage
    s = frame.meters_per_unit
    return np.column_stack(
        [(pts[:, 1] - cy) * s, -(pts[:, 0] - cx) * s, (pts[:, 2] - frame.z_ground_stage) * s]
    )


def convert(
    usdz_path: Path = DEFAULT_USDZ,
    out_dir: Path = DEFAULT_OUT_DIR,
    dims_path: Path = DEFAULT_DIMS,
    chassis_tris: int = 24000,
    wheel_tris: int = 7000,
    gate_stl_dir: Path | None = None,
    full_glb_path: Path | None = None,
) -> dict:
    parts, mats, meta = load_usd_parts(usdz_path)
    hub_aabbs = [
        (tuple(p["verts_stage"].min(axis=0)), tuple(p["verts_stage"].max(axis=0)))
        for p in parts
        if "Mecanum_hub" in p["name"]
    ]
    roller_z_min = min(
        float(p["verts_stage"][:, 2].min()) for p in parts if "Mecanum_roller" in p["name"]
    )
    frame, hubs_gz = compute_frame_from_hubs(
        hub_aabbs, roller_z_min, meters_per_unit=meta["meters_per_unit"]
    )
    validate_wheel_symmetry(hubs_gz)

    all_v, all_f, face_pid = [], [], []
    v_off = 0
    named_parts: dict[str, dict] = {}
    roller_radii, wheel_widths, roller_widths = [], [], []
    non_wheel_min_z = float("inf")
    for idx, p in enumerate(parts):
        v = _transform_array(p["verts_stage"], frame)
        p["verts_gz"] = v
        bmin, bmax = v.min(axis=0), v.max(axis=0)
        ctr = (bmin + bmax) / 2.0
        p["link"] = classify_part_link(p["name"], tuple(ctr))
        if p["name"] in NAMED_PARTS:
            named_parts[p["name"]] = {
                "min_m": bmin.tolist(),
                "max_m": bmax.tolist(),
                "center_m": ctr.tolist(),
                "size_m": (bmax - bmin).tolist(),
            }
        if p["link"] == "chassis":
            non_wheel_min_z = min(non_wheel_min_z, float(bmin[2]))
        elif "Mecanum_roller" in p["name"]:
            hx, _, hz = hubs_gz[p["link"]]
            roller_radii.append(float(np.hypot(v[:, 0] - hx, v[:, 2] - hz).max()))
            roller_widths.append(float(bmax[1] - bmin[1]))
        all_v.append(v)
        all_f.append(p["faces"] + v_off)
        face_pid.append(np.full(len(p["faces"]), idx, dtype=np.int32))
        v_off += len(v)

    for wl in WHEEL_LINKS:
        ys = np.concatenate([p["verts_gz"][:, 1] for p in parts if p["link"] == wl])
        wheel_widths.append(float(ys.max() - ys.min()))

    full_mesh = trimesh.Trimesh(np.vstack(all_v), np.vstack(all_f), process=False)
    seen = visible_part_indices(full_mesh, np.concatenate(face_pid))

    by_link: dict[str, list[dict]] = defaultdict(list)
    for pid in sorted(seen):
        by_link[parts[pid]["link"]].append(parts[pid])

    out_dir.mkdir(parents=True, exist_ok=True)
    dec_links: dict[str, tuple[trimesh.Trimesh, list[str]]] = {}
    used_mats: set[str] = set()
    for link_name in ("chassis", *WHEEL_LINKS):
        target = chassis_tris if link_name == "chassis" else wheel_tris
        dec, dmats = decimate_link(by_link[link_name], target)
        dec_links[link_name] = (dec, dmats)
        used_mats.update(dmats)

    mtl_name = "payload_car.mtl"
    write_mtl(out_dir / mtl_name, mats, used_mats)
    for link_name, (dec, dmats) in dec_links.items():
        offset = np.zeros(3) if link_name == "chassis" else np.asarray(hubs_gz[link_name], float)
        write_link_obj(out_dir / f"{link_name}.obj", mtl_name, dec, dmats, mats, offset)
    export_scene_glb(out_dir / "payload_car.glb", dec_links, mats)

    if full_glb_path is not None:
        full_links: dict[str, tuple[trimesh.Trimesh, list[str]]] = {}
        for link_name in ("chassis", *WHEEL_LINKS):
            lparts = [p for p in parts if p["link"] == link_name]
            vs, fs, fm = [], [], []
            vo = 0
            for p in lparts:
                vs.append(p["verts_gz"])
                fs.append(p["faces"] + vo)
                fm.extend([p["mat"]] * len(p["faces"]))
                vo += len(p["verts_gz"])
            full_links[link_name] = (
                trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False),
                fm,
            )
        export_scene_glb(full_glb_path, full_links, mats)

    if gate_stl_dir is not None:
        gate_stl_dir.mkdir(parents=True, exist_ok=True)
        full_mesh.export(gate_stl_dir / "orig.stl")
        trimesh.util.concatenate([m for m, _ in dec_links.values()]).export(
            gate_stl_dir / "cand.stl"
        )

    dec_tris = int(sum(len(m.faces) for m, _ in dec_links.values()))
    meta.update(
        origin_xy_stage=frame.origin_xy_stage,
        z_ground_stage=frame.z_ground_stage,
        decimated_triangles=dec_tris,
    )
    dims = measure_car_dimensions(
        (tuple(full_mesh.bounds[0]), tuple(full_mesh.bounds[1])),
        hubs_gz,
        wheel_outer_radius_m=max(roller_radii),
        wheel_width_m=float(np.mean(wheel_widths)),
        roller_width_m=float(np.mean(roller_widths)),
        ground_clearance_non_wheel_m=non_wheel_min_z,
        named_parts_gz=named_parts,
        source_meta=meta,
    )
    dims_path.write_text(json.dumps(dims, indent=2) + "\n", encoding="utf-8")
    return dims


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_USDZ)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dims-out", type=Path, default=DEFAULT_DIMS)
    ap.add_argument("--chassis-tris", type=int, default=24000)
    ap.add_argument("--wheel-tris", type=int, default=7000)
    ap.add_argument(
        "--gate-stl-dir",
        type=Path,
        default=None,
        help="Also write orig.stl + cand.stl for meshgate",
    )
    ap.add_argument(
        "--full-glb", type=Path, default=None, help="Optional full-resolution GLB export path"
    )
    args = ap.parse_args()
    dims = convert(
        args.input,
        args.out_dir,
        args.dims_out,
        args.chassis_tris,
        args.wheel_tris,
        args.gate_stl_dir,
        args.full_glb,
    )
    print(
        f"Converted {dims['source']['raw_triangles']} -> {dims['source']['decimated_triangles']} tris; "
        f"L={dims['overall_length_m'] * 1e3:.2f}mm W={dims['overall_width_m'] * 1e3:.2f}mm "
        f"H={dims['overall_height_m'] * 1e3:.2f}mm; wheelbase={dims['wheelbase_m'] * 1e3:.2f}mm "
        f"track={dims['wheel_track_m'] * 1e3:.2f}mm wheel_d={dims['wheel_diameter_m'] * 1e3:.2f}mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
