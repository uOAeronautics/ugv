"""Build simulation/gazebo/payload_prior.json from the payload car USDZ CAD + datasheet priors.

Usage (isolated CAD deps, no torch):
    uv run --no-project --python 3.12 --with usd-core --with trimesh --with numpy --with scipy \
        python -I simulation/gazebo/build_payload_prior.py
"""

from __future__ import annotations

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

from usd_car_geometry import WHEEL_LINKS, classify_part_link, compute_frame_from_hubs  # noqa: E402

DEFAULT_USDZ = HERE / "assets/car/payload_assembly/payload_assembly.usdz"
DEFAULT_OUT = HERE / "payload_prior.json"


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


def _classify_component(
    name: str, mat: str, link: str, ctr_m: np.ndarray, vol_cm3: float
) -> tuple[str, str]:
    x, y, z = ctr_m * 1e3
    if link != "chassis":
        if "Mecanum_hub" in name:
            return (link, "wheel_hub_pla")
        if "Mecanum_roller" in name:
            return (link, "wheel_roller_tpu")
        if "Roller_axle" in name or "M2x8" in name:
            return (link, "wheel_steel")
        if "Roller_sleeve" in name:
            return (link, "wheel_sleeve_plastic")
        return (link, "wheel_other")

    # Purchased envelope assemblies (mass from datasheet range prior; position/shape from CAD).
    if name.startswith("_2S_LiPo"):
        return ("chassis", "lipo_2s_envelope")
    if name.startswith(("BNO055", "IMU_PCB", "STEMMA_QT")):
        return ("chassis", "bno055_module")
    if name.startswith(("camera_PCB", "M12_holder", "M12_lens", "FPC_connector", "CSI_22_pin")):
        return ("chassis", "camera_module")
    if (
        name.startswith("Body")
        and abs(abs(x) - 35.4) < 6.0
        and 22.0 <= abs(y) <= 58.0
        and 13.0 <= z <= 27.0
    ):
        corner = f"{'front' if x >= 0 else 'rear'}_{'left' if y >= 0 else 'right'}"
        return ("chassis", f"n20_motor_{corner}")
    if name.startswith(("PCB_56x41", "RJ45", "HDMI", "USB_A", "USB_C", "GPIO_2x20", "riser")):
        return ("chassis", "sbc_56x41_module")
    if name.startswith("Body") and z > 25.0 and x > 0.0 and abs(y) < 35.0:
        return ("chassis", "sbc_56x41_module")

    # CAD-volume structural, PCB, wiring, and fastener parts.
    if name.startswith(("fairing", "battery_tray", "bracket")):
        return ("chassis", "printed_pla_petg")
    if name.startswith("velcro_strap"):
        return ("chassis", "velcro_straps")
    if name.startswith(("PHR_", "SHR_", "XT30_housing", "XT30__battery", "M2_5_nylon", "forward_")):
        return ("chassis", "plastic_connectors")
    if "wire" in name or "battery_lead" in name or name.startswith("FPC_ribbon"):
        return ("chassis", "copper_wires")
    if name.startswith("Body") and "Opaque80124105" in mat:
        return ("chassis", "baseboard_fr4_pcb")
    if name.startswith(("M2_5x8", "M2_5_nut", "M2x6")):
        return ("chassis", "steel_fasteners")
    if name.startswith("Body") and "SteelSatin" in mat:
        if (abs(abs(x) - 46.0) < 1.0 and abs(abs(y) - 46.0) < 1.0) or (
            abs(abs(y) - 39.0) < 1.0 and abs(vol_cm3 - 0.0475) < 0.01
        ):
            return ("chassis", "steel_fasteners")
        return ("chassis", "baseboard_headers_smd")
    if name.startswith("XT30_solder") or (name.startswith("Body") and "BrassPolished" in mat):
        return ("chassis", "brass_inserts")
    raise RuntimeError(
        f"Unclassified CAD prim: name={name!r} mat={mat!r} link={link!r} ctr={ctr_m}"
    )


PART_MASS_SPEC: dict[str, tuple[str, str]] = {
    "printed_pla_petg": ("cad_volume_density", "pla_petg_fdm"),
    "baseboard_fr4_pcb": ("cad_volume_density", "fr4_pcb"),
    "baseboard_headers_smd": ("cad_volume_density", "baseboard_headers_smd"),
    "steel_fasteners": ("cad_volume_density", "steel_fasteners_axles"),
    "brass_inserts": ("cad_volume_density", "brass_inserts"),
    "copper_wires": ("cad_volume_density", "copper_wire_insulated"),
    "plastic_connectors": ("cad_volume_density", "plastic_connectors_sleeves"),
    "velcro_straps": ("cad_volume_density", "velcro_straps"),
    "wheel_hub_pla": ("cad_volume_density", "pla_petg_fdm"),
    "wheel_roller_tpu": ("cad_volume_density", "tpu_roller_fdm"),
    "wheel_sleeve_plastic": ("cad_volume_density", "plastic_connectors_sleeves"),
    "wheel_steel": ("cad_volume_density", "steel_fasteners_axles"),
    "lipo_2s_envelope": ("purchased_envelope", "lipo_2s_72x35x17"),
    "n20_motor_front_left": ("purchased_envelope", "n20_gearmotor_each"),
    "n20_motor_front_right": ("purchased_envelope", "n20_gearmotor_each"),
    "n20_motor_rear_left": ("purchased_envelope", "n20_gearmotor_each"),
    "n20_motor_rear_right": ("purchased_envelope", "n20_gearmotor_each"),
    "bno055_module": ("purchased_envelope", "bno055_breakout"),
    "camera_module": ("purchased_envelope", "camera_module_40x40_m12"),
    "sbc_56x41_module": ("purchased_envelope", "sbc_56x41"),
}


def extract_cad_parts(usdz_path: Path) -> tuple[list[dict], dict]:
    stage = Usd.Stage.Open(str(usdz_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usdz_path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    mats = {str(p.GetPath()): p.GetName() for p in stage.Traverse() if p.IsA(UsdShade.Material)}
    xcache = UsdGeom.XformCache()

    raw = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        um = UsdGeom.Mesh(prim)
        pts = um.GetPointsAttr().Get()
        fvc = um.GetFaceVertexCountsAttr().Get()
        fvi = um.GetFaceVertexIndicesAttr().Get()
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
        raw.append(
            {
                "name": prim.GetName(),
                "verts_stage": wpts,
                "faces": faces,
                "mat": mats.get(mat_path, "default"),
            }
        )

    hub_aabbs = [
        (tuple(p["verts_stage"].min(axis=0)), tuple(p["verts_stage"].max(axis=0)))
        for p in raw
        if "Mecanum_hub" in p["name"]
    ]
    roller_z_min = min(
        float(p["verts_stage"][:, 2].min()) for p in raw if "Mecanum_roller" in p["name"]
    )
    frame, hubs_gz = compute_frame_from_hubs(hub_aabbs, roller_z_min, meters_per_unit=mpu)
    cx, cy = frame.origin_xy_stage
    s = frame.meters_per_unit

    grouped: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    roller_rings: list[float] = []
    roller_radii: list[float] = []
    roller_lengths: list[float] = []
    roller_hands: dict[str, list[int]] = defaultdict(list)

    total_wt = 0
    total_hull = 0
    for p in raw:
        pts = p["verts_stage"]
        v = np.column_stack(
            [(pts[:, 1] - cy) * s, -(pts[:, 0] - cx) * s, (pts[:, 2] - frame.z_ground_stage) * s]
        )
        bmin, bmax = v.min(axis=0), v.max(axis=0)
        ctr = (bmin + bmax) / 2.0
        link = classify_part_link(p["name"], tuple(ctr))
        mesh = trimesh.Trimesh(v, p["faces"], process=True)
        wt = bool(mesh.is_watertight)
        if wt:
            total_wt += 1
            vol = abs(float(mesh.volume))
            com = np.asarray(mesh.center_mass, dtype=np.float64)
            mi = np.asarray(mesh.moment_inertia, dtype=np.float64)
            if mesh.volume < 0:
                mi = -mi
        else:
            total_hull += 1
            hull = mesh.convex_hull
            vol = abs(float(hull.volume))
            com = np.asarray(hull.center_mass, dtype=np.float64)
            mi = np.asarray(hull.moment_inertia, dtype=np.float64)
            if hull.volume < 0:
                mi = -mi

        if "Mecanum_roller" in p["name"] and link in WHEEL_LINKS:
            hx, _, hz = hubs_gz[link]
            vc = v - ctr
            _, vecs = np.linalg.eigh(vc.T @ vc / len(vc))
            axis = vecs[:, 2]
            if axis[1] < 0:
                axis = -axis
            proj = vc @ axis
            roller_lengths.append(float(proj.max() - proj.min()))
            perp = vc - np.outer(proj, axis)
            roller_radii.append(float(np.linalg.norm(perp, axis=1).max()))
            roller_rings.append(float(np.hypot(ctr[0] - hx, ctr[2] - hz)))
            ang = float(np.arctan2(ctr[0] - hx, ctr[2] - hz))
            tang = np.array([np.cos(ang), 0.0, -np.sin(ang)])
            roller_hands[link].append(int(np.sign(float(np.dot(axis, tang)) * float(axis[1]))))

        key = _classify_component(p["name"], p["mat"], link, ctr, vol * 1e6)
        grouped[key].append((vol, com, mi, wt, bmin, bmax))

    cad_parts: list[dict] = []
    for (link, cls), items in sorted(grouped.items()):
        mass_model, ref_key = PART_MASS_SPEC[cls]
        tot_vol = float(sum(x[0] for x in items))
        com = sum(x[0] * x[1] for x in items) / max(tot_vol, 1e-18)
        inertia_tot = np.zeros((3, 3), dtype=np.float64)
        for vol, c, mi, _, _, _ in items:
            d = c - com
            shift = float(np.dot(d, d)) * np.eye(3) - np.outer(d, d)
            inertia_tot += mi + vol * shift
        i_per_kg = inertia_tot / max(tot_vol, 1e-18)
        bmin = np.min([x[4] for x in items], axis=0)
        bmax = np.max([x[5] for x in items], axis=0)
        wt_count = sum(1 for x in items if x[3])
        hull_count = len(items) - wt_count
        method = "watertight" if hull_count == 0 else "watertight+convex_hull"
        entry: dict = {
            "id": f"{link}.{cls}",
            "link": link,
            "component": cls,
            "mass_model": mass_model,
            "geometry": {
                "mesh_count": len(items),
                "watertight_count": wt_count,
                "convex_hull_count": hull_count,
                "volume_method": method,
                "volume_m3": round(tot_vol, 12),
                "com_m": [round(float(x), 6) for x in com],
                "bbox_size_m": [round(float(x), 6) for x in (bmax - bmin)],
                "inertia_per_kg_m2": [
                    round(float(i_per_kg[0, 0]), 10),
                    round(float(i_per_kg[1, 1]), 10),
                    round(float(i_per_kg[2, 2]), 10),
                    round(float(i_per_kg[0, 1]), 10),
                    round(float(i_per_kg[0, 2]), 10),
                    round(float(i_per_kg[1, 2]), 10),
                ],
                "source": "cad-volume",
            },
        }
        if mass_model == "cad_volume_density":
            entry["material_class"] = ref_key
        else:
            entry["catalogue_part"] = ref_key
        cad_parts.append(entry)

    # Verify CAD roller handedness relative to mecanum_geometry.HANDEDNESS = (+1, -1, -1, +1).
    fl_hand = roller_hands["front_left_wheel"][0]
    cad_hand_sign = -1 if fl_hand < 0 else 1
    wheel_outer_r = float(np.mean(roller_rings) + np.mean(roller_radii))
    cad_summary = {
        "raw_meshes": len(raw),
        "watertight_meshes": total_wt,
        "convex_hull_fallback_meshes": total_hull,
        "wheel_roller_count_per_wheel": len(roller_hands["front_left_wheel"]),
        "wheel_roller_ring_radius_m": round(float(np.mean(roller_rings)), 6),
        "wheel_roller_cad_radius_m": round(float(np.mean(roller_radii)), 6),
        "wheel_roller_contact_radius_m": round(0.01995 - float(np.mean(roller_rings)), 6),
        "wheel_roller_length_m": round(float(np.mean(roller_lengths)), 6),
        "wheel_outer_radius_from_rollers_m": round(wheel_outer_r, 6),
        "cad_raw_mecanum_hand_sign": cad_hand_sign,
        "mecanum_hand_sign": 1,
        "source": "cad-volume",
        "notes": (
            "1241/1242 USD mesh prims are watertight; 1 tiny SMD pad (0.0005 cm^3 on sbc_56x41_module) "
            "uses convex-hull fallback. Raw USD places A hubs on FL/RR and B hubs on FR/RL "
            "(cad_raw_mecanum_hand_sign=-1, O-pattern on top / X-pattern on floor, yaw moment arm "
            "only |(track-wheelbase)/2| = 27.9 mm). Drivable prior uses standard top-view X-pattern "
            "(mecanum_hand_sign=+1, O-pattern on floor, yaw moment arm (track+wheelbase)/2 = 98.7 mm, "
            "matching mecanum_geometry.HANDEDNESS); verify wheel mounting on physical car."
        ),
    }
    return cad_parts, cad_summary


def build_prior(usdz_path: Path = DEFAULT_USDZ) -> dict:
    cad_parts, cad_summary = extract_cad_parts(usdz_path)
    sha256 = hashlib.sha256(usdz_path.read_bytes()).hexdigest()

    material_densities = {
        "pla_petg_fdm": {
            "dist": "uniform",
            "low": 450.0,
            "high": 1270.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": (
                "FDM PLA (1240 kg/m^3 solid) or PETG (1270 kg/m^3 solid) with unknown slicer infill "
                "(20% sparse infill+walls ~450 kg/m^3 up to 100% solid PETG 1270 kg/m^3)."
            ),
        },
        "tpu_roller_fdm": {
            "dist": "uniform",
            "low": 650.0,
            "high": 1250.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "TPU 95A rollers (35%-100% effective infill; solid TPU 1200-1250 kg/m^3).",
        },
        "fr4_pcb": {
            "dist": "uniform",
            "low": 1800.0,
            "high": 2100.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "FR4 glass-epoxy PCB substrate with copper planes (typical 1850-2100 kg/m^3).",
        },
        "baseboard_headers_smd": {
            "dist": "uniform",
            "low": 1200.0,
            "high": 2800.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": (
                "Baseboard pin headers, JST sockets, inductors, and ICs modeled as solid CAD blocks "
                "(hollow plastic housings + metal pins/silicone dies)."
            ),
        },
        "steel_fasteners_axles": {
            "dist": "uniform",
            "low": 7750.0,
            "high": 8050.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "Carbon or 304 stainless steel screws, nuts, and 2x23 mm roller axles.",
        },
        "brass_inserts": {
            "dist": "uniform",
            "low": 8400.0,
            "high": 8700.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "C360 brass heat-set inserts and XT30 solder cups.",
        },
        "copper_wire_insulated": {
            "dist": "uniform",
            "low": 2500.0,
            "high": 5500.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "Stranded copper wire + PVC/silicone insulation effective bulk density.",
        },
        "plastic_connectors_sleeves": {
            "dist": "uniform",
            "low": 900.0,
            "high": 1420.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "Nylon/PBT/POM connector housings, spacers, and roller bearing sleeves.",
        },
        "velcro_straps": {
            "dist": "uniform",
            "low": 150.0,
            "high": 450.0,
            "unit": "kg/m^3",
            "source": "datasheet-range-prior",
            "notes": "Woven nylon hook-and-loop battery straps (CAD models solid bounding envelope).",
        },
    }

    purchased_masses = {
        "lipo_2s_72x35x17": {
            "dist": "uniform",
            "low": 0.030,
            "high": 0.078,
            "unit": "kg",
            "label": "prior",
            "source": "datasheet-range-prior",
            "citation": (
                "Typical 2S 7.4V 450-1300 mAh LiPo pack datasheet mass range fitting inside the "
                "72x35x17 mm CAD tray envelope (e.g. GNB/Tattu/Turnigy 2S packs)."
            ),
        },
        "n20_gearmotor_each": {
            "dist": "uniform",
            "low": 0.009,
            "high": 0.012,
            "unit": "kg",
            "label": "prior",
            "source": "datasheet-range-prior",
            "citation": (
                "Typical Pololu / GA12-N20 12x10 mm micro metal gearmotor datasheet range "
                "(9.5-10.5 g nominal; 9-12 g across shaft lengths and gearbox ratios)."
            ),
        },
        "bno055_breakout": {
            "dist": "uniform",
            "low": 0.0025,
            "high": 0.0045,
            "unit": "kg",
            "label": "prior",
            "source": "datasheet-range-prior",
            "citation": "Typical Adafruit / SparkFun BNO055 STEMMA QT IMU breakout datasheet range (~3 g).",
        },
        "camera_module_40x40_m12": {
            "dist": "uniform",
            "low": 0.008,
            "high": 0.022,
            "unit": "kg",
            "label": "prior",
            "source": "datasheet-range-prior",
            "citation": "Typical 38x38/40x40 board camera PCB + M12 lens holder + M12 lens datasheet range.",
        },
        "sbc_56x41": {
            "dist": "uniform",
            "low": 0.015,
            "high": 0.042,
            "unit": "kg",
            "label": "prior",
            "source": "datasheet-range-prior",
            "citation": (
                "Typical compact 56x41 mm SBC / CM4+mini-carrier / Radxa-Zero / Pi-Zero-class "
                "assembly with RJ45/HDMI/USB/GPIO ports datasheet range."
            ),
        },
    }

    n20_variants = [
        {
            "id": "n20_50_1_6v",
            "gear_ratio": 50,
            "rated_voltage_v": 6.0,
            "no_load_rpm_low": 500.0,
            "no_load_rpm_high": 650.0,
            "stall_torque_kgf_cm_low": 0.35,
            "stall_torque_kgf_cm_high": 0.80,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 50:1 LP/MP/HP 6V & generic GA12-N20 6V 50:1 datasheet range.",
        },
        {
            "id": "n20_100_1_6v",
            "gear_ratio": 100,
            "rated_voltage_v": 6.0,
            "no_load_rpm_low": 250.0,
            "no_load_rpm_high": 330.0,
            "stall_torque_kgf_cm_low": 0.70,
            "stall_torque_kgf_cm_high": 1.50,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 100:1 LP/MP/HP 6V & generic GA12-N20 6V 100:1 datasheet range.",
        },
        {
            "id": "n20_150_1_6v",
            "gear_ratio": 150,
            "rated_voltage_v": 6.0,
            "no_load_rpm_low": 170.0,
            "no_load_rpm_high": 220.0,
            "stall_torque_kgf_cm_low": 1.00,
            "stall_torque_kgf_cm_high": 2.20,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 150:1 LP/MP/HP 6V & generic GA12-N20 6V 150:1 datasheet range.",
        },
        {
            "id": "n20_210_1_6v",
            "gear_ratio": 210,
            "rated_voltage_v": 6.0,
            "no_load_rpm_low": 120.0,
            "no_load_rpm_high": 160.0,
            "stall_torque_kgf_cm_low": 1.40,
            "stall_torque_kgf_cm_high": 2.80,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 210:1 / generic 200:1 6V GA12-N20 datasheet range.",
        },
        {
            "id": "n20_298_1_6v",
            "gear_ratio": 298,
            "rated_voltage_v": 6.0,
            "no_load_rpm_low": 80.0,
            "no_load_rpm_high": 115.0,
            "stall_torque_kgf_cm_low": 2.00,
            "stall_torque_kgf_cm_high": 4.00,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 298:1 / generic 300:1 6V GA12-N20 datasheet range.",
        },
        {
            "id": "n20_50_1_12v",
            "gear_ratio": 50,
            "rated_voltage_v": 12.0,
            "no_load_rpm_low": 500.0,
            "no_load_rpm_high": 650.0,
            "stall_torque_kgf_cm_low": 0.40,
            "stall_torque_kgf_cm_high": 0.90,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 50:1 12V & generic GA12-N20 12V 50:1 datasheet range (derated to 2S pack).",
        },
        {
            "id": "n20_100_1_12v",
            "gear_ratio": 100,
            "rated_voltage_v": 12.0,
            "no_load_rpm_low": 250.0,
            "no_load_rpm_high": 330.0,
            "stall_torque_kgf_cm_low": 0.80,
            "stall_torque_kgf_cm_high": 1.80,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 100:1 12V & generic GA12-N20 12V 100:1 datasheet range (derated to 2S pack).",
        },
        {
            "id": "n20_150_1_12v",
            "gear_ratio": 150,
            "rated_voltage_v": 12.0,
            "no_load_rpm_low": 170.0,
            "no_load_rpm_high": 220.0,
            "stall_torque_kgf_cm_low": 1.20,
            "stall_torque_kgf_cm_high": 2.50,
            "weight": 1.0,
            "source": "datasheet-range-prior",
            "citation": "Pololu 150:1 12V & generic GA12-N20 12V 150:1 datasheet range (derated to 2S pack).",
        },
    ]

    supply_and_drivetrain = {
        "battery_pack_voltage_v": {
            "dist": "uniform",
            "low": 7.0,
            "high": 8.2,
            "nominal_v": 7.4,
            "unit": "V",
            "source": "datasheet-range-prior",
            "notes": "2S LiPo operating voltage range around 7.4 V nominal.",
        },
        "supply_efficiency": {
            "dist": "uniform",
            "low": 0.65,
            "high": 0.95,
            "unit": "ratio",
            "source": "datasheet-range-prior",
            "notes": "Unknown H-bridge voltage drop (BJT vs MOSFET), wiring loss, and battery sag.",
        },
        "roller_mu": {
            "dist": "uniform",
            "low": 0.35,
            "high": 1.10,
            "unit": "1",
            "source": "datasheet-range-prior",
            "notes": (
                "TPU roller-to-floor friction coefficient; spans low-grip dusty indoor floor to tacky "
                "TPU and covers Freenove Gazebo reference (0.80-0.85)."
            ),
        },
        "roller_bearing_damping_nm_s_per_rad": {
            "dist": "log_uniform",
            "low": 1.0e-6,
            "high": 1.0e-4,
            "unit": "N*m*s/rad",
            "source": "datasheet-range-prior",
            "notes": (
                "Passive roller bearing viscous drag; covers Freenove mecanum_geometry default (2e-6) "
                "and pilot02 fitted value (3.19e-5)."
            ),
        },
        "roller_bearing_friction_nm": {
            "dist": "log_uniform",
            "low": 5.0e-7,
            "high": 2.0e-5,
            "unit": "N*m",
            "source": "datasheet-range-prior",
            "notes": "Passive roller bearing Coulomb friction; covers Freenove value (1e-6).",
        },
        "wheel_deadzone": {
            "dist": "uniform",
            "low": 0.0,
            "high": 0.20,
            "unit": "duty",
            "source": "datasheet-range-prior",
            "notes": (
                "Motor stiction/PWM deadband; covers Freenove measured L7 floor deadbands "
                "(20261113-181040-probe-axes: fwd 0.05, strafe 0.13, yaw 0.10-0.16; pilot02 0.0)."
            ),
        },
        "wheel_duty_exponent": {
            "dist": "uniform",
            "low": 0.60,
            "high": 1.15,
            "unit": "1",
            "source": "datasheet-range-prior",
            "notes": "Nonlinear PWM-to-effort exponent; covers Freenove pilot02 (0.596) through linear (1.0).",
        },
        "wheel_gain_spread": {
            "dist": "uniform",
            "low": 0.92,
            "high": 1.08,
            "unit": "ratio",
            "source": "datasheet-range-prior",
            "notes": "Per-wheel motor/gearbox manufacturing spread around nominal gain.",
        },
        "wheel_drive_tau_s": {
            "dist": "uniform",
            "low": 0.03,
            "high": 0.28,
            "unit": "s",
            "source": "datasheet-range-prior",
            "notes": "First-order drive response lag; covers Freenove pilot02 (0.218 s) and L7 onset (0.07-0.37 s).",
        },
        "wheel_brake_tau_s": {
            "dist": "uniform",
            "low": 0.02,
            "high": 0.18,
            "unit": "s",
            "source": "datasheet-range-prior",
            "notes": "Zero-duty brake take-up lag; covers Freenove pilot02 (0.12 s).",
        },
        "brake_damping_nm_s_per_rad": {
            "dist": "log_uniform",
            "low": 0.0005,
            "high": 0.005,
            "unit": "N*m*s/rad",
            "source": "datasheet-range-prior",
            "notes": "Shorted-terminal dynamic braking damping; covers Freenove (0.003).",
        },
        "brake_resistance_stall_fraction": {
            "dist": "uniform",
            "low": 0.01,
            "high": 0.08,
            "unit": "ratio",
            "source": "datasheet-range-prior",
            "notes": "Dry gearbox/brake drag as fraction of stall torque; covers Freenove (0.002/0.0785 = 0.025).",
        },
        "carrier_mass_kg": {
            "value": 0.0005,
            "unit": "kg",
            "source": "datasheet-range-prior",
            "notes": "Virtual radial-compliance carrier mass per wheel (subtracted from chassis to preserve total mass).",
        },
        "tire_radial_stiffness_n_per_m": {
            "dist": "uniform",
            "low": 1500.0,
            "high": 5000.0,
            "unit": "N/m",
            "source": "datasheet-range-prior",
            "notes": "Equivalent radial compliance stiffness for 4-wheel contact load sharing (Freenove uses 5000 N/m).",
        },
        "tire_radial_damping_ns_per_m": {
            "dist": "uniform",
            "low": 8.0,
            "high": 25.0,
            "unit": "N*s/m",
            "source": "datasheet-range-prior",
            "notes": "Equivalent radial compliance damping (Freenove uses 10 N*s/m).",
        },
        "tire_radial_travel_m": {
            "value": 0.0015,
            "unit": "m",
            "source": "datasheet-range-prior",
            "notes": "Radial compliance prismatic travel limit (+/-1.5 mm).",
        },
    }

    # Compute exact bottom-up mass distribution summary (50,000 draws, deterministic seed 0).
    rng = np.random.default_rng(0)
    n_draws = 50000
    d_draws = {k: rng.uniform(v["low"], v["high"], n_draws) for k, v in material_densities.items()}
    p_draws = {k: rng.uniform(v["low"], v["high"], n_draws) for k, v in purchased_masses.items()}
    total_mass = np.zeros(n_draws, dtype=np.float64)
    for part in cad_parts:
        if part["mass_model"] == "cad_volume_density":
            total_mass += part["geometry"]["volume_m3"] * d_draws[part["material_class"]]
        else:
            total_mass += p_draws[part["catalogue_part"]]

    upper_bound_kg = 0.300
    trunc = total_mass[total_mass <= upper_bound_kg]
    if len(trunc) < 100:
        raise RuntimeError(f"Too few bottom-up draws below {upper_bound_kg} kg: {len(trunc)}")

    prior = {
        "schema_version": "1",
        "status": "PRIOR_NOT_MEASURED",
        "warning": (
            "Monte Carlo prior for unmeasured payload_car physical parameters. "
            "NOT a calibrated or measured vehicle model. Do not trust controllers tuned on these "
            "samples on physical hardware."
        ),
        "allowed_sources": ["owner", "cad-volume", "datasheet-range-prior"],
        "cad_path": "assets/car/payload_assembly/payload_assembly.usdz",
        "cad_sha256": sha256,
        "total_mass_upper_bound_kg": {
            "value": upper_bound_kg,
            "unit": "kg",
            "quote": "probably under 300g",
            "quote_date": "2026-11-27",
            "source": "owner",
        },
        "cad_geometry_summary": cad_summary,
        "material_densities_kg_per_m3": material_densities,
        "purchased_envelope_masses_kg": purchased_masses,
        "cad_parts": cad_parts,
        "bottom_up_mass_estimate_kg": {
            "untruncated": {
                "n_draws": n_draws,
                "p05": round(float(np.percentile(total_mass, 5)), 6),
                "p50": round(float(np.percentile(total_mass, 50)), 6),
                "p95": round(float(np.percentile(total_mass, 95)), 6),
                "fraction_le_upper_bound": round(float(len(trunc) / n_draws), 6),
                "unit": "kg",
                "source": "cad-volume",
                "notes": "Bottom-up CAD part volumes x datasheet density ranges + purchased envelope datasheet mass ranges.",
            },
            "truncated_at_owner_bound": {
                "n_accepted": int(len(trunc)),
                "p05": round(float(np.percentile(trunc, 5)), 6),
                "p50": round(float(np.percentile(trunc, 50)), 6),
                "p95": round(float(np.percentile(trunc, 95)), 6),
                "max": round(float(trunc.max()), 6),
                "unit": "kg",
                "source": "owner",
                "notes": "Bottom-up distribution truncated to total_mass_kg <= 0.300 kg per owner statement.",
            },
        },
        "n20_motor_variants": n20_variants,
        "supply_and_drivetrain": supply_and_drivetrain,
    }
    return prior


def main() -> int:
    prior = build_prior(DEFAULT_USDZ)
    DEFAULT_OUT.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    u = prior["bottom_up_mass_estimate_kg"]["untruncated"]
    t = prior["bottom_up_mass_estimate_kg"]["truncated_at_owner_bound"]
    print(
        f"Wrote {DEFAULT_OUT}\n"
        f"  CAD meshes: {prior['cad_geometry_summary']['raw_meshes']} "
        f"(watertight={prior['cad_geometry_summary']['watertight_meshes']}, "
        f"convex_hull={prior['cad_geometry_summary']['convex_hull_fallback_meshes']})\n"
        f"  Untruncated mass: median={u['p50'] * 1e3:.1f}g, 5-95%=[{u['p05'] * 1e3:.1f}, {u['p95'] * 1e3:.1f}]g "
        f"(frac<=300g={u['fraction_le_upper_bound']:.3f})\n"
        f"  Truncated (<=300g): median={t['p50'] * 1e3:.1f}g, 5-95%=[{t['p05'] * 1e3:.1f}, {t['p95'] * 1e3:.1f}]g"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
