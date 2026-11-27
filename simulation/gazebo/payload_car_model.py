"""Gazebo SDF builder for the payload car, driven by measured CAD dimensions.

Frame convention (matching assembly.py): +X forward, +Y left, +Z up, metres;
origin (0, 0, 0) at the centre of the four wheel hubs projected to ground contact
(z = 0 at the lowest roller vertex).

Mass, inertia, N20 motor parameters, camera intrinsics and BNO055 IMU noise are
UNKNOWN until the physical vehicle is weighed and characterised. This module never
invents those values or reuses Freenove parameters silently.
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMENSIONS_PATH = HERE / "payload_dimensions.json"
MESH_REL_DIR = Path("assets/car/payload_assembly")
WHEEL_LINKS = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)

# TODO(payload-car-characterisation): measure physical vehicle on a scale and replace None.
# See research/tasks/2026-11-27-characterise-payload-car.md.
UNMEASURED_CHASSIS_MASS_KG_TODO: float | None = None
UNMEASURED_WHEEL_MASS_KG_TODO: float | None = None


def load_payload_dimensions(path: Path = DIMENSIONS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_box_inertial(
    link, node, mass_kg: float, center: tuple[float, float, float], size: tuple[float, float, float]
):
    sx, sy, sz = size
    inertial = node(link, "inertial")
    node(inertial, "mass", f"{mass_kg:.6f}")
    node(inertial, "pose", " ".join(f"{v:.6f}" for v in (*center, 0.0, 0.0, 0.0)))
    inertia = node(inertial, "inertia")
    node(inertia, "ixx", f"{mass_kg * (sy * sy + sz * sz) / 12.0:.9f}")
    node(inertia, "iyy", f"{mass_kg * (sx * sx + sz * sz) / 12.0:.9f}")
    node(inertia, "izz", f"{mass_kg * (sx * sx + sy * sy) / 12.0:.9f}")
    for tag in ("ixy", "ixz", "iyz"):
        node(inertia, tag, "0")


def _add_cylinder_y_inertial(link, node, mass_kg: float, radius: float, length: float):
    """Cylinder aligned with link +Y (wheel spin axis)."""
    inertial = node(link, "inertial")
    node(inertial, "mass", f"{mass_kg:.6f}")
    inertia = node(inertial, "inertia")
    trans = mass_kg * (3.0 * radius * radius + length * length) / 12.0
    spin = mass_kg * radius * radius / 2.0
    node(inertia, "ixx", f"{trans:.9f}")
    node(inertia, "iyy", f"{spin:.9f}")
    node(inertia, "izz", f"{trans:.9f}")
    for tag in ("ixy", "ixz", "iyz"):
        node(inertia, tag, "0")


def add_payload_car(
    world,
    node,
    *,
    pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.001, 0.0, 0.0, 0.0),
    dimensions_path: Path = DIMENSIONS_PATH,
    chassis_mass_kg: float | None = UNMEASURED_CHASSIS_MASS_KG_TODO,
    wheel_mass_kg: float | None = UNMEASURED_WHEEL_MASS_KG_TODO,
    require_measured_physics: bool = False,
):
    """Append the payload car model to an SDF `<world>` element."""
    dims = load_payload_dimensions(dimensions_path)
    missing_mass = chassis_mass_kg is None or wheel_mass_kg is None
    if missing_mass and require_measured_physics:
        raise ValueError(
            "Payload car mass/inertia and N20 motor parameters are unmeasured (TODO). "
            "Pass measured --chassis-mass-kg and --wheel-mass-kg, or see "
            "research/tasks/2026-11-27-characterise-payload-car.md."
        )
    if missing_mass:
        print(
            "WARNING: payload_car mass, inertia, N20 motor dynamics, camera intrinsics and "
            "BNO055 IMU noise are UNKNOWN (unmeasured). Emitting <static>true</static> "
            "(geometry/visual and collision model only) without <inertial> or drive plugins "
            "so Gazebo does not silently assign 1 kg per link.",
            file=sys.stderr,
        )

    model = node(world, "model", name="payload_car")
    model.append(
        ET.Comment(
            " Payload car visual meshes and collision envelopes from measured CAD "
            "(simulation/gazebo/payload_dimensions.json). Mass, inertia, N20 motor curves, "
            "camera intrinsics and BNO055 IMU noise are UNKNOWN (TODO). "
        )
    )
    if missing_mass:
        model.append(
            ET.Comment(
                " Emitted as <static>true</static> (geometry/visual and collision model only) "
                "until mass and motor parameters are measured "
                "(research/tasks/2026-11-27-characterise-payload-car.md); "
                "dynamic links without <inertial> would otherwise receive an implicit 1 kg mass "
                "from Gazebo. "
            )
        )
        node(model, "static", "true")
    node(model, "pose", " ".join(map(str, pose)))

    # --- Chassis link ---
    chassis = node(model, "link", name="chassis")
    node(chassis, "pose", "0 0 0 0 0 0")
    fairing = dims["fairing"]
    if not missing_mass:
        _add_box_inertial(
            chassis,
            node,
            chassis_mass_kg,
            tuple(fairing["center_m"]),
            tuple(fairing["size_m"]),
        )
    else:
        chassis.append(
            ET.Comment(
                " TODO: chassis mass/inertia UNKNOWN (UNMEASURED_CHASSIS_MASS_KG_TODO is None) "
            )
        )

    vis = node(chassis, "visual", name="chassis_visual")
    mesh = node(node(vis, "geometry"), "mesh")
    node(mesh, "uri", (MESH_REL_DIR / "chassis.obj").as_posix())

    col = node(chassis, "collision", name="fairing_collision")
    node(col, "pose", " ".join(f"{v:.6f}" for v in (*fairing["center_m"], 0.0, 0.0, 0.0)))
    box = node(node(col, "geometry"), "box")
    node(box, "size", " ".join(f"{v:.6f}" for v in fairing["size_m"]))

    cam = dims.get("camera")
    if cam:
        lens_col = node(chassis, "collision", name="m12_lens_collision")
        node(
            lens_col,
            "pose",
            " ".join(f"{v:.6f}" for v in (*cam["lens_center_m"], 0.0, math.pi / 2.0, 0.0)),
        )
        cyl = node(node(lens_col, "geometry"), "cylinder")
        node(cyl, "radius", f"{cam['lens_diameter_m'] / 2.0:.6f}")
        node(cyl, "length", f"{cam['lens_length_m']:.6f}")
        chassis.append(
            ET.Comment(
                f" TODO: camera sensor at measured M12 lens front {cam['lens_front_face_m']} m "
                "omitted because sensor model and intrinsics are UNKNOWN "
            )
        )

    imu = dims.get("imu")
    if imu:
        chassis.append(
            ET.Comment(
                f" TODO: BNO055 IMU sensor at measured pose {imu['center_m']} m "
                "omitted because noise profile and mounting matrix are UNKNOWN "
            )
        )

    # --- Four mecanum wheel links + revolute joints ---
    wheel_r = float(dims["wheel_radius_m"])
    wheel_w = float(dims["wheel_width_m"])
    for link_name in WHEEL_LINKS:
        hub = dims["wheel_hubs_m"][link_name]
        wlink = node(model, "link", name=link_name)
        node(wlink, "pose", " ".join(f"{v:.6f}" for v in (*hub, 0.0, 0.0, 0.0)))
        if not missing_mass:
            _add_cylinder_y_inertial(wlink, node, wheel_mass_kg, wheel_r, wheel_w)
        else:
            wlink.append(
                ET.Comment(
                    " TODO: wheel mass/inertia UNKNOWN (UNMEASURED_WHEEL_MASS_KG_TODO is None) "
                )
            )
        wvis = node(wlink, "visual", name=f"{link_name}_visual")
        wmesh = node(node(wvis, "geometry"), "mesh")
        node(wmesh, "uri", (MESH_REL_DIR / f"{link_name}.obj").as_posix())

        wcol = node(wlink, "collision", name=f"{link_name}_collision")
        node(wcol, "pose", f"0 0 0 {math.pi / 2.0:.6f} 0 0")
        wcyl = node(node(wcol, "geometry"), "cylinder")
        node(wcyl, "radius", f"{wheel_r:.6f}")
        node(wcyl, "length", f"{wheel_w:.6f}")

        joint = node(model, "joint", name=f"{link_name}_joint", type="revolute")
        node(joint, "parent", "chassis")
        node(joint, "child", link_name)
        axis = node(joint, "axis")
        node(axis, "xyz", "0 1 0")
        limit = node(axis, "limit")
        node(limit, "lower", "-1e16")
        node(limit, "upper", "1e16")

    if not missing_mass:
        odom = node(
            model,
            "plugin",
            filename="gz-sim-odometry-publisher-system",
            name="gz::sim::systems::OdometryPublisher",
        )
        node(odom, "dimensions", 2)
        node(odom, "odom_topic", "/model/payload_car/odometry")
    return model, dims
