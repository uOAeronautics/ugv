"""Gazebo SDF builder for the payload car, driven by measured CAD dimensions.

Frame convention (matching assembly.py): +X forward, +Y left, +Z up, metres;
origin (0, 0, 0) at the centre of the four wheel hubs projected to ground contact
(z = 0 at the lowest roller vertex).

Mass, inertia, N20 motor parameters, camera intrinsics and BNO055 IMU noise are
UNKNOWN until the physical vehicle is weighed and characterised. By default this
module emits `<static>true</static>` with no `<inertial>` or drive plugins.
When an explicit Monte Carlo prior `sample` (from `payload_prior.py`) is passed,
it emits a dynamic drivable model with the drawn prior record embedded in the SDF
and labelled `PRIOR_SAMPLE_NOT_MEASURED`.
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
WHEEL_HANDEDNESS_BASE = {
    "front_left_wheel": 1,
    "front_right_wheel": -1,
    "rear_left_wheel": -1,
    "rear_right_wheel": 1,
}

# TODO(payload-car-characterisation): measure physical vehicle on a scale and replace None.
# See research/tasks/2026-11-27-characterise-payload-car.md.
UNMEASURED_CHASSIS_MASS_KG_TODO: float | None = None
UNMEASURED_WHEEL_MASS_KG_TODO: float | None = None


def load_payload_dimensions(path: Path = DIMENSIONS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_tensor_inertial(
    link,
    node,
    mass_kg: float,
    center: tuple[float, float, float] | list[float],
    tensor: tuple[float, ...] | list[float],
):
    inertial = node(link, "inertial")
    node(inertial, "mass", f"{mass_kg:.6f}")
    node(inertial, "pose", " ".join(f"{float(v):.6f}" for v in (*center, 0.0, 0.0, 0.0)))
    inertia = node(inertial, "inertia")
    for tag, value in zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), tensor, strict=True):
        node(inertia, tag, f"{float(value):.12g}")


def _add_box_inertial(
    link, node, mass_kg: float, center: tuple[float, float, float], size: tuple[float, float, float]
):
    sx, sy, sz = size
    _add_tensor_inertial(
        link,
        node,
        mass_kg,
        center,
        (
            mass_kg * (sy * sy + sz * sz) / 12.0,
            mass_kg * (sx * sx + sz * sz) / 12.0,
            mass_kg * (sx * sx + sy * sy) / 12.0,
            0.0,
            0.0,
            0.0,
        ),
    )


def _add_cylinder_y_inertial(link, node, mass_kg: float, radius: float, length: float):
    """Cylinder aligned with link +Y (wheel spin axis)."""
    trans = mass_kg * (3.0 * radius * radius + length * length) / 12.0
    spin = mass_kg * radius * radius / 2.0
    _add_tensor_inertial(link, node, mass_kg, (0.0, 0.0, 0.0), (trans, spin, trans, 0.0, 0.0, 0.0))


def _add_sampled_passive_rollers(model, node, dims: dict, sample: dict) -> int:
    """Append 4 x 7 collision-only passive 45-degree rollers matching the CAD geometry."""
    contact = sample["contact"]
    wheels = sample["mass"]["wheels"]
    n_rollers = int(contact["wheel_roller_count_per_wheel"])
    ring = float(contact["wheel_roller_ring_radius_m"])
    rr = float(contact["wheel_roller_contact_radius_m"])
    # Use cylinder collision (matching Freenove mecanum_geometry.py) with barrel-core length
    # (~0.625 x CAD tip-to-tip length) so adjacent tilted rollers clear the floor smoothly and
    # DART's ODE LCP solver avoids capsule-on-box duplicate contact singularities.
    cyl_len = max(0.008, float(contact["wheel_roller_length_m"]) * 0.625)
    mu = float(contact["roller_mu"])
    damping = float(contact["roller_bearing_damping_nm_s_per_rad"])
    friction = float(contact["roller_bearing_friction_nm"])
    hand_sign = int(contact["mecanum_hand_sign"])

    count = 0
    for link_name in WHEEL_LINKS:
        hub = dims["wheel_hubs_m"][link_name]
        w_spec = wheels[link_name]
        hand = WHEEL_HANDEDNESS_BASE[link_name] * hand_sign
        for i in range(n_rollers):
            angle = i * math.tau / n_rollers
            axis = (
                math.cos(angle) / math.sqrt(2.0),
                hand / math.sqrt(2.0),
                -math.sin(angle) / math.sqrt(2.0),
            )
            yaw = math.atan2(axis[1], axis[0])
            pitch = math.acos(max(-1.0, min(1.0, axis[2])))
            pos = (hub[0] + ring * math.sin(angle), hub[1], hub[2] + ring * math.cos(angle))
            rname = f"{link_name}_roller_{i}"
            rlink = node(model, "link", name=rname)
            node(rlink, "pose", " ".join(f"{v:.6f}" for v in (*pos, 0.0, pitch, yaw)))
            _add_tensor_inertial(
                rlink,
                node,
                float(w_spec["each_roller_mass_kg"]),
                (0.0, 0.0, 0.0),
                w_spec["roller_inertia_kg_m2"],
            )
            col = node(rlink, "collision", name=f"{rname}_collision")
            cyl = node(node(col, "geometry"), "cylinder")
            node(cyl, "radius", f"{rr:.6f}")
            node(cyl, "length", f"{cyl_len:.6f}")
            ode = node(node(node(col, "surface"), "friction"), "ode")
            node(ode, "mu", f"{mu:.6f}")
            node(ode, "mu2", f"{mu:.6f}")
            joint = node(model, "joint", name=f"{rname}_joint", type="revolute")
            node(joint, "parent", link_name)
            node(joint, "child", rname)
            axle = node(joint, "axis")
            node(axle, "xyz", "0 0 1")
            dyn = node(axle, "dynamics")
            node(dyn, "damping", f"{damping:.9f}")
            node(dyn, "friction", f"{friction:.9f}")
            count += 1
    return count


def _add_sampled_drive_plugin(model, node, dims: dict, sample: dict) -> None:
    motor = sample["motor"]
    contact = sample["contact"]
    drive = node(model, "plugin", filename="ugv_tt_motor", name="ugv::TTMotorSystem")
    node(drive, "topic_prefix", "/model/payload_car")
    node(drive, "mecanum", "true")
    node(drive, "mecanum_hand_sign", contact["mecanum_hand_sign"])
    node(drive, "wheelbase", f"{float(dims['wheelbase_m']):.6f}")
    node(drive, "wheel_separation", f"{float(dims['wheel_track_m']):.6f}")
    node(drive, "wheel_radius", f"{float(dims['wheel_radius_m']):.6f}")
    node(drive, "stall_torque", f"{float(motor['stall_torque_nm']):.6f}")
    node(drive, "no_load_speed", f"{float(motor['no_load_speed_radps']):.6f}")
    node(drive, "brake_damping", f"{float(motor['brake_damping_nm_s_per_rad']):.6f}")
    node(drive, "brake_resistance", f"{float(motor['brake_resistance_nm']):.6f}")
    node(drive, "velocity_feedback", "0.06")
    node(drive, "velocity_integral", "0.6")
    for field in (
        "wheel_deadzone",
        "wheel_duty_exponent",
        "wheel_voltage_gain",
        "wheel_reverse_voltage_gain",
    ):
        node(drive, field, " ".join(f"{float(v):.6f}" for v in motor[field]))
    node(drive, "wheel_drive_tau", " ".join(f"{float(v):.6f}" for v in motor["wheel_drive_tau_s"]))
    node(drive, "wheel_brake_tau", " ".join(f"{float(v):.6f}" for v in motor["wheel_brake_tau_s"]))
    node(drive, "wheel_gain_stddev", "0 0 0 0")
    node(drive, "random_seed", int(sample["seed"]))
    node(drive, "calibration_status", sample["status"])
    node(drive, "prior_sample_seed", int(sample["seed"]))
    node(drive, "prior_sha256", sample.get("prior_sha256", ""))
    node(drive, "prior_sample_json", json.dumps(sample, sort_keys=True))


def add_payload_car(
    world,
    node,
    *,
    pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.001, 0.0, 0.0, 0.0),
    dimensions_path: Path = DIMENSIONS_PATH,
    chassis_mass_kg: float | None = UNMEASURED_CHASSIS_MASS_KG_TODO,
    wheel_mass_kg: float | None = UNMEASURED_WHEEL_MASS_KG_TODO,
    require_measured_physics: bool = False,
    sample: dict | None = None,
):
    """Append the payload car model to an SDF `<world>` element."""
    dims = load_payload_dimensions(dimensions_path)
    is_sampled = sample is not None
    missing_mass = (chassis_mass_kg is None or wheel_mass_kg is None) and not is_sampled
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
    if is_sampled:
        model.append(
            ET.Comment(
                f" PRIOR SAMPLE, not measured (seed={sample['seed']}, "
                f"total_mass_kg={sample['mass']['total_mass_kg']}, "
                f"motor={sample['motor']['variant_id']}). Do NOT treat as a calibrated car. "
            )
        )
    else:
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
    if is_sampled:
        ch = sample["mass"]["chassis"]
        _add_tensor_inertial(chassis, node, ch["mass_kg"], ch["center_m"], ch["inertia_kg_m2"])
    elif not missing_mass:
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

    # --- Four mecanum wheel links + revolute joints (+ radial compliance & passive rollers when sampled) ---
    wheel_r = float(dims["wheel_radius_m"])
    wheel_w = float(dims["wheel_width_m"])
    for link_name in WHEEL_LINKS:
        hub = dims["wheel_hubs_m"][link_name]
        wlink = node(model, "link", name=link_name)
        node(wlink, "pose", " ".join(f"{v:.6f}" for v in (*hub, 0.0, 0.0, 0.0)))
        if is_sampled:
            w_spec = sample["mass"]["wheels"][link_name]
            _add_tensor_inertial(
                wlink,
                node,
                float(w_spec["hub_mass_kg"]),
                (0.0, 0.0, 0.0),
                w_spec["hub_inertia_kg_m2"],
            )
        elif not missing_mass:
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

        if is_sampled:
            # Inner hub core collision only so the 7 passive rollers make the floor contact.
            ring = float(sample["contact"]["wheel_roller_ring_radius_m"])
            wcol = node(wlink, "collision", name=f"{link_name}_hub_core_collision")
            node(wcol, "pose", f"0 0 0 {math.pi / 2.0:.6f} 0 0")
            wcyl = node(node(wcol, "geometry"), "cylinder")
            node(wcyl, "radius", f"{max(0.006, ring - 0.0025):.6f}")
            node(wcyl, "length", f"{wheel_w * 0.8:.6f}")

            # Radial-compliance carrier for 4-wheel floor load sharing (same topology as Freenove).
            carrier_name = f"{link_name}_carrier"
            carrier = node(model, "link", name=carrier_name)
            node(carrier, "pose", " ".join(f"{v:.6f}" for v in (*hub, 0.0, 0.0, 0.0)))
            _add_tensor_inertial(
                carrier,
                node,
                float(w_spec["carrier_mass_kg"]),
                (0.0, 0.0, 0.0),
                w_spec["carrier_inertia_kg_m2"],
            )
            spring = node(model, "joint", name=f"{link_name}_radial_compliance", type="prismatic")
            node(spring, "parent", "chassis")
            node(spring, "child", carrier_name)
            sa = node(spring, "axis")
            node(sa, "xyz", "0 0 1")
            sl = node(sa, "limit")
            travel = float(sample["contact"]["tire_radial_travel_m"])
            node(sl, "lower", f"{-travel:.6f}")
            node(sl, "upper", f"{travel:.6f}")
            sd = node(sa, "dynamics")
            node(sd, "spring_reference", "0")
            node(
                sd,
                "spring_stiffness",
                f"{float(sample['contact']['tire_radial_stiffness_n_per_m']):.3f}",
            )
            node(sd, "damping", f"{float(sample['contact']['tire_radial_damping_ns_per_m']):.3f}")

            joint = node(model, "joint", name=f"{link_name}_joint", type="revolute")
            node(joint, "parent", carrier_name)
            node(joint, "child", link_name)
            axis = node(joint, "axis")
            node(axis, "xyz", "0 1 0")
            limit = node(axis, "limit")
            node(limit, "lower", "-1e16")
            node(limit, "upper", "1e16")
            node(limit, "effort", f"{float(sample['motor']['stall_torque_nm']):.6f}")
            node(limit, "velocity", f"{float(sample['motor']['no_load_speed_radps']):.6f}")
            dyn = node(axis, "dynamics")
            node(dyn, "friction", "0.0002")
            node(dyn, "damping", "0.00001")
        else:
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

    if is_sampled:
        _add_sampled_passive_rollers(model, node, dims, sample)
        _add_sampled_drive_plugin(model, node, dims, sample)

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
