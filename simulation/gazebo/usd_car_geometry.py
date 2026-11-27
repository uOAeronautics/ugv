"""Frame transform, hub centring, link classification and dimension metrics for USD car CAD.

Pure standard library so unit tests and SDF generators can import it without usd-core or numpy.
Gazebo car frame convention (matching simulation/gazebo/assembly.py):
  +X forward (camera/front), +Y left, +Z up, metres;
  origin (0, 0, 0) at the mean XY of the four wheel hubs on the ground plane (z = 0 at lowest roller).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

WHEEL_LINKS = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
WHEEL_PART_KEYWORDS = (
    "Mecanum_hub",
    "Mecanum_roller",
    "Roller_sleeve",
    "Roller_axle",
    "M2x8",
)


@dataclass(frozen=True)
class FrameTransform:
    """Rigid USD-stage-to-Gazebo transform (scale + horizontal shift + Z-axis rotation)."""

    origin_xy_stage: tuple[float, float]
    z_ground_stage: float
    meters_per_unit: float = 0.01
    forward_axis: str = "+Y"

    def point_to_gazebo(self, pt: tuple[float, float, float]) -> tuple[float, float, float]:
        """Map one USD world point (stage units, geometric Z-up) into Gazebo metres."""
        cx, cy = self.origin_xy_stage
        s = self.meters_per_unit
        x_u, y_u, z_u = pt
        if self.forward_axis == "+Y":
            # Rz(-90 deg): +Y_usd -> +X_gz (forward), -X_usd -> +Y_gz (left), +Z_usd -> +Z_gz (up).
            return ((y_u - cy) * s, -(x_u - cx) * s, (z_u - self.z_ground_stage) * s)
        if self.forward_axis == "+X":
            return ((x_u - cx) * s, (y_u - cy) * s, (z_u - self.z_ground_stage) * s)
        if self.forward_axis == "-Y":
            return (-(y_u - cy) * s, (x_u - cx) * s, (z_u - self.z_ground_stage) * s)
        if self.forward_axis == "-X":
            return (-(x_u - cx) * s, -(y_u - cy) * s, (z_u - self.z_ground_stage) * s)
        raise ValueError(f"Unsupported forward_axis: {self.forward_axis!r}")

    def bounds_to_gazebo(
        self,
        bmin: tuple[float, float, float],
        bmax: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Transform an axis-aligned USD bounding box into an axis-aligned Gazebo bounding box."""
        p0 = self.point_to_gazebo(bmin)
        p1 = self.point_to_gazebo(bmax)
        return (
            (min(p0[0], p1[0]), min(p0[1], p1[1]), min(p0[2], p1[2])),
            (max(p0[0], p1[0]), max(p0[1], p1[1]), max(p0[2], p1[2])),
        )


def compute_frame_from_hubs(
    hub_aabbs_stage: Iterable[tuple[tuple[float, float, float], tuple[float, float, float]]],
    z_ground_stage: float,
    meters_per_unit: float = 0.01,
    forward_axis: str = "+Y",
) -> tuple[FrameTransform, dict[str, tuple[float, float, float]]]:
    """Compute the car origin and four wheel hub centres from hub AABBs in USD world coords.

    Each corner may have multiple hub meshes (e.g. inner N20-side hub + outer hub plate);
    they are grouped into four quadrants around the overall hub centroid so arbitrary stage
    translations are handled without hardcoded coordinates.
    """
    aabbs = list(hub_aabbs_stage)
    if len(aabbs) < 4:
        raise ValueError(f"Expected at least 4 wheel hub AABBs, got {len(aabbs)}")
    ctrs = [((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2) for mn, mx in aabbs]
    mid_x = sum(c[0] for c in ctrs) / len(ctrs)
    mid_y = sum(c[1] for c in ctrs) / len(ctrs)

    # First rotate around the provisional midpoint to classify FL/FR/RL/RR in Gazebo axes.
    provisional = FrameTransform(
        origin_xy_stage=(mid_x, mid_y),
        z_ground_stage=z_ground_stage,
        meters_per_unit=meters_per_unit,
        forward_axis=forward_axis,
    )
    grouped: dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float]]]] = {}
    for aabb, ctr in zip(aabbs, ctrs, strict=True):
        gx, gy, _ = provisional.point_to_gazebo(ctr)
        key = f"{'front' if gx >= 0 else 'rear'}_{'left' if gy >= 0 else 'right'}_wheel"
        grouped.setdefault(key, []).append(aabb)

    if set(grouped) != set(WHEEL_LINKS):
        raise ValueError(f"Expected all 4 wheel corners {WHEEL_LINKS}, found {sorted(grouped)}")

    corner_stage_ctrs: dict[str, tuple[float, float, float]] = {}
    for key, items in grouped.items():
        mn = tuple(min(item[0][i] for item in items) for i in range(3))
        mx = tuple(max(item[1][i] for item in items) for i in range(3))
        corner_stage_ctrs[key] = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)

    origin_x = sum(c[0] for c in corner_stage_ctrs.values()) / 4.0
    origin_y = sum(c[1] for c in corner_stage_ctrs.values()) / 4.0
    frame = FrameTransform(
        origin_xy_stage=(origin_x, origin_y),
        z_ground_stage=z_ground_stage,
        meters_per_unit=meters_per_unit,
        forward_axis=forward_axis,
    )
    hubs_gz = {k: frame.point_to_gazebo(c) for k, c in corner_stage_ctrs.items()}
    return frame, hubs_gz


def classify_part_link(
    prim_name: str,
    center_gz: tuple[float, float, float],
    wheel_y_min_m: float = 0.05,
) -> str:
    """Assign a USD mesh prim to one of the four wheel links or to `chassis`."""
    if any(kw in prim_name for kw in WHEEL_PART_KEYWORDS) and abs(center_gz[1]) >= wheel_y_min_m:
        end = "front" if center_gz[0] >= 0 else "rear"
        side = "left" if center_gz[1] >= 0 else "right"
        return f"{end}_{side}_wheel"
    return "chassis"


def r6(value: float) -> float:
    return round(float(value), 6)


def measure_car_dimensions(
    overall_bounds_gz: tuple[tuple[float, float, float], tuple[float, float, float]],
    wheel_hubs_gz: dict[str, tuple[float, float, float]],
    wheel_outer_radius_m: float,
    wheel_width_m: float,
    roller_width_m: float,
    ground_clearance_non_wheel_m: float,
    named_parts_gz: dict[str, dict],
    source_meta: dict,
) -> dict:
    """Build the measured dimensions JSON payload (metres, Gazebo car frame)."""
    bmin, bmax = overall_bounds_gz
    fl = wheel_hubs_gz["front_left_wheel"]
    fr = wheel_hubs_gz["front_right_wheel"]
    rl = wheel_hubs_gz["rear_left_wheel"]
    rr = wheel_hubs_gz["rear_right_wheel"]

    wheelbase_m = ((fl[0] - rl[0]) + (fr[0] - rr[0])) / 2.0
    wheel_track_m = ((fl[1] - fr[1]) + (rl[1] - rr[1])) / 2.0
    hub_height_m = sum(c[2] for c in wheel_hubs_gz.values()) / 4.0

    lens = named_parts_gz.get("M12_lens__1_")
    cam_pcb = named_parts_gz.get("camera_PCB__1_")
    imu = named_parts_gz.get("BNO055__1_")
    fairing = named_parts_gz.get("fairing")
    lipo = named_parts_gz.get("_2S_LiPo_72x35x17__envelope_")

    camera_info: dict | None = None
    if lens is not None:
        camera_info = {
            "lens_part": "M12_lens__1_",
            "lens_center_m": [r6(v) for v in lens["center_m"]],
            "lens_front_face_m": [
                r6(lens["max_m"][0]),
                r6(lens["center_m"][1]),
                r6(lens["center_m"][2]),
            ],
            "lens_height_m": r6(lens["center_m"][2]),
            "lens_diameter_m": r6(lens["size_m"][1]),
            "lens_length_m": r6(lens["size_m"][0]),
            "pcb_center_m": [r6(v) for v in cam_pcb["center_m"]] if cam_pcb else None,
            "pcb_size_m": [r6(v) for v in cam_pcb["size_m"]] if cam_pcb else None,
        }

    return {
        "model_name": "payload_car",
        "frame": {
            "convention": "+X forward, +Y left, +Z up, metres",
            "origin": "centre of the four wheel hubs projected to ground contact (z=0 at lowest roller vertex)",
            "source_usd_meters_per_unit": source_meta["meters_per_unit"],
            "source_usd_declared_up_axis": source_meta["declared_up_axis"],
            "source_usd_geometric_up_axis": "+Z",
            "source_usd_forward_axis": "+Y",
            "source_usd_origin_xy_stage": [r6(v) for v in source_meta["origin_xy_stage"]],
            "source_usd_z_ground_stage": r6(source_meta["z_ground_stage"]),
        },
        "source": {
            "cad_path": "assets/car/payload_assembly/payload_assembly.usdz",
            "sha256": source_meta["sha256"],
            "raw_meshes": source_meta["raw_meshes"],
            "raw_triangles": source_meta["raw_triangles"],
            "decimated_triangles": source_meta.get("decimated_triangles"),
        },
        "overall_length_m": r6(bmax[0] - bmin[0]),
        "overall_width_m": r6(bmax[1] - bmin[1]),
        "overall_height_m": r6(bmax[2] - bmin[2]),
        "bbox_min_m": [r6(v) for v in bmin],
        "bbox_max_m": [r6(v) for v in bmax],
        "wheel_type": "mecanum_7_roller",
        "wheel_diameter_m": r6(2.0 * wheel_outer_radius_m),
        "wheel_radius_m": r6(wheel_outer_radius_m),
        "wheel_width_m": r6(wheel_width_m),
        "roller_width_m": r6(roller_width_m),
        "wheel_track_m": r6(wheel_track_m),
        "wheelbase_m": r6(wheelbase_m),
        "wheel_hub_height_m": r6(hub_height_m),
        "wheel_hubs_m": {k: [r6(v) for v in wheel_hubs_gz[k]] for k in WHEEL_LINKS},
        "ground_clearance_min_m": r6(ground_clearance_non_wheel_m),
        "fairing": (
            {
                "center_m": [r6(v) for v in fairing["center_m"]],
                "size_m": [r6(v) for v in fairing["size_m"]],
                "bottom_z_m": r6(fairing["min_m"][2]),
                "top_z_m": r6(fairing["max_m"][2]),
            }
            if fairing
            else None
        ),
        "camera": camera_info,
        "imu": (
            {
                "part": "BNO055__1_",
                "center_m": [r6(v) for v in imu["center_m"]],
                "height_m": r6(imu["center_m"][2]),
            }
            if imu
            else None
        ),
        "battery_envelope": (
            {
                "part": "_2S_LiPo_72x35x17__envelope_",
                "center_m": [r6(v) for v in lipo["center_m"]],
                "size_m": [r6(v) for v in lipo["size_m"]],
            }
            if lipo
            else None
        ),
        "unknown_hardware_parameters": {
            "total_mass_kg": None,
            "chassis_mass_kg": None,
            "wheel_mass_kg": None,
            "inertia_tensor_kg_m2": None,
            "motor_gear_ratio_and_torque": "UNKNOWN (N20 gearmotor variant unverified)",
            "motor_driver_board_and_pinout": "UNKNOWN",
            "sbc_model_and_os": "UNKNOWN (CAD shows PCB_56x41 with HDMI/RJ45/USB/2x20 GPIO)",
            "camera_sensor_and_intrinsics": "UNKNOWN (CAD shows M12_lens on 40x40 camera_PCB)",
            "imu_noise_and_mounting_matrix": "UNKNOWN (CAD shows BNO055 on IMU_PCB)",
        },
    }


def validate_wheel_symmetry(
    wheel_hubs_gz: dict[str, tuple[float, float, float]], tol_m: float = 1e-4
) -> None:
    """Sanity-check that the centred wheel hubs are symmetric about X=0 and Y=0."""
    fl, fr = wheel_hubs_gz["front_left_wheel"], wheel_hubs_gz["front_right_wheel"]
    rl, rr = wheel_hubs_gz["rear_left_wheel"], wheel_hubs_gz["rear_right_wheel"]
    if abs(fl[0] + rl[0]) > tol_m or abs(fr[0] + rr[0]) > tol_m:
        raise ValueError(f"Wheel hubs not centred in X: {wheel_hubs_gz}")
    if abs(fl[1] + fr[1]) > tol_m or abs(rl[1] + rr[1]) > tol_m:
        raise ValueError(f"Wheel hubs not centred in Y: {wheel_hubs_gz}")
    if not all(math.isfinite(v) for c in wheel_hubs_gz.values() for v in c):
        raise ValueError(f"Non-finite wheel hub coordinates: {wheel_hubs_gz}")
