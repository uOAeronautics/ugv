"""Unit tests for the payload car USD frame/centring math, assets, and Gazebo SDF generator."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate  # noqa: E402
import lockstep_env  # noqa: E402
import native_motor_parameters  # noqa: E402
from drive_guard import PAYLOAD_CAR_NOT_DRIVABLE_MSG, require_drivable_car  # noqa: E402
from payload_car_model import WHEEL_LINKS, add_payload_car, load_payload_dimensions  # noqa: E402
from usd_car_geometry import (  # noqa: E402
    FrameTransform,
    classify_part_link,
    compute_frame_from_hubs,
    validate_wheel_symmetry,
)

# SHA-256 of the committed pre-swap Freenove worlds (baseline commit before 2026-11-27 swap).
HEAD_FREENOVE_DRONE_WORLD_SHA256 = (
    "02d31c93f56134b4d914a610c06632f2eb17921e1690278fef737adc133d524a"
)
HEAD_FREENOVE_LAB_WORLD_SHA256 = "fae8493f432a1be8a08dc24c5bbccda7debf7aa90fa11cf2d9a5af958c075b89"


class PayloadConverterGeometryTest(unittest.TestCase):
    def test_usd_to_gazebo_frame_transform_applies_units_rotation_and_ground_contact(self):
        # Four synthetic wheel hubs in USD cm coordinates, arbitrarily offset at (-20, +20, +0.82),
        # with +Y forward, +Z up, and lowest roller at z = -1.174 cm.
        hubs = [
            ((-27.45, 21.62, -1.10), (-25.20, 25.46, 2.74)),  # front_left
            ((-14.80, 21.62, -1.10), (-12.55, 25.46, 2.74)),  # front_right
            ((-27.45, 14.54, -1.10), (-25.20, 18.38, 2.74)),  # rear_left
            ((-14.80, 14.54, -1.10), (-12.55, 18.38, 2.74)),  # rear_right
        ]
        frame, hubs_gz = compute_frame_from_hubs(hubs, z_ground_stage=-1.174, meters_per_unit=0.01)
        validate_wheel_symmetry(hubs_gz)

        self.assertEqual(frame.origin_xy_stage, (-20.0, 20.0))
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][0], 0.0354, places=6)
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][1], 0.06325, places=6)
        self.assertAlmostEqual(hubs_gz["front_right_wheel"][1], -0.06325, places=6)
        self.assertAlmostEqual(hubs_gz["rear_left_wheel"][0], -0.0354, places=6)
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][2], 0.01994, places=6)

        # Forward +Y in USD must map to +X in Gazebo; lowest roller must land at z=0.
        front_pt = frame.point_to_gazebo((-20.0, 26.36, -1.174))
        self.assertAlmostEqual(front_pt[0], 0.0636, places=6)
        self.assertAlmostEqual(front_pt[1], 0.0, places=6)
        self.assertAlmostEqual(front_pt[2], 0.0, places=6)

        # -X in USD (left side) must map to +Y in Gazebo (left side).
        left_pt = frame.point_to_gazebo((-25.0, 20.0, 0.82))
        self.assertAlmostEqual(left_pt[0], 0.0, places=6)
        self.assertAlmostEqual(left_pt[1], 0.05, places=6)

    def test_arbitrary_stage_translation_is_invariant(self):
        dx, dy, dz = 137.5, -42.25, 19.0
        base_hubs = [
            ((-27.45 + dx, 21.62 + dy, -1.10 + dz), (-25.20 + dx, 25.46 + dy, 2.74 + dz)),
            ((-14.80 + dx, 21.62 + dy, -1.10 + dz), (-12.55 + dx, 25.46 + dy, 2.74 + dz)),
            ((-27.45 + dx, 14.54 + dy, -1.10 + dz), (-25.20 + dx, 18.38 + dy, 2.74 + dz)),
            ((-14.80 + dx, 14.54 + dy, -1.10 + dz), (-12.55 + dx, 18.38 + dy, 2.74 + dz)),
        ]
        _, hubs_gz = compute_frame_from_hubs(
            base_hubs, z_ground_stage=-1.174 + dz, meters_per_unit=0.01
        )
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][0], 0.0354, places=6)
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][1], 0.06325, places=6)
        self.assertAlmostEqual(hubs_gz["front_left_wheel"][2], 0.01994, places=6)

    def test_classify_part_link_separates_wheels_and_chassis(self):
        self.assertEqual(
            classify_part_link("Mecanum_roller__TPU_", (0.035, 0.063, 0.02)), "front_left_wheel"
        )
        self.assertEqual(
            classify_part_link("Mecanum_hub_outer__B_", (-0.035, -0.063, 0.02)),
            "rear_right_wheel",
        )
        self.assertEqual(classify_part_link("fairing", (0.0, 0.0, 0.037)), "chassis")
        # Inner M2x8 screw not near wheel track stays on chassis
        self.assertEqual(classify_part_link("M2x8", (0.01, 0.01, 0.01)), "chassis")
        self.assertEqual(classify_part_link("M2x8", (0.035, -0.064, 0.02)), "front_right_wheel")

    def test_bounds_to_gazebo_preserves_min_le_max(self):
        frame = FrameTransform(origin_xy_stage=(-20.0, 20.0), z_ground_stage=-1.0)
        bmin, bmax = frame.bounds_to_gazebo((-25.0, 15.0, -1.0), (-15.0, 25.0, 4.0))
        for i in range(3):
            self.assertLessEqual(bmin[i], bmax[i])
        self.assertEqual(bmin, (-0.05, -0.05, 0.0))
        self.assertEqual(bmax, (0.05, 0.05, 0.05))


class PayloadCarAssetsAndSdfTest(unittest.TestCase):
    def test_measured_dimensions_and_exported_meshes_exist(self):
        dims = load_payload_dimensions()
        self.assertEqual(dims["model_name"], "payload_car")
        self.assertAlmostEqual(dims["bbox_min_m"][2], 0.0, places=6)
        self.assertAlmostEqual(dims["overall_width_m"], 0.149, places=4)
        self.assertAlmostEqual(dims["wheelbase_m"], 0.0708, places=4)
        self.assertAlmostEqual(dims["wheel_track_m"], 0.1265, places=4)
        self.assertGreater(dims["camera"]["lens_front_face_m"][0], 0.05)
        self.assertLessEqual(dims["source"]["decimated_triangles"], 60000)

        # Unmeasured hardware parameters must stay explicitly unknown, never fabricated.
        unknown = dims["unknown_hardware_parameters"]
        self.assertIsNone(unknown["total_mass_kg"])
        self.assertIsNone(unknown["chassis_mass_kg"])
        self.assertIsNone(unknown["wheel_mass_kg"])
        self.assertIsNone(unknown["inertia_tensor_kg_m2"])

        asset_dir = HERE / "assets/car/payload_assembly"
        self.assertTrue((asset_dir / "payload_assembly.usdz").is_file())
        self.assertTrue((asset_dir / "SOURCE.md").is_file())
        self.assertTrue((asset_dir / "payload_car.mtl").is_file())
        self.assertTrue((asset_dir / "payload_car.glb").is_file())
        for link_name in ("chassis", *WHEEL_LINKS):
            obj = asset_dir / f"{link_name}.obj"
            self.assertTrue(obj.is_file(), f"Missing {obj}")
            lines = obj.read_text(encoding="utf-8").splitlines()
            v_count = sum(1 for ln in lines if ln.startswith("v "))
            f_lines = [ln for ln in lines if ln.startswith("f ")]
            self.assertGreater(v_count, 10)
            self.assertGreater(len(f_lines), 10)
            self.assertIn("mtllib payload_car.mtl", lines[:5])

    def test_default_generator_builds_payload_car_without_freenove_standins(self):
        args = type(
            "Args",
            (),
            dict(
                car="payload",
                drone=False,
                room=False,
                marker_layout="real",
                chassis_mass_kg=None,
                wheel_mass_kg=None,
                require_measured_physics=False,
                motor_profile=None,
                nominal_motors=False,
            ),
        )()
        root, _ = generate.build_world(args)
        model = root.find('./world/model[@name="payload_car"]')
        self.assertIsNotNone(model)
        self.assertIsNone(root.find('./world/model[@name="freenove"]'))

        # Payload car with unmeasured mass is emitted as static (visual/collision only)
        # so Gazebo cannot silently invent a 1 kg mass per link.
        self.assertEqual(model.findtext("static"), "true")

        # No Freenove motor plugin, odometry plugin, or fabricated inertial tags.
        plugins = {p.get("name") for p in model.findall("plugin")}
        self.assertNotIn("ugv::TTMotorSystem", plugins)
        self.assertEqual(plugins, set())
        self.assertEqual(model.findall("./link/inertial"), [])

        # Four wheel links + four revolute joints about +Y.
        joints = model.findall('./joint[@type="revolute"]')
        self.assertEqual(len(joints), 4)
        for j in joints:
            self.assertEqual(j.findtext("axis/xyz"), "0 1 0")

    def test_committed_payload_worlds_are_static_without_invented_inertial(self):
        for name in ("payload_world.sdf", "drone_docking_world.sdf", "lab_docking_world.sdf"):
            with self.subTest(world=name):
                root = ET.parse(HERE / name).getroot()
                model = root.find('./world/model[@name="payload_car"]')
                self.assertIsNotNone(model)
                self.assertEqual(model.findtext("static"), "true")
                self.assertEqual(model.findall("./link/inertial"), [])
                self.assertIsNone(model.find('plugin[@name="ugv::TTMotorSystem"]'))

    def test_require_measured_physics_fails_visibly_when_mass_unknown(self):
        world = ET.Element("world", name="test")
        with self.assertRaises(ValueError):
            add_payload_car(world, generate.node, require_measured_physics=True)

    def test_freenove_regeneration_matches_committed_baseline_byte_for_byte(self):
        with tempfile.TemporaryDirectory(prefix="ugv-freenove-identity-") as tmp:
            drone_out = Path(tmp) / "drone_docking_world.sdf"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "generate.py"),
                    "--car",
                    "freenove",
                    "--drone",
                    "--output",
                    str(drone_out),
                ],
                check=True,
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                hashlib.sha256(drone_out.read_bytes()).hexdigest(),
                HEAD_FREENOVE_DRONE_WORLD_SHA256,
            )

            lab_out = Path(tmp) / "lab_docking_world.sdf"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "generate.py"),
                    "--car",
                    "freenove",
                    "--room",
                    "--drone",
                    "--output",
                    str(lab_out),
                ],
                check=True,
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                hashlib.sha256(lab_out.read_bytes()).hexdigest(),
                HEAD_FREENOVE_LAB_WORLD_SHA256,
            )

    def test_drive_guard_rejects_payload_world_and_allows_freenove(self):
        for name in ("payload_world.sdf", "drone_docking_world.sdf", "lab_docking_world.sdf"):
            with self.subTest(world=name):
                with self.assertRaises(RuntimeError) as ctx:
                    require_drivable_car(world=HERE / name)
                self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        for car in ("payload", "payload_car"):
            with self.subTest(car=car):
                with self.assertRaises(RuntimeError) as ctx:
                    require_drivable_car(car=car)
                self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        with self.assertRaises(RuntimeError) as ctx:
            require_drivable_car(topics=["/world/payload_lab/stats", "/model/payload_car/odometry"])
        self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        with tempfile.TemporaryDirectory(prefix="ugv-guard-") as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                lockstep_env.headless_world(HERE / "drone_docking_world.sdf", Path(tmp) / "out.sdf")
            self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        payload_root = ET.parse(HERE / "payload_world.sdf")
        with self.assertRaises(RuntimeError) as ctx:
            native_motor_parameters.extract_native_parameters(payload_root)
        self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        import vision_in_loop as vil

        with self.assertRaises(RuntimeError) as ctx:
            vil.selftest(car="payload")
        self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

        # Archived Freenove world remains drivable and passes the guard.
        require_drivable_car(world=HERE / "freenove_world.sdf")


if __name__ == "__main__":
    unittest.main()
