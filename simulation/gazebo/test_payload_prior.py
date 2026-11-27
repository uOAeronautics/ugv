"""Unit tests for payload_prior.json, deterministic sampling, SDF generation, and drive guard."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from drive_guard import (  # noqa: E402
    PAYLOAD_CAR_NOT_DRIVABLE_MSG,
    PRIOR_SAMPLE_BANNER,
    extract_prior_sample_record,
    require_drivable_car,
)
from payload_prior import (  # noqa: E402
    ALLOWED_SOURCES,
    PRIOR_PATH,
    PRIOR_STATUS,
    load_prior,
    sample_payload_parameters,
    validate_prior_sources,
)


class PayloadPriorTest(unittest.TestCase):
    def test_prior_loads_and_every_entry_has_valid_source(self):
        prior = load_prior(PRIOR_PATH)
        self.assertEqual(validate_prior_sources(prior), [])
        self.assertEqual(prior["status"], "PRIOR_NOT_MEASURED")
        self.assertEqual(set(prior["allowed_sources"]), set(ALLOWED_SOURCES))

        ub = prior["total_mass_upper_bound_kg"]
        self.assertAlmostEqual(ub["value"], 0.300, places=6)
        self.assertEqual(ub["source"], "owner")
        self.assertIn("probably under 300g", ub["quote"])

        geo = prior["cad_geometry_summary"]
        self.assertEqual(geo["source"], "cad-volume")
        self.assertEqual(geo["raw_meshes"], 1242)
        self.assertEqual(
            geo["watertight_meshes"] + geo["convex_hull_fallback_meshes"], geo["raw_meshes"]
        )

        # Negative check: validator catches missing or disallowed source tags.
        bad = {"x": {"low": 1.0, "high": 2.0}}
        self.assertTrue(validate_prior_sources(bad))
        bad_src = {"x": {"low": 1.0, "high": 2.0, "source": "guessed"}}
        self.assertTrue(validate_prior_sources(bad_src))

        # Purchased envelope parts and N20 variants must all cite datasheet-range-prior.
        for key in (
            "lipo_2s_72x35x17",
            "n20_gearmotor_each",
            "bno055_breakout",
            "camera_module_40x40_m12",
            "sbc_56x41",
        ):
            self.assertIn(key, prior["purchased_envelope_masses_kg"])
            self.assertEqual(
                prior["purchased_envelope_masses_kg"][key]["source"], "datasheet-range-prior"
            )
        self.assertGreaterEqual(len(prior["n20_motor_variants"]), 5)
        for v in prior["n20_motor_variants"]:
            self.assertEqual(v["source"], "datasheet-range-prior")

    def test_sampler_is_deterministic_per_seed_and_varies_across_seeds(self):
        prior = load_prior()
        a = sample_payload_parameters(42, prior)
        b = sample_payload_parameters(42, prior)
        self.assertEqual(a, b)
        self.assertEqual(a["status"], PRIOR_STATUS)
        self.assertEqual(a["seed"], 42)

        samples = [sample_payload_parameters(s, prior) for s in range(20)]
        masses = {s["mass"]["total_mass_kg"] for s in samples}
        variants = {s["motor"]["variant_id"] for s in samples}
        self.assertGreater(len(masses), 10)
        self.assertGreater(len(variants), 2)

    def test_mass_truncated_at_300g_and_com_inertia_derived_from_cad_parts(self):
        prior = load_prior()
        fairing_center_z = 0.037041
        for seed in range(100):
            s = sample_payload_parameters(seed, prior)
            total = s["mass"]["total_mass_kg"]
            self.assertGreater(total, 0.15)
            self.assertLessEqual(total, 0.300 + 1e-9)

            # Link masses must account for total mass (chassis + 4 carriers + 4 hubs + 28 rollers).
            ch_m = s["mass"]["chassis"]["mass_kg"]
            n_r = s["contact"]["wheel_roller_count_per_wheel"]
            wheels_sum = sum(
                w["hub_mass_kg"] + w["carrier_mass_kg"] + n_r * w["each_roller_mass_kg"]
                for w in s["mass"]["wheels"].values()
            )
            self.assertAlmostEqual(ch_m + wheels_sum, total, places=4)

            # Chassis COM comes from CAD part positions, not the fairing box center.
            ch_com = s["mass"]["chassis"]["center_m"]
            self.assertNotAlmostEqual(ch_com[2], fairing_center_z, places=3)

            # Positive-definite inertia tensor on chassis.
            ixx, iyy, izz, ixy, ixz, iyz = s["mass"]["chassis"]["inertia_kg_m2"]
            self.assertGreater(ixx, 0)
            self.assertGreater(iyy, 0)
            self.assertGreater(izz, 0)
            det = (
                ixx * iyy * izz
                + 2 * ixy * ixz * iyz
                - ixx * iyz * iyz
                - iyy * ixz * ixz
                - izz * ixy * ixy
            )
            self.assertGreater(det, 0)


class PayloadSampleGeneratorAndGuardTest(unittest.TestCase):
    def test_generate_sample_seed_writes_drivable_sdf_sidecar_and_passes_guard(self):
        with tempfile.TemporaryDirectory(prefix="ugv-sample-gen-") as tmp:
            out_sdf = Path(tmp) / "sampled_s7.sdf"
            res = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "generate.py"),
                    "--car",
                    "payload",
                    "--sample-seed",
                    "7",
                    "--output",
                    str(out_sdf),
                ],
                check=True,
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            self.assertIn(PRIOR_SAMPLE_BANNER, res.stdout)
            sidecar = out_sdf.with_suffix(".sample.json")
            self.assertTrue(sidecar.is_file())
            sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_data["seed"], 7)
            self.assertEqual(sidecar_data["status"], PRIOR_STATUS)
            self.assertLessEqual(sidecar_data["mass"]["total_mass_kg"], 0.300)

            root = ET.parse(out_sdf).getroot()
            model = root.find('./world/model[@name="payload_car"]')
            self.assertIsNotNone(model)
            self.assertIsNone(model.find("static"))
            # 1 chassis + 4 carriers + 4 wheel hubs + 28 passive rollers = 37 links, all with inertial.
            links = model.findall("link")
            self.assertEqual(len(links), 37)
            self.assertTrue(all(link.find("inertial") is not None for link in links))
            self.assertAlmostEqual(
                sum(float(link.findtext("inertial/mass")) for link in links),
                sidecar_data["mass"]["total_mass_kg"],
                places=4,
            )

            # Embedded sample record matches sidecar JSON.
            rec = extract_prior_sample_record(out_sdf)
            self.assertEqual(rec, sidecar_data)

            # Guard accepts sampled world and prints PRIOR SAMPLE banner.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                returned = require_drivable_car(
                    world=out_sdf,
                    car="payload",
                    topics=["/model/payload_car/odometry", "/world/payload_lab/stats"],
                )
            self.assertEqual(returned["seed"], 7)
            self.assertIn(PRIOR_SAMPLE_BANNER, buf.getvalue())

            # Validate SDF syntax with Gazebo when gz is available.
            if shutil.which("gz"):
                check = subprocess.run(
                    ["gz", "sdf", "-k", str(out_sdf)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

            # Tampering with or removing the sample record causes guard to reject again.
            drive = model.find('plugin[@name="ugv::TTMotorSystem"]')
            drive.remove(drive.find("prior_sample_json"))
            with self.assertRaises(RuntimeError) as ctx:
                require_drivable_car(world=root)
            self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    def test_generate_without_sample_seed_stays_static_and_freenove_rejects_seed(self):
        with tempfile.TemporaryDirectory(prefix="ugv-static-gen-") as tmp:
            static_sdf = Path(tmp) / "static.sdf"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "generate.py"),
                    "--car",
                    "payload",
                    "--output",
                    str(static_sdf),
                ],
                check=True,
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            self.assertFalse(static_sdf.with_suffix(".sample.json").exists())
            with self.assertRaises(RuntimeError) as ctx:
                require_drivable_car(world=static_sdf)
            self.assertEqual(str(ctx.exception), PAYLOAD_CAR_NOT_DRIVABLE_MSG)

            bad = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "generate.py"),
                    "--car",
                    "freenove",
                    "--sample-seed",
                    "0",
                    "--output",
                    str(Path(tmp) / "bad.sdf"),
                ],
                cwd=HERE,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_montecarlo_results_file_exists_and_carries_prior_status(self):
        mc_path = HERE / "payload_montecarlo.json"
        self.assertTrue(mc_path.is_file())
        data = json.loads(mc_path.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], PRIOR_STATUS)
        self.assertGreaterEqual(data["n_samples"], 32)
        self.assertEqual(len(data["samples"]), data["n_samples"])
        self.assertTrue(all(r.get("run_id") for r in data["samples"]))
        self.assertGreaterEqual(data["sanity_fraction"], 0.0)
        self.assertLessEqual(data["sanity_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
