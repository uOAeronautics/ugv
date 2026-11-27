"""Stdlib-only loader, validator, and deterministic Monte Carlo sampler for payload_prior.json.

Produces drivable prior parameter sets for Gazebo without inventing a single hardware
measurement. Every drawn sample is explicitly marked PRIOR_SAMPLE_NOT_MEASURED, derives
COM and full 6-component inertia tensors from CAD part positions/shapes via the parallel-axis
theorem, and enforces the owner's <= 0.300 kg upper bound by truncation (rejection sampling).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from mass_properties import box, combine

HERE = Path(__file__).resolve().parent
PRIOR_PATH = HERE / "payload_prior.json"
ALLOWED_SOURCES = frozenset({"owner", "cad-volume", "datasheet-range-prior"})
PRIOR_STATUS = "PRIOR_SAMPLE_NOT_MEASURED"
WHEEL_LINKS = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
GRAVITY_KGF_CM_TO_NM = 0.0980665


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_prior_sources(obj: object, path: str = "$") -> list[str]:
    """Return a list of schema violations if any numeric entry lacks a valid `"source"`."""
    errors: list[str] = []
    if isinstance(obj, dict):
        has_direct_number = any(
            _is_number(v) or (isinstance(v, list) and any(_is_number(x) for x in v))
            for v in obj.values()
        )
        if has_direct_number:
            src = obj.get("source")
            if src not in ALLOWED_SOURCES:
                errors.append(
                    f"{path}: numeric object has invalid/missing source={src!r} "
                    f"(must be one of {sorted(ALLOWED_SOURCES)})"
                )
        for key, val in obj.items():
            errors.extend(validate_prior_sources(val, f"{path}.{key}"))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if _is_number(item):
                continue
            errors.extend(validate_prior_sources(item, f"{path}[{idx}]"))
    return errors


def load_prior(path: Path = PRIOR_PATH) -> dict:
    raw = Path(path).read_bytes()
    prior = json.loads(raw.decode("utf-8"))
    errors = validate_prior_sources(prior)
    if errors:
        raise ValueError("Invalid payload prior sources:\n" + "\n".join(errors))
    prior["_sha256"] = hashlib.sha256(raw).hexdigest()
    return prior


def _draw_spec(rng: random.Random, spec: dict) -> float:
    if "value" in spec:
        return float(spec["value"])
    dist = spec["dist"]
    lo, hi = float(spec["low"]), float(spec["high"])
    if dist == "uniform":
        return rng.uniform(lo, hi)
    if dist == "log_uniform":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    raise ValueError(f"Unsupported distribution: {dist!r}")


def _check_inertia_valid(inertia: list[float], label: str) -> None:
    a, b, c, d, e, f = inertia
    if not all(math.isfinite(x) for x in inertia):
        raise ValueError(f"Non-finite inertia on {label}: {inertia}")
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Non-positive diagonal inertia on {label}: {inertia}")
    if a * b - d * d <= 0:
        raise ValueError(f"Non-PD 2x2 inertia minor on {label}: {inertia}")
    det = a * b * c + 2 * d * e * f - a * f * f - b * e * e - c * d * d
    if det <= 0:
        raise ValueError(f"Non-PD inertia tensor on {label}: {inertia}")
    half = (a + b + c) / 2.0
    u, v, w = half - a, half - b, half - c
    if u <= 0 or v <= 0 or w <= 0 or u * v - d * d <= 0:
        raise ValueError(f"Inertia triangle inequality violated on {label}: {inertia}")
    tdet = u * v * w - 2 * d * e * f - u * f * f - v * e * e - w * d * d
    if tdet <= 0:
        raise ValueError(f"Inertia triangle det violated on {label}: {inertia}")


def _draw_mass_sample(rng: random.Random, prior: dict) -> tuple[dict, dict, float, int]:
    """Rejection-sample bottom-up CAD + catalogue masses truncated at owner upper bound."""
    upper_kg = float(prior["total_mass_upper_bound_kg"]["value"])
    dens_specs = prior["material_densities_kg_per_m3"]
    cat_specs = prior["purchased_envelope_masses_kg"]
    cad_parts = prior["cad_parts"]

    for attempt in range(1, 5000):
        densities = {k: _draw_spec(rng, spec) for k, spec in dens_specs.items()}
        cat_masses = {k: _draw_spec(rng, spec) for k, spec in cat_specs.items()}
        part_masses: dict[str, float] = {}
        total_kg = 0.0
        for part in cad_parts:
            if part["mass_model"] == "cad_volume_density":
                m = float(part["geometry"]["volume_m3"]) * densities[part["material_class"]]
            else:
                m = cat_masses[part["catalogue_part"]]
            part_masses[part["id"]] = m
            total_kg += m
        if total_kg <= upper_kg:
            return densities, part_masses, total_kg, attempt
    raise RuntimeError(f"Failed to draw a sample <= {upper_kg} kg in 5000 attempts")


def sample_payload_parameters(seed: int, prior: dict | None = None) -> dict:
    """Deterministically draw one full dynamic parameter set from the payload prior."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if prior is None:
        prior = load_prior()
    rng = random.Random(seed)
    densities, part_masses, total_mass_kg, rejection_attempts = _draw_mass_sample(rng, prior)

    sd = prior["supply_and_drivetrain"]
    cad_geo = prior["cad_geometry_summary"]
    n_rollers = int(cad_geo["wheel_roller_count_per_wheel"])
    roller_r = float(cad_geo["wheel_roller_contact_radius_m"])
    roller_len = float(cad_geo["wheel_roller_length_m"])
    carrier_mass_kg = _draw_spec(rng, sd["carrier_mass_kg"])

    # Combine CAD parts into chassis link (shifted to car origin) and 4 wheel links (local hub frame).
    chassis_parts: list[dict] = []
    wheel_hub_parts: dict[str, list[dict]] = {wl: [] for wl in WHEEL_LINKS}
    wheel_roller_mass: dict[str, float] = {wl: 0.0 for wl in WHEEL_LINKS}

    for part in prior["cad_parts"]:
        pid = part["id"]
        mass = part_masses[pid]
        geo = part["geometry"]
        com = list(geo["com_m"])
        inertia = [mass * float(v) for v in geo["inertia_per_kg_m2"]]
        link = part["link"]
        if link == "chassis":
            chassis_parts.append({"mass": mass, "center": com, "inertia": inertia})
        elif part["component"] == "wheel_roller_tpu":
            wheel_roller_mass[link] += mass
        else:
            wheel_hub_parts[link].append({"mass": mass, "center": com, "inertia": inertia})

    chassis_raw = combine(chassis_parts)
    # Subtract the 4 tiny virtual radial-compliance carrier masses from chassis uniformly
    # so the sum of all SDF link masses equals total_mass_kg exactly.
    chassis_mass_after_carriers = chassis_raw["mass"] - 4.0 * carrier_mass_kg
    if chassis_mass_after_carriers <= 0.02:
        raise ValueError("Sampled chassis mass too small after subtracting carrier masses")
    scale = chassis_mass_after_carriers / chassis_raw["mass"]
    chassis_inertial = {
        "mass_kg": round(chassis_mass_after_carriers, 6),
        "center_m": [round(v, 6) for v in chassis_raw["center"]],
        "inertia_kg_m2": [round(v * scale, 10) for v in chassis_raw["inertia"]],
    }
    _check_inertia_valid(chassis_inertial["inertia_kg_m2"], "chassis")

    # Compute each wheel hub inertial (expressed relative to wheel hub COM ~ wheel origin)
    # and per-roller mass/inertia (7 rollers per wheel).
    wheels_inertial: dict[str, dict] = {}
    all_assembly_parts = [
        {
            "mass": chassis_inertial["mass_kg"],
            "center": chassis_inertial["center_m"],
            "inertia": chassis_inertial["inertia_kg_m2"],
        }
    ]
    for wl in WHEEL_LINKS:
        hub_combined = combine(wheel_hub_parts[wl])
        hub_mass = hub_combined["mass"]
        hub_inertia = hub_combined["inertia"]
        _check_inertia_valid(hub_inertia, wl)
        each_roller_mass = wheel_roller_mass[wl] / n_rollers
        r_trans = each_roller_mass * (3.0 * roller_r * roller_r + roller_len * roller_len) / 12.0
        r_spin = each_roller_mass * roller_r * roller_r / 2.0
        carrier = box(carrier_mass_kg, [0.0, 0.0, 0.0], [0.008, 0.008, 0.008])
        wheels_inertial[wl] = {
            "hub_mass_kg": round(hub_mass, 6),
            "hub_center_world_m": [round(v, 6) for v in hub_combined["center"]],
            "hub_inertia_kg_m2": [round(v, 10) for v in hub_inertia],
            "each_roller_mass_kg": round(each_roller_mass, 6),
            "roller_inertia_kg_m2": [
                round(r_trans, 10),
                round(r_trans, 10),
                round(r_spin, 10),
                0.0,
                0.0,
                0.0,
            ],
            "carrier_mass_kg": round(carrier_mass_kg, 6),
            "carrier_inertia_kg_m2": [round(v, 10) for v in carrier["inertia"]],
            "total_wheel_mass_kg": round(hub_mass + wheel_roller_mass[wl], 6),
        }
        all_assembly_parts.append(hub_combined)
        all_assembly_parts.append(dict(carrier, center=hub_combined["center"]))
        all_assembly_parts.append(
            {
                "mass": wheel_roller_mass[wl],
                "center": hub_combined["center"],
                "inertia": [
                    wheel_roller_mass[wl] * (0.015**2 / 2.0 + roller_len**2 / 12.0),
                    wheel_roller_mass[wl] * (0.015**2),
                    wheel_roller_mass[wl] * (0.015**2 / 2.0 + roller_len**2 / 12.0),
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

    total_combined = combine(all_assembly_parts)
    _check_inertia_valid(total_combined["inertia"], "total_assembly")

    # Discrete N20 motor variant + continuous datasheet range within variant.
    variants = prior["n20_motor_variants"]
    weights = [float(v["weight"]) for v in variants]
    variant = rng.choices(variants, weights=weights, k=1)[0]
    rated_v = float(variant["rated_voltage_v"])
    rated_rpm = rng.uniform(float(variant["no_load_rpm_low"]), float(variant["no_load_rpm_high"]))
    rated_stall_kgf_cm = rng.uniform(
        float(variant["stall_torque_kgf_cm_low"]), float(variant["stall_torque_kgf_cm_high"])
    )

    pack_v = _draw_spec(rng, sd["battery_pack_voltage_v"])
    supply_eff = _draw_spec(rng, sd["supply_efficiency"])
    effective_v = pack_v * supply_eff
    v_scale = effective_v / rated_v
    no_load_rpm_eff = rated_rpm * v_scale
    no_load_speed_radps = no_load_rpm_eff * 2.0 * math.pi / 60.0
    stall_torque_nm = rated_stall_kgf_cm * GRAVITY_KGF_CM_TO_NM * v_scale

    roller_mu = _draw_spec(rng, sd["roller_mu"])
    roller_bearing_damping = _draw_spec(rng, sd["roller_bearing_damping_nm_s_per_rad"])
    roller_bearing_friction = _draw_spec(rng, sd["roller_bearing_friction_nm"])
    deadzone_base = _draw_spec(rng, sd["wheel_deadzone"])
    duty_exp = _draw_spec(rng, sd["wheel_duty_exponent"])
    drive_tau = _draw_spec(rng, sd["wheel_drive_tau_s"])
    brake_tau = _draw_spec(rng, sd["wheel_brake_tau_s"])
    brake_damping = _draw_spec(rng, sd["brake_damping_nm_s_per_rad"])
    brake_res_frac = _draw_spec(rng, sd["brake_resistance_stall_fraction"])
    brake_resistance = stall_torque_nm * brake_res_frac
    wheel_gains = [round(_draw_spec(rng, sd["wheel_gain_spread"]), 6) for _ in range(4)]
    wheel_rev_gains = [round(g * rng.uniform(0.95, 1.05), 6) for g in wheel_gains]
    radial_stiffness = _draw_spec(rng, sd["tire_radial_stiffness_n_per_m"])
    radial_damping = _draw_spec(rng, sd["tire_radial_damping_ns_per_m"])
    radial_travel = _draw_spec(rng, sd["tire_radial_travel_m"])

    return {
        "status": PRIOR_STATUS,
        "warning": "PRIOR SAMPLE, not measured",
        "seed": seed,
        "prior_sha256": prior.get("_sha256", ""),
        "rejection_attempts": rejection_attempts,
        "mass": {
            "total_mass_kg": round(total_mass_kg, 6),
            "upper_bound_kg": float(prior["total_mass_upper_bound_kg"]["value"]),
            "assembly_com_m": [round(v, 6) for v in total_combined["center"]],
            "assembly_inertia_kg_m2": [round(v, 10) for v in total_combined["inertia"]],
            "chassis": chassis_inertial,
            "wheels": wheels_inertial,
            "sampled_densities_kg_per_m3": {k: round(v, 3) for k, v in densities.items()},
            "part_masses_kg": {k: round(v, 6) for k, v in part_masses.items()},
        },
        "motor": {
            "variant_id": variant["id"],
            "gear_ratio": int(variant["gear_ratio"]),
            "rated_voltage_v": rated_v,
            "rated_no_load_rpm": round(rated_rpm, 3),
            "rated_stall_torque_kgf_cm": round(rated_stall_kgf_cm, 4),
            "pack_voltage_v": round(pack_v, 4),
            "supply_efficiency": round(supply_eff, 4),
            "effective_voltage_v": round(effective_v, 4),
            "no_load_rpm_effective": round(no_load_rpm_eff, 3),
            "no_load_speed_radps": round(no_load_speed_radps, 6),
            "stall_torque_nm": round(stall_torque_nm, 6),
            "wheel_deadzone": [round(deadzone_base, 6)] * 4,
            "wheel_duty_exponent": [round(duty_exp, 6)] * 4,
            "wheel_voltage_gain": wheel_gains,
            "wheel_reverse_voltage_gain": wheel_rev_gains,
            "wheel_drive_tau_s": [round(drive_tau, 6)] * 4,
            "wheel_brake_tau_s": [round(brake_tau, 6)] * 4,
            "brake_damping_nm_s_per_rad": round(brake_damping, 6),
            "brake_resistance_nm": round(brake_resistance, 6),
        },
        "contact": {
            "roller_mu": round(roller_mu, 6),
            "roller_bearing_damping_nm_s_per_rad": round(roller_bearing_damping, 9),
            "roller_bearing_friction_nm": round(roller_bearing_friction, 9),
            "wheel_roller_count_per_wheel": n_rollers,
            "wheel_roller_ring_radius_m": float(cad_geo["wheel_roller_ring_radius_m"]),
            "wheel_roller_contact_radius_m": roller_r,
            "wheel_roller_length_m": roller_len,
            "mecanum_hand_sign": int(cad_geo["mecanum_hand_sign"]),
            "tire_radial_stiffness_n_per_m": round(radial_stiffness, 3),
            "tire_radial_damping_ns_per_m": round(radial_damping, 3),
            "tire_radial_travel_m": round(radial_travel, 6),
        },
    }
