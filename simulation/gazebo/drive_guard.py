"""Shared guard preventing Gazebo drive/measurement tools from misbehaving on unmeasured payload_car.

Unmeasured (static) payload_car worlds are always rejected. Prior-sampled payload_car worlds
generated via `generate.py --car payload --sample-seed N` carry an embedded sample record
(`PRIOR_SAMPLE_NOT_MEASURED`) and are permitted with an explicit banner warning.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

PAYLOAD_CAR_NOT_DRIVABLE_MSG = (
    "payload car has no measured mass or motor model "
    "(research/tasks/2026-11-27-characterise-payload-car.md); "
    "regenerate with `python3 generate.py --car payload --sample-seed N` for a prior sample "
    "or `python3 generate.py --car freenove ...` to drive the archived Freenove car"
)
PRIOR_SAMPLE_BANNER = "PRIOR SAMPLE, not measured"
PRIOR_SAMPLE_STATUS = "PRIOR_SAMPLE_NOT_MEASURED"

_PAYLOAD_TOPIC_MARKERS = ("/model/payload_car/", "/world/payload_lab/")


def _root_of(world: Path | str | ET.Element | ET.ElementTree) -> ET.Element:
    if isinstance(world, ET.Element):
        return world
    if isinstance(world, ET.ElementTree):
        root = world.getroot()
        if root is None:
            raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)
        return root
    return ET.parse(Path(world)).getroot()


def extract_prior_sample_record(
    world: Path | str | ET.Element | ET.ElementTree | None,
) -> dict | None:
    """Return the parsed prior-sample record if `world` is a valid drivable sampled payload world."""
    if world is None:
        return None
    root = _root_of(world)
    model = root.find('.//model[@name="payload_car"]')
    if model is None:
        return None
    if (model.findtext("static") or "").strip().lower() in {"true", "1"}:
        return None
    drive = model.find('plugin[@name="ugv::TTMotorSystem"]')
    if drive is None:
        return None
    links = model.findall("link")
    if not links or any(link.find("inertial") is None for link in links):
        return None
    status = (drive.findtext("calibration_status") or "").strip()
    seed_txt = (drive.findtext("prior_sample_seed") or "").strip()
    sample_txt = (drive.findtext("prior_sample_json") or "").strip()
    if status != PRIOR_SAMPLE_STATUS or not seed_txt or not sample_txt:
        return None
    try:
        seed = int(seed_txt)
        record = json.loads(sample_txt)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("status") != PRIOR_SAMPLE_STATUS:
        return None
    if record.get("seed") != seed:
        return None
    return record


def require_drivable_car(
    *,
    world: Path | str | ET.Element | ET.ElementTree | None = None,
    car: str | None = None,
    topics: Iterable[str] | None = None,
) -> dict | None:
    """Raise RuntimeError if target car/world/session is an unmeasured payload_car.

    Returns the prior sample record dict (and prints `PRIOR SAMPLE, not measured`) when
    accepting a sampled payload_car world; returns None for Freenove.
    """
    sample_record = extract_prior_sample_record(world) if world is not None else None

    if car is not None and car.strip().lower() in {"payload", "payload_car"}:
        if sample_record is None:
            raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if topics is not None and sample_record is None:
        for topic in topics:
            if any(marker in topic for marker in _PAYLOAD_TOPIC_MARKERS):
                raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if world is None:
        return None

    root = _root_of(world)
    if root.find('.//model[@name="payload_car"]') is not None:
        if sample_record is None:
            raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)
        mass_g = float(sample_record.get("mass", {}).get("total_mass_kg", 0.0)) * 1000.0
        variant = sample_record.get("motor", {}).get("variant_id", "unknown")
        print(
            f"{PRIOR_SAMPLE_BANNER} (seed={sample_record['seed']}, "
            f"mass={mass_g:.1f}g, motor={variant})",
            flush=True,
        )
        return sample_record

    freenove = root.find('.//model[@name="freenove"]')
    if freenove is None:
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if (freenove.findtext("static") or "").strip().lower() in {"true", "1"}:
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if freenove.find('plugin[@name="ugv::TTMotorSystem"]') is None:
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    links = freenove.findall("link")
    if links and any(link.find("inertial") is None for link in links):
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)
    return None
