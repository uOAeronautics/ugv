"""Shared guard preventing Gazebo drive/measurement tools from misbehaving on payload_car."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

PAYLOAD_CAR_NOT_DRIVABLE_MSG = (
    "payload car has no measured mass or motor model "
    "(research/tasks/2026-11-27-characterise-payload-car.md); "
    "regenerate with `python3 generate.py --car freenove ...` "
    "to drive the archived Freenove car"
)

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


def require_drivable_car(
    *,
    world: Path | str | ET.Element | ET.ElementTree | None = None,
    car: str | None = None,
    topics: Iterable[str] | None = None,
) -> None:
    """Raise RuntimeError if the target car, SDF world, or live Gazebo session is payload_car."""
    if car is not None and car.strip().lower() in {"payload", "payload_car"}:
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if topics is not None:
        for topic in topics:
            if any(marker in topic for marker in _PAYLOAD_TOPIC_MARKERS):
                raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

    if world is None:
        return

    root = _root_of(world)
    if root.find('.//model[@name="payload_car"]') is not None:
        raise RuntimeError(PAYLOAD_CAR_NOT_DRIVABLE_MSG)

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
