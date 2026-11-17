"""Checks for the generated STRL-001 web assets and the SDF decomposition."""

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_strl_assets as assets  # noqa: E402

MANIFEST = assets.MANIFEST_PATH
DECOMPOSITION = assets.DECOMPOSITION_PATH


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), "run tools/build_strl_assets.py"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def decomposition() -> dict:
    assert DECOMPOSITION.exists(), "run tools/build_strl_assets.py"
    return json.loads(DECOMPOSITION.read_text(encoding="utf-8"))


def test_manifest_parses_and_has_parts(manifest):
    mesh_parts = [part for part in manifest["parts"] if part["file"]]
    assert 9 <= len(mesh_parts) <= 10
    assert manifest["totalTris"] > 0
    assert manifest["totalTris"] == sum(part["tris"] for part in manifest["parts"])


def test_every_part_file_exists(manifest):
    for part in manifest["parts"]:
        if part["file"] is None:
            continue
        path = MANIFEST.parent / part["file"]
        assert path.exists(), f"{part['id']} -> {part['file']} missing"
        assert assets.is_ascii_stl(path) is False
        data = path.read_bytes()
        (count,) = struct.unpack("<I", data[80:84])
        assert len(data) == 84 + 50 * count, f"{path.name} is not a valid binary STL"
        assert count == part["tris"]


def test_manifest_total_bytes_matches_files(manifest):
    total = sum(
        (MANIFEST.parent / part["file"]).stat().st_size
        for part in manifest["parts"]
        if part["file"]
    )
    assert total == manifest["totalBytes"]


def test_decomposition_matches_sdf(decomposition):
    sdf = assets.parse_sdf()
    assert decomposition["links"] == sdf["links"]
    assert decomposition["joints"] == sdf["joints"]
    assert decomposition["tree"] == sdf["tree"]

    assert len(decomposition["links"]) == 11
    assert len(decomposition["joints"]) == 10
    assert decomposition["tree"]["root"] == "base_link"

    types = {joint["type"] for joint in decomposition["joints"]}
    assert types == {"fixed", "revolute"}

    # Every mesh the SDF references exists on disk and was converted.
    converted = {part["file"] for part in json.loads(MANIFEST.read_text())["parts"] if part["file"]}
    for link in decomposition["links"]:
        if link["mesh"] is None:
            continue
        assert (assets.MODEL_DIR / "meshes" / link["mesh"]).exists()
        assert assets.binary_name(link["mesh"]) in converted

    groups = decomposition["groups"]
    assert set(groups) == {"airframe", "control", "propulsion", "sensor"}
    assert sum(len(names) for names in groups.values()) == len(decomposition["links"])


def test_ascii_stl_converts_to_binary(tmp_path):
    ascii_stl = tmp_path / "tri.stl"
    ascii_stl.write_text(
        "solid tri\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid tri\n",
        encoding="ascii",
    )
    assert assets.is_ascii_stl(ascii_stl) is True
    triangles = assets.read_stl(ascii_stl)
    assert len(triangles) == 1

    first = tmp_path / "a.bin.stl"
    second = tmp_path / "b.bin.stl"
    size = assets.write_binary_stl(first, triangles)
    assets.write_binary_stl(second, triangles)
    assert size == 84 + 50
    assert first.read_bytes() == second.read_bytes()
    assert assets.is_ascii_stl(first) is False
    assert assets.read_stl(first) == triangles


def test_manifest_carries_the_real_tag_constellation(manifest):
    tags = {tag["id"]: tag for tag in manifest["tags"]}
    assert set(tags) == {14, 6, 3}
    for tag in tags.values():
        assert abs(sum(c * c for c in tag["normal"]) - 1.0) < 1e-9
        assert len(tag["pos"]) == 3
