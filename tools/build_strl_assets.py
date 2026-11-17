"""Build the STRL-001 web assets from the Gazebo reference package.

Reads the reference SDF and its ASCII STL meshes, converts every mesh to binary
STL, and writes the assembly decomposition plus the browser manifest.  The SDF is
the single source of truth: geometry, poses and joint wiring are parsed, never
hand-copied.

    uv run python tools/build_strl_assets.py
"""

from __future__ import annotations

import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "simulation/gazebo/strl_001"
SDF_PATH = MODEL_DIR / "model.sdf"
DECOMPOSITION_PATH = MODEL_DIR / "decomposition.json"
OUT_DIR = ROOT / "src/ugv/lab/static/models/strl001"
MANIFEST_PATH = OUT_DIR / "manifest.json"
TAGS_PATH = MODEL_DIR / "tags.json"

BINARY_HEADER = b"strl001 binary STL"
FLOATS = 6


# ---------------------------------------------------------------- SDF parsing


def _floats(text: str | None, count: int = FLOATS) -> list[float]:
    """Parse a whitespace-separated pose vector, defaulting to zeros."""
    if not text:
        return [0.0] * count
    values = [float(v) for v in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}: {text!r}")
    return values


def _inertia(link: ET.Element) -> dict[str, float]:
    node = link.find("inertial/inertia")
    if node is None:
        return {}
    return {child.tag: float(child.text or 0.0) for child in node}


def _mass(link: ET.Element) -> float:
    node = link.find("inertial/mass")
    return float(node.text) if node is not None and node.text else 0.0


def _visual_mesh(link: ET.Element) -> tuple[str | None, list[float] | None]:
    """Return (mesh basename, scale) for the link's first mesh visual."""
    mesh = link.find("visual/geometry/mesh")
    if mesh is None:
        return None, None
    uri = mesh.findtext("uri")
    if not uri:
        return None, None
    scale_text = mesh.findtext("scale")
    return Path(uri).name, (_floats(scale_text, 3) if scale_text else None)


def parse_sdf(path: Path = SDF_PATH) -> dict:
    """Parse the reference SDF into links, joints and the assembly tree."""
    model = ET.parse(path).getroot().find("model")
    if model is None:
        raise ValueError(f"no <model> in {path}")

    links: list[dict] = []
    for link in model.findall("link"):
        mesh, scale = _visual_mesh(link)
        links.append(
            {
                "name": link.get("name"),
                "mesh": mesh,
                "scale": scale,
                "pose": _floats(link.findtext("pose")),
                "mass": _mass(link),
                "inertia": _inertia(link),
            }
        )

    joints: list[dict] = []
    for joint in model.findall("joint"):
        axis = joint.find("axis")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.findtext("parent"),
                "child": joint.findtext("child"),
                "axis": _floats(axis.findtext("xyz"), 3) if axis is not None else [0.0, 0.0, 0.0],
                "pose": _floats(joint.findtext("pose")),
            }
        )

    tree = build_tree(links, joints)
    return {
        "model": model.get("name"),
        "links": links,
        "joints": joints,
        "tree": tree,
        "groups": group_links([link["name"] for link in links]),
    }


def build_tree(links: list[dict], joints: list[dict]) -> dict:
    """Root is the only link that is never a joint child; children map parent -> kids."""
    names = [link["name"] for link in links]
    children = {name: [] for name in names}
    child_names = set()
    for joint in joints:
        children[joint["parent"]].append(joint["child"])
        child_names.add(joint["child"])
    roots = [name for name in names if name not in child_names]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    return {"root": roots[0], "children": children}


def group_links(names: list[str]) -> dict[str, list[str]]:
    """Classify links by role using the model's naming convention."""
    groups: dict[str, list[str]] = {"airframe": [], "control": [], "propulsion": [], "sensor": []}
    for name in names:
        if name == "base_link":
            groups["airframe"].append(name)
        elif "aileron" in name or "ruddervator" in name:
            groups["control"].append(name)
        elif "nacelle" in name or "prop" in name:
            groups["propulsion"].append(name)
        else:
            groups["sensor"].append(name)
    return groups


# ---------------------------------------------------------------- STL handling


def is_ascii_stl(path: Path) -> bool:
    """Detect ASCII vs binary STL.

    ASCII files start with 'solid' and contain facet records.  Binary files may also
    start with 'solid', so a file whose size matches 84 + 50 * triangle_count is
    treated as binary regardless of its header.
    """
    with path.open("rb") as handle:
        head = handle.read(1024)
    if len(head) >= 84:
        (count,) = struct.unpack("<I", head[80:84])
        if path.stat().st_size == 84 + 50 * count:
            return False
    if head.lstrip()[:5].lower() != b"solid":
        return False
    return b"facet" in head


def read_ascii_stl(path: Path) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Return [(normal, (v0, v1, v2))] for an ASCII STL, as flat float tuples."""
    triangles = []
    normal: tuple[float, ...] = (0.0, 0.0, 0.0)
    vertices: list[float] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "facet" and len(parts) >= 5:
            normal = tuple(float(v) for v in parts[2:5])
            vertices = []
        elif parts[0] == "vertex" and len(parts) >= 4:
            vertices.extend(float(v) for v in parts[1:4])
            if len(vertices) == 9:
                triangles.append((normal, tuple(vertices)))
                vertices = []
    return triangles


def read_binary_stl(path: Path) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Return [(normal, (v0, v1, v2))] for a binary STL."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: truncated binary STL")
    (count,) = struct.unpack("<I", data[80:84])
    triangles = []
    offset = 84
    for _ in range(count):
        values = struct.unpack("<12f", data[offset : offset + 48])
        triangles.append((values[0:3], values[3:12]))
        offset += 50
    return triangles


def read_stl(path: Path) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    return read_ascii_stl(path) if is_ascii_stl(path) else read_binary_stl(path)


def write_binary_stl(path: Path, triangles: list) -> int:
    """Write a binary STL with a fixed header so repeated runs produce identical bytes."""
    header = BINARY_HEADER.ljust(80, b"\0")
    chunks = [header, struct.pack("<I", len(triangles))]
    for normal, vertices in triangles:
        chunks.append(struct.pack("<12f", *normal, *vertices))
        chunks.append(b"\0\0")
    payload = b"".join(chunks)
    path.write_bytes(payload)
    return len(payload)


# ---------------------------------------------------------------- assembly


def binary_name(mesh: str) -> str:
    """body.stl -> body.bin.stl"""
    return f"{Path(mesh).stem}.bin.stl"


def joint_for(link_name: str, joints: list[dict]) -> dict | None:
    for joint in joints:
        if joint["child"] == link_name:
            return {
                "type": joint["type"],
                "axis": joint["axis"],
                "parent": joint["parent"],
                "child": joint["child"],
            }
    return None


def build_parts(sdf: dict, mesh_dir: Path, out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Convert every referenced mesh; return (manifest parts, size table rows)."""
    parts: list[dict] = []
    rows: list[dict] = []
    for link in sdf["links"]:
        mesh = link["mesh"]
        if mesh is None:
            parts.append(
                {
                    "id": link["name"],
                    "file": None,
                    "tris": 0,
                    "pose": link["pose"],
                    "joint": joint_for(link["name"], sdf["joints"]),
                }
            )
            continue
        source = mesh_dir / mesh
        if not source.exists():
            raise FileNotFoundError(f"SDF references missing mesh: {source}")
        triangles = read_stl(source)
        if not triangles:
            raise ValueError(f"{source}: no triangles parsed")
        target = out_dir / binary_name(mesh)
        written = write_binary_stl(target, triangles)
        before = source.stat().st_size
        rows.append(
            {
                "part": link["name"],
                "mesh": mesh,
                "tris": len(triangles),
                "before": before,
                "after": written,
            }
        )
        parts.append(
            {
                "id": link["name"],
                "file": target.name,
                "tris": len(triangles),
                "pose": link["pose"],
                "joint": joint_for(link["name"], sdf["joints"]),
            }
        )
    return parts, rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    print(f"{'part':<24}{'mesh':<24}{'tris':>8}{'ascii B':>12}{'binary B':>12}{'ratio':>8}")
    for row in rows:
        ratio = row["before"] / row["after"] if row["after"] else 0.0
        print(
            f"{row['part']:<24}{row['mesh']:<24}{row['tris']:>8}"
            f"{row['before']:>12}{row['after']:>12}{ratio:>7.2f}x"
        )
    total_before = sum(row["before"] for row in rows)
    total_after = sum(row["after"] for row in rows)
    print(f"{'total':<48}{sum(r['tris'] for r in rows):>8}{total_before:>12}{total_after:>12}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sdf = parse_sdf()
    write_json(DECOMPOSITION_PATH, sdf)

    parts, rows = build_parts(sdf, MODEL_DIR / "meshes", OUT_DIR)
    manifest = {
        "parts": parts,
        "tree": sdf["tree"],
        "tags": json.loads(TAGS_PATH.read_text(encoding="utf-8"))["tags"],
        "totalTris": sum(part["tris"] for part in parts),
        "totalBytes": sum(row["after"] for row in rows),
    }
    write_json(MANIFEST_PATH, manifest)

    print_table(rows)
    print(
        f"{len(sdf['links'])} links, {len(sdf['joints'])} joints, "
        f"{manifest['totalTris']} triangles -> {MANIFEST_PATH.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
