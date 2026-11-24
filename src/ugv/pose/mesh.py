"""STRL-001 airframe mesh and the rigid keypoint skeleton used for pose estimation.

Frame (from `simulation/gazebo/strl_001/tags.json`): model frame, metres, x to the nose, y to
the left wing, z up. The part STLs in `simulation/gazebo/strl_001/meshes/` are already in this
frame (each part's vertices sit at its link pose), so parts are concatenated without a transform.
Control surfaces are hinged in the SDF; here they are held at zero deflection, so the skeleton
is rigid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[3] / "simulation" / "gazebo" / "strl_001" / "meshes"
PARTS = (
    "body",
    "nacelle_left",
    "nacelle_right",
    "prop_left",
    "prop_right",
    "prop_rear",
    "aileron_left",
    "aileron_right",
    "ruddervator_left",
    "ruddervator_right",
)
_STL = np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])


def load_stl(path: Path) -> np.ndarray:
    """Binary STL to triangles, shape (N, 3, 3), float64 metres."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL with {n} triangles")
    return np.frombuffer(raw, _STL, count=n, offset=84)["v"].reshape(n, 3, 3).astype(np.float64)


@dataclass(frozen=True)
class Airframe:
    """Triangles of the whole airframe and the part index of each triangle."""

    tris: np.ndarray  # (N, 3, 3) model frame, metres
    part: np.ndarray  # (N,) index into PARTS

    @property
    def centroids(self) -> np.ndarray:
        return self.tris.mean(axis=1)


def load_airframe(mesh_dir: Path = MESH_DIR) -> Airframe:
    tris, part = [], []
    for i, name in enumerate(PARTS):
        t = load_stl(mesh_dir / f"{name}.stl")
        tris.append(t)
        part.append(np.full(len(t), i))
    return Airframe(np.concatenate(tris), np.concatenate(part))


def keypoints(frame: Airframe, near: bool = False) -> dict[str, np.ndarray]:
    """Named 3D keypoints (model frame, metres) picked from mesh extremes, so they follow the mesh.

    Nose and tail on the centreline, wingtips, the three rotor hubs, the two V-tail tips, the nose
    wheel and the two main wheels (12 points, asymmetric enough to fix left from right). Replace an entry with a surveyed point when one exists; the
    pose solver only needs the names to agree between this map and the detector.

    `near=True` appends the three nose-tyre points (`tyre_left`, `tyre_right`, `tyre_top`): the
    wheel's left, right and top extremes seen from behind, which stay in view at bay range when
    the rest of the airframe is cut off by the picture. 15 points; the first 12 are unchanged.
    """
    v = frame.tris.reshape(-1, 3)
    pv = np.repeat(frame.part, 3)

    def part_centre(name: str) -> np.ndarray:
        return v[pv == PARTS.index(name)].mean(axis=0)

    def part_extreme(name: str, axis: int, sign: int) -> np.ndarray:
        pts = v[pv == PARTS.index(name)]
        return pts[(sign * pts[:, axis]).argmax()]

    body = v[pv == 0]
    centre = body[np.abs(body[:, 1]) < 0.03]  # fuselage centreline strip
    kp = {
        "nose": centre[centre[:, 0].argmax()],
        "tail": centre[centre[:, 0].argmin()],
        "wingtip_left": v[v[:, 1].argmax()],
        "wingtip_right": v[v[:, 1].argmin()],
        "hub_left": part_centre("prop_left"),
        "hub_right": part_centre("prop_right"),
        "hub_rear": part_centre("prop_rear"),
        "vtail_tip_left": part_extreme("ruddervator_left", 2, 1),
        "vtail_tip_right": part_extreme("ruddervator_right", 2, 1),
    }
    # Landing gear: the lowest 2% of body vertices. The nose gear is one wheel on the
    # centreline; the main gear is a left/right pair behind it.
    low = body[body[:, 2] <= np.quantile(body[:, 2], 0.02)]
    split = 0.5 * (low[:, 0].min() + low[:, 0].max())
    front, rear = low[low[:, 0] >= split], low[low[:, 0] < split]
    kp["gear_nose"] = front.mean(axis=0)
    kp["gear_main_left"] = rear[rear[:, 1] >= 0].mean(axis=0)
    kp["gear_main_right"] = rear[rear[:, 1] < 0].mean(axis=0)
    if near:
        floor = float(frame.tris[:, :, 2].min())
        g = kp["gear_nose"]
        wheel = v[(np.linalg.norm(v[:, :2] - g[:2], axis=1) < 0.05) & (v[:, 2] - floor < 0.06)]
        kp["tyre_left"] = wheel[wheel[:, 1].argmax()]
        kp["tyre_right"] = wheel[wheel[:, 1].argmin()]
        kp["tyre_top"] = wheel[wheel[:, 2].argmax()]
    return kp
