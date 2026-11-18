"""Build the Freenove car web asset from the Gazebo assembly.

`simulation/gazebo/assembly.py` is the source of truth for the car's visual parts (boxes and
cylinders, SDF conventions: pose = x y z roll pitch yaw, cylinder axis along z). This writes them,
grouped as in the SDF (chassis plus one group per wheel), to the Lab dashboard:

    uv run python tools/build_car_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAZEBO = ROOT / "simulation/gazebo"
OUT = ROOT / "src/ugv/lab/static/models/freenove/assembly.json"


def build() -> dict:
    sys.path.insert(0, str(GAZEBO))
    from assembly import COLORS, assembly

    p = json.loads((GAZEBO / "dimensions.json").read_text())
    groups = assembly(p)
    r4 = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    return {
        "source": "simulation/gazebo/assembly.py",
        "frame": "x forward, y left, z up, metres; car origin on the floor under the chassis centre",
        "colors": {k: r4(v) for k, v in COLORS.items()},
        "groups": {
            name: {
                "origin": r4(g["origin"]),
                "parts": [
                    {
                        "shape": q["shape"],
                        "pos": r4(q["pos"]),
                        "size": r4(q["size"]),
                        "rot": r4(q["rot"]),
                        "color": q["color"],
                    }
                    for q in g["parts"]
                ],
            }
            for name, g in groups.items()
        },
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    n = sum(len(g["parts"]) for g in data["groups"].values())
    print(f"{OUT.relative_to(ROOT)}: {len(data['groups'])} groups, {n} parts")


if __name__ == "__main__":
    main()
