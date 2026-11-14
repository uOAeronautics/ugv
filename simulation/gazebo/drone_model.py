"""Detailed photo-informed stationary quadcopter parking target.

X is along the two landing skids, Y left, Z up; origin at floor level.
The flat part schema matches assembly.py, adding shape=mesh and mesh=<path>.
Collision follows separate tubes/plates: the central drive-under space stays open.
"""

import json
import math
import random
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRONE_COLORS = {
    "carbon": (0.027, 0.032, 0.035),
    "carbon_edge": (0.012, 0.016, 0.019),
    "foam": (0.018, 0.019, 0.022),
    "red": (0.69, 0.025, 0.055),
    "silver": (0.58, 0.63, 0.66),
    "steel": (0.28, 0.32, 0.34),
    "white": (0.93, 0.94, 0.90),
    "black": (0.006, 0.008, 0.009),
    "copper": (0.55, 0.22, 0.075),
    "green": (0.045, 0.24, 0.10),
    "yellow": (0.98, 0.62, 0.015),
    "blue": (0.015, 0.25, 0.65),
    "wire_white": (0.75, 0.76, 0.67),
    "gold": (0.67, 0.40, 0.08),
}
MARKER_LAYOUTS = ("real", "legacy")
DEFAULT_MARKER_LAYOUT = "real"
REAL_MARKER_IDS = (14, 6, 3)
LEGACY_MARKER_IDS = (12, 9, 13, 14)
# Real loose paper cards (observed 2026-11-13): 50mm tag with ~1.5mm bottom quiet zone,
# ~3mm sides, and ~18mm top hanging margin (left, right, bottom, top in card frame).
REAL_PAPER_BOUNDS = (-0.028, 0.028, -0.0265, 0.043)


def _rot_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _euler_xyz(m):
    ry = math.asin(max(-1.0, min(1.0, -m[2][0])))
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(m[2][1], m[2][2])
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = 0.0
        rz = math.atan2(-m[0][1], m[1][1])
    return (rx, ry, rz)


def _apply_card_tilt(rot, rng, tilt_deg):
    lo, hi = tilt_deg

    def signed():
        return math.radians(rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi))

    # Pitch/tilt around card X and roll around card Z (loose hanging paper card).
    return _euler_xyz(_mat_mul(_rot_xyz(*rot), _rot_xyz(signed(), 0.0, signed())))


def _prop_mesh():
    """Cambered tapered two-blade propeller, closed triangle surface, SI units."""
    folder = HERE / "assets" / "drone"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "propeller.obj"
    verts = []
    faces = []
    # Root, broad shoulder, curved swept tip. Each section has upper/lower lip.
    stations = [
        (0.009, 0.008, 0),
        (0.027, 0.014, 0.003),
        (0.061, 0.024, 0.005),
        (0.105, 0.022, 0.008),
        (0.148, 0.014, 0.012),
        (0.181, 0.007, 0.016),
        (0.190, 0.001, 0.017),
    ]
    for direction in (1, -1):
        start = len(verts)
        for x, chord, sweep in stations:
            pitch = 0.21 * (1 - x / 0.24)
            for side, zlip in ((-1, -0.001), (1, -0.001), (1, 0.001), (-1, 0.001)):
                y = sweep + side * chord
                z = side * chord * pitch + zlip
                verts.append((direction * x, direction * y, z))
        for i in range(len(stations) - 1):
            for k in range(4):
                a = start + i * 4 + k
                b = start + i * 4 + (k + 1) % 4
                c = start + (i + 1) * 4 + (k + 1) % 4
                d = start + (i + 1) * 4 + k
                faces.extend(((a, b, c), (a, c, d)))
        faces.extend(((start, start + 2, start + 1), (start, start + 3, start + 2)))
        j = start + (len(stations) - 1) * 4
        faces.extend(((j, j + 1, j + 2), (j, j + 2, j + 3)))
    text = "# Photo estimated 380mm two blade propeller\n"
    text += "".join(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n" for v in verts)
    text += "".join(f"f {f[0] + 1:d} {f[1] + 1:d} {f[2] + 1:d}\n" for f in faces)
    # Gazebo generation and Blender previews may request this asset concurrently.
    # Preserve unchanged assets, and publish changed contents as one atomic rename.
    try:
        unchanged = path.read_text(encoding="utf-8") == text
    except FileNotFoundError:
        unchanged = False
    if not unchanged:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=folder,
                prefix=".propeller-",
                suffix=".obj",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(text)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return "assets/drone/propeller.obj"


def drone_assembly(
    layout=DEFAULT_MARKER_LAYOUT, randomize_tilt=False, rng=None, tilt_deg=(10.0, 25.0)
):
    """Return reusable visual/collision part dictionaries (all dimensions SI).

    layout='real' (default, 2026-11-13): DICT_4X4_50 IDs 14 (+y_sim entry, 25 deg pitch),
    6 (+x_sim side), 3 (-x_sim side); fourth -y_sim tag omitted (unreadable behind wires).
    layout='legacy': original 4-card constellation (9, 12, 13, 14) for reproducibility.
    """
    if layout not in MARKER_LAYOUTS:
        raise ValueError(f"layout must be one of {MARKER_LAYOUTS}, got {layout!r}")
    p = json.loads((HERE / "drone_dimensions.json").read_text())
    parts = []

    def add(name, shape, pos, size, color, rot=(0, 0, 0), collision=False, **extra):
        parts.append(
            dict(
                name=name,
                shape=shape,
                pos=tuple(pos),
                size=tuple(size),
                color=color,
                rot=tuple(rot),
                collision=collision,
                **extra,
            )
        )

    def box(n, xyz, size, c, rot=(0, 0, 0), collision=False):
        add(n, "box", xyz, size, c, rot, collision)

    def cyl(n, xyz, r, h, c, rot=(0, 0, 0), collision=False):
        add(n, "cylinder", xyz, (r, h), c, rot, collision)

    def rod(n, a, b, r, c, collision=False):
        dx, dy, dz = [b[i] - a[i] for i in range(3)]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        cyl(
            n,
            [(a[i] + b[i]) / 2 for i in range(3)],
            r,
            length,
            c,
            (0, math.acos(max(-1, min(1, dz / length))), math.atan2(dy, dx)),
            collision,
        )

    def wire(n, points, c, r=0.0015):
        for i, (a, b) in enumerate(zip(points, points[1:], strict=False)):
            rod(n + "_" + str(i), a, b, r, c)

    # Continuous black skid cores with individual foam sleeves and red annular seams.
    for s in (-1, 1):
        y = s * p["skid_center_spacing"] / 2
        rod(f"skid_core_{s}", (-0.235, y, 0.016), (0.235, y, 0.016), 0.015, "carbon", True)
        for j in range(23):
            x = -0.22 + j * 0.02
            cyl(f"foam_{s}_{j}", (x, y, 0.017), 0.015, 0.0175, "foam", (0, math.pi / 2, 0))
            cyl(
                f"red_ring_{s}_{j}",
                (x + 0.009, y, 0.017),
                0.0151,
                0.0018,
                "red",
                (0, math.pi / 2, 0),
            )
        for end in (-1, 1):
            cyl(
                f"endcap_{s}_{end}",
                (end * 0.237, y, 0.017),
                0.0105,
                0.006,
                "steel",
                (0, math.pi / 2, 0),
            )
        # One splayed tube joins the CENTER of each skid to the central deck.
        # Photos 1/3/5 show T-shaped landing gear, rather than a four-leg cage.
        a = (0, s * 0.095, 0.407)
        b = (0, y, 0.029)
        rod(f"landing_leg_{s}", a, b, 0.011, "carbon", True)
        rod(f"leg_upper_collar_{s}", a, (0, s * 0.103, 0.373), 0.014, "green")
        rod(f"leg_lower_collar_{s}", b, (0, s * 0.164, 0.066), 0.013, "carbon_edge")
        box(f"leg_clamp_{s}", (0, y, 0.032), (0.032, 0.035, 0.015), "carbon_edge")
        cyl(f"leg_clamp_bolt_{s}", (0, y, 0.042), 0.003, 0.004, "silver")

    # Stacked circular carbon sandwich, exposed silver spacers and perimeter bolts.
    for z in (0.410, 0.438):
        cyl(
            "center_plate_" + str(z),
            (0, 0, z),
            p["deck_diameter"] / 2,
            0.004,
            "carbon",
            collision=True,
        )
        for j in range(12):
            a = j * math.tau / 12
            x = 0.105 * math.cos(a)
            y = 0.105 * math.sin(a)
            cyl(f"plate_bolt_{z}_{j}", (x, y, z + 0.003), 0.0032, 0.002, "silver")
            if z == 0.410:
                cyl(f"plate_spacer_{j}", (x, y, 0.424), 0.003, 0.024, "steel")
    propmesh = _prop_mesh()
    # Four radial carbon booms, bolted root saddles, vented motor cages.
    for i, a in enumerate((math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)):
        c, s = math.cos(a), math.sin(a)
        x, y = 0.36 * c, 0.36 * s
        rod(f"boom_{i}", (0.08 * c, 0.08 * s, 0.43), (x, y, 0.43), 0.013, "carbon", True)
        for rr in (0.105, 0.145):
            box(
                f"boom_saddle_{i}_{rr}",
                (rr * c, rr * s, 0.43),
                (0.032, 0.039, 0.044),
                "carbon_edge",
                (0, 0, a),
            )
            for q in (-1, 1):
                xx = rr * c - q * 0.015 * s
                yy = rr * s + q * 0.015 * c
                cyl(f"boom_bolt_{i}_{rr}_{q}", (xx, yy, 0.454), 0.0028, 0.003, "silver")
        box(f"motor_base_{i}", (x, y, 0.431), (0.065, 0.057, 0.005), "carbon", (0, 0, a), True)
        for k in (-1, 1):
            box(
                f"motor_cage_{i}_{k}",
                (x - k * 0.028 * s, y + k * 0.028 * c, 0.451),
                (0.063, 0.004, 0.030),
                "carbon_edge",
                (0, 0, a),
            )
            for j in (-1, 0, 1):
                # Silver heads give the elongated motor brackets a perforated appearance.
                cyl(
                    f"cage_fastener_{i}_{k}_{j}",
                    (x + j * 0.019 * c - k * 0.031 * s, y + j * 0.019 * s + k * 0.031 * c, 0.448),
                    0.003,
                    0.002,
                    "steel",
                    (math.pi / 2, 0, a),
                )
        cyl(f"motor_bottom_{i}", (x, y, 0.461), 0.025, 0.008, "steel")
        cyl(f"motor_can_{i}", (x, y, 0.479), 0.024, 0.026, "black", collision=True)
        cyl(f"motor_top_{i}", (x, y, 0.494), 0.023, 0.004, "silver")
        for j in range(12):
            t = j * math.tau / 12
            box(
                f"motor_winding_{i}_{j}",
                (x + 0.015 * math.cos(t), y + 0.015 * math.sin(t), 0.496),
                (0.004, 0.006, 0.005),
                "copper",
                (0, 0, t),
            )
        cyl(f"prop_hub_{i}", (x, y, 0.505), 0.013, 0.012, "silver")
        add(
            f"propeller_{i}",
            "mesh",
            (x, y, 0.508),
            (1, 1, 1),
            "white",
            (0, 0, a + 0.25 + i * 0.49),
            mesh=propmesh,
        )
        # Rounded conical spinner built from rings avoids external mesh dependency.
        for j in range(9):
            cyl(
                f"spinner_{i}_{j}",
                (x, y, 0.516 + j * 0.0022),
                0.0085 * (1 - j / 10) ** 0.65,
                0.0024,
                "silver",
            )
        wire(
            f"esc_wire_{i}",
            [
                (0.08 * c, 0.08 * s, 0.447),
                (0.16 * c, 0.16 * s, 0.452),
                (0.29 * c, 0.29 * s, 0.447),
                (x, y, 0.459),
            ],
            "black",
            0.002,
        )
        box(f"esc_{i}", (0.19 * c, 0.19 * s, 0.445), (0.055, 0.019, 0.009), "black", (0, 0, a))

    # Flight controller electronics, little connectors, ribbon sockets, tangled harness.
    box("autopilot", (0, 0.014, 0.457), (0.075, 0.044, 0.019), "black", collision=True)
    box("controller_label", (-0.008, 0.014, 0.467), (0.035, 0.027, 0.0008), "white")
    box("receiver", (0.061, -0.017, 0.455), (0.041, 0.028, 0.015), "green")
    box("receiver_shield", (0.061, -0.017, 0.464), (0.025, 0.020, 0.003), "silver")
    for j in range(9):
        box(
            f"signal_socket_{j}", (-0.038 + j * 0.009, 0.042, 0.465), (0.006, 0.008, 0.008), "black"
        )
        cyl(f"socket_pin_{j}", (-0.038 + j * 0.009, 0.042, 0.470), 0.0006, 0.003, "gold")
    for i, c in enumerate(("red", "blue", "yellow", "green", "wire_white", "black") * 3):
        xx = -0.075 + (i % 7) * 0.021
        yy = -0.065 + (i % 4) * 0.033
        wire(
            f"harness_{i}",
            [
                (xx, yy, 0.446),
                (xx + 0.009, yy - 0.014, 0.466 + i % 3 * 0.006),
                (-xx * 0.6, 0.024 + (i % 4) * 0.005, 0.479 + i % 5 * 0.003),
                (0.025, -0.025, 0.461),
            ],
            c,
            0.00075,
        )
    rod("radio_antenna", (-0.055, -0.025, 0.47), (-0.17, -0.08, 0.471), 0.003, "black")
    # Tall GPS puck on a slender carbon mast, with visible loose red/black wiring.
    rod("gps_mast", (-0.033, 0.025, 0.444), (-0.033, 0.025, 0.637), 0.0035, "carbon", True)
    cyl("gps_stalk_foot", (-0.033, 0.025, 0.449), 0.009, 0.010, "black")
    cyl("gps_puck", (-0.033, 0.025, 0.665), 0.027, 0.055, "black", collision=True)
    cyl("gps_puck_rim", (-0.033, 0.025, 0.640), 0.028, 0.005, "carbon_edge")
    cyl("gps_cap", (-0.033, 0.025, 0.693), 0.026, 0.002, "carbon")
    for j, c in enumerate(("red", "black", "wire_white")):
        wire(
            f"gps_cable_{j}",
            [
                (-0.033 + j * 0.003, 0.025, 0.638),
                (-0.041 + j * 0.003, 0.029, 0.592),
                (-0.032 + j * 0.003, 0.024, 0.549),
                (-0.025 + j * 0.003, 0.021, 0.47),
            ],
            c,
            0.0008,
        )

    # Open hanging payload cage. Its bottom is above the UGV, no floor-to-deck box.
    for x in (-0.046, 0.046):
        for y in (-0.035, 0.035):
            rod(f"payload_post_{x}_{y}", (x, y, 0.406), (x, y, 0.267), 0.0025, "carbon", True)
    box("payload_shelf", (0, 0, 0.266), (0.108, 0.085, 0.004), "carbon", collision=True)
    box("payload_silver", (0, 0, 0.28), (0.065, 0.055, 0.025), "silver", collision=True)
    box("gimbal_servo", (0.008, -0.052, 0.248), (0.042, 0.022, 0.033), "black", collision=True)
    box("servo_label", (0.008, -0.064, 0.248), (0.021, 0.001, 0.021), "white")
    cyl("servo_horn", (0.008, -0.066, 0.248), 0.011, 0.004, "gold", (math.pi / 2, 0, 0))
    box("payload_camera", (0.037, 0, 0.239), (0.022, 0.042, 0.028), "black", collision=True)
    cyl("payload_lens", (0.051, 0, 0.239), 0.009, 0.008, "black", (0, math.pi / 2, 0))
    # Dangling power connectors kept near one side of overhead payload.
    for i, c in enumerate(("red", "black")):
        wire(
            f"power_dangle_{i}",
            [
                (0.057, -0.07 + i * 0.004, 0.422),
                (0.075, -0.10 + i * 0.004, 0.35),
                (0.09, -0.11 + i * 0.004, 0.315),
            ],
            c,
            0.0022,
        )
    box("yellow_xt60", (0.092, -0.11, 0.308), (0.021, 0.015, 0.022), "yellow", (0, 0.3, 0.15))
    for j in (-1, 1):
        cyl("xt60_socket_" + str(j), (0.092 + j * 0.004, -0.11, 0.295), 0.0025, 0.003, "gold")
    wire(
        "servo_cable",
        [(-0.012, -0.04, 0.245), (-0.03, -0.05, 0.21), (-0.019, -0.06, 0.196)],
        "black",
        0.0011,
    )

    # Real DICT_4X4_50 markers, 50mm black square.
    grids = {
        0: ("1011", "0101", "0011", "0010"),
        1: ("0000", "1111", "1001", "1010"),
        3: ("1001", "1001", "0100", "0110"),
        6: ("1001", "1110", "0010", "1110"),
        9: ("1100", "1111", "0101", "0110"),
        12: ("0000", "1110", "1011", "0111"),
        13: ("0010", "1010", "0000", "1111"),
        14: ("0010", "0100", "1011", "0001"),
    }

    def card(name, pos, rot, size, marker_id, paper_bounds=None):
        # Local XY plane, normal +Z; rotate whole card into vertical plane.
        rx, ry, rz = rot

        def transform(v):
            x, y, z = v
            y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
            x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
            x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
            return tuple(pos[i] + (x, y, z)[i] for i in range(3))

        if paper_bounds is None:
            box(name + "_paper", pos, (size, size, 0.0008), "white", rot)
        else:
            left_b, r, b, t = paper_bounds
            box(
                name + "_paper",
                transform((0.5 * (left_b + r), 0.5 * (b + t), 0.0)),
                (r - left_b, t - b, 0.0008),
                "white",
                rot,
            )
        cell = 0.05 / 6
        # One printed face per physical card; ink stays centred at pos.
        for face in (1,):
            box(
                name + "_ink_" + str(face),
                transform((0, 0, face * 0.0006)),
                (0.05, 0.05, 0.0003),
                "black",
                rot,
            )
            for row, line in enumerate(grids[marker_id], 1):
                for col, bit in enumerate(line, 1):
                    if bit == "1":
                        xx = (col - 2.5) * cell * face
                        box(
                            f"{name}_pixel_{face}_{row}_{col}",
                            transform((xx, (2.5 - row) * cell, face * 0.0009)),
                            (cell, cell, 0.0003),
                            "white",
                            rot,
                        )

    if layout == "legacy":
        card(
            "payload_marker_front",
            (-0.058, -0.005, 0.202),
            (math.pi / 2, 0, -math.pi / 2),
            0.068,
            13,
        )
        card(
            "payload_marker_rear", (0.058, -0.005, 0.202), (math.pi / 2, 0, math.pi / 2), 0.068, 14
        )
        for s in (-1, 1):
            card(
                "leg_marker_" + str(s),
                (0, s * 0.168, 0.092),
                (-s * math.pi / 2, 0, 0),
                0.068,
                9 if s == -1 else 12,
            )
    else:
        # Gazebo drone frame G maps to rl/sim.py frame S via x_G=-y_S, y_G=+x_S, z_G=z_S.
        # Fourth -y_S card omitted (ID unreadable behind wires).
        tilt_rng = (rng if rng is not None else random.Random(0)) if randomize_tilt else None
        specs = (
            (
                "entry_marker_14",
                (0.0, -0.028, 0.14),
                (math.pi / 2 - math.radians(25.0), 0.0, -math.pi / 2),
                14,
            ),
            ("side_marker_6", (0.09, 0.14, 0.15), (math.pi / 2, 0.0, math.pi), 6),
            ("side_marker_3", (0.0, -0.13, 0.16), (math.pi / 2, 0.0, 0.0), 3),
        )
        for cname, pos, rot, mid in specs:
            if tilt_rng is not None:
                rot = _apply_card_tilt(rot, tilt_rng, tilt_deg)
            card(cname, pos, rot, 0.068, mid, paper_bounds=REAL_PAPER_BOUNDS)
    return parts


def add_drone(
    world,
    node,
    pose=(0.8, 0, 0, 0, 0, 0),
    layout=DEFAULT_MARKER_LAYOUT,
    randomize_tilt=False,
    rng=None,
    tilt_deg=(10.0, 25.0),
):
    """Insert static photo model with only local solid collision components."""
    model = node(world, "model", name="photo_quadcopter")
    node(model, "static", "true")
    node(model, "pose", " ".join(map(str, pose)))
    link = node(model, "link", name="drone_frame")
    for part in drone_assembly(
        layout=layout, randomize_tilt=randomize_tilt, rng=rng, tilt_deg=tilt_deg
    ):
        for kind in ("visual", "collision") if part["collision"] else ("visual",):
            obj = node(link, kind, name=part["name"] + "_" + kind)
            node(obj, "pose", " ".join(map(str, (*part["pos"], *part["rot"]))))
            shape = node(node(obj, "geometry"), part["shape"])
            if part["shape"] == "box":
                node(shape, "size", " ".join(map(str, part["size"])))
            elif part["shape"] == "mesh":
                node(shape, "uri", part["mesh"])
                node(shape, "scale", "1 1 1")
            else:
                node(shape, "radius", part["size"][0])
                node(shape, "length", part["size"][1])
            if kind == "visual":
                mat = node(obj, "material")
                rgba = " ".join(map(str, (*DRONE_COLORS[part["color"]], 1)))
                node(mat, "ambient", rgba)
                node(mat, "diffuse", rgba)
                node(
                    mat,
                    "specular",
                    ".22 .22 .22 1" if part["color"] == "silver" else ".025 .025 .025 1",
                )
    return model
