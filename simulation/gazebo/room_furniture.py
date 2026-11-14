"""Photo-informed workshop furniture. SI dimensions, estimated from room video.
Reusable flat parts for Gazebo and Blender. Static collision on structural parts.
"""

import math
import os
import tempfile
from pathlib import Path

FURNITURE_COLORS = {
    "room_red": (0.63, 0.012, 0.027),
    "room_red_edge": (0.36, 0.008, 0.015),
    "room_black": (0.018, 0.021, 0.025),
    "room_rubber": (0.026, 0.029, 0.034),
    "room_grey": (0.27, 0.30, 0.32),
    "room_metal": (0.48, 0.52, 0.54),
    "room_wood": (0.72, 0.61, 0.40),
    "room_wood_edge": (0.52, 0.42, 0.27),
    "room_cardboard": (0.43, 0.30, 0.17),
    "room_tape": (0.67, 0.54, 0.32),
    "room_white": (0.86, 0.87, 0.82),
    "room_screen": (0.025, 0.055, 0.065),
    "room_blue": (0.018, 0.17, 0.32),
    "room_green": (0.03, 0.23, 0.11),
    "room_orange": (0.86, 0.24, 0.025),
    "room_cork": (0.43, 0.31, 0.19),
}
HERE = Path(__file__).resolve().parent


def _atomic_text(path, text):
    """Publish complete mesh bytes without racing concurrent scene readers."""
    if path.exists() and path.read_text() == text:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _chair_shell():
    """Moulded polypropylene seat/back with curved edges and lumbar sweep."""
    path = HERE / "assets/room/chair_shell.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = []
    faces = []
    # Longitudinal seat-to-back profile, rear is local +Y.
    profile = [
        (-0.235, 0.445),
        (-0.20, 0.436),
        (-0.10, 0.426),
        (0.02, 0.426),
        (0.12, 0.44),
        (0.18, 0.49),
        (0.205, 0.58),
        (0.22, 0.68),
        (0.24, 0.79),
        (0.25, 0.83),
    ]
    for layer in (-0.005, 0.005):
        for j, (y, z) in enumerate(profile):
            half = 0.225 if j < 7 else 0.218 - (j - 7) * 0.008
            for k in range(9):
                u = -1 + k / 4
                verts.append((u * half, y, z + 0.025 * u * u + layer))
    n = len(profile) * 9
    for layer in range(2):
        off = layer * n
        for j in range(len(profile) - 1):
            for k in range(8):
                a = off + j * 9 + k
                b = a + 1
                c = a + 10
                d = a + 9
                faces.extend(((a, b, c), (a, c, d)) if layer else ((a, c, b), (a, d, c)))
    for j in range(len(profile) - 1):
        for k in (0, 8):
            a = j * 9 + k
            b = (j + 1) * 9 + k
            faces.extend(((a, b, b + n), (a, b + n, a + n)))
    for j in (0, len(profile) - 1):
        for k in range(8):
            a = j * 9 + k
            b = a + 1
            faces.extend(((a, b, b + n), (a, b + n, a + n)))
    _atomic_text(
        path,
        "# Curved moulded sled chair shell\n"
        + "".join(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
        + "".join(f"f {f[0] + 1:d} {f[1] + 1:d} {f[2] + 1:d}\n" for f in faces),
    )
    return "assets/room/chair_shell.obj"


def furniture_assembly():
    parts = []
    shell = _chair_shell()

    def group(prefix, x, y, yaw=0):
        def add(n, shape, pos, size, color, rot=(0, 0, 0), collision=False, **extra):
            xx, yy, z = pos
            c = math.cos(yaw)
            s = math.sin(yaw)
            parts.append(
                dict(
                    name=prefix + "_" + n,
                    shape=shape,
                    pos=(x + c * xx - s * yy, y + s * xx + c * yy, z),
                    size=size,
                    color="room_" + color,
                    rot=(rot[0], rot[1], rot[2] + yaw),
                    collision=collision,
                    **extra,
                )
            )

        def box(n, p, s, c, rot=(0, 0, 0), collision=True):
            add(n, "box", p, s, c, rot, collision)

        def cyl(n, p, r, h, c, rot=(0, 0, 0), collision=False):
            add(n, "cylinder", p, (r, h), c, rot, collision)

        def rod(n, a, b, r, c, collision=True):
            d = [b[i] - a[i] for i in range(3)]
            h = math.sqrt(sum(v * v for v in d))
            cyl(
                n,
                tuple((a[i] + b[i]) / 2 for i in range(3)),
                r,
                h,
                c,
                (0, math.acos(d[2] / h), math.atan2(d[1], d[0])),
                collision,
            )

        return add, box, cyl, rod

    def table(n, x, y, yaw=0, w=1.6, d=0.8):
        a, b, c, r = group(n, x, y, yaw)
        b("laminate", (0, 0, 0.742), (w, d, 0.036), "wood")
        b("edge", (0, 0, 0.72), (w + 0.004, d + 0.004, 0.012), "wood_edge")
        for s in (-1, 1):
            xx = s * (w / 2 - 0.18)
            b("upright" + str(s), (xx, 0, 0.385), (0.075, 0.075, 0.64), "metal")
            b("foot" + str(s), (xx, 0, 0.055), (0.065, d - 0.10, 0.045), "metal")
            for k in (-1, 1):
                b(
                    f"endpad{s}{k}",
                    (xx, k * (d / 2 - 0.08), 0.026),
                    (0.085, 0.08, 0.027),
                    "rubber",
                )
        b("crossbar", (0, 0, 0.51), (w - 0.30, 0.045, 0.08), "grey")
        return a, b, c, r

    def chair(n, x, y, yaw=0, black=False):
        a, b, c, r = group(n, x, y, yaw)
        a("shell", "mesh", (0, 0, 0), (1, 1, 1), "black" if black else "red", mesh=shell)
        b(
            "seat_collision",
            (0, 0, 0.44),
            (0.43, 0.40, 0.025),
            "red" if not black else "black",
            collision=True,
        )
        for s in (-1, 1):
            r("front" + str(s), (s * 0.205, -0.18, 0.43), (s * 0.23, -0.25, 0.025), 0.011, "black")
            r("rear" + str(s), (s * 0.205, 0.16, 0.48), (s * 0.23, 0.29, 0.025), 0.011, "black")
            r("sled" + str(s), (s * 0.23, -0.25, 0.025), (s * 0.23, 0.29, 0.025), 0.011, "black")

    def office(n, x, y, yaw=0):
        a, b, c, r = group(n, x, y, yaw)
        c("column", (0, 0, 0.29), 0.032, 0.37, "metal", collision=True)
        c("collar", (0, 0, 0.17), 0.049, 0.12, "black")
        for k in range(5):
            t = k * math.tau / 5
            ex = 0.29 * math.cos(t)
            ey = 0.29 * math.sin(t)
            r("spoke" + str(k), (0, 0, 0.13), (ex, ey, 0.075), 0.019, "black")
            c("caster" + str(k), (ex, ey, 0.045), 0.043, 0.05, "rubber", (math.pi / 2, 0, t))
        b("seat", (0, 0, 0.47), (0.49, 0.47, 0.075), "black")
        b("back", (0, 0.22, 0.72), (0.46, 0.07, 0.43), "black", (0.12, 0, 0))
        for s in (-1, 1):
            r("armstem" + str(s), (s * 0.23, 0.11, 0.49), (s * 0.29, 0.11, 0.65), 0.013, "grey")
            b("armpad" + str(s), (s * 0.29, 0.015, 0.65), (0.06, 0.30, 0.04), "black")

    def workstation(n, x, y, yaw=0):
        a, b, c, r = table(n, x, y, yaw, w=1.65, d=0.75)
        for j, xx in enumerate((-0.37, 0.25)):
            b("monitor_foot" + str(j), (xx, 0.16, 0.777), (0.25, 0.18, 0.018), "black")
            b("monitor_stand" + str(j), (xx, 0.19, 0.90), (0.04, 0.04, 0.24), "grey")
            b("monitor" + str(j), (xx, 0.205, 1.09), (0.55, 0.045, 0.33), "black")
            b(
                "display" + str(j),
                (xx, 0.179, 1.09),
                (0.505, 0.004, 0.285),
                "screen",
                collision=False,
            )
        b("keyboard", (0, -0.17, 0.775), (0.44, 0.15, 0.02), "black")
        for row in range(4):
            for k in range(12):
                b(
                    f"key{row}_{k}",
                    (-0.195 + k * 0.035, -0.215 + row * 0.029, 0.788),
                    (0.027, 0.021, 0.006),
                    "grey",
                    collision=False,
                )
        b("mouse", (0.36, -0.16, 0.783), (0.05, 0.085, 0.025), "black")
        b("tower", (0.62, 0.12, 0.30), (0.22, 0.42, 0.54), "black")
        for k in range(5):
            b(
                "tower_vent" + str(k),
                (0.62, -0.095, 0.35 + k * 0.022),
                (0.15, 0.005, 0.006),
                "grey",
                collision=False,
            )
        cs = math.cos(yaw)
        sn = math.sin(yaw)
        office(n + "_chair", x + sn * 0.7, y - cs * 0.7, yaw)

    def carton(b, n, p, size):
        b(n, p, size, "cardboard")
        b(
            n + "_tape",
            (p[0], p[1], p[2] + size[2] / 2 + 0.001),
            (0.05, size[1], 0.002),
            "tape",
            collision=False,
        )

    def toolchest(n, x, y, yaw=0, w=0.9, h=1.05):
        a, b, c, r = group(n, x, y, yaw)
        b("body", (0, 0, h / 2 + 0.08), (w, 0.50, h), "red")
        b("top", (0, 0, h + 0.09), (w + 0.025, 0.53, 0.025), "black")
        for j in range(7):
            z = 0.19 + j * (h - 0.12) / 7
            b("drawer" + str(j), (0, -0.256, z), (w - 0.05, 0.018, 0.10), "red_edge")
            b(
                "pull" + str(j),
                (0, -0.277, z + 0.025),
                (w - 0.15, 0.025, 0.018),
                "metal",
                collision=False,
            )
        for sx in (-1, 1):
            for sy in (-1, 1):
                c(
                    f"wheel{sx}{sy}",
                    (sx * (w / 2 - 0.09), sy * 0.19, 0.065),
                    0.06,
                    0.045,
                    "rubber",
                    (math.pi / 2, 0, 0),
                    True,
                )

    def shelf(n, x, y, yaw=0):
        a, b, c, r = group(n, x, y, yaw)
        for sx in (-1, 1):
            for sy in (-1, 1):
                b(f"post{sx}{sy}", (sx * 0.72, sy * 0.24, 1.05), (0.04, 0.04, 2.1), "grey")
        for j in range(5):
            z = 0.12 + j * 0.46
            b("shelf" + str(j), (0, 0, z), (1.5, 0.55, 0.045), "wood")
            for k in range(3):
                xx = -0.48 + k * 0.47
                if (j + k) % 2:
                    carton(b, f"carton{j}_{k}", (xx, 0, z + 0.16), (0.38, 0.40, 0.28))
                else:
                    b(f"bin{j}_{k}", (xx, 0, z + 0.13), (0.38, 0.41, 0.22), "blue")
                    b(
                        f"label{j}_{k}",
                        (xx, -0.207, z + 0.13),
                        (0.15, 0.003, 0.06),
                        "white",
                        collision=False,
                    )

    # Perimeter arrangement leaves a generous central manoeuvring area.
    for i, (x, y) in enumerate(
        ((2.0, 3.5), (3.65, 3.5), (2.0, 4.35), (3.65, 4.35), (-8.3, 0.5), (-8.3, 1.4))
    ):
        a, b, c, r = table("teaching_table" + str(i), x, y)
        if i in (0, 2, 4):
            b("notebook", (-0.3, 0.07, 0.773), (0.28, 0.21, 0.015), "white", collision=False)
            b("electronics", (0.3, -0.08, 0.79), (0.21, 0.13, 0.043), "green", collision=False)
            for k in range(4):
                b(
                    "chip" + str(k),
                    (0.24 + k * 0.035, -0.08, 0.815),
                    (0.022, 0.04, 0.009),
                    "black",
                    collision=False,
                )
        if i < 2:
            chair("chair_a" + str(i), x - 0.4, y - 0.75, 0)
        elif i < 4:
            chair("chair_b" + str(i), x + 0.4, y + 0.75, math.pi)
        else:
            chair("chair_a" + str(i), x - 0.4, y - 0.75, 0, black=i == 4)
    # Outer end chairs around the joined teaching tables, never between tops.
    for j, y in enumerate((3.5, 4.35)):
        chair("teaching_end_left" + str(j), 0.45, y, -math.pi / 2)
        chair("teaching_end_right" + str(j), 5.15, y, math.pi / 2)
    for i, x in enumerate((6.5, 8.2, 9.9)):
        workstation("computer" + str(i), x, 5.15)
    for i, (x, y, yaw) in enumerate(((-6, -3, 0), (-7.15, -3, 0), (5, -4.8, 0))):
        toolchest("tools" + str(i), x, y, yaw)
    shelf("supply_shelf", 10.9, -4.9)
    shelf("rear_shelf", -9.9, -4.9)
    # Shop island with grey welded frame and broad wood work surface.
    a, b, c, r = table("shop_island", 7, -3, w=3.4, d=1.25)
    b("lower_shelf", (0, 0, 0.24), (3.15, 1.1, 0.05), "grey")
    for i, xx in enumerate((-0.95, -0.20, 0.7)):
        carton(b, "box" + str(i), (xx, 0.1, 0.91), (0.55, 0.45, 0.28))
    b("bench_vise", (1.25, -0.43, 0.87), (0.22, 0.17, 0.19), "blue")
    b("vise_jaws", (1.25, -0.43, 0.975), (0.26, 0.13, 0.04), "metal")
    r("vise_handle", (1.25, -0.56, 0.85), (1.25, -0.56, 1.06), 0.009, "metal", False)
    # Tall grey cupboards and black equipment lockers.
    for j, x in enumerate((-10.5, -9.55, -8.6)):
        a, b, c, r = group("locker" + str(j), x, 5.48)
        b("body", (0, 0, 1.0), (0.88, 0.65, 2.0), "grey" if j < 2 else "black")
        for s in (-1, 1):
            b(
                "door" + str(s),
                (s * 0.214, -0.334, 1.02),
                (0.416, 0.018, 1.91),
                "metal" if j < 2 else "black",
            )
            b("handle" + str(s), (s * 0.075, -0.36, 1.0), (0.015, 0.025, 0.15), "black")
            for k in range(4):
                b(
                    f"vent{s}{k}",
                    (s * 0.214, -0.346, 1.75 - k * 0.035),
                    (0.25, 0.005, 0.009),
                    "grey",
                    collision=False,
                )
    # Cork partition with Canadian flag seen in the workshop panorama.
    a, b, c, r = group("flag_partition", 6.8, -4.7, math.pi)
    b("panel", (0, 0, 1.20), (3.4, 0.10, 1.5), "cork")
    for s in (-1, 1):
        b("post" + str(s), (s * 1.73, 0, 0.95), (0.09, 0.14, 1.9), "grey")
        b("foot" + str(s), (s * 1.73, 0, 0.065), (0.12, 0.75, 0.08), "grey")
    b("flag", (0, -0.057, 1.25), (1.35, 0.008, 0.675), "white", collision=False)
    for s in (-1, 1):
        b(
            "flag_red" + str(s),
            (s * 0.50, -0.064, 1.25),
            (0.35, 0.008, 0.675),
            "red",
            collision=False,
        )
    # Geometric maple silhouette assembled from rotated angular lobes.
    b("maple_core", (0, -0.072, 1.25), (0.19, 0.005, 0.25), "red", collision=False)
    for s in (-1, 1):
        b(
            "maple_lobe" + str(s),
            (s * 0.12, -0.072, 1.28),
            (0.14, 0.005, 0.19),
            "red",
            (0, s * 0.50, 0),
            False,
        )
        b(
            "maple_outer" + str(s),
            (s * 0.19, -0.072, 1.24),
            (0.13, 0.005, 0.10),
            "red",
            (0, s * 0.65, 0),
            False,
        )
    b("maple_top", (0, -0.072, 1.40), (0.085, 0.005, 0.15), "red", collision=False)
    b("maple_stem", (0, -0.072, 1.08), (0.02, 0.005, 0.12), "red", collision=False)
    # Storage tubs and buckets under the partition.
    for j in range(3):
        b(
            "tub" + str(j),
            (-1 + j * 0.6, 0, 0.17),
            (0.48, 0.42, 0.28),
            "blue" if j == 0 else "black",
        )
    c("bucket", (0.96, 0, 0.20), 0.15, 0.32, "white")
    c("bucket_lid", (0.96, 0, 0.37), 0.16, 0.025, "orange")
    # Foam aircraft and unfinished foam stock stored on the workshop wall.
    a, b, c, r = group("foam_aircraft", -3.8, -5.65)
    b("fuselage", (0, 0, 2.0), (0.10, 0.08, 1.25), "white", collision=False)
    for side in (-1, 1):
        b(
            "wing" + str(side),
            (side * 0.34, 0, 2.02),
            (0.72, 0.035, 0.27),
            "blue",
            (0, -side * 0.35, 0),
            False,
        )
        b(
            "tail" + str(side),
            (side * 0.15, 0, 1.52),
            (0.35, 0.035, 0.15),
            "blue",
            (0, -side * 0.25, 0),
            False,
        )
    b("cockpit", (0, -0.06, 2.22), (0.08, 0.065, 0.23), "black", collision=False)
    a, b, c, r = group("white_foam_stock", -5.4, -5.42)
    for j in range(4):
        b(
            "board" + str(j),
            (j * 0.09, 0, 0.72),
            (0.07, 0.54, 1.45),
            "white",
            (0, 0.10 + j * 0.025, 0.025 * j),
        )
    # Reference-view accessories: open laptops, power leads and workshop storage.
    for j, (xx, yy) in enumerate(((2.15, 3.55), (3.85, 3.48))):
        a, b, c, r = group("ref_laptop" + str(j), xx, yy, 0.06 - j * 0.16)
        b("base", (0, 0, 0.781), (0.34, 0.24, 0.018), "metal")
        b("keyboard", (0, 0.015, 0.792), (0.29, 0.12, 0.004), "black", collision=False)
        b("trackpad", (0, -0.075, 0.792), (0.09, 0.045, 0.004), "grey", collision=False)
        b("screen", (0, 0.123, 0.907), (0.34, 0.014, 0.235), "black", (-0.12, 0, 0))
        b("display", (0, 0.113, 0.907), (0.311, 0.003, 0.205), "screen", (-0.12, 0, 0), False)
        for k in range(5):
            b(
                "screen_line" + str(k),
                (-0.05, 0.109, 0.96 - k * 0.022),
                (0.12 + k * 0.015, 0.002, 0.004),
                "metal",
                (-0.12, 0, 0),
                False,
            )
    a, b, c, r = group("ref_power", 2.77, 3.54)
    b("strip", (0, 0, 0.785), (0.07, 0.34, 0.035), "white")
    for k in range(4):
        b(
            "socket" + str(k),
            (0, -0.12 + k * 0.078, 0.805),
            (0.035, 0.045, 0.004),
            "grey",
            collision=False,
        )
    cable = [
        (0.02, 0.18, 0.79),
        (0.13, 0.30, 0.78),
        (0.30, 0.39, 0.78),
        (0.42, 0.42, 0.76),
        (0.48, 0.48, 0.65),
        (0.49, 0.49, 0.06),
        (0.38, 0.35, 0.018),
        (0.25, 0.27, 0.018),
    ]
    for k, (v, w) in enumerate(zip(cable, cable[1:], strict=False)):
        r("lead" + str(k), v, w, 0.004, "black", False)
    a, b, c, r = group("ref_backpack", 3.4, 2.5, -0.25)
    b("body", (0, 0, 0.25), (0.30, 0.19, 0.46), "black", (0.12, 0, 0))
    b("pocket", (0, -0.11, 0.20), (0.25, 0.07, 0.25), "rubber", (0.12, 0, 0))
    for side in (-1, 1):
        r(
            "zip" + str(side),
            (side * 0.13, -0.15, 0.10),
            (side * 0.13, -0.14, 0.33),
            0.004,
            "grey",
            False,
        )
        r(
            "strap" + str(side),
            (side * 0.11, 0.105, 0.39),
            (side * 0.13, 0.19, 0.08),
            0.013,
            "black",
            False,
        )
    r("handle", (-0.06, 0, 0.49), (0.06, 0, 0.49), 0.013, "black", False)
    for j, (xx, yy) in enumerate(((-6.4, -2.9), (-7.8, -3.2))):
        a, b, c, r = group("ref_stool" + str(j), xx, yy)
        c("seat", (0, 0, 0.53), 0.19, 0.07, "black", collision=True)
        c("post", (0, 0, 0.31), 0.025, 0.40, "metal", collision=True)
        for k in range(5):
            t = k * math.tau / 5
            ex = 0.25 * math.cos(t)
            ey = 0.25 * math.sin(t)
            r("spoke" + str(k), (0, 0, 0.12), (ex, ey, 0.07), 0.013, "grey")
            c("caster" + str(k), (ex, ey, 0.04), 0.035, 0.035, "black", (math.pi / 2, 0, t))
    a, b, c, r = group("ref_stacked_tables", 10.5, -3.7, 0.08)
    for j in range(5):
        xx = (j - 2) * 0.095
        b("top" + str(j), (xx, 0, 0.95), (0.055, 1.58, 0.80), "wood", (0, 0.06, 0))
        for sy in (-1, 1):
            r(
                f"folded_leg{j}{sy}",
                (xx - 0.035, sy * 0.60, 0.57),
                (xx - 0.035, sy * 0.60, 1.29),
                0.021,
                "metal",
            )
    for sy in (-1, 1):
        b("rack_foot" + str(sy), (0, sy * 0.65, 0.06), (0.90, 0.07, 0.07), "grey")
    for j in range(5):
        a, b, c, r = group("ref_stacked_chair" + str(j), 8, -4.7 + j * 0.045, 0.1)
        a("shell", "mesh", (0, 0, j * 0.075), (1, 1, 1), "black", mesh=shell)
        for side in (-1, 1):
            r(
                "front" + str(side),
                (side * 0.205, -0.18, 0.43 + j * 0.075),
                (side * 0.23, -0.25, 0.025 + j * 0.075),
                0.011,
                "black",
            )
            r(
                "rear" + str(side),
                (side * 0.205, 0.16, 0.48 + j * 0.075),
                (side * 0.23, 0.29, 0.025 + j * 0.075),
                0.011,
                "black",
            )
            r(
                "sled" + str(side),
                (side * 0.23, -0.25, 0.025 + j * 0.075),
                (side * 0.23, 0.29, 0.025 + j * 0.075),
                0.011,
                "black",
            )
    return parts
