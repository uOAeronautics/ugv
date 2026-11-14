"""South workshop objects inferred from the room rotation video (SI metres)."""

import math

SOUTH_COLORS = {
    "detail_white": (0.84, 0.85, 0.81),
    "detail_frame": (0.34, 0.38, 0.40),
    "detail_dark": (0.035, 0.042, 0.045),
    "detail_wood": (0.66, 0.55, 0.36),
    "detail_blue": (0.035, 0.18, 0.32),
    "detail_grey": (0.47, 0.50, 0.51),
    "detail_card": (0.46, 0.33, 0.20),
    "detail_tape": (0.67, 0.55, 0.34),
}


def south_details():
    out = []

    def group(prefix, x, y):
        def add(n, shape, p, size, c, rot=(0, 0, 0), collision=False):
            out.append(
                dict(
                    name="d_south_" + prefix + "_" + n,
                    shape=shape,
                    pos=(x + p[0], y + p[1], p[2]),
                    size=size,
                    color="detail_" + c,
                    rot=rot,
                    collision=collision,
                )
            )

        def box(n, p, size, c, rot=(0, 0, 0), collision=True):
            add(n, "box", p, size, c, rot, collision)

        def cyl(n, p, r, h, c, rot=(0, 0, 0), collision=False):
            add(n, "cylinder", p, (r, h), c, rot, collision)

        def rod(n, a, b, r, c, collision=True):
            d = tuple(b[k] - a[k] for k in range(3))
            length = math.sqrt(sum(v * v for v in d))
            cyl(
                n,
                tuple((a[k] + b[k]) / 2 for k in range(3)),
                r,
                length,
                c,
                (0, math.acos(d[2] / length), math.atan2(d[1], d[0])),
                collision,
            )

        return box, cyl, rod

    b, c, r = group("board", 1.8, -6.2)
    b("surface", (0, 0, 1.15), (1.1, 0.045, 1.30), "white")
    for sx in (-1, 1):
        b("upright" + str(sx), (sx * 0.57, 0, 0.98), (0.035, 0.06, 1.73), "frame")
        b("foot" + str(sx), (sx * 0.57, 0, 0.10), (0.06, 0.64, 0.045), "frame")
        for sy in (-1, 1):
            c(
                "wheel%s%s" % (sx, sy),
                (sx * 0.57, sy * 0.27, 0.055),
                0.045,
                0.04,
                "dark",
                (math.pi / 2, 0, 0),
                True,
            )
    for z in (0.49, 1.81):
        b("rail" + str(z), (0, 0, z), (1.16, 0.06, 0.03), "frame")
    b("pen_tray", (0, 0.065, 0.49), (1.06, 0.12, 0.025), "frame")
    b("eraser", (0.24, 0.07, 0.52), (0.14, 0.055, 0.03), "dark", collision=False)
    # Irregular writing rows and clipped paper visible on the reference board.
    for j in range(22):
        b(
            "writing" + str(j),
            (-0.42 + (j % 3) * 0.29, 0.025, 1.65 - (j // 3) * 0.14),
            (0.16 + 0.025 * (j % 2), 0.003, 0.005),
            "dark",
            collision=False,
        )
    b("top_clip", (0.30, 0.035, 1.82), (0.24, 0.012, 0.045), "card", collision=False)
    b("clipped_paper", (0.31, 0.04, 1.74), (0.22, 0.002, 0.17), "white", collision=False)
    b, c, r = group("drawer_bench", 4, -5)
    b("carcass", (0, 0, 0.45), (1.3, 1.8, 0.84), "frame")
    b("worktop", (0, 0, 0.905), (1.38, 1.87, 0.065), "wood")
    for k in range(7):
        z = 0.13 + k * 0.105
        b("drawer" + str(k), (-0.66, 0, z), (0.025, 1.68, 0.092), "grey")
        b("handle" + str(k), (-0.69, 0, z + 0.015), (0.035, 0.80, 0.018), "dark", collision=False)
    for sy in (-1, 1):
        b("gantry_post" + str(sy), (0.55, sy * 0.80, 1.37), (0.045, 0.045, 0.91), "frame")
    b("gantry_beam", (0.55, 0, 1.82), (0.045, 1.65, 0.045), "frame")
    b("gantry_shelf", (0.38, 0, 1.78), (0.40, 1.70, 0.035), "wood")
    for k, (xx, yy, ww, dd, hh) in enumerate(
        (
            (-0.22, -0.45, 0.56, 0.49, 0.35),
            (0.16, 0.28, 0.62, 0.68, 0.42),
            (0.25, -0.58, 0.25, 0.30, 0.18),
        )
    ):
        z = 0.94 + hh / 2
        b("box" + str(k), (xx, yy, z), (ww, dd, hh), "card")
        b(
            "tape" + str(k),
            (xx, yy, z + hh / 2 + 0.002),
            (0.055, dd, 0.004),
            "tape",
            collision=False,
        )
    b, c, r = group("folded", 0, -10.8)
    for j in range(3):
        xx = -0.25 + j * 0.18
        b("table" + str(j), (xx, 0, 0.93), (0.055, 1.75, 0.78), "white", (0, 0.09, 0))
        for sy in (-1, 1):
            r(
                "table_leg%s%s" % (j, sy),
                (xx - 0.05, sy * 0.65, 0.58),
                (xx - 0.05, sy * 0.65, 1.26),
                0.018,
                "frame",
            )
    for j in range(3):
        xx = -1.6 + j * 0.43
        for sx in (-1, 1):
            r(
                "frame_leg%s%s" % (j, sx),
                (xx + sx * 0.18, 0.50, 0.04),
                (xx + sx * 0.18, 0.50, 1.36),
                0.016,
                "dark",
            )
        r("frame_top" + str(j), (xx - 0.18, 0.5, 1.36), (xx + 0.18, 0.5, 1.36), 0.016, "dark")
        r("frame_base" + str(j), (xx - 0.18, 0.5, 0.12), (xx + 0.18, 0.5, 0.12), 0.016, "dark")
    for j in range(2):
        xx = 1 + j * 0.53
        b("chairseat" + str(j), (xx, 0, 0.45), (0.42, 0.41, 0.035), "white")
        b("chairback" + str(j), (xx, -0.19, 0.69), (0.41, 0.045, 0.43), "white", (-0.09, 0, 0))
        for sx in (-1, 1):
            for sy in (-1, 1):
                r(
                    "chairleg%s%s%s" % (j, sx, sy),
                    (xx + sx * 0.17, sy * 0.15, 0.44),
                    (xx + sx * 0.21, sy * 0.20, 0.025),
                    0.015,
                    "white",
                )
    b, c, r = group("rack", -5, -10.5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b("post%s%s" % (sx, sy), (sx * 0.63, sy * 0.28, 0.97), (0.035, 0.035, 1.94), "dark")
    for j in range(4):
        z = 0.15 + j * 0.50
        b("shelf" + str(j), (0, 0, z), (1.3, 0.60, 0.035), "dark")
        for k in range(2):
            xx = -0.32 + k * 0.64
            b("bin%d_%d" % (j, k), (xx, 0, z + 0.16), (0.51, 0.48, 0.28), "dark")
            b(
                "label%d_%d" % (j, k),
                (xx, 0.246, z + 0.16),
                (0.18, 0.005, 0.07),
                "white",
                collision=False,
            )
            b("lid%d_%d" % (j, k), (xx, 0, z + 0.31), (0.54, 0.50, 0.025), "grey", collision=False)
            b(
                "latch%d_%d" % (j, k),
                (xx + 0.18, 0.252, z + 0.27),
                (0.055, 0.012, 0.055),
                "dark",
                collision=False,
            )
    b, c, r = group("tray_stack", -3.95, -10.5)
    for j in range(12):
        z = 0.07 + j * 0.085
        b("base" + str(j), (0, 0, z), (0.55, 0.66, 0.018), "grey")
        for sx in (-1, 1):
            b("side%s%s" % (j, sx), (sx * 0.265, 0, z + 0.04), (0.02, 0.66, 0.075), "grey")
        for sy in (-1, 1):
            b("end%s%s" % (j, sy), (0, sy * 0.32, z + 0.04), (0.53, 0.02, 0.075), "grey")
    b, c, r = group("trash", -3, -9)
    c("body", (0, 0, 0.29), 0.22, 0.56, "dark", collision=True)
    c("rim", (0, 0, 0.58), 0.235, 0.045, "grey")
    c("inner", (0, 0, 0.605), 0.197, 0.006, "dark")
    # The drawer island is a short cabinet, not a full-length workshop table.
    for part in out:
        if part["name"].startswith("d_south_drawer_bench_"):
            x, y, z = part["pos"]
            part["pos"] = (x, -5 + (y + 5) * 0.72, z)
            if part["shape"] == "box":
                a, b, c = part["size"]
                part["size"] = (a, b * 0.72, c)
    return out
