"""Door-side meeting bay observed in the second walking reference (metres)."""

import math

WALK_COLORS = {
    "walk_grey": (0.30, 0.33, 0.34),
    "walk_dark": (0.026, 0.03, 0.032),
    "walk_paper": (0.81, 0.80, 0.73),
    "walk_red": (0.65, 0.02, 0.025),
    "walk_blue": (0.025, 0.18, 0.34),
    "walk_screen": (0.22, 0.34, 0.44),
    "walk_wood": (0.58, 0.43, 0.27),
}


def walk_parts():
    p = []

    def box(n, xyz, size, c="walk_dark", rot=(0, 0, 0), collision=False):
        p.append(
            dict(
                name="walk_" + n,
                shape="box",
                pos=xyz,
                size=size,
                color=c,
                rot=rot,
                collision=collision,
            )
        )

    def rod(n, a, b, r=0.006, c="walk_dark"):
        d = [b[i] - a[i] for i in range(3)]
        h = math.sqrt(sum(v * v for v in d))
        p.append(
            dict(
                name="walk_" + n,
                shape="cylinder",
                pos=tuple((a[i] + b[i]) / 2 for i in range(3)),
                size=(r, h),
                color=c,
                rot=(0, math.acos(d[2] / h), math.atan2(d[1], d[0])),
                collision=False,
            )
        )

    # Open laptop and grey prototype on the joined meeting tables.
    box("laptop_base", (-3.45, -3.20, 0.775), (0.34, 0.25, 0.018), "walk_grey")
    box("laptop_lid", (-3.45, -3.08, 0.90), (0.34, 0.018, 0.25), "walk_dark", (-0.14, 0, 0))
    box("laptop_display", (-3.45, -3.093, 0.90), (0.31, 0.004, 0.22), "walk_screen", (-0.14, 0, 0))
    box("folded_prototype", (-3.4, -4.18, 0.84), (0.46, 0.32, 0.15), "walk_grey", (0, 0.08, 0.15))
    for j, (x, y) in enumerate(((-3.1, -4.5), (-3.7, -3.7), (-4.0, -4.3))):
        box("paper" + str(j), (x, y, 0.771), (0.29, 0.21, 0.003), "walk_paper", (0, 0, j * 0.21))
    box("red_sheet", (-3.0, -3.8, 0.774), (0.30, 0.23, 0.003), "walk_red")
    box("radio", (-2.94, -4.7, 0.80), (0.19, 0.24, 0.065), "walk_blue")
    for k in (-1, 1):
        box("radio_stick" + str(k), (-2.94 + k * 0.055, -4.7, 0.845), (0.016, 0.016, 0.04))
    rod("radio_antenna", (-2.94, -4.8, 0.82), (-2.94, -4.96, 1.00), 0.004)
    for j, (x, y) in enumerate(((-3.0, -3.4), (-3.1, -4.2), (-4.05, -3.65))):
        for k in range(16):
            a = k * math.tau / 16
            b = (k + 1) * math.tau / 16
            rod(
                f"lead_{j}_{k}",
                (x + 0.06 * math.cos(a), y + 0.085 * math.sin(a), 0.782),
                (x + 0.06 * math.cos(b), y + 0.085 * math.sin(b), 0.782),
                0.0025,
            )
    # Tall folded rolling grid frames immediately beside the south-west door.
    for j in range(2):
        x = -5.30 + j * 0.12
        y = -9.9
        for sy in (-1, 1):
            rod(f"rack_post{j}_{sy}", (x, y + sy * 0.48, 0.10), (x, y + sy * 0.48, 1.77), 0.018)
        for k in range(10):
            rod(
                f"rack_cross{j}_{k}",
                (x, y - 0.48, 0.16 + k * 0.17),
                (x, y + 0.48, 0.16 + k * 0.17),
                0.008,
                "walk_grey",
            )
        for k in range(5):
            rod(
                f"rack_grid{j}_{k}",
                (x, y - 0.43 + k * 0.215, 0.16),
                (x, y - 0.43 + k * 0.215, 1.70),
                0.005,
                "walk_grey",
            )
        for sy in (-1, 1):
            box(f"rack_foot{j}_{sy}", (x, y + sy * 0.48, 0.08), (0.55, 0.045, 0.04))
    box(
        "leaning_door_board",
        (-5.16, -10.6, 0.73),
        (0.06, 0.42, 1.43),
        "walk_wood",
        (0, -0.10, 0),
        True,
    )
    return p
