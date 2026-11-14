"""Video-anchor detail pass in cardinal coordinates: E+X,N+Y, 12x24m."""

import math
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETAIL_COLORS = {
    "detail_black": (0.018, 0.022, 0.026),
    "detail_metal": (0.28, 0.31, 0.32),
    "detail_silver": (0.58, 0.62, 0.63),
    "detail_wood": (0.58, 0.43, 0.25),
    "detail_pale": (0.72, 0.65, 0.47),
    "detail_red": (0.58, 0.012, 0.025),
    "detail_white": (0.81, 0.82, 0.78),
    "detail_blue": (0.02, 0.12, 0.31),
    "detail_orange": (0.76, 0.19, 0.018),
    "detail_yellow": (0.81, 0.63, 0.04),
    "detail_screen": (0.015, 0.028, 0.035),
    "detail_green": (0.025, 0.16, 0.07),
    "detail_cork": (0.42, 0.31, 0.20),
    "detail_paper": (0.77, 0.76, 0.69),
}


def _mesh(name, vertices, faces):
    path = HERE / "assets" / "room" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join("v %.6f %.6f %.6f\n" % v for v in vertices) + "".join(
        "f %s\n" % " ".join(str(i + 1) for i in f) for f in faces
    )
    if not path.exists() or path.read_text() != data:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix="." + name + ".", suffix=".tmp", delete=False
            ) as f:
                tmp = Path(f.name)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
    return "assets/room/" + name


def room_details():
    p = []

    def box(n, pos, size, c="detail_black", r=(0, 0, 0), collision=False):
        p.append(
            dict(
                name="detail_" + n,
                shape="box",
                pos=pos,
                size=size,
                color=c,
                rot=r,
                collision=collision,
            )
        )

    def cyl(n, pos, rad, h, c="detail_black", r=(0, 0, 0), collision=False):
        p.append(
            dict(
                name="detail_" + n,
                shape="cylinder",
                pos=pos,
                size=(rad, h),
                color=c,
                rot=r,
                collision=collision,
            )
        )

    def rod(n, a, b, rad=0.012, c="detail_metal", collision=False):
        d = [b[i] - a[i] for i in range(3)]
        h = math.sqrt(sum(v * v for v in d))
        cyl(
            n,
            tuple((a[i] + b[i]) / 2 for i in range(3)),
            rad,
            h,
            c,
            (0, math.acos(d[2] / h), math.atan2(d[1], d[0])),
            collision,
        )

    def mesh(n, path, pos, c, r=(0, 0, 0)):
        p.append(
            dict(
                name="detail_" + n,
                shape="mesh",
                mesh=path,
                pos=pos,
                size=(1, 1, 1),
                color=c,
                rot=r,
                collision=False,
            )
        )

    def monitor(n, x, y, z, normal="south"):
        # South-facing normal -Y; east-wall screens face west -X.
        box(n + "_stand", (x, y, z + 0.08), (0.035, 0.04, 0.16), "detail_black")
        box(n + "_foot", (x, y - 0.03, z + 0.005), (0.20, 0.16, 0.014), "detail_black")
        if normal == "south":
            box(n + "_housing", (x, y, z + 0.30), (0.48, 0.045, 0.29), "detail_black")
            box(n + "_screen", (x, y - 0.025, z + 0.30), (0.45, 0.003, 0.255), "detail_screen")
        else:
            box(n + "_housing", (x, y, z + 0.30), (0.045, 0.48, 0.29), "detail_black")
            box(n + "_screen", (x - 0.025, y, z + 0.30), (0.003, 0.45, 0.255), "detail_screen")

    def bench(n, x, y, w=2.0, d=0.8):
        box(n + "_top", (x, y, 0.86), (w, d, 0.055), "detail_pale", collision=True)
        for sx in (-1, 1):
            for sy in (-1, 1):
                box(
                    f"{n}_leg_{sx}_{sy}",
                    (x + sx * (w / 2 - 0.08), y + sy * (d / 2 - 0.07), 0.43),
                    (0.045, 0.045, 0.84),
                    "detail_metal",
                    collision=True,
                )
        box(
            n + "_rail",
            (x, y + d / 2 - 0.07, 0.25),
            (w - 0.1, 0.035, 0.045),
            "detail_metal",
            collision=True,
        )

    # East computing station and five narrow black lockers, immediately beyond
    # the whiteboards in the clockwise sweep. Avoid generic three-desk row.
    box("east_desk_top", (5.05, -3, 0.76), (0.72, 1.65, 0.035), "detail_pale", collision=True)
    for yy in (-3.65, -2.35):
        box(
            "east_desk_leg_" + str(yy),
            (5.05, yy, 0.39),
            (0.06, 0.06, 0.72),
            "detail_silver",
            collision=True,
        )
        box(
            "east_desk_foot_" + str(yy),
            (5.05, yy, 0.045),
            (0.62, 0.06, 0.04),
            "detail_silver",
            collision=True,
        )
    monitor("east_monitor_a", 5.14, -3.43, 0.79, "west")
    monitor("east_monitor_b", 5.14, -2.90, 0.79, "west")
    box("east_keyboard", (4.91, -2.95, 0.792), (0.16, 0.40, 0.018), "detail_black")
    box("controller_base", (4.94, -3.45, 0.81), (0.23, 0.24, 0.10), "detail_black")
    # Steering-wheel controller: hollow polygon rim with three spokes.
    center = (4.86, -3.45, 0.97)
    rad = 0.115
    for j in range(20):
        aa = j * math.tau / 20
        bb = (j + 1) * math.tau / 20
        rod(
            "steer_rim_" + str(j),
            (center[0], center[1] + rad * math.cos(aa), center[2] + rad * math.sin(aa)),
            (center[0], center[1] + rad * math.cos(bb), center[2] + rad * math.sin(bb)),
            0.012,
            "detail_black",
        )
    for j in range(3):
        a = j * math.tau / 3
        rod(
            "steer_spoke_" + str(j),
            center,
            (center[0], center[1] + rad * math.cos(a), center[2] + rad * math.sin(a)),
            0.013,
            "detail_metal",
        )
    cyl("steer_center", center, 0.032, 0.035, "detail_black", (0, math.pi / 2, 0))
    for j in range(5):
        y = -4.1 - j * 0.36
        box(
            "locker_body_" + str(j), (5.50, y, 1.0), (0.62, 0.35, 2), "detail_black", collision=True
        )
        box("locker_seam_" + str(j), (5.178, y - 0.172, 1.0), (0.006, 0.008, 1.90), "detail_metal")
        box(
            "locker_latch_" + str(j), (5.17, y + 0.095, 1.03), (0.018, 0.025, 0.10), "detail_silver"
        )
        for k in range(4):
            box(
                f"locker_vent_{j}_{k}",
                (5.175, y, 1.76 - k * 0.035),
                (0.006, 0.21, 0.008),
                "detail_metal",
            )
    # West-side small aircraft and instruments visible between tripod and shop.
    bench("west_folding", -5.0, 2.0, 1.55, 0.60)
    box("west_crate", (-5.0, 2.05, 1.05), (0.48, 0.36, 0.32), "detail_black", collision=True)
    box(
        "west_crate_open_lid", (-5.0, 2.23, 1.27), (0.48, 0.035, 0.34), "detail_black", (0.35, 0, 0)
    )
    for j in range(3):
        box(
            "west_instrument_" + str(j),
            (-5.5 + j * 0.28, 1.84, 0.925),
            (0.18, 0.12, 0.08),
            "detail_white",
        )
    # Red folding X stand, small silver aircraft supported at waist height.
    for yy in (0.65, 0.95):
        rod("red_x_a_" + str(yy), (-5.2, yy, 0.03), (-4.35, yy, 0.86), 0.012, "detail_red", True)
        rod("red_x_b_" + str(yy), (-4.35, yy, 0.03), (-5.2, yy, 0.86), 0.012, "detail_red", True)
    rod("redstand_top", (-5.2, 0.65, 0.86), (-5.2, 0.95, 0.86), 0.012, "detail_red")
    box("small_plane_fuselage", (-4.76, 0.80, 0.91), (0.90, 0.045, 0.045), "detail_silver")
    box(
        "small_plane_wing",
        (-4.8, 0.80, 0.93),
        (0.22, 1.18, 0.018),
        "detail_silver",
        (0, 0.03, 0.08),
    )
    box("small_plane_tail", (-4.42, 0.80, 0.96), (0.11, 0.35, 0.015), "detail_red")
    # Northwest black fabrication table with aircraft and a rolling stool.
    bench("northwest_shop", -4.5, 6.4, 2.3, 1.0)
    box("northwest_shop_lower", (-4.5, 6.4, 0.22), (2.1, 0.8, 0.04), "detail_black", collision=True)
    box("large_airframe_body", (-4.5, 6.4, 0.97), (1.55, 0.08, 0.09), "detail_silver")
    box("large_airframe_wing", (-4.5, 6.4, 1.00), (0.38, 1.45, 0.027), "detail_silver")
    box("large_airframe_tail", (-3.85, 6.4, 1.01), (0.16, 0.53, 0.02), "detail_red")
    # North back wall workstations, power tools and open wooden shelves.
    for i, x in enumerate((-3.8, -1.35)):
        bench("north_bench" + str(i), x, 10.8, 2.15, 0.85)
        for k in range(2):
            monitor(f"north_monitor_{i}_{k}", x - 0.42 + k * 0.56, 11.01, 0.90)
        box("north_keyboard_" + str(i), (x, 10.56, 0.91), (0.43, 0.15, 0.018), "detail_black")
        box("north_pcb_" + str(i), (x + 0.6, 10.55, 0.92), (0.25, 0.18, 0.05), "detail_green")
        cyl("north_stool_" + str(i), (x, 9.9, 0.49), 0.18, 0.07, "detail_black", collision=True)
        rod(
            "north_stool_post_" + str(i),
            (x, 9.9, 0.08),
            (x, 9.9, 0.46),
            0.025,
            "detail_silver",
            True,
        )
        for a in range(5):
            q = a * math.tau / 5
            rod(
                f"north_stool_foot_{i}_{a}",
                (x, 9.9, 0.10),
                (x + 0.24 * math.cos(q), 9.9 + 0.24 * math.sin(q), 0.055),
                0.012,
                "detail_silver",
            )
        for k in range(5):
            box(
                f"north_clutter_{i}_{k}",
                (x - 0.70 + k * 0.27, 10.7, 0.95 + 0.015 * (k % 2)),
                (0.14, 0.16, 0.10),
                "detail_white" if k % 2 else "detail_orange",
            )
    # Open wood shelves behind flagpartition: asymmetric boxes and loose stock.
    for sx in (-1, 1):
        box(
            "north_shelf_side_" + str(sx),
            (2.4 + sx * 0.7, 11.47, 1.02),
            (0.05, 0.48, 2.04),
            "detail_wood",
            collision=True,
        )
    for j in range(6):
        z = 0.10 + j * 0.37
        box(
            "north_shelf_" + str(j),
            (2.4, 11.47, z),
            (1.45, 0.50, 0.04),
            "detail_wood",
            collision=True,
        )
        for k in range(3):
            box(
                f"north_shelf_stock_{j}_{k}",
                (1.95 + k * 0.45, 11.44, z + 0.10),
                (0.34, 0.36, 0.15),
                "detail_white" if (j + k) % 2 else "detail_pale",
            )
    cyl("orange_shop_vac", (-0.1, 11.0, 0.30), 0.22, 0.43, "detail_orange", collision=True)
    cyl("shop_vac_lid", (-0.1, 11, 0.54), 0.24, 0.07, "detail_black")
    rod("shop_vac_hose_a", (-0.1, 10.78, 0.45), (0.32, 10.45, 0.21), 0.025, "detail_black")
    rod("shop_vac_hose_b", (0.32, 10.45, 0.21), (0.55, 10.54, 0.06), 0.025, "detail_black")
    for j in range(4):
        cyl(
            "back_container_" + str(j),
            (0.35 + j * 0.30, 11.5, 0.24),
            0.11,
            0.45,
            "detail_white" if j % 2 else "detail_black",
            collision=True,
        )
    # Distinct blue delta wing on the north wall, not a conventional crossplane.
    delta = _mesh(
        "north_delta_wing.obj",
        [
            (-0.65, 0, -0.5),
            (0.65, 0, -0.5),
            (0, 0, 0.62),
            (-0.13, -0.035, -0.51),
            (0.13, -0.035, -0.51),
            (0, -0.035, 0.62),
        ],
        [(0, 1, 2), (3, 5, 4), (0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2), (0, 3, 4), (0, 4, 1)],
    )
    mesh("north_delta", delta, (-3.45, 11.89, 2.5), "detail_blue")
    box("delta_centerline", (-3.45, 11.85, 2.45), (0.025, 0.015, 0.76), "detail_silver")
    for j in range(3):
        box(
            "north_spare_wing_" + str(j),
            (-4.95, 11.88, 2.15 + j * 0.18),
            (0.76, 0.05, 0.075),
            "detail_white",
        )
    # Two wheeled room-divider panels between camera and back wall, flag on left.
    fx, fy = 3.0, 3.0
    for i in range(2):
        xx = fx - 0.82 + i * 1.66
        box(
            "partition_cork_" + str(i),
            (xx, fy, 1.3),
            (1.58, 0.085, 0.86),
            "detail_cork",
            collision=True,
        )
        box(
            "partition_bottom_" + str(i),
            (xx, fy, 0.55),
            (1.58, 0.10, 0.56),
            "detail_metal",
            collision=True,
        )
        for ss in (-1, 1):
            px = xx + ss * 0.82
            box(
                f"partition_post_{i}_{ss}",
                (px, fy, 1.0),
                (0.055, 0.12, 1.96),
                "detail_metal",
                collision=True,
            )
            box(
                f"partition_foot_{i}_{ss}",
                (px, fy, 0.07),
                (0.065, 0.62, 0.055),
                "detail_metal",
                collision=True,
            )
            for ssy in (-1, 1):
                cyl(
                    f"partition_castor_{i}_{ss}_{ssy}",
                    (px, fy + ssy * 0.25, 0.035),
                    0.035,
                    0.025,
                    "detail_black",
                    (0, math.pi / 2, 0),
                )
        box("partition_top_" + str(i), (xx, fy, 1.8), (1.68, 0.12, 0.07), "detail_metal")
        for j in range(5):
            box(
                f"partition_yellow_clip_{i}_{j}",
                (xx - 0.60 + j * 0.30, fy, 1.88),
                (0.07, 0.08, 0.12),
                "detail_yellow",
            )
    xx = fx - 0.82
    box("flag_white", (xx, fy - 0.049, 1.31), (1.13, 0.008, 0.65), "detail_white")
    for ss in (-1, 1):
        box(
            "flag_red_" + str(ss),
            (xx + ss * 0.425, fy - 0.055, 1.31),
            (0.28, 0.008, 0.65),
            "detail_red",
        )
    outline = [
        (0, 0.30),
        (0.05, 0.17),
        (0.13, 0.22),
        (0.11, 0.10),
        (0.25, 0.15),
        (0.21, 0.04),
        (0.29, 0.0),
        (0.13, -0.08),
        (0.16, -0.14),
        (0.025, -0.11),
        (0.025, -0.25),
        (-0.025, -0.25),
        (-0.025, -0.11),
        (-0.16, -0.14),
        (-0.13, -0.08),
        (-0.29, 0),
        (-0.21, 0.04),
        (-0.25, 0.15),
        (-0.11, 0.10),
        (-0.13, 0.22),
        (-0.05, 0.17),
    ]
    maple = _mesh(
        "canadian_maple_detail.obj",
        [(0, 0, 0)] + [(x, 0, z) for x, z in outline],
        [(0, j + 1, (j + 1) % len(outline) + 1) for j in range(len(outline))],
    )
    mesh("flag_maple", maple, (xx, fy - 0.062, 1.31), "detail_red")
    # Open trays, a bucket and small blue bag under the two panels.
    for j in range(4):
        box(
            "partition_storage_" + str(j),
            (fx - 1.25 + j * 0.64, fy, 0.16),
            (0.48, 0.38, 0.25),
            "detail_blue" if j == 0 else "detail_black",
        )
    cyl("partition_bucket", (fx + 1.20, fy - 0.14, 0.24), 0.14, 0.40, "detail_white")
    cyl("partition_bucket_lid", (fx + 1.20, fy - 0.14, 0.45), 0.15, 0.035, "detail_orange")
    try:
        from room_south_details import SOUTH_COLORS, south_details

        DETAIL_COLORS.update(SOUTH_COLORS)
        p.extend(south_details())
    except ImportError:
        pass
    # Apply the final video-anchor locations while preserving local detail.
    for part in p:
        name = part["name"]
        x, y, z = part["pos"]
        if name.startswith(("detail_east_", "detail_controller_", "detail_steer_")):
            part["pos"] = (x + 0.05, y - 1.0, z)
        elif name.startswith("detail_locker_"):
            part["pos"] = (x - 0.20, y - 0.80, z)
        elif name.startswith("detail_northwest_shop_") or name.startswith("detail_large_airframe_"):
            part["pos"] = (x + 0.5, y - 1.4, z)
    p = [
        part
        for part in p
        if not part["name"].startswith(("detail_northwest_shop_", "detail_large_airframe_"))
    ]
    from room_west_bays import BAY_COLORS, bay_parts

    DETAIL_COLORS.update(BAY_COLORS)
    p.extend(bay_parts())
    return p
