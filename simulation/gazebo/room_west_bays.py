"""West window bays refined against reference seconds 11,13,15.
Cardinal SI frame E+X,N+Y. Replace generic northwest shop/airframe pieces.
"""

import math

BAY_COLORS = {
    "bay_black": (0.015, 0.019, 0.022),
    "bay_steel": (0.16, 0.18, 0.19),
    "bay_silver": (0.56, 0.59, 0.60),
    "bay_white": (0.78, 0.79, 0.75),
    "bay_red": (0.61, 0.02, 0.025),
    "bay_wood": (0.55, 0.39, 0.22),
    "bay_pale": (0.72, 0.63, 0.44),
    "bay_blue": (0.025, 0.12, 0.25),
    "bay_lime": (0.56, 0.72, 0.035),
    "bay_cyan": (0.015, 0.28, 0.49),
    "bay_paper": (0.67, 0.67, 0.61),
    "bay_wire": (0.055, 0.065, 0.07),
}


def bay_parts():
    parts = []

    def box(n, p, s, c="bay_black", r=(0, 0, 0), collision=False):
        parts.append(
            dict(
                name="westbay_" + n, shape="box", pos=p, size=s, color=c, rot=r, collision=collision
            )
        )

    def cyl(n, p, rad, h, c="bay_black", r=(0, 0, 0), collision=False):
        parts.append(
            dict(
                name="westbay_" + n,
                shape="cylinder",
                pos=p,
                size=(rad, h),
                color=c,
                rot=r,
                collision=collision,
            )
        )

    def rod(n, a, b, rad=0.01, c="bay_silver", collision=False):
        d = [b[i] - a[i] for i in range(3)]
        h = math.sqrt(sum(q * q for q in d))
        cyl(
            n,
            tuple((a[i] + b[i]) / 2 for i in range(3)),
            rad,
            h,
            c,
            (0, math.acos(d[2] / h), math.atan2(d[1], d[0])),
            collision,
        )

    def ring(n, x, y, z, rad, tube, c="bay_silver", plane="xy"):
        for j in range(20):
            a = j * math.tau / 20
            b = (j + 1) * math.tau / 20

            def point(v):
                return (
                    (x + rad * math.cos(v), y + rad * math.sin(v), z)
                    if plane == "xy"
                    else (x, y + rad * math.cos(v), z + rad * math.sin(v))
                )

            rod(n + str(j), point(a), point(b), tube, c)

    # A coherent inward-projecting steel work island; the panorama sees its
    # broad front at13s and its narrow end plus foam-stock at15s.
    x, y = -3.8, 4.4
    box("shop_top", (x, y, 0.90), (1.10, 2.20, 0.055), "bay_pale", collision=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            xx = x + sx * 0.47
            yy = y + sy * 1.01
            box(
                f"shop_leg_{sx}_{sy}",
                (xx, yy, 0.45),
                (0.055, 0.055, 0.88),
                "bay_steel",
                collision=True,
            )
    for sx in (-1, 1):
        box(
            "shop_side_rail" + str(sx),
            (x + sx * 0.47, y, 0.26),
            (0.045, 2.03, 0.045),
            "bay_steel",
            collision=True,
        )
    box("shop_lower_shelf", (x, y, 0.25), (0.94, 1.96, 0.035), "bay_black", collision=True)
    box(
        "shop_under_case",
        (x - 0.12, y + 0.22, 0.43),
        (0.55, 0.7, 0.31),
        "bay_black",
        collision=True,
    )
    # Aluminium/foam airframe lying on the top. Slender fuselage, open wing
    # spars/ribs and a white skin panel distinguish it from a generic cross.
    rod(
        "airframe_fuselage",
        (x + 0.08, y - 0.88, 0.98),
        (x + 0.08, y + 0.83, 1.01),
        0.024,
        "bay_silver",
    )
    for yy in (y - 0.16, y + 0.07):
        rod(
            "airframe_spar_" + str(yy),
            (x - 0.69, yy, 1.01),
            (x + 0.69, yy, 1.01),
            0.012,
            "bay_silver",
        )
    for j in range(9):
        xx = x - 0.65 + j * 0.16
        rod(
            "airframe_wing_rib_" + str(j),
            (xx, y - 0.18, 1.008),
            (xx, y + 0.14, 1.015),
            0.008,
            "bay_silver",
        )
    box("airframe_skin", (x - 0.27, y - 0.02, 1.029), (0.56, 0.33, 0.012), "bay_white")
    box("airframe_tail", (x + 0.08, y + 0.68, 1.04), (0.46, 0.16, 0.013), "bay_white")
    box("airframe_fin", (x + 0.08, y + 0.74, 1.12), (0.018, 0.18, 0.18), "bay_red", (0, 0.08, 0))
    box("airframe_redstripe", (x - 0.26, y - 0.20, 1.04), (0.57, 0.045, 0.009), "bay_red")
    # Second angular loose airframe/wing sections behind the main plane.
    box(
        "spare_wing_left",
        (x - 0.24, y + 0.57, 0.99),
        (0.43, 0.44, 0.025),
        "bay_white",
        (0, 0.08, -0.32),
    )
    box(
        "spare_wing_red",
        (x - 0.30, y + 0.57, 1.008),
        (0.32, 0.06, 0.012),
        "bay_red",
        (0, 0.08, -0.32),
    )
    # Backstop project display, white lower field and broad charcoal title band.
    box("display_board", (-4.15, 5.62, 1.47), (1.60, 0.045, 0.93), "bay_white", collision=True)
    box("display_header", (-4.15, 5.59, 1.83), (1.60, 0.009, 0.15), "bay_steel")
    for j in range(8):
        box(
            "display_title_stroke" + str(j),
            (-4.65 + j * 0.13, 5.583, 1.83),
            (0.085, 0.002, 0.017),
            "bay_white",
        )
    for j in range(6):
        box(
            "display_note_" + str(j),
            (-4.55 + (j % 3) * 0.38, 5.586, 1.55 - (j // 3) * 0.16),
            (0.27, 0.002, 0.012),
            "bay_black",
        )
    # Tall upright red/white foam aircraft parts between pier and project board.
    for j in range(3):
        xx = -4.95 + j * 0.15
        box(
            "vertical_foam_" + str(j),
            (xx, 5.42, 1.62),
            (0.10, 0.055, 1.18),
            "bay_white",
            (0, 0.035 * j, 0),
        )
        box("vertical_red_" + str(j), (xx, 5.385, 1.91), (0.095, 0.006, 0.38), "bay_red")
    box("upright_tailplane", (-4.8, 5.43, 1.62), (0.60, 0.055, 0.18), "bay_red")
    # Three individual long foam boards leaning against the front-right end.
    for j in range(3):
        box(
            "leaning_foam_" + str(j),
            (-2.93 + j * 0.035, 3.94 + j * 0.13, 0.74),
            (0.075, 0.42, 1.47),
            "bay_white",
            (0, -0.12, 0.035 * j),
            True,
        )
    # Separate small RC aircraft on narrow silver T support in front of bench.
    xx, yy = -3.35, 3.13
    rod("rc_stand_upright", (xx, yy, 0.05), (xx, yy, 0.78), 0.013, "bay_silver", True)
    rod("rc_stand_foot", (xx - 0.30, yy, 0.04), (xx + 0.30, yy, 0.04), 0.016, "bay_silver", True)
    rod("rc_stand_foot2", (xx, yy - 0.20, 0.04), (xx, yy + 0.20, 0.04), 0.014, "bay_silver", True)
    box("rc_fuselage", (xx, yy, 0.83), (0.83, 0.06, 0.055), "bay_white", (0, 0.035, -0.12))
    box("rc_wing", (xx - 0.05, yy, 0.86), (0.25, 0.92, 0.016), "bay_white", (0, 0, -0.12))
    box("rc_wing_red", (xx - 0.05, yy + 0.26, 0.872), (0.25, 0.27, 0.006), "bay_red", (0, 0, -0.12))
    box("rc_tail", (xx + 0.33, yy, 0.87), (0.11, 0.34, 0.015), "bay_red")
    box("rc_vertical_tail", (xx + 0.35, yy, 0.94), (0.13, 0.018, 0.14), "bay_white")
    cyl("rc_nose", (xx - 0.43, yy, 0.83), 0.03, 0.07, "bay_blue", (0, math.pi / 2, 0))
    # Black backed shop stool on chrome five-leg base with circular foot ring.
    xx, yy = -4.55, 2.95
    cyl("shop_stool_seat", (xx, yy, 0.56), 0.185, 0.07, "bay_black", collision=True)
    rod("shop_stool_post", (xx, yy, 0.09), (xx, yy, 0.53), 0.023, "bay_silver", True)
    ring("shop_stool_ring_", xx, yy, 0.25, 0.21, 0.010)
    box("shop_stool_back", (xx, yy + 0.14, 0.79), (0.29, 0.04, 0.23), "bay_black", (0, 0.05, 0))
    for sx in (-1, 1):
        rod(
            "shop_stool_back_post" + str(sx),
            (xx + sx * 0.10, yy + 0.13, 0.55),
            (xx + sx * 0.10, yy + 0.14, 0.79),
            0.012,
            "bay_silver",
        )
    for j in range(5):
        a = j * math.tau / 5
        end = (xx + 0.25 * math.cos(a), yy + 0.25 * math.sin(a), 0.055)
        rod("shop_stool_foot" + str(j), (xx, yy, 0.10), end, 0.014, "bay_silver")
        cyl("shop_stool_caster" + str(j), end, 0.03, 0.035, "bay_black", (math.pi / 2, 0, a))
    # Add realistic contents to the existing west folding instrument table.
    # Its open black case remains supplied by room_details.py.
    for j in range(4):
        box(
            "instrument_box_" + str(j),
            (-5.48 + j * 0.25, 1.82, 0.91),
            (0.18, 0.14, 0.07),
            "bay_white",
        )
        cyl(
            "instrument_knob_" + str(j),
            (-5.48 + j * 0.25, 1.73, 0.925),
            0.014,
            0.02,
            "bay_black",
            (math.pi / 2, 0, 0),
        )
    box("instrument_laptop_base", (-4.66, 1.84, 0.91), (0.28, 0.20, 0.013), "bay_steel")
    box(
        "instrument_laptop_screen",
        (-4.66, 1.94, 1.02),
        (0.28, 0.02, 0.22),
        "bay_black",
        (-0.16, 0, 0),
    )
    for j in range(2):
        ring(
            "instrument_lead_coil_" + str(j) + "_",
            -5.10,
            1.89,
            0.96 + j * 0.01,
            0.075,
            0.003,
            "bay_wire",
        )
    rod("instrument_power_lead", (-5.1, 1.9, 0.95), (-4.6, 1.84, 0.95), 0.003, "bay_wire")
    # Thin stock and tools visually connect the bench to the nearby red tool chest.
    for j in range(4):
        box("bench_tool_" + str(j), (-3.45, 4.1 + j * 0.22, 0.96), (0.21, 0.07, 0.03), "bay_steel")
    box("bench_foam_strip", (-3.9, 4.6, 1.05), (0.10, 1.1, 0.025), "bay_white", (0, 0, 0.12))
    # Walk-through refinement: actual wing test article is a deep lattice beam,
    # not a string of little ribs on two poles. Two freestanding trestles support it.
    bx, by = -4.25, 0.10
    for sy in (-0.92, 0.92):
        for sx in (-1, 1):
            rod(
                f"wing_trestle_leg_{sy}_{sx}",
                (bx + sx * 0.32, by + sy, 0.035),
                (bx + sx * 0.10, by + sy, 1.05),
                0.024,
                "bay_wood",
                True,
            )
        box(
            "wing_trestle_foot_" + str(sy),
            (bx, by + sy, 0.035),
            (0.82, 0.08, 0.07),
            "bay_wood",
            collision=True,
        )
        rod(
            "wing_trestle_cross_" + str(sy),
            (bx - 0.25, by + sy, 0.49),
            (bx + 0.25, by + sy, 0.49),
            0.018,
            "bay_wood",
        )
    # Four corner longerons and repeated diagonal/trapezoid ribs, central plywood.
    for xx in (-0.16, 0.16):
        for zz in (1.04, 1.36):
            rod(
                f"wing_longeron_{xx}_{zz}",
                (bx + xx, by - 1.56, zz),
                (bx + xx, by + 1.56, zz),
                0.009,
                "bay_pale",
            )
    for j in range(13):
        yy = by - 1.56 + j * 0.26
        for xx in (-0.16, 0.16):
            rod(
                f"wing_vertical_{j}_{xx}",
                (bx + xx, yy, 1.04),
                (bx + xx, yy, 1.36),
                0.008,
                "bay_pale",
            )
            if j < 12:
                rod(
                    f"wing_diagonal_{j}_{xx}",
                    (bx + xx, yy, 1.04),
                    (bx + xx, yy + 0.26, 1.36),
                    0.008,
                    "bay_pale",
                )
        rod(
            "wing_top_rib_" + str(j),
            (bx - 0.16, yy, 1.36),
            (bx + 0.16, yy, 1.36),
            0.008,
            "bay_pale",
        )
        rod(
            "wing_lower_rib_" + str(j),
            (bx - 0.16, yy, 1.04),
            (bx + 0.16, yy, 1.04),
            0.008,
            "bay_pale",
        )
    box("wing_plywood_center", (bx + 0.168, by + 0.10, 1.19), (0.012, 0.48, 0.30), "bay_pale")
    # Separate long pegboard/tool bench behind the fabrication island.
    tx, ty = -5.03, 6.62
    box("toolbench_top", (tx, ty, 0.88), (0.88, 2.75, 0.05), "bay_white", collision=True)
    box("toolbench_lower", (tx, ty, 0.20), (0.76, 2.62, 0.04), "bay_steel", collision=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(
                f"toolbench_leg_{sx}_{sy}",
                (tx + sx * 0.37, ty + sy * 1.28, 0.44),
                (0.055, 0.055, 0.86),
                "bay_steel",
                collision=True,
            )
    box("tool_pegboard", (tx - 0.43, ty, 1.35), (0.045, 2.65, 0.85), "bay_wood", collision=True)
    # Rows of perforations and strongly colored hanging holders/clamps.
    for j in range(15):
        yy = ty - 1.20 + j * 0.17
        for k in range(4):
            cyl(
                f"peg_hole_{j}_{k}",
                (tx - 0.401, yy, 1.09 + k * 0.16),
                0.008,
                0.003,
                "bay_black",
                (0, math.pi / 2, 0),
            )
        if j % 2 == 0:
            box("tool_holder_" + str(j), (tx - 0.35, yy, 1.39), (0.13, 0.09, 0.19), "bay_lime")
            rod(
                "tool_shaft_" + str(j),
                (tx - 0.25, yy, 1.33),
                (tx - 0.25, yy, 1.71),
                0.010,
                "bay_silver",
            )
            box(
                "tool_handle_" + str(j),
                (tx - 0.25, yy, 1.63),
                (0.055, 0.05, 0.13),
                "bay_cyan" if j % 4 else "bay_red",
            )
    # Parts drawer towers at each end, clear trays and piled loose tools.
    for sy in (-1, 1):
        yy = ty + sy * 1.01
        box("parts_drawer_tower_" + str(sy), (tx + 0.02, yy, 1.12), (0.34, 0.31, 0.45), "bay_black")
        for j in range(6):
            box(
                f"parts_drawer_{sy}_{j}",
                (tx + 0.20, yy, 0.96 + j * 0.065),
                (0.012, 0.27, 0.05),
                "bay_silver",
            )
            box(
                f"parts_pull_{sy}_{j}",
                (tx + 0.215, yy, 0.96 + j * 0.065),
                (0.02, 0.07, 0.012),
                "bay_black",
            )
    for j in range(12):
        yy = ty - 0.85 + j * 0.15
        box(
            "loose_tool_" + str(j),
            (tx + 0.14, yy, 0.94),
            (0.28, 0.045, 0.025),
            "bay_cyan" if j % 3 == 0 else "bay_silver",
            (0, 0, 0.3 * (j % 3)),
        )
    for j in range(3):
        box(
            "clear_parts_tray_" + str(j),
            (tx + 0.05, ty - 0.45 + j * 0.37, 0.96),
            (0.37, 0.28, 0.065),
            "bay_paper",
        )
    # Two recognizable open-frame desktop 3D printers on the rear side table.
    px, py = -5.02, 9.12
    box("printer_bench", (px, py, 0.87), (0.85, 1.77, 0.045), "bay_pale", collision=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(
                f"printer_benchleg_{sx}_{sy}",
                (px + sx * 0.34, py + sy * 0.76, 0.43),
                (0.045, 0.045, 0.85),
                "bay_black",
                collision=True,
            )
    for j in range(2):
        yy = py - 0.44 + j * 0.85
        box("printer_base_" + str(j), (px, yy, 0.96), (0.45, 0.52, 0.13), "bay_black")
        box("printer_bed_" + str(j), (px, yy, 1.07), (0.34, 0.36, 0.025), "bay_silver")
        for sx in (-1, 1):
            box(
                f"printer_tower_{j}_{sx}",
                (px + sx * 0.21, yy + 0.17, 1.29),
                (0.035, 0.04, 0.55),
                "bay_black",
            )
        rod(
            "printer_toprail_" + str(j),
            (px - 0.21, yy + 0.17, 1.56),
            (px + 0.21, yy + 0.17, 1.56),
            0.018,
            "bay_black",
        )
        rod(
            "printer_xrail_" + str(j),
            (px - 0.20, yy + 0.17, 1.32),
            (px + 0.20, yy + 0.17, 1.32),
            0.012,
            "bay_silver",
        )
        box("printer_head_" + str(j), (px, yy + 0.12, 1.30), (0.10, 0.09, 0.10), "bay_cyan")
        cyl(
            "filament_spool_" + str(j),
            (px - 0.27, yy + 0.1, 1.18),
            0.11,
            0.06,
            "bay_black",
            (0, math.pi / 2, 0),
        )
    # Lattice-wing building work island with adhesives, bottles and paper.
    # The walkthrough reveals a second cluttered build surface north of the
    # earlier aircraft/display bench, not a bare single airframe table.
    ax, ay = -3.25, 7.40
    box("build_island_top", (ax, ay, 0.91), (1.65, 2.10, 0.06), "bay_black", collision=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(
                f"build_island_leg_{sx}_{sy}",
                (ax + sx * 0.72, ay + sy * 0.94, 0.45),
                (0.065, 0.065, 0.88),
                "bay_steel",
                collision=True,
            )
    box("build_island_lower", (ax, ay, 0.20), (1.48, 1.87, 0.045), "bay_steel", collision=True)
    for sx in (-1, 1):
        rod(
            "build_wing_spar" + str(sx),
            (ax + sx * 0.21, ay - 0.95, 1.02),
            (ax + sx * 0.21, ay + 0.95, 1.02),
            0.009,
            "bay_pale",
        )
    for j in range(16):
        yy = ay - 0.95 + j * 0.126
        rod(
            "build_wing_rib" + str(j),
            (ax - 0.25, yy, 1.01),
            (ax + 0.25, yy, 1.04),
            0.008,
            "bay_pale",
        )
        if j < 15:
            rod(
                "build_wing_diag" + str(j),
                (ax - 0.21, yy, 1.02),
                (ax + 0.21, yy + 0.126, 1.02),
                0.006,
                "bay_pale",
            )
    for j in range(8):
        xx = ax - 0.62 + (j % 2) * 1.20
        yy = ay - 0.75 + (j // 2) * 0.46
        cyl(
            "adhesive_bottle_" + str(j),
            (xx, yy, 1.03),
            0.035,
            0.18,
            "bay_white" if j % 2 else "bay_red",
        )
        cyl("bottle_cap_" + str(j), (xx, yy, 1.13), 0.025, 0.025, "bay_lime")
        box(
            "paper_scrap_" + str(j),
            (xx + 0.11, yy + 0.08, 0.947),
            (0.19, 0.15, 0.004),
            "bay_paper",
            (0, 0, j * 0.28),
        )
    cyl("tall_build_bottle", (ax + 0.57, ay + 0.28, 1.17), 0.055, 0.42, "bay_white")
    box("brown_supply_bag", (ax + 0.55, ay + 0.76, 1.15), (0.24, 0.25, 0.40), "bay_wood")
    # Correct the small supported RC plane to the distinctive red-white-blue livery.
    box(
        "rc_blue_fuselage", (-3.35, 3.13, 0.865), (0.71, 0.06, 0.035), "bay_cyan", (0, 0.035, -0.12)
    )
    for sy in (-1, 1):
        box(
            "rc_blue_wingtip" + str(sy),
            (-3.40, 3.13 + sy * 0.40, 0.882),
            (0.25, 0.10, 0.009),
            "bay_cyan",
            (0, 0, -0.12),
        )
    # A larger red toolchest with an open top lid and pulled-out lower drawer.
    cx, cy = -2.1, 4.7
    box("open_toolchest_body", (cx, cy, 0.58), (0.76, 0.51, 1.04), "bay_red", collision=True)
    for j in range(9):
        zz = 0.18 + j * 0.085
        box(
            "open_toolchest_drawer_" + str(j), (cx, cy - 0.265, zz), (0.70, 0.025, 0.065), "bay_red"
        )
        box(
            "open_toolchest_pull_" + str(j),
            (cx, cy - 0.285, zz + 0.015),
            (0.58, 0.025, 0.015),
            "bay_silver",
        )
    box("toolchest_top_tray", (cx, cy, 1.12), (0.74, 0.49, 0.08), "bay_black")
    box("toolchest_open_lid", (cx, cy + 0.19, 1.38), (0.76, 0.035, 0.40), "bay_red", (-0.18, 0, 0))
    for j in range(3):
        box(
            "toolchest_lid_badge_" + str(j),
            (cx - 0.22 + j * 0.22, cy + 0.16, 1.38),
            (0.16, 0.008, 0.07),
            "bay_silver",
        )
    box(
        "toolchest_open_bottom",
        (cx, cy - 0.39, 0.10),
        (0.69, 0.37, 0.07),
        "bay_black",
        collision=True,
    )
    box("toolchest_bottom_front", (cx, cy - 0.59, 0.14), (0.71, 0.025, 0.16), "bay_red")
    box("toolchest_gloves", (cx - 0.15, cy - 0.40, 0.17), (0.22, 0.17, 0.07), "bay_white")
    # Walking aisle views the drawer fronts from the east.
    for part in parts:
        if "toolchest_" in part["name"]:
            x, y, z = part["pos"]
            dx = x - cx
            dy = y - cy
            part["pos"] = (cx - dy, cy + dx, z)
            rx, ry, rz = part["rot"]
            part["rot"] = (rx, ry, rz + math.pi / 2)
    return parts
