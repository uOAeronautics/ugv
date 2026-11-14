"""Photo-informed visual parts shared by Gazebo and Blender. All values in metres.

X points toward the camera/front, Y left, Z up. Small components are approximate.
"""

import math
from tt_geometry import add_tt_motor
from camera_model import camera_geometry, load_camera

COLORS = {
    "pcb": (0.018, 0.024, 0.03),
    "green": (0.025, 0.26, 0.10),
    "blue": (0.025, 0.11, 0.32),
    "rubber": (0.016, 0.019, 0.022),
    "yellow_dark": (0.62, 0.32, 0.006),
    "yellow": (0.95, 0.55, 0.008),
    "silver": (0.52, 0.56, 0.59),
    "gold": (0.56, 0.34, 0.07),
    "chip": (0.012, 0.014, 0.018),
    "orange": (0.78, 0.28, 0.025),
    "white": (0.82, 0.84, 0.79),
    "red": (0.55, 0.018, 0.012),
    "lens": (0.02, 0.045, 0.065),
}


def assembly(p):
    inch = 0.0254
    L = p["chassis_length_in"] * inch
    W = p["chassis_width_in"] * inch
    R = p["wheel_diameter_in"] * inch / 2
    tire = p.get("wheel_width_mm_assumed", (p["overall_width_in"] * inch - W) * 500) / 1000
    track = p.get("wheel_track_mm_assumed", (p["overall_width_in"] * inch - tire) * 1000) / 1000
    wb = p["overall_length_in"] * inch - 2 * R
    board = p["main_board_height_in"] * inch
    thick = p["board_thickness_mm_assumed"] / 1000
    motorbottom = p["motor_bottom_clearance_in"] * inch
    groups = {"chassis": {"origin": (0, 0, 0), "parts": []}}

    def part(group, name, shape, pos, size, color, rot=(0, 0, 0), collision=False):
        groups[group]["parts"].append(
            dict(
                name=name,
                shape=shape,
                pos=pos,
                size=size,
                color=color,
                rot=rot,
                collision=collision,
            )
        )

    def box(name, pos, size, color, rot=(0, 0, 0), collision=False):
        part("chassis", name, "box", pos, size, color, rot, collision)

    def cyl(name, pos, r, h, color, rot=(0, 0, 0), collision=False):
        part("chassis", name, "cylinder", pos, (r, h), color, rot, collision)

    box("main_board", (0, 0, board - thick / 2), (L, W, thick), "pcb", collision=True)
    # Four TT gearmotors: yellow gearbox and inboard silver can on each side.
    for end, x in [("front", wb / 2), ("rear", -wb / 2)]:
        direction = -1 if end == "front" else 1
        for side, s in [("left", 1), ("right", -1)]:
            tag = end + "_" + side
            y = s * (W / 2 - 0.011)
            add_tt_motor(groups["chassis"]["parts"], tag, x, y, direction, s, motorbottom, R)
            group = tag + "_wheel"
            groups[group] = {"origin": (x, s * track / 2, R), "parts": []}
            from mecanum_geometry import wheel_parts

            groups[group]["parts"].extend(wheel_parts(R, tire))

    # Estimated low central LiPo envelope; mass and placement in mass_distribution.json.
    box("lipo_pack", (-0.002, 0, 0.030), (0.085, 0.044, 0.020), "silver", collision=True)
    for x in (-0.042, 0.038):
        box("lipo_wrap_" + str(x), (x, 0, 0.030), (0.005, 0.0445, 0.0205), "yellow")
    box("battery_retaining_strap", (-0.002, 0, 0.0195), (0.012, 0.046, 0.001), "chip")
    for y in (-0.0225, 0.0225):
        box("battery_strap_side_" + str(y), (-0.002, y, 0.031), (0.012, 0.001, 0.023), "chip")
    box("battery_power_lead", (0.044, 0.008, 0.034), (0.013, 0.002, 0.002), "red")
    box("battery_connector", (0.052, 0.008, 0.034), (0.006, 0.006, 0.005), "yellow")

    # Raised Raspberry Pi near the rear; positions estimated from top and side views.
    pi_z = board + 0.016
    for x in (-0.082, -0.020):
        for y in (-0.025, 0.025):
            cyl("standoff_" + str((x, y)), (x, y, board + 0.007), 0.002, 0.014, "gold")
            cyl("pi_screw_" + str((x, y)), (x, y, pi_z + 0.0013), 0.0025, 0.0015, "silver")
    box("raspberry_pi", (-0.048, 0, pi_z), (0.085, 0.056, 0.0016), "green", collision=True)
    box("pi_processor", (-0.045, -0.002, pi_z + 0.0025), (0.015, 0.015, 0.003), "chip")
    for i, y in enumerate((-0.018, 0.004)):
        box("usb_stack_" + str(i), (-0.082, y, pi_z + 0.008), (0.018, 0.016, 0.015), "silver")
        box("usb_opening_" + str(i), (-0.0913, y, pi_z + 0.008), (0.0008, 0.012, 0.010), "chip")
    box("ethernet", (-0.08, 0.024, pi_z + 0.008), (0.021, 0.016, 0.015), "silver")
    for x in (-0.059, -0.040):
        box("hdmi_" + str(x), (x, -0.026, pi_z + 0.003), (0.009, 0.007, 0.005), "silver")
    box("gpio_header", (-0.041, 0.022, pi_z + 0.005), (0.050, 0.005, 0.008), "chip")
    # Arduino Nano on tall pin headers over the Pi.
    nano_z = pi_z + 0.022
    box("nano_board", (-0.054, 0.003, nano_z), (0.043, 0.018, 0.0016), "blue", collision=True)
    box("nano_radio", (-0.047, 0.002, nano_z + 0.0025), (0.012, 0.010, 0.003), "silver")
    box("nano_usb", (-0.075, 0.003, nano_z + 0.003), (0.006, 0.008, 0.004), "silver")
    for x in range(12):
        for y in (-0.005, 0.011):
            box(
                "nano_pin_" + str((x, y)),
                (-0.072 + x * 0.003, y, nano_z - 0.009),
                (0.0006, 0.0006, 0.018),
                "gold",
            )
    # New upright white 40x40mm camera stand, 40mm behind board front.
    camera = load_camera()
    mount = camera["mount"]
    geometry = camera_geometry(p, camera)
    cx, cz = geometry["stand_x"], geometry["center_z"]
    depth, rail = mount["depth_m_assumed"], mount["rail_width_m_assumed"]
    for side in (-1, 1):
        box(
            "camera_stand_side_" + str(side),
            (cx, side * (mount["width_m"] - rail) / 2, cz),
            (depth, rail, mount["height_m"]),
            "white",
            collision=True,
        )
        box(
            "camera_stand_edge_" + str(side),
            (cx, 0, cz + side * (mount["height_m"] - rail) / 2),
            (depth, mount["width_m"] - 2 * rail, rail),
            "white",
            collision=True,
        )
    box(
        "camera_stand_foot",
        (cx - 0.003, 0, board + 0.001),
        (0.012, 0.040, 0.002),
        "white",
        collision=True,
    )
    box("camera_pcb", (cx + 0.003, 0, cz), (0.0016, 0.035, 0.035), "green", collision=True)
    box("camera_lens_base", (cx + 0.006, 0, cz), (0.004, 0.017, 0.017), "chip")
    cyl("camera_lens_barrel", (cx + 0.010, 0, cz), 0.007, 0.0104, "chip", (0, math.pi / 2, 0), True)
    cyl(
        "camera_glass",
        (geometry["optical_xyz"][0] - 0.0003, 0, cz),
        0.0055,
        0.0006,
        "lens",
        (0, math.pi / 2, 0),
    )
    for y in (-0.0145, 0.0145):
        for z in (-0.0145, 0.0145):
            cyl(
                "camera_screw_" + str((y, z)),
                (cx + 0.0045, y, cz + z),
                0.0018,
                0.0014,
                "silver",
                (0, math.pi / 2, 0),
            )
    # Retained servo geometry sits behind the fixed stand; position is estimated.
    servo_x = cx - 0.020
    box("front_servo", (servo_x, 0, board + 0.009), (0.023, 0.013, 0.017), "chip", collision=True)
    cyl("servo_pivot", (servo_x, 0, board + 0.019), 0.003, 0.005, "white")
    box("servo_horn", (servo_x, 0, board + 0.022), (0.004, 0.022, 0.002), "white")
    # Photo-informed ribbon loop rises from the Pi to the stand's top connector.
    path = [
        (-0.015, pi_z + 0.003),
        (-0.003, board + 0.067),
        (0.023, board + 0.070),
        (cx - 0.004, board + 0.039),
    ]
    for i, ((ax, az), (bx, bz)) in enumerate(zip(path, path[1:])):
        box(
            "camera_ribbon_" + str(i),
            ((ax + bx) / 2, 0, (az + bz) / 2),
            (math.hypot(bx - ax, bz - az), 0.015, 0.0003),
            "orange",
            (0, -math.atan2(bz - az, bx - ax), 0),
        )
    cyl("buzzer", (0.071, -0.030, board + 0.0055), 0.006, 0.011, "chip")
    cyl("buzzer_label", (0.071, -0.030, board + 0.0112), 0.0053, 0.0003, "white")
    for x in (-0.026, -0.011):
        box("switch_base_" + str(x), (x, -0.037, board + 0.003), (0.009, 0.008, 0.005), "white")
        box("switch_blue_" + str(x), (x, -0.037, board + 0.006), (0.004, 0.004, 0.002), "blue")
    for x in (-0.082, -0.048, -0.014, 0.025, 0.080):
        for y in (-0.039, 0.039):
            box("led_" + str((x, y)), (x, y, board + 0.001), (0.0035, 0.0035, 0.0015), "white")
    for x in (-0.085, 0.078):
        for y in (-0.033, 0.033):
            cyl("board_screw_" + str((x, y)), (x, y, board + 0.001), 0.0025, 0.002, "silver")
    for i in range(12):
        box(
            "front_gpio_" + str(i),
            (-0.003 + i * 0.0025, 0.029, board + 0.004),
            (0.001, 0.002, 0.007),
            "gold",
        )
    return groups
