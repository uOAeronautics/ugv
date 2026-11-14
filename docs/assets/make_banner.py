"""ugv banner: a mecanum car slides into the drone's landing-skid slot, tags at the head.

Run: uv run --with pillow --with numpy python make_banner.py
Policy frame, metres: drone at the origin, skids along y, anchor tag 14 at
y = -0.08 (tags 6 and 3 beside it, offsets from calibration/runs/l7/tagmap.json),
skid tips near y = +0.16, inside gap 0.31 m, car 0.17 m wide, parking goal
(0, +0.04). Bit patterns are DICT_4X4_50. The approach starts from the ~1 m
staging pose; the path between is schematic (a mecanum car can strafe and turn
at once). Skid radius and tag-card outline are drawn, not measured.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw

W, H, SS = 1600, 480, 2
INK, PAPER, MUTE, LINE, ACC = (
    (14, 19, 26),
    (236, 231, 218),
    (107, 118, 134),
    (38, 48, 60),
    (242, 169, 59),
)

FONT = """
u 2 #...# #...# #...# #...# .####
g 2 .#### #...# #...# .#### ....# #...# .###.
v 2 #...# #...# #...# .#.#. ..#..
0 0 .###. #...# #..## #.#.# ##..# #...# .###.
1 0 ..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.
3 0 .###. #...# ....# ..##. ....# #...# .###.
4 0 ...#. ..##. .#.#. #..#. ##### ...#. ...#.
5 0 ##### #.... ####. ....# ....# #...# .###.
6 0 .###. #.... #.... ####. #...# #...# .###.
9 0 .###. #...# #...# .#### ....# ...#. .##..
. 6 ..#..
x 2 #...# .#.#. ..#.. .#.#. #...#
c 2 .#### #.... #.... #.... .####
a 2 .###. ....# .#### #...# .####
r 2 #.##. ##..# #.... #.... #....
l 0 .##.. ..#.. ..#.. ..#.. ..#.. ..#.. .###.
o 2 .###. #...# #...# #...# .###.
t 1 .#... ####. .#... .#... .#... ..###
s 2 .#### #.... .###. ....# ####.
2 0 .###. #...# ....# ...#. ..#.. .#... #####
m 2 ##.#. #.#.# #.#.# #.#.# #.#.#
"""
GLYPHS = {}
for ln in FONT.strip().splitlines():
    c, r0, *rows = ln.split()
    GLYPHS[c] = (int(r0), rows)


def adv(c, s):
    return (3 if c == "." else 6) * s


def text(d, x, y, s, scale, fill):
    """Pixel lettering on a 5-wide grid; y is the top of the ascender row."""
    for ch in s:
        r0, rows = GLYPHS.get(ch, (0, []))
        for j, row in enumerate(rows):
            for i, c in enumerate(row):
                if c == "#":
                    px, py = x + i * scale, y + (r0 + j) * scale
                    d.rectangle([px, py, px + scale - 1, py + scale - 1], fill=fill)
        x += adv(ch, scale)


def text_w(s, scale):
    return sum(adv(c, scale) for c in s) - scale


# --- geometry ---------------------------------------------------------------
ARUCO = {  # DICT_4X4_50 inner 4x4 cells, 1 = white
    14: ["0010", "0100", "1011", "0001"],
    6: ["1001", "1110", "0010", "1110"],
    3: ["1001", "1001", "0100", "0110"],
}
TAGS = {
    14: (0.0, -0.08),
    6: (0.088, -0.13),
    3: (-0.077, -0.129),
}  # (x, y), m; offsets of 6, 3 from the l7 tag map
TAG = 0.05  # tag edge, m
SKID_X, SKID_R = 0.185, 0.03  # skid centre lateral offset (inside gap 0.31 m), capsule radius
SKID_Y = (-0.115, 0.13)  # capsule centre-line ends, so the tips reach about y = +0.16
GOAL = (0.0, 0.04)  # camera reference at the goal; body centre sits 0.08 m behind it
CAR_L, CAR_W = 0.22, 0.17

S = 700 * SS  # px per metre
OX, OY = 1400 * SS, 250 * SS  # canvas position of the origin
GREY = (150, 160, 175, 255)


def P(x, y):  # policy frame (x lateral, y along the skids) -> canvas; y runs leftwards
    return (OX - y * S, OY - x * S)


def layer():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def disc(d, c, r, **kw):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)


def car(d, cx, cy, phi, fill, edge):
    """Mecanum car footprint centred at (cx, cy); phi = heading measured from the -y direction."""
    ux, uy = math.sin(phi), -math.cos(phi)
    vx, vy = -uy, ux

    def T(u, v):
        return P(cx + u * ux + v * vx, cy + u * uy + v * vy)

    d.polygon(
        [
            T(-CAR_L / 2, -CAR_W / 2 + 0.03),
            T(CAR_L / 2, -CAR_W / 2 + 0.03),
            T(CAR_L / 2, CAR_W / 2 - 0.03),
            T(-CAR_L / 2, CAR_W / 2 - 0.03),
        ],
        fill=fill,
        outline=edge,
    )
    for u in (-0.072, 0.072):
        for v in (-CAR_W / 2 + 0.005, CAR_W / 2 - 0.005):
            d.polygon(
                [
                    T(u - 0.03, v - 0.02),
                    T(u + 0.03, v - 0.02),
                    T(u + 0.03, v + 0.02),
                    T(u - 0.03, v + 0.02),
                ],
                fill=INK + (255,),
                outline=edge,
            )
            sg = 1 if u * v > 0 else -1  # roller diagonals form an X across the car
            for k in (-0.018, -0.006, 0.006, 0.018):
                d.line(
                    [T(u + k - 0.008 * sg, v - 0.02), T(u + k + 0.008 * sg, v + 0.02)],
                    fill=edge,
                    width=SS,
                )


def approach(s):
    """Schematic path: body centre y = 1.0 -> 0.12 m; the 0.14 m lateral offset is slid out by y = 0.5."""
    t = min(s / 0.55, 1.0)
    return 0.14 * (1 - t * t * (3 - 2 * t)), 1.0 - 0.88 * s


def heading(s):
    x, y = approach(s)
    x2, y2 = approach(min(s + 0.01, 1)) if s < 0.99 else approach(1)
    x1, y1 = approach(max(s - 0.01, 0))
    return math.atan2(x2 - x1, y1 - y2)


def main():
    base = Image.new("RGBA", (W * SS, H * SS), INK + (255,))

    # 10 cm grid (bold every 50 cm), faded toward the edges
    g, d = layer()
    for k in range(-5, 6):
        d.line(
            [P(k * 0.1, -0.5), P(k * 0.1, 1.6)],
            fill=(48, 60, 75, 255 if k % 5 == 0 else 150),
            width=SS,
        )
    for k in range(-5, 17):
        d.line(
            [P(-0.5, k * 0.1), P(0.5, k * 0.1)],
            fill=(48, 60, 75, 255 if k % 5 == 0 else 150),
            width=SS,
        )
    ga = np.array(g).astype(float)
    yy, xx = np.mgrid[0 : H * SS, 0 : W * SS] / SS
    ga[..., 3] *= np.clip(1.3 - ((xx - 1020) / 600) ** 2 - ((yy - 250) / 330) ** 2, 0, 1)
    base = Image.alpha_composite(base, Image.fromarray(ga.astype(np.uint8)))

    # range ticks along the bottom edge
    g, d = layer()
    for m, lab in ((0, "0"), (0.5, "0.5"), (1.0, "1")):
        x = P(0, m)[0]
        d.line([(x, 452 * SS), (x, 460 * SS)], fill=GREY, width=SS)
        text(d, int(x - text_w(lab, 2 * SS) / 2), 463 * SS, lab, 2 * SS, GREY)
    base = Image.alpha_composite(base, g)

    # camera view cone (half-angle 33.5 deg, rl/sim.py) from the staging pose toward the tags
    g, d = layer()
    sx, sy = approach(0)
    ax, ay = sx, sy - CAR_L / 2
    aim = math.atan2(TAGS[14][0] - ax, ay - TAGS[14][1])
    for k in range(60):
        r0, r1 = k * 0.0175, (k + 1) * 0.0175
        pts = [
            P(
                ax + r * math.sin(aim + sg * math.radians(33.5)),
                ay - r * math.cos(aim + sg * math.radians(33.5)),
            )
            for r, sg in ((r0, -1), (r1, -1), (r1, 1), (r0, 1))
        ]
        d.polygon(pts, fill=ACC + (int(16 * (1 - k / 60)),))
    for sg in (-1, 1):
        for k in range(60):
            r0, r1 = k * 0.0175, (k + 1) * 0.0175
            d.line(
                [
                    P(
                        ax + r0 * math.sin(aim + sg * math.radians(33.5)),
                        ay - r0 * math.cos(aim + sg * math.radians(33.5)),
                    ),
                    P(
                        ax + r1 * math.sin(aim + sg * math.radians(33.5)),
                        ay - r1 * math.cos(aim + sg * math.radians(33.5)),
                    ),
                ],
                fill=ACC + (int(80 * (1 - k / 60)),),
                width=SS,
            )
    base = Image.alpha_composite(base, g)

    # tag card and the three tags at real relative scale
    g, d = layer()
    c0, c1 = P(0.125, -0.175), P(-0.125, -0.045)
    d.rounded_rectangle(
        [min(c0[0], c1[0]), min(c0[1], c1[1]), max(c0[0], c1[0]), max(c0[1], c1[1])],
        6 * SS,
        fill=(24, 31, 42, 255),
        outline=(74, 86, 102, 255),
        width=SS,
    )
    h = TAG * S / 2
    for tid, (x, y) in TAGS.items():
        c = P(x, y)
        d.rectangle([c[0] - h, c[1] - h, c[0] + h, c[1] + h], fill=PAPER + (255,))
        cell = 2 * h / 6
        d.rectangle(
            [c[0] - h + cell, c[1] - h + cell, c[0] + h - cell, c[1] + h - cell], fill=INK + (255,)
        )
        for r, row in enumerate(ARUCO[tid]):
            for k, v in enumerate(row):
                if v == "1":
                    x0, y0 = c[0] - h + (k + 1) * cell, c[1] - h + (r + 1) * cell
                    d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=PAPER + (255,))
        lab = str(tid)
        text(d, int(P(0, -0.175)[0]) + 10 * SS, int(c[1]) - 7 * SS, lab, 2 * SS, GREY)
    base = Image.alpha_composite(base, g)

    # landing skids: capsules, inside gap 0.31 m
    g, d = layer()
    edge, body = (120, 134, 154, 255), (30, 39, 52, 255)
    for sx_ in (-SKID_X, SKID_X):
        a, b = P(sx_, SKID_Y[0]), P(sx_, SKID_Y[1])
        for col, r in ((edge, SKID_R), (body, SKID_R - 0.0035)):
            d.line([a, b], fill=col, width=int(2 * r * S))
            for e in (a, b):
                disc(d, e, r * S, fill=col)
    ya = SKID_Y[1] + SKID_R + 0.045  # dimension line just beyond the skid tips
    d.line([P(-SKID_X + SKID_R, ya), P(SKID_X - SKID_R, ya)], fill=PAPER + (255,), width=SS)
    for sx_ in (-SKID_X + SKID_R, SKID_X - SKID_R):
        d.line([P(sx_, ya - 0.012), P(sx_, ya + 0.012)], fill=PAPER + (255,), width=SS)
    text(
        d,
        int(P(0, ya)[0]) - 10 * SS - text_w("0.31 m", 2 * SS),
        int(P(SKID_X - SKID_R, ya)[1]) - 7 * SS,
        "0.31 m",
        2 * SS,
        PAPER + (255,),
    )
    base = Image.alpha_composite(base, g)

    # approach: dotted centre line, car footprints from faint (staging) to solid (parked)
    g, d = layer()
    for i in range(0, 101, 2):
        disc(d, P(*approach(i / 100)), 2.2 * SS, fill=ACC + (235,))
    for k, s_ in enumerate((0.0, 0.33, 0.64)):
        x, y = approach(s_)
        car(d, x, y, heading(s_), ACC + ((40 + 40 * k) // 2,), ACC + (90 + 60 * k,))
    car(d, 0.0, 0.12, 0.0, ACC + (70,), ACC + (255,))
    gc = P(*GOAL)
    d.line([(gc[0] - 9 * SS, gc[1]), (gc[0] + 9 * SS, gc[1])], fill=ACC + (255,), width=SS)
    d.line([(gc[0], gc[1] - 9 * SS), (gc[0], gc[1] + 9 * SS)], fill=ACC + (255,), width=SS)
    base = Image.alpha_composite(base, g)

    # title: pixel lettering with a hard offset shadow, accent rule, caption
    g, d = layer()
    sc, x0, y0 = 20 * SS, 78 * SS, 140 * SS
    text(d, x0 + sc // 2, y0 + sc // 2, "ugv", sc, (0, 0, 0, 150))
    text(d, x0, y0, "ugv", sc, PAPER + (255,))
    d.rectangle(
        [x0, y0 + 9 * sc + 10 * SS, x0 + text_w("ugv", sc), y0 + 9 * sc + 22 * SS],
        fill=ACC + (255,),
    )
    base = Image.alpha_composite(base, g)

    out = base.resize((W, H), Image.LANCZOS)
    mask = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W * SS - 1, H * SS - 1], 16 * SS, fill=255)
    out.putalpha(mask.resize((W, H), Image.LANCZOS))
    ImageDraw.Draw(out).rounded_rectangle([0, 0, W - 1, H - 1], 16, outline=LINE + (255,), width=1)
    out.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png"), optimize=True)


if __name__ == "__main__":
    main()
