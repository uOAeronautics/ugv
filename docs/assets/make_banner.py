"""ugv README banner: the parking bay drawn as a topographic field-survey chart.

Run: uv run --with pillow --with numpy python docs/assets/make_banner.py
Needs fontconfig plus Noto Sans Mono, DejaVu Serif and Fira Sans Condensed.

Policy frame, metres: drone at the origin, skids along y, inside gap 0.31 m,
ArUco tags 14/6/3 at the head. The bay is a contour saddle (0.32 -> 0.30), the
tags are triangulation stations, and the approach is a dashed traverse chained
ST 0+00 -> 0+00.5 -> GOAL 0+01. The chart is schematic (NTS); the rotated stamp
carries the RL ladder L0-L7 with L7 filled.
"""

from __future__ import annotations

import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 480
INK, RED, MUTE, PAPER = (35, 42, 51), (181, 68, 44), (138, 133, 119), (242, 237, 225)
_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(family: str, size: int) -> ImageFont.FreeTypeFont:
    key = (family, size)
    if key not in _FONTS:
        path = subprocess.run(
            ["fc-match", "-f", "%{file}", family], check=True, capture_output=True, text=True
        ).stdout
        _FONTS[key] = ImageFont.truetype(path, size)
    return _FONTS[key]


def layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def grain(im: Image.Image, amount: int = 4, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.array(im).astype(np.int16)
    a[..., :3] = np.clip(
        a[..., :3] + rng.integers(-amount, amount + 1, a.shape[:2])[..., None], 0, 255
    )
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def mecanum(d: ImageDraw.ImageDraw, cx: float, cy: float, L: float, Wd: float) -> None:
    """Top-down car footprint, heading along +y; roller diagonals form an X."""
    d.polygon(
        [
            (cx - Wd / 2 + 6, cy - L / 2),
            (cx + Wd / 2 - 6, cy - L / 2),
            (cx + Wd / 2 - 6, cy + L / 2),
            (cx - Wd / 2 + 6, cy + L / 2),
        ],
        outline=INK + (235,),
        width=3,
    )
    for u in (-L * 0.32, L * 0.32):
        for v in (-Wd / 2 + 2, Wd / 2 - 2):
            d.rectangle(
                [cx + v - 7, cy + u - 14, cx + v + 7, cy + u + 14], outline=INK + (235,), width=3
            )
            sg = 1 if u * v > 0 else -1
            for k in (-9, -3, 3, 9):
                d.line(
                    [(cx + v - 7, cy + u + k - 4 * sg), (cx + v + 7, cy + u + k + 4 * sg)],
                    fill=INK + (235,),
                    width=3,
                )


def main() -> None:
    im = grain(Image.new("RGBA", (W, H), PAPER + (255,)))

    # contour saddle around the bay, knocked out under the stamp and GOAL label
    g, d = layer()
    cx, cy = 1150, 250
    for k in range(9):
        r0 = 60 + k * 34
        pts = []
        for t in np.linspace(0, 2 * math.pi, 220):
            wob = (
                1
                + 0.10 * math.sin(3 * t + k * 0.7)
                + 0.06 * math.sin(5 * t - k)
                + 0.04 * math.sin(8 * t + 2 * k)
            )
            pts.append((cx + r0 * wob * 1.35 * math.cos(t), cy + r0 * wob * 0.82 * math.sin(t)))
        d.line(pts + [pts[0]], fill=INK + (70 if k % 3 else 120,), width=2 if k % 3 else 3)
    for k, lab in ((2, "0.30"), (5, "0.31"), (8, "0.32")):
        lx, ly = cx - (60 + k * 34) * 1.35 - 52, cy - 10
        d.rectangle([lx - 4, ly - 2, lx + 44, ly + 22], fill=PAPER + (255,))
        d.text((lx, ly), lab, font=font("DejaVu Serif:style=Bold", 18), fill=INK + (215,))
    clip = Image.new("L", (W, H), 0)
    ImageDraw.Draw(clip).rectangle([28, 28, W - 29, H - 29], fill=255)
    ko = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ko).rectangle([1216, 316, 1320, 348], fill=255)
    sm = Image.new("L", (372, 128), 0)
    ImageDraw.Draw(sm).rectangle([0, 0, 371, 127], fill=255)
    ko.paste(sm.rotate(-6, expand=True, resample=Image.BICUBIC), (640, 36))
    ga = np.array(g)
    ga[..., 3] = (
        ga[..., 3].astype(float) * np.minimum(np.array(clip), 255 - np.array(ko)) / 255
    ).astype(np.uint8)
    im.alpha_composite(Image.fromarray(ga, "RGBA"))

    g, d = layer()
    for x in range(80, W, 80):
        d.line([(x, 26), (x, 34)], fill=INK + (120,), width=2)
        d.line([(x, H - 34), (x, H - 26)], fill=INK + (120,), width=2)
    for y in range(80, H, 80):
        d.line([(26, y), (34, y)], fill=INK + (120,), width=2)
        d.line([(W - 34, y), (W - 26, y)], fill=INK + (120,), width=2)
    d.rectangle([18, 18, W - 19, H - 19], outline=INK + (200,), width=3)
    d.rectangle([26, 26, W - 27, H - 27], outline=INK + (120,), width=1)

    # dashed traverse with survey chainage
    p0, p1, n = (620, 400), (1150, 250), 26
    for i in range(n):
        t0, t1 = i / n, (i + 0.55) / n
        a = (
            p0[0] + (p1[0] - p0[0]) * t0,
            p0[1] + (p1[1] - p0[1]) * t0 + 26 * math.sin(math.pi * t0),
        )
        b = (
            p0[0] + (p1[0] - p0[0]) * t1,
            p0[1] + (p1[1] - p0[1]) * t1 + 26 * math.sin(math.pi * t1),
        )
        d.line([a, b], fill=RED + (220,), width=3)
    for i, lab in ((0, "ST 0+00"), (13, "0+00.5")):
        t = i / n
        p = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t + 26 * math.sin(math.pi * t))
        d.ellipse(
            [p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5],
            fill=PAPER + (255,),
            outline=RED + (255,),
            width=2,
        )
        d.text((p[0] - 24, p[1] - 30), lab, font=font("DejaVu Serif", 15), fill=INK + (220,))
    d.ellipse(
        [p1[0] - 5, p1[1] - 5, p1[0] + 5, p1[1] + 5],
        fill=PAPER + (255,),
        outline=RED + (255,),
        width=2,
    )

    # skid parcel, gap dimension, car at the goal
    for sx in (-52, 52):
        d.rounded_rectangle(
            [1150 + sx - 12, 168, 1150 + sx + 12, 332],
            12,
            fill=INK + (30,),
            outline=INK + (220,),
            width=3,
        )
        for yy in range(180, 330, 12):
            d.line([(1150 + sx - 10, yy + 8), (1150 + sx + 10, yy)], fill=INK + (90,), width=1)
    d.line([(1098, 150), (1202, 150)], fill=INK + (230,), width=2)
    for xx in (1098, 1202):
        d.line([(xx, 144), (xx, 156)], fill=INK + (230,), width=2)
    d.text((1112, 122), "0.31 m", font=font("DejaVu Serif:style=Bold", 20), fill=INK + (255,))
    mecanum(d, 1150, 250, 96, 74)
    d.line([(1141, 250), (1159, 250)], fill=RED + (255,), width=2)
    d.line([(1150, 241), (1150, 259)], fill=RED + (255,), width=2)

    # triangulation stations for the three tags
    for (tx, ty), tid in (((1330, 120), 14), ((1440, 250), 6), ((1330, 380), 3)):
        d.polygon(
            [(tx, ty - 12), (tx - 11, ty + 8), (tx + 11, ty + 8)], outline=RED + (230,), width=2
        )
        d.line([(tx, ty - 4), (tx, ty + 4)], fill=RED + (230,), width=2)
        d.text((tx + 16, ty - 10), f"TAG {tid}", font=font("DejaVu Serif", 16), fill=INK + (220,))
        d.line([(tx - 11, ty + 8), (1202, 250)], fill=MUTE + (110,), width=1)

    rx, ry = 1490, 90
    for k in range(8):
        a = k * math.pi / 4
        r = 30 if k % 2 == 0 else 16
        d.polygon(
            [
                (rx, ry),
                (rx + r * math.cos(a - 0.16), ry + r * math.sin(a - 0.16)),
                (rx + (r + 12) * math.cos(a), ry + (r + 12) * math.sin(a)),
                (rx + r * math.cos(a + 0.16), ry + r * math.sin(a + 0.16)),
            ],
            fill=INK + (200 if k % 2 == 0 else 90,),
        )
    d.text((rx - 6, ry - 52), "N", font=font("DejaVu Serif:style=Bold", 18), fill=INK + (230,))
    d.text(
        (74, 420),
        "traverse & bay schematic · NTS",
        font=font("DejaVu Serif", 17),
        fill=MUTE + (230,),
    )

    d.text((70, 100), "ugv", font=font("Noto Sans Mono:style=Black", 150), fill=INK + (255,))
    d.rectangle([74, 306, 402, 316], fill=RED + (255,))
    d.text(
        (74, 332), "a mecanum car parks itself", font=font("DejaVu Serif", 26), fill=INK + (230,)
    )
    d.text(
        (74, 366),
        "under a drone, from sim to floor",
        font=font("DejaVu Serif", 26),
        fill=INK + (230,),
    )
    d.text(
        (74, 60),
        "FIELD CHART · SHEET L7 · uOAeronautics",
        font=font("Fira Sans Condensed:style=Bold", 20),
        fill=MUTE + (255,),
    )
    # GOAL label last, haloed over the tag leader that crosses it
    d.rectangle([1220, 320, 1316, 344], fill=PAPER + (255,))
    d.text((1224, 322), "GOAL 0+01", font=font("DejaVu Serif", 15), fill=INK + (220,))
    im.alpha_composite(g)

    # rotated survey stamp carrying the RL ladder
    st = Image.new("RGBA", (372, 128), (0, 0, 0, 0))
    sd = ImageDraw.Draw(st)
    sd.rectangle([4, 4, 367, 123], fill=PAPER + (255,), outline=RED + (210,), width=3)
    sd.text(
        (18, 12),
        "L7 FIELD SURVEY",
        font=font("Fira Sans Condensed:style=Bold", 30),
        fill=RED + (225,),
    )
    sd.text((18, 50), "REV L7 · SIM TO FLOOR", font=font("DejaVu Serif", 19), fill=RED + (205,))
    for i in range(8):
        x = 18 + i * 42
        lit = i == 7
        sd.rectangle(
            [x, 82, x + 34, 106],
            outline=RED + (220 if lit else 120,),
            width=2,
            fill=RED + (210 if lit else 0,),
        )
        sd.text(
            (x + 8, 86),
            f"L{i}",
            font=font("Noto Sans Mono", 15),
            fill=PAPER + (255,) if lit else RED + (170,),
        )
    im.alpha_composite(st.rotate(-6, expand=True, resample=Image.BICUBIC), (640, 36))

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], 16, fill=255)
    im.putalpha(mask)
    ImageDraw.Draw(im).rounded_rectangle(
        [0, 0, W - 1, H - 1], 16, outline=(36, 46, 66, 255), width=2
    )
    im.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png"), optimize=True)


if __name__ == "__main__":
    main()
