"""ugv README banner: the parking bay as a quiet topographic chart.

Run: uv run --with pillow --with numpy python docs/assets/make_banner.py
Needs fontconfig plus Noto Sans Mono.

Policy frame, metres: drone at the origin, skids along y, inside gap 0.31 m.
The bay is a contour saddle; the dashed approach curves in from the staging
point and ends tangent to the parked heading (along the skid axis, yaw 0).
The chip row is the repo's real sim-to-real ladder L0-L7, L7 = the real car.
"""

from __future__ import annotations

import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 480
INK, RED, MUTE, PAPER = (35, 42, 51), (181, 68, 44), (138, 133, 119), (242, 237, 225)
RULE_Y, RULE_GAP = 282, (150, 260)  # rule crosses the g descender, knocked out around it
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


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
    )


def mecanum(d: ImageDraw.ImageDraw, cx: float, cy: float, L: float, Wd: float) -> None:
    """Top-down car footprint, heading along the skid axis; roller diagonals form an X."""
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

    # contour saddle around the bay
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
    clip = Image.new("L", (W, H), 0)
    ImageDraw.Draw(clip).rectangle([28, 28, W - 29, H - 29], fill=255)
    ga = np.array(g)
    ga[..., 3] = (ga[..., 3].astype(float) * np.array(clip) / 255).astype(np.uint8)
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

    # approach: curves in from staging, ends tangent to the parked heading
    p0, p1, p2, p3 = (620, 400), (900, 380), (1150, 430), (1150, 322)
    ts = np.linspace(0, 1, 120)
    pts = [bezier(p0, p1, p2, p3, t) for t in ts]
    for i in range(0, len(pts) - 1, 4):
        d.line([pts[i], pts[min(i + 2, len(pts) - 1)]], fill=RED + (220,), width=3)
    d.ellipse(
        [p0[0] - 5, p0[1] - 5, p0[0] + 5, p0[1] + 5],
        fill=PAPER + (255,),
        outline=RED + (255,),
        width=2,
    )

    # skid bay and the parked car, aligned to the skid axis
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
    mecanum(d, 1150, 250, 96, 74)
    d.line([(1141, 250), (1159, 250)], fill=RED + (255,), width=2)
    d.line([(1150, 241), (1150, 259)], fill=RED + (255,), width=2)

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

    # wordmark; rule passes under u and v, cut around the g descender
    d.text((70, 100), "ugv", font=font("Noto Sans Mono:style=Black", 150), fill=INK + (255,))
    d.rectangle([74, RULE_Y, RULE_GAP[0], RULE_Y + 10], fill=RED + (255,))
    d.rectangle([RULE_GAP[1], RULE_Y, 344, RULE_Y + 10], fill=RED + (255,))

    # the real sim-to-real ladder, L7 = the real car
    for i in range(8):
        x = 74 + i * 42
        lit = i == 7
        d.rectangle(
            [x, 340, x + 34, 364],
            outline=RED + (220 if lit else 120,),
            width=2,
            fill=RED + (210 if lit else 0,),
        )
        d.text(
            (x + 8, 344),
            f"L{i}",
            font=font("Noto Sans Mono", 15),
            fill=PAPER + (255,) if lit else RED + (170,),
        )
    im.alpha_composite(g)

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], 16, fill=255)
    im.putalpha(mask)
    ImageDraw.Draw(im).rounded_rectangle(
        [0, 0, W - 1, H - 1], 16, outline=(36, 46, 66, 255), width=2
    )
    im.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png"), optimize=True)


if __name__ == "__main__":
    main()
