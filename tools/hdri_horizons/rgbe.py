"""Shared Radiance RGBE decode + tonemapped PNG encode for the horizon tools.

The lab ships .hdr files on demand; this module reads them back for the offline
predictor and the labeling dashboard. PNG is written with stdlib zlib only.
"""

import struct
import zlib


def load_hdr(path):
    with open(path, "rb") as f:
        d = f.read()
    pos = 0

    def line():
        nonlocal pos
        start = pos
        while d[pos] != 10:
            pos += 1
        s = d[start:pos].decode("ascii", "replace")
        pos += 1
        return s

    if line() != "#?RADIANCE":
        raise ValueError(f"not a Radiance file: {path}")
    for _ in iter(line, ""):
        pass
    import re

    m = re.match(r"^-Y\s+(\d+)\s+\+X\s+(\d+)$", line())
    if not m:
        raise ValueError("bad resolution line")
    h, w = int(m.group(1)), int(m.group(2))
    rgb = bytearray()
    for _ in range(h):
        if d[pos] == 2 and d[pos + 1] == 2 and (d[pos + 2] & 0x80) == 0:
            pos += 4
            linebuf = [0] * (w * 4)
            for c in range(4):
                x = 0
                while x < w:
                    n = d[pos]
                    pos += 1
                    if n > 128:
                        v = d[pos]
                        pos += 1
                        for i in range(n - 128):
                            linebuf[(x + i) * 4 + c] = v
                        x += n - 128
                    else:
                        for i in range(n):
                            linebuf[(x + i) * 4 + c] = d[pos]
                            pos += 1
                        x += n
            rgb.extend(linebuf)
        else:
            rgb.extend(d[pos : pos + w * 4])
            pos += w * 4
    return w, h, rgb


def luminance_grid(rgb, w, h):
    grid = []
    for y in range(h):
        row = []
        base = y * w * 4
        for x in range(w):
            i = base + x * 4
            e = rgb[i + 3]
            if e == 0:
                row.append(0.0)
                continue
            import math

            f = 2.0 ** (e - 136)  # e-128 exponent, RGBE /256, log10 later
            lum = (0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2]) * f
            row.append(math.log10(lum + 1e-3))
        grid.append(row)
    return grid


def tonemap_png(rgb, w, h, max_size=512):
    """Compress RGBE into an 8-bit PNG with a global shoulder curve v = x/(x+1)."""
    scale = min(1.0, max_size / max(w, h))
    out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))

    def curve(x):  # fixed gain of 4: typical HDR interiors sit mid-tone, windows clip
        x *= 4.0
        return min(255, int(255.0 * x / (x + 1.0)))

    scan = bytearray()
    for oy in range(out_h):
        scan.append(0)  # filter: none
        y = min(h - 1, int(oy / scale))
        for ox in range(out_w):
            x = min(w - 1, int(ox / scale))
            i = (y * w + x) * 4
            e = rgb[i + 3]
            if e == 0:
                scan.extend((0, 0, 0))
                continue
            f = 2.0 ** (e - 136)
            scan.extend((curve(rgb[i] * f), curve(rgb[i + 1] * f), curve(rgb[i + 2] * f)))
    return encode_png(out_w, out_h, bytes(scan))


def encode_png(width, height, rgb_scanlines):
    """Encode raw filtered RGB scanlines into a PNG blob."""

    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = zlib.compress(rgb_scanlines, 6)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
