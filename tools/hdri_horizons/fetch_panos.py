#!/usr/bin/env python3
"""Download the extended panorama set (ambientCG Environment HDRIs + OpenHDRI) and merge it
into src/ugv/lab/static/hdri.json, which the lab reads on every challenge page load.

All downloaded files land under runs/cache/hdri_v2/ (gitignored); the tracked index only
carries ids, names and credits. Poly Haven's 998-map index is kept in static/hdri-polyhaven.json.

Run:  uv run python tools/hdri_horizons/fetch_panos.py [--ambientcg-2k 2K] [--skip-openhdri] [--skip-ambientcg]
"""

import argparse
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "cache" / "hdri_v2"
INDEX = ROOT / "src" / "ugv" / "lab" / "static" / "hdri.json"
POLYHAVEN_BACKUP = ROOT / "src" / "ugv" / "lab" / "static" / "hdri-polyhaven.json"

AMBIENTCG_API = "https://ambientcg.com/api/v2/full_json?type=hdri&limit=100&offset={offset}&include=downloadData"
AMBIENTCG_CATEGORIES = {
    "Day Environment HDRI",
    "Evening Environment HDRI",
    "Indoor Environment HDRI",
    "Night Environment HDRI",
    "Morning Environment HDRI",
}
OPENHDRI_LIST = [
    (
        "indoor",
        [
            "fly-studio-05",
            "fly-studio-04",
            "fly-studio-03",
            "fly-studio-02",
            "fly-studio-01",
            "big-studio-04",
            "big-studio-03",
            "big-studio-02",
            "big-studio-01",
            "studio-2nd-floor-snoot",
            "studio-2nd-floor",
            "studio-2nd-floor-cyclorama",
            "chapel-interior",
            "historic-flour-mill",
            "fluorescent-lit-machinery-room",
            "hilberts-mill-conference-room",
            "ludwikowice-underpass",
        ],
    ),
    (
        "outdoor",
        [
            "sunlit-castle-fountain",
            "gruntowa-may-cumulus",
            "polanica-fountain-terrace",
            "polanica-fountain-square",
            "polanica-bandshell-morning",
            "gruntowa-april-pre-sunset",
            "gruntowa-april-golden-hour",
            "egg-hill-april-late-afternoon",
            "jugow-jana-overcast",
            "jugow-glowna-overcast",
            "egg-hill-distant-valley-glow",
            "egg-hill-cloud-break",
            "wlodyka-bunker-entrance",
            "egg-hill-cloudy",
            "egg-hill-sunset-clear-sky",
            "zum-bergkeller-ruins",
            "brick-wall-alley",
            "overcast-frozen-river",
            "rafalonka-winter-meadow",
            "hausdorf-winter-view",
            "gruntowa-autumn-sunclouds",
            "skarpa-winter-forest",
            "rubinkowo-sandbox-overcast",
            "egg-hill-autumn-noon",
            "molke-sunny-road",
            "naroznik-autumn-haze",
            "pilgrim-farmland-october-sun",
            "pilgrim-farmland-overcast",
            "broumov-airport-runway",
            "autumn-bridge-arch",
            "rock-theatre-viewpoint",
            "gruntowa-meadow-overcast",
            "calvary-hill-chapels",
            "stairway-to-basilica",
            "west-basilica-courtyard",
            "sunny-industrial-yard",
            "factory-road-turnaround",
            "jawornik-meadow-noon",
            "neogothic-church-exterior",
            "gruntowa-meadow-clear-sky",
            "gruntowa-meadow-afternoon",
            "overcast-afternoon-quarry",
            "broumov-airport-sunset",
        ],
    ),
    ("urban", []),
    ("nature", []),
    ("studio", []),
]


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    subprocess.run(
        [
            "curl",
            "-sfL",
            "--retry",
            "3",
            "--max-time",
            "600",
            "-A",
            "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0",
            "-o",
            str(dst),
            url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def to_2k_hdr(exr_path, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["magick", str(exr_path), "-resize", "2048x1024", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ambientcg_entries(size):
    assets = []
    for page in range(5):
        data = fetch_json(AMBIENTCG_API.format(offset=page * 100))
        assets.extend(data["foundAssets"])
        print(f"ambientCG page {page}: {len(assets)}")
    out = []
    for a in assets:
        cat = a.get("displayCategory") or ""
        if cat not in AMBIENTCG_CATEGORIES:
            continue
        dl = None
        folders = a.get("downloadFolders") or {}
        zip_info = (
            (folders.get("default") or {}).get("downloadFiletypeCategories", {}).get("zip", {})
        )
        for d in zip_info.get("downloads", []):
            if d.get("attribute") == size:
                dl = d["downloadLink"]
                break
        if dl:
            out.append((a["assetId"], cat, a.get("displayName", a["assetId"]), dl))
    return out


def fetch_ambientcg(size, exr_keep=False):
    entries = ambientcg_entries(size)
    print(f"ambientCG selected: {len(entries)}")
    results = []

    def work(item):
        asset_id, cat, name, url = item
        base = CACHE / "ambientcg"
        exr = base / f"{asset_id}_2K_HDR.exr"
        try:
            z = base / f"{asset_id}_2K.zip"
            download(url, z)
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.lower().endswith("_hdr.exr"):
                        zf.extract(n, base)
                        shutil.move(base / n, exr)
                        break
                else:
                    raise RuntimeError("no HDR .exr in zip")
            z.unlink()
            hdr = CACHE / f"acg_{asset_id.lower()}_2k.hdr"
            to_2k_hdr(exr, hdr)
            exr.unlink()
            return {
                "id": f"acg_{asset_id.lower()}",
                "name": name,
                "env": "indoor" if cat == "Indoor Environment HDRI" else "outdoor",
                "cat": cat,
                "source": "ambientcg.com",
                "license": "CC0",
            }
        except Exception as e:
            print(f"FAIL {asset_id}: {e}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(work, entries), 1):
            if r:
                results.append(r)
            if i % 25 == 0:
                print(f"{i}/{len(entries)} done", flush=True)
    return results


def fetch_openhdri():
    results = []
    for category, slugs in OPENHDRI_LIST:
        for slug in slugs:
            slug = slug.strip()
            if not slug:
                continue
            name = " ".join(w.capitalize() for w in re.split(r"[-_]", slug))
            exr = CACHE / "openhdri" / f"{slug}_2K.exr"
            try:
                subprocess.run(
                    [
                        "curl",
                        "-sfL",
                        "--retry",
                        "3",
                        "--max-time",
                        "600",
                        "-A",
                        "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0",
                        "-e",
                        "https://openhdri.org/",
                        "-o",
                        str(exr),
                        f"https://dl.openhdri.org/file/openexr/{slug}_2K.exr",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                hdr = CACHE / f"ohl_{slug}_2k.hdr"
                to_2k_hdr(exr, hdr)
                exr.unlink()
                env = (
                    "indoor"
                    if category == "indoor"
                    else ("urban" if category == "urban" else "outdoor")
                )
                results.append(
                    {
                        "id": f"ohl_{slug}",
                        "name": name,
                        "env": env,
                        "cat": f"OpenHDRI {category.capitalize()}",
                        "source": "openhdri.org",
                        "license": "CC0",
                    }
                )
            except Exception as e:
                print(f"FAIL {slug}: {e}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambientcg-2k", default="2K")
    ap.add_argument("--skip-openhdri", action="store_true")
    ap.add_argument("--skip-ambientcg", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    entries = []
    if not args.skip_ambientcg:
        entries += fetch_ambientcg(args.ambientcg_2k)
    if not args.skip_openhdri:
        entries += fetch_openhdri()
    print(f"fetched {len(entries)} panoramas")

    current_index = json.loads(INDEX.read_text()) if INDEX.exists() else {"assets": []}
    if current_index.get("assets") and not POLYHAVEN_BACKUP.exists():
        POLYHAVEN_BACKUP.write_text(
            json.dumps(
                {
                    **current_index,
                    "source": "polyhaven.com",
                    "_comment": "Poly Haven CC0 set, kept for reference; the active set is static/hdri.json",
                },
                indent=1,
            )
            + "\n"
        )
        print(f"backed up Poly Haven index to {POLYHAVEN_BACKUP}")

    merged = entries
    INDEX.write_text(
        json.dumps(
            {"source": "OpenHDRI + ambientCG", "license": "CC0 (both)", "assets": merged}, indent=1
        )
        + "\n"
    )
    print(f"wrote {INDEX} ({len(merged)} assets)")


if __name__ == "__main__":
    main()
