"""FastAPI app for the Lab dashboard. Read-only over the repo except the HDRI cache under
`runs/cache/`; it never talks to the car.

uv run ugv-lab            # http://127.0.0.1:8768
"""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ugv.lab import data, sim

STATIC = Path(__file__).resolve().parent / "static"
HDRI_CACHE = Path(__file__).resolve().parents[3] / "runs" / "cache" / "hdri"
HDRI_V2_CACHE = Path(__file__).resolve().parents[3] / "runs" / "cache" / "hdri_v2"
HDRI_ID = re.compile(r"^[a-z0-9_-]+$")
HDRI_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{id}_1k.hdr"


class NewSession(BaseModel):
    seed: int
    start: list[float] | None = None
    preset: str | None = None
    quality: str = "bad"
    env: str = "lab"
    exposure: float = 0.0  # fixed stops over correct exposure; 0 = auto-exposure


class Step(BaseModel):
    ticks: int
    duty: list[float] | None = None
    auto: bool = False


class Exposure(BaseModel):
    stops: float  # fixed stops over correct exposure; 0 = auto


class CameraFrame(BaseModel):
    detections: list[list[float]] | None = None  # [[keypoint index, u, v], ...] or None when lost
    lost: bool = False
    fx: dict | None = None  # how the browser rendered it (exposure, blur, ...), for the HUD


def create_app() -> FastAPI:
    app = FastAPI(title="ugv lab")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def revalidate(request, call_next):
        # The lab is edited while it runs: make browsers revalidate modules instead of mixing versions.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith(("/static/", "/runs/")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/ledger")
    def ledger() -> dict:
        return data.ledger()

    @app.get("/api/park/evals")
    def park_evals() -> list:
        return data.park_evals()

    @app.get("/api/park/traces/{name}")
    def park_traces(name: str) -> list:
        t = data.park_traces(name)
        if t is None:
            raise HTTPException(404, "no traces for this eval")
        return t

    @app.get("/api/park/searches")
    def park_searches() -> list:
        return data.park_searches()

    @app.get("/api/park/world")
    def park_world() -> dict:
        return {"bay": data.bay(), "plant": data.plant()}

    @app.get("/api/park/plan")
    def park_plan() -> dict:
        return {**data.plan(), "skeleton": sim.skeleton()}

    @app.get("/api/park/hdri/{hid}.hdr")
    def park_hdri(hid: str) -> FileResponse:
        """Panoramas, fetched once and cached under runs/ (never committed).

        Namespaced ids (acg_…, ohl_…) are the OpenHDRI/ambientCG CC0 set, pre-downloaded and
        converted to 2K Radiance by tools/hdri_horizons/fetch_panos.py. Plain ids are Poly Haven
        (CC0), fetched on first use and cached.
        """
        if not HDRI_ID.match(hid):
            raise HTTPException(404, "unknown HDRI")
        v2 = (
            HDRI_V2_CACHE / f"{hid}_2k.hdr"
            if (hid.startswith("acg_") or hid.startswith("ohl_"))
            else None
        )
        if v2 is not None:
            if not v2.exists():
                raise HTTPException(
                    404, f"{hid} not pre-fetched; run tools/hdri_horizons/fetch_panos.py"
                )
            return FileResponse(v2, media_type="image/vnd.radiance")
        path = HDRI_CACHE / f"{hid}_1k.hdr"
        if not path.exists():
            HDRI_CACHE.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            try:
                with urllib.request.urlopen(HDRI_URL.format(id=hid), timeout=60) as r:
                    with tmp.open("wb") as f:
                        shutil.copyfileobj(r, f)
                tmp.replace(path)
            except (urllib.error.URLError, TimeoutError) as e:
                tmp.unlink(missing_ok=True)
                raise HTTPException(502, f"could not fetch {hid}: {e}") from e
        return FileResponse(path, media_type="image/vnd.radiance")

    @app.post("/api/park/session")
    def park_session(n: NewSession) -> dict:
        if n.preset and n.preset not in sim.PRESETS:
            raise HTTPException(400, f"preset must be one of {sorted(sim.PRESETS)}")
        if n.quality not in sim.QUALITY or (
            n.env not in sim.ENVIRONMENTS and not n.env.startswith("hdri:")
        ):
            raise HTTPException(
                400, f"quality in {list(sim.QUALITY)}, env in {list(sim.ENVIRONMENTS)} or hdri:<id>"
            )
        if not 0.0 <= n.exposure <= 4.0:
            raise HTTPException(400, "exposure is fixed stops over correct, 0 (auto) to 4")
        try:
            return sim.SESSIONS.create(n.seed, n.start, n.preset, n.quality, n.env, n.exposure)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/park/session/{sid}/step")
    def park_step(sid: str, st: Step) -> dict:
        try:
            return sim.SESSIONS.step(sid, st.ticks, st.duty, st.auto)
        except KeyError as e:
            raise HTTPException(404, "session expired; start a new round") from e

    @app.post("/api/park/session/{sid}/exposure")
    def park_exposure(sid: str, x: Exposure) -> dict:
        """Record a live exposure change; the browser applies it to the pixels it analyses."""
        if not 0.0 <= x.stops <= 4.0:
            raise HTTPException(400, "exposure is fixed stops over correct, 0 (auto) to 4")
        try:
            return sim.SESSIONS.set_exposure(sid, x.stops)
        except KeyError as e:
            raise HTTPException(404, "session expired; start a new round") from e

    @app.post("/api/park/session/{sid}/camera")
    def park_camera(sid: str, c: CameraFrame) -> dict:
        """One camera frame measured from the actual rendered pixels; the estimator fuses it."""
        if c.detections is not None:
            if len(c.detections) > 15:
                raise HTTPException(400, "at most 15 keypoints")
            for d in c.detections:
                if len(d) != 3:
                    raise HTTPException(400, "each detection is [keypoint_index, u, v]")
                i, u, v = d
                if (
                    int(i) != i
                    or not 0 <= int(i) < 15
                    or not 0.0 <= u <= 1280
                    or not 0.0 <= v <= 720
                ):
                    raise HTTPException(400, f"bad keypoint index {i} or pixel ({u}, {v})")
            c.detections = [[int(d[0]), float(d[1]), float(d[2])] for d in c.detections]
        try:
            return sim.SESSIONS.camera_frame(sid, c.detections, c.lost, c.fx)
        except KeyError as e:
            raise HTTPException(404, "session expired; start a new round") from e

    @app.get("/api/pose/mesh")
    def pose_mesh() -> list:
        return data.mesh()

    @app.get("/api/pose")
    def pose() -> dict:
        return data.pose_runs()

    @app.get("/runs/{rel:path}")
    def run_file(rel: str) -> FileResponse:
        p = data.run_file(rel)
        if p is None:
            raise HTTPException(404, "not found")
        return FileResponse(p)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Lab dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8768)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
