"""FastAPI app for the Lab dashboard. Read-only over the repo; it never talks to the car.

uv run ugv-lab            # http://127.0.0.1:8768
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ugv.lab import data, sim

STATIC = Path(__file__).resolve().parent / "static"


class NewSession(BaseModel):
    seed: int
    start: list[float] | None = None
    preset: str | None = None
    quality: str = "bad"
    env: str = "lab"


class Step(BaseModel):
    ticks: int
    duty: list[float] | None = None
    auto: bool = False


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

    @app.post("/api/park/session")
    def park_session(n: NewSession) -> dict:
        if n.preset and n.preset not in sim.PRESETS:
            raise HTTPException(400, f"preset must be one of {sorted(sim.PRESETS)}")
        if n.quality not in sim.QUALITY or n.env not in sim.ENVIRONMENTS:
            raise HTTPException(
                400, f"quality in {list(sim.QUALITY)}, env in {list(sim.ENVIRONMENTS)}"
            )
        try:
            return sim.SESSIONS.create(n.seed, n.start, n.preset, n.quality, n.env)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/park/session/{sid}/step")
    def park_step(sid: str, st: Step) -> dict:
        try:
            return sim.SESSIONS.step(sid, st.ticks, st.duty, st.auto)
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
