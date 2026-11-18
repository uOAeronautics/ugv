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


class Drive(BaseModel):
    seed: int
    level: str
    start: list[float]


class Frame(BaseModel):
    seed: int
    level: str
    car: list[float]
    tick: int


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

    @app.get("/api/park/round")
    def park_round(seed: int, level: str = "nominal") -> dict:
        _level(level)
        return sim.new_round(seed, level)

    @app.post("/api/park/autopilot")
    def park_autopilot(d: Drive) -> dict:
        _level(d.level)
        return sim.autopilot(d.seed, d.level, d.start)

    @app.post("/api/park/perceive")
    def park_perceive(f: Frame) -> dict:
        _level(f.level)
        return sim.perceive(f.seed, f.level, f.car, f.tick)

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


def _level(level: str) -> None:
    if level not in sim.LEVELS:
        raise HTTPException(400, f"level must be one of {sorted(sim.LEVELS)}")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Lab dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8768)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
