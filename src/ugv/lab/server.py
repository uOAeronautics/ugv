"""FastAPI app for the Lab dashboard. Read-only over the repo; it never talks to the car.

uv run ugv-lab            # http://127.0.0.1:8768
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ugv.lab import data

STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="ugv lab")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

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
