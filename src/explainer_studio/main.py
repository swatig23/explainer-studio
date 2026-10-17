"""FastAPI app: one endpoint that runs the pipeline, plus the static
front-end (the mockup, upgraded to call this for real data instead of
hardcoded JS arrays).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .agents.ingest import DocumentTooLongError
from .models import ExplainerRequest, ExplainerResult
from .pipeline import run_pipeline

app = FastAPI(title="Vernacular Explainer Studio")


@app.post("/api/generate", response_model=ExplainerResult)
async def generate(request: ExplainerRequest) -> ExplainerResult:
    try:
        return await run_pipeline(request)
    except DocumentTooLongError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("[api] pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
