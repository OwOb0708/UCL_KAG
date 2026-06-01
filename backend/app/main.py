from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    GraphFact,
    HealthResponse,
    StatusResponse,
)
from app.services.gdrive_loader import GoogleDriveLoader
from app.services.kag_service import KAGService
from app.services.sync_service import ingest_drive_folder, periodic_sync

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_workspace = Path(__file__).resolve().parents[1]
_frontend = _workspace / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

_status: dict = {"indexing_status": "starting", "indexed_docs": 0}


@app.on_event("startup")
async def startup() -> None:
    if not settings.openai_base_url or not settings.openai_api_key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required.")

    kag = KAGService(
        openspg_host=settings.openspg_host,
        project_id=settings.kag_project_id,
        namespace=settings.kag_namespace,
        schema_path=str(_workspace / "schema" / "UCLLab.schema"),
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        chat_model=settings.openai_chat_model,
        planner_model=settings.openai_planner_model or settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
    )
    await kag.initialize()
    app.state.kag = kag

    drive = GoogleDriveLoader(
        service_account_json=settings.google_service_account_json,
        scopes=settings.google_drive_scope_list,
    )
    app.state.drive = drive

    _status["indexing_status"] = "ready"

    if settings.gdrive_folder_id:
        asyncio.create_task(_background_ingest(kag, drive, settings.gdrive_folder_id))
        asyncio.create_task(
            periodic_sync(
                folder_id=settings.gdrive_folder_id,
                drive=drive,
                kag=kag,
                interval_hours=settings.sync_interval_hours,
                status_ref=_status,
            )
        )


async def _background_ingest(kag: KAGService, drive: GoogleDriveLoader, folder_id: str) -> None:
    _status["indexing_status"] = "indexing"
    try:
        n, _ = await ingest_drive_folder(folder_id, drive, kag)
        _status["indexed_docs"] = n
        print(f"[startup] initial ingest done: {n} documents")
    except Exception as exc:
        print(f"[startup] ingest error: {exc}")
        _status["indexing_status"] = f"error: {exc}"
        return
    _status["indexing_status"] = "ready"


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def homepage():
    idx = _frontend / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "Frontend not mounted"}


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(**_status)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    kag: KAGService = app.state.kag
    try:
        answer, facts, trace = await kag.solve(payload.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        citations=[],
        graph_facts=[
            GraphFact(
                subject=f.get("subject", ""),
                predicate=f.get("predicate", ""),
                object=f.get("object", ""),
                source=f.get("source"),
            )
            for f in facts
        ],
        trace=trace,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    pass
