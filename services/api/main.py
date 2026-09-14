"""
FastAPI entrypoint for the application.

Phase 0/1 scope: health checks + upload endpoint that kicks off a Temporal
ingestion workflow + a WebSocket stub that will later stream LLM responses
with citations (Phase 3).
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from temporalio.client import Client as TemporalClient

from config import (
    PROJECT_NAME,
    TEMPORAL_ADDRESS,
    UPLOAD_DIR,
)

log = structlog.get_logger()

os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create a Temporal client only when the dependency is reachable.

    This lets tests and local health probes run in a clean degraded mode
    rather than failing during app startup when no Temporal worker service is
    running yet.
    """
    app.state.temporal_client = None
    app.state.temporal_connected = False

    try:
        app.state.temporal_client = await TemporalClient.connect(TEMPORAL_ADDRESS)
        app.state.temporal_connected = True
        log.info("temporal_client_connected", address=TEMPORAL_ADDRESS)
    except Exception as exc:  # noqa: BLE001
        app.state.temporal_client = None
        app.state.temporal_connected = False
        log.warning("temporal_client_unavailable", address=TEMPORAL_ADDRESS, error=str(exc))

    yield
    log.info("shutting_down")


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/dependencies")
async def health_dependencies() -> JSONResponse:
    """Lightweight check that downstream infra is reachable.

    Returns a clean degraded status if Temporal is not running instead of
    crashing the whole process during FastAPI app startup or tests.
    """
    checks = {"temporal": "unavailable"}
    if app.state.temporal_client is None:
        return JSONResponse(checks)

    try:
        async for _ in app.state.temporal_client.list_workflows():
            break
        checks["temporal"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["temporal"] = "unavailable"
        log.warning("temporal_dependency_probe_failed", error=str(exc))
    return JSONResponse(checks)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Accepts a document, stores it, and starts the ingestion workflow
    when the Temporal client is reachable.

    In offline or test contexts, the request degrades cleanly with a 503
    instead of crashing the FastAPI app during startup or import.
    """
    if app.state.temporal_client is None:
        raise HTTPException(
            status_code=503,
            detail="Temporal service unavailable; workflow cannot be started",
        )

    doc_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    try:
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}") from exc

    workflow_id = f"ingest-{doc_id}"
    try:
        await app.state.temporal_client.start_workflow(
            "IngestDocumentWorkflow",
            args=[dest_path, file.filename],
            id=workflow_id,
            task_queue="ingestion-task-queue",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ingestion_workflow_start_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Temporal workflow could not start: {exc}",
        ) from exc

    log.info("ingestion_started", doc_id=doc_id, workflow_id=workflow_id)
    return {
        "doc_id": doc_id,
        "workflow_id": workflow_id,
        "status": "queued",
    }


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Phase 3 will replace this stub with: retrieve -> rerank -> stream
    tokens from the LLM gateway -> emit citations. For now it just echoes
    to prove the socket lifecycle and history handling work."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            log.info("chat_message_received", message=message)
            # TODO Phase 3: push to LangGraph agent, stream response chunks
            await websocket.send_json(
                {"type": "token", "content": f"(stub) you said: {message}"}
            )
            await websocket.send_json({"type": "done", "citations": []})
    except WebSocketDisconnect:
        log.info("chat_ws_disconnected")
