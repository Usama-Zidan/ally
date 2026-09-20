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
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from temporalio.client import Client as TemporalClient

from config import (
    MAX_UPLOAD_BYTES,
    PROJECT_NAME,
    TEMPORAL_ADDRESS,
    UPLOAD_DIR,
)
from services.auth.dependencies import get_current_tenant, get_current_tenant_ws
from services.retrieval.reranker import warm_reranker

log = structlog.get_logger()

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Only extensions the ingestion pipeline actually knows how to route
# (see workflows.activities.detect_document_type). Rejecting anything
# else here, before a Temporal workflow is even started, avoids silently
# accepting files the pipeline will never process correctly.
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create a Temporal client and warm the reranker only when reachable.

    This lets tests and local health probes run in a clean degraded mode
    rather than failing during app startup when no Temporal worker service is
    running yet. The reranker is warmed here (rather than lazily on first
    request) so the first real query doesn't pay the ~10s cross-encoder
    load time — see services.retrieval.reranker.warm_reranker.
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

    try:
        await run_in_threadpool(warm_reranker)
        log.info("reranker_warmed")
    except Exception as exc:  # noqa: BLE001
        # Not fatal at startup — Phase 3's chat endpoint will retry lazily
        # on first use, just with the latency hit this warmup exists to
        # avoid. Don't block the app from serving /health while a GPU/model
        # host is still coming up.
        log.warning("reranker_warmup_failed", error=str(exc))

    yield
    log.info("shutting_down")


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/dependencies")
async def health_dependencies() -> JSONResponse:
    """Report whether the configured Temporal service is reachable."""
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
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_current_tenant),
) -> dict:
    """Store an uploaded document and enqueue its ingestion workflow.

    Requires a valid ``X-API-Key`` header (see services.auth). The
    resolved tenant_id is passed into the workflow so every chunk this
    document produces is scoped to that tenant in both Postgres and
    Qdrant — see workflows.ingestion_workflow.IngestDocumentWorkflow.

    Returns identifiers for the document and queued workflow. Raises HTTP
    400 for an invalid/oversized upload, HTTP 401 for a missing/invalid
    API key, or HTTP 503 when Temporal is unavailable or cannot start the
    workflow.
    """
    if app.state.temporal_client is None:
        raise HTTPException(
            status_code=503,
            detail="Temporal service unavailable; workflow cannot be started",
        )

    safe_filename = os.path.basename(file.filename or "")
    extension = os.path.splitext(safe_filename)[1].lower()
    if not safe_filename or extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {extension!r}; allowed: "
                f"{sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
            ),
        )

    doc_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{safe_filename}")

    try:
        # Streamed in fixed-size chunks and capped at MAX_UPLOAD_BYTES so a
        # single large upload can't exhaust worker memory the way
        # `await file.read()` (loading the whole file at once) would, and
        # so an oversized upload is rejected — and its partial temp file
        # cleaned up — before it fills disk.
        bytes_written = 0
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}") from exc

    workflow_id = f"ingest-{doc_id}"
    try:
        await app.state.temporal_client.start_workflow(
            "IngestDocumentWorkflow",
            args=[dest_path, safe_filename, tenant_id],
            id=workflow_id,
            task_queue="ingestion-task-queue",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ingestion_workflow_start_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Temporal workflow could not start: {exc}",
        ) from exc

    log.info("ingestion_started", doc_id=doc_id, workflow_id=workflow_id, tenant_id=tenant_id)
    return {
        "doc_id": doc_id,
        "workflow_id": workflow_id,
        "status": "queued",
    }


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Phase 3 will replace this stub with: retrieve -> rerank -> stream
    tokens from the LLM gateway -> emit citations. For now it just echoes
    to prove the socket lifecycle and history handling work.

    Requires ``?api_key=`` as a query parameter (browsers can't set custom
    headers during the WebSocket handshake, so this can't reuse the
    X-API-Key header the REST endpoints use). The resolved tenant_id will
    scope every retrieve() call once Phase 3 wires this stub up to the
    retrieval pipeline.
    """
    tenant_id = await get_current_tenant_ws(websocket.query_params.get("api_key"))
    if tenant_id is None:
        await websocket.close(code=1008, reason="Missing or invalid API key")
        return

    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            log.info("chat_message_received", tenant_id=tenant_id, message=message)
            # TODO Phase 3: push to LangGraph agent, stream response chunks
            # scoped to retrieve(..., tenant_id=tenant_id)
            await websocket.send_json(
                {"type": "token", "content": f"(stub) you said: {message}"}
            )
            await websocket.send_json({"type": "done", "citations": []})
    except WebSocketDisconnect:
        log.info("chat_ws_disconnected", tenant_id=tenant_id)
