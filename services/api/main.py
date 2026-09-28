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
from typing import cast

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
    CHAT_RETRIEVAL_TOP_K,
    MAX_UPLOAD_BYTES,
    PROJECT_NAME,
    TEMPORAL_ADDRESS,
    UPLOAD_DIR,
)
from services.auth.dependencies import get_current_tenant, get_current_tenant_ws
from services.chat import history as chat_history
from services.chat.condense import condense_query
from services.chat.prompt import NO_CONTEXT_MESSAGE, build_messages
from services.llm_gateway.router import LLMGatewayError, stream_chat_completion
from services.retrieval.pipeline import retrieve_async
from services.retrieval.reranker import warm_reranker

log = structlog.get_logger()

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Only extensions the ingestion pipeline actually knows how to route
# (see workflows.activities.detect_document_type). Rejecting anything
# else here, before a Temporal workflow is even started, avoids silently
# accepting files the pipeline will never process correctly.
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}

# How much of a retrieved chunk is echoed to the client alongside its
# filename/page/score, so a citation can be read without a second lookup.
SOURCE_PREVIEW_CHARS = 400


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
    """Real-time RAG chat: retrieve -> rerank -> stream tokens from the
    LLM gateway -> emit page-level citations.

    Requires ``?api_key=`` as a query parameter (browsers can't set custom
    headers during the WebSocket handshake, so this can't reuse the
    X-API-Key header the REST endpoints use). Every retrieve() call and
    every history read/write is scoped to the resolved tenant_id.

    An optional ``?conversation_id=`` query parameter resumes an existing
    conversation's history; when omitted, a new id is generated and sent
    back in the initial "ready" message so the client can reconnect into
    the same conversation later.

    Protocol (server -> client), per user message:
        {"type": "ready", "conversation_id": "..."}      once, on connect
        {"type": "sources", "sources": [...]}             per turn, before any tokens
        {"type": "token", "content": "..."}               zero or more, streamed
        {"type": "done", "citations": [...]}               once per turn
        {"type": "error", "detail": "..."}                 instead of done, on failure

    Each entry in "sources"/"citations" is
    {"index", "filename", "page_number", "score", "text"}, where
    "index" is the number the model cites inline as "[1]" and "text" is
    a SOURCE_PREVIEW_CHARS preview of the passage sent to the model.
    """
    tenant_id = await get_current_tenant_ws(websocket.query_params.get("api_key"))
    if tenant_id is None:
        await websocket.close(code=1008, reason="Missing or invalid API key")
        return

    conversation_id = websocket.query_params.get("conversation_id") or str(uuid.uuid4())

    await websocket.accept()
    await websocket.send_json({"type": "ready", "conversation_id": conversation_id})

    try:
        while True:
            user_query = await websocket.receive_text()
            log.info(
                "chat_message_received",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )

            try:
                history = await run_in_threadpool(
                    chat_history.get_history, tenant_id, conversation_id
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("chat_history_read_failed", tenant_id=tenant_id, conversation_id=conversation_id, error=str(exc))
                await websocket.send_json({"type": "error", "detail": "Chat history is temporarily unavailable."})
                history = []
                
            history = cast(list[dict], history)

            # Retrieval-only rewrite: on a follow-up turn ("what about page
            # 2?"), the raw user_query is missing the context a bare vector/
            # BM25 search needs. condense_query resolves that using prior
            # turns; the LLM prompt below still gets the original phrasing.
            retrieval_query = await condense_query(history, user_query)
            if retrieval_query != user_query:
                log.info(
                    "query_condensed",
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    original=user_query,
                    condensed=retrieval_query,
                )

            try:
                sources = await retrieve_async(
                    retrieval_query, tenant_id, top_k=CHAT_RETRIEVAL_TOP_K
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("chat_retrieval_failed", tenant_id=tenant_id, error=str(exc))
                await websocket.send_json(
                    {"type": "error", "detail": "Search is temporarily unavailable."}
                )
                continue

            source_summaries = [
                {
                    "index": index,
                    "filename": source["filename"],
                    "page_number": source["page_number"],
                    "score": source.get("score"),
                    # A short preview of the passage the model was given.
                    # Without it a client can render "[1] hr_policy.pdf
                    # page 3" but not the sentence it points at, so a
                    # reader who wants to check a citation has nothing to
                    # check it against. Truncated because the full chunk is
                    # already in the model's context and repeating it in
                    # the wire message would double the size of every
                    # turn's sources payload.
                    "text": source["text"][:SOURCE_PREVIEW_CHARS],
                }
                for index, source in enumerate(sources, start=1)
            ]
            await websocket.send_json({"type": "sources", "sources": source_summaries})

            if not sources:
                # Skip the LLM call entirely: an empty context block
                # invites the model to answer from outside knowledge
                # despite the system prompt, and it burns a request for
                # an answer we already know should be "I don't know."
                await websocket.send_json({"type": "token", "content": NO_CONTEXT_MESSAGE})
                await websocket.send_json({"type": "done", "citations": []})
                try:
                    await run_in_threadpool(
                        chat_history.append_turn,
                        tenant_id,
                        conversation_id,
                        "user",
                        user_query,
                    )
                    await run_in_threadpool(
                        chat_history.append_turn,
                        tenant_id,
                        conversation_id,
                        "assistant",
                        NO_CONTEXT_MESSAGE,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("chat_history_write_failed", tenant_id=tenant_id, conversation_id=conversation_id, error=str(exc))
                continue

            messages = build_messages(
                cast(list[dict], history), sources, user_query
            )

            answer_parts: list[str] = []
            try:
                async for delta in stream_chat_completion(messages):
                    answer_parts.append(delta)
                    await websocket.send_json({"type": "token", "content": delta})
            except LLMGatewayError as exc:
                log.warning("chat_generation_failed", tenant_id=tenant_id, error=str(exc))
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue

            answer = "".join(answer_parts)
            await websocket.send_json({"type": "done", "citations": source_summaries})

            try:
                await run_in_threadpool(
                    chat_history.append_turn, tenant_id, conversation_id, "user", user_query
                )
                await run_in_threadpool(
                    chat_history.append_turn,
                    tenant_id,
                    conversation_id,
                    "assistant",
                    answer,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("chat_history_write_failed", tenant_id=tenant_id, conversation_id=conversation_id, error=str(exc))
                continue
    except WebSocketDisconnect:
        log.info("chat_ws_disconnected", tenant_id=tenant_id, conversation_id=conversation_id)
