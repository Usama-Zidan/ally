"""
FastAPI entrypoint for the application.

Health, Temporal document upload, streaming chat and the PC companion
transport. Chat turn orchestration is delegated to services.chat.use_cases.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator, cast

import structlog
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from temporalio.client import Client as TemporalClient

from config import runtime_settings
from services.agent.graph import route_message
from services.agent.classifier import build_classifier
from services.agent.graph import build_graph
from services.agent.pc_agent_bridge import PCAgentBridge
from services.agent.registry import build_default_registry
from services.auth.dependencies import get_ingestion_tenant
from services.auth.devices import DeviceRegistry, PairedDevice
from services.auth.admission import PairingAdmission
from services.auth.principal import Principal
from services.auth.tickets import PostgresTicketStore
from services.api.routes.authentication import router as authentication_router
from services.api.routes.devices import router as devices_router
from services.api.websocket.chat import (
    chat_ws as chat_ws,
    handle_chat_session,
    router as chat_websocket_router,
)
from services.api.websocket.pc_agent import (
    handle_pc_agent_session,
    pc_agent_ws as pc_agent_ws,
    router as pc_agent_websocket_router,
)
from services.api.websocket.security import SocketSession, WebSocketSecurity
from services.chat.mongo_history import MongoHistoryStore
from services.chat.ports import HistoryStore
from services.chat.condense import condense_query
from services.chat.prompt import NO_CONTEXT_MESSAGE as NO_CONTEXT_MESSAGE
from services.chat.prompt import build_messages as build_messages
from services.chat.use_cases import (
    SOURCE_PREVIEW_CHARS as SOURCE_PREVIEW_CHARS,
    ChatUseCase,
    ChatUseCaseDependencies,
)
from services.ingestion.upload import (
    ALLOWED_UPLOAD_EXTENSIONS as ALLOWED_UPLOAD_EXTENSIONS,
    LocalUploadStorage,
    S3UploadStorage,
    UploadFailure,
    UploadUseCase,
)
from services.llm_gateway.router import LLMGateway, LLMGatewayError as LLMGatewayError
from services.llm_gateway.router import stream_chat_completion
from services.retrieval.pipeline import retrieve_async
from services.retrieval.qdrant_store import QdrantStore
from services.retrieval.model_runtime import RetrievalModels
from services.retrieval.model_loading import configure_model_inference
from services.retrieval.reranker import warm_reranker
from services.runtime.settings import RuntimeSettings
from services.runtime.task_drain import DrainableTasks

log = structlog.get_logger()
_DEFAULT_STREAM_COMPLETION = stream_chat_completion


def _stream_completion_for(app: FastAPI) -> Callable[..., AsyncGenerator[str, None]]:
    # Preserve the historical module-level test/extension seam while normal
    # requests use the gateway owned by their FastAPI application.
    if stream_chat_completion is not _DEFAULT_STREAM_COMPLETION:
        return stream_chat_completion
    gateway = getattr(app.state, "llm_gateway", None)
    return gateway.stream_chat_completion if gateway is not None else stream_chat_completion


api_router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create a Temporal client and warm the reranker only when reachable.

    This lets tests and local health probes run in a clean degraded mode
    rather than failing during app startup when no Temporal worker service is
    running yet. The reranker is warmed here (rather than lazily on first
    request) so the first real query doesn't pay the ~10s cross-encoder
    load time — see services.retrieval.reranker.warm_reranker.
    """
    snapshot = getattr(app.state, "runtime_settings", runtime_settings)
    app.state.runtime_settings = snapshot
    configure_model_inference(snapshot.application)
    os.makedirs(snapshot.application.upload_dir, exist_ok=True)
    security_settings = snapshot.security
    async with AsyncExitStack() as resources:
        qdrant_store = QdrantStore(snapshot.application)
        app.state.qdrant_store = qdrant_store
        resources.push_async_callback(run_in_threadpool, qdrant_store.close)
        app.state.retrieval_models = RetrievalModels(
            snapshot.application.embedding_model, snapshot.application.reranker_model
        )
        history_store = MongoHistoryStore(snapshot.history)
        app.state.history_store = history_store
        resources.push_async_callback(run_in_threadpool, history_store.close)
        app.state.websocket_security = WebSocketSecurity(
            security_settings, PostgresTicketStore(security_settings)
        )
        resources.push_async_callback(app.state.websocket_security.close)
        app.state.device_registry = DeviceRegistry()
        app.state.pairing_admission = PairingAdmission(security_settings)

        async def check_paired_device(device: PairedDevice) -> None:
            await run_in_threadpool(app.state.device_registry.require_active, device)

        pc_agent_bridge = PCAgentBridge(check_device=check_paired_device)
        app.state.pc_agent_bridge = pc_agent_bridge
        resources.push_async_callback(pc_agent_bridge.close)
        tool_registry = build_default_registry(pc_agent_bridge)
        app.state.tool_registry = tool_registry
        app.state.llm_gateway = LLMGateway(snapshot.application)
        classifier = build_classifier(
            snapshot.application.intent_classifier,
            completion=app.state.llm_gateway.complete_chat,
        )
        app.state.agent_graph = build_graph(classifier=classifier, registry=tool_registry)
        app.state.retrieval_work = DrainableTasks()
        app.state.chat_use_case = _build_chat_use_case(app)
        await _start_optional_services(app)
        try:
            yield
        finally:
            log.info("shutting_down")
            await app.state.websocket_security.close()
            await app.state.pc_agent_bridge.close()
            await app.state.retrieval_work.close()


async def _start_optional_services(app: FastAPI) -> None:
    settings = cast(RuntimeSettings, app.state.runtime_settings).application
    app.state.temporal_client = None
    app.state.temporal_connected = False

    try:
        async with asyncio.timeout(settings.temporal_connect_timeout_seconds):
            app.state.temporal_client = await TemporalClient.connect(settings.temporal_address)
        app.state.temporal_connected = True
        log.info("temporal_client_connected")
    except Exception as exc:  # noqa: BLE001
        app.state.temporal_client = None
        app.state.temporal_connected = False
        log.warning("temporal_client_unavailable", error_type=type(exc).__name__)

    try:
        await run_in_threadpool(warm_reranker, settings.reranker_model)
        log.info("reranker_warmed")
    except Exception as exc:  # noqa: BLE001
        # Not fatal at startup — Phase 3's chat endpoint will retry lazily
        # on first use, just with the latency hit this warmup exists to
        # avoid. Don't block the app from serving /health while a GPU/model
        # host is still coming up.
        log.warning("reranker_warmup_failed", error_type=type(exc).__name__)

    app.state.upload_use_case = _build_upload_use_case(app)


@api_router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@api_router.get("/health/dependencies")
async def health_dependencies(request: Request) -> JSONResponse:
    """Report whether the configured Temporal service is reachable."""
    checks = {"temporal": "unavailable"}
    state = request.app.state
    if state.temporal_client is None:
        return JSONResponse(checks)

    try:
        async with asyncio.timeout(
            state.runtime_settings.application.temporal_probe_timeout_seconds
        ):
            async for _ in state.temporal_client.list_workflows():
                break
        checks["temporal"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["temporal"] = "unavailable"
        log.warning("temporal_dependency_probe_failed", error_type=type(exc).__name__)
    return JSONResponse(checks)


@api_router.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_ingestion_tenant),
) -> dict:
    """Authenticate the upload and delegate storage/workflow orchestration."""
    try:
        use_case = getattr(request.app.state, "upload_use_case", None)
        if not isinstance(use_case, UploadUseCase):
            use_case = _build_upload_use_case(request.app)
        return await use_case.execute(file, tenant_id)
    except UploadFailure as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _build_upload_use_case(app: FastAPI) -> UploadUseCase:
    """Bind upload configuration and the lifespan-owned workflow client."""
    settings = cast(RuntimeSettings, app.state.runtime_settings).application
    storage = (
        S3UploadStorage(settings.ingestion_s3_bucket, settings.aws_region)
        if settings.ingestion_storage_backend == "s3"
        else LocalUploadStorage()
    )
    return UploadUseCase(
        upload_dir=settings.upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
        storage=storage,
        workflow=getattr(app.state, "temporal_client", None),
        make_id=lambda: str(uuid.uuid4()),
    )


def _build_chat_use_case(app: FastAPI) -> ChatUseCase:
    """Bind app-owned resources to the chat turn use case."""
    state = app.state
    retrieval_work = getattr(state, "retrieval_work", None)
    if not isinstance(retrieval_work, DrainableTasks):
        retrieval_work = DrainableTasks()

    async def route(tenant_id: str, query: str, principal: Principal) -> dict[str, Any]:
        return await route_message(
            tenant_id,
            query,
            principal=principal,
            graph=getattr(state, "agent_graph", None),
        )

    def stream_factory() -> Callable[..., AsyncGenerator[str, None]]:
        return _stream_completion_for(app)

    async def condense(
        history: list[dict],
        query: str,
        stream_completion: Callable[..., AsyncGenerator[str, None]],
    ) -> str:
        return await condense_query(history, query, stream_completion)

    async def retrieve(query: str, tenant_id: str) -> list[dict]:
        settings = state.runtime_settings.application
        return await retrieval_work.run(
            retrieve_async(
                query,
                tenant_id,
                top_k=settings.chat_retrieval_top_k,
                qdrant_store=getattr(state, "qdrant_store", None),
                retrieval_models=getattr(state, "retrieval_models", None),
            )
        )

    return ChatUseCase(
        ChatUseCaseDependencies(
            history=cast(HistoryStore, state.history_store),
            route_message=route,
            stream_completion=stream_factory,
            condense_query=condense,
            retrieve=retrieve,
        )
    )


async def _chat_session(websocket: WebSocket, session: SocketSession) -> None:
    """Compatibility seam for existing tests and integrations."""
    use_case = getattr(websocket.app.state, "chat_use_case", None)
    if not isinstance(use_case, ChatUseCase):
        use_case = _build_chat_use_case(websocket.app)
    await handle_chat_session(websocket, session, use_case)


async def _pc_agent_session(websocket: WebSocket, session: SocketSession) -> None:
    """Compatibility seam for existing tests and integrations."""
    await handle_pc_agent_session(websocket, session)


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    """Build an isolated FastAPI application and its lifecycle dependencies."""
    snapshot = settings or runtime_settings
    application = FastAPI(title=snapshot.application.project_name, lifespan=lifespan)
    application.state.runtime_settings = snapshot
    application.include_router(authentication_router)
    application.include_router(devices_router)
    application.include_router(api_router)
    application.include_router(chat_websocket_router)
    application.include_router(pc_agent_websocket_router)
    return application


app = create_app()
