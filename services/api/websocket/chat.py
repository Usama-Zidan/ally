"""Authenticated chat WebSocket transport."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.api.websocket.security import SocketSession, get_websocket_security
from services.chat.ports import HistoryInputError, validate_identity
from services.chat.use_cases import ChatUseCase

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    security = get_websocket_security(websocket)
    async with security.session(websocket, "chat") as session:
        if session is not None:
            await handle_chat_session(websocket, session)


async def handle_chat_session(
    websocket: WebSocket,
    session: SocketSession,
    use_case: ChatUseCase | None = None,
) -> None:
    """Validate the authenticated identity and run chat turns."""
    principal = session.principal
    tenant_id = principal.tenant_id
    supplied_conversation_id = websocket.query_params.get("conversation_id")
    conversation_id = supplied_conversation_id or str(uuid.uuid4())

    if supplied_conversation_id and "chat:history" not in principal.scopes:
        await websocket.close(code=1008, reason="Conversation history access is not permitted")
        return
    try:
        validate_identity(tenant_id, conversation_id)
    except HistoryInputError:
        await websocket.close(code=1008, reason="Invalid conversation identity")
        return

    await session.send_json({"type": "ready", "conversation_id": conversation_id})
    active_use_case: Any = use_case or getattr(websocket.app.state, "chat_use_case", None)
    if active_use_case is None:
        raise RuntimeError("Chat use case has not been initialized")
    try:
        await active_use_case.run(session, conversation_id)
    except WebSocketDisconnect:
        log.info(
            "chat_ws_disconnected",
            extra={"tenant_id": tenant_id, "conversation_id": conversation_id},
        )
