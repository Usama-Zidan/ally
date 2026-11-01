"""Authenticated PC companion WebSocket transport."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from services.agent.pc_agent_bridge import PCAgentError
from services.api.websocket.security import SocketSession, get_websocket_security
from services.auth.principal import AuthorizationDenied

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/pc-agent")
async def pc_agent_ws(websocket: WebSocket) -> None:
    security = get_websocket_security(websocket)
    async with security.session(websocket, "pc-agent") as session:
        if session is not None:
            await handle_pc_agent_session(websocket, session)


async def handle_pc_agent_session(websocket: WebSocket, session: SocketSession) -> None:
    """Authenticate a paired device and relay its command results."""
    tenant_id = session.principal.tenant_id

    try:
        handshake = await asyncio.wait_for(session.receive_json(), timeout=10)
    except TimeoutError:
        await websocket.close(code=1003, reason="Expected a handshake message")
        return

    if not isinstance(handshake, dict) or handshake.get("type") != "handshake":
        await websocket.close(code=1003, reason="First message must be a handshake")
        return

    capabilities = handshake.get("capabilities", {})
    try:
        device = await run_in_threadpool(
            websocket.app.state.device_registry.bind,
            session.principal,
            handshake.get("device_id"),
        )
    except AuthorizationDenied:
        await websocket.close(code=1008, reason="A paired device credential is required")
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("pc_device_binding_unavailable", extra={"error_type": type(exc).__name__})
        await websocket.close(code=1013, reason="Device verification is temporarily unavailable")
        return

    bridge: Any = websocket.app.state.pc_agent_bridge
    try:
        try:
            await bridge.register(tenant_id, websocket, capabilities, device=device)
        except ValueError:
            await websocket.close(code=1003, reason="Invalid capabilities")
            return
        except PCAgentError:
            await websocket.close(code=1013, reason="Paired device verification failed")
            return
        await session.send_json({"type": "ready", "device_id": device.device_id})
        while True:
            message = await session.receive_json()
            if isinstance(message, dict) and message.get("type") == "result":
                await bridge.handle_result(tenant_id, message, websocket)
            else:
                log.warning("pc_agent_unknown_message_type", extra={"tenant_id": tenant_id})
    except WebSocketDisconnect:
        log.info("pc_agent_ws_disconnected", extra={"tenant_id": tenant_id})
    finally:
        bridge.unregister(tenant_id, websocket)
