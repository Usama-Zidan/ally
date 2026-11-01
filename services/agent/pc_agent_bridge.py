"""
Bridges the backend to each tenant's local Windows companion agent.

The connection direction is inverted from a normal tool integration: the
backend can't call into a device sitting behind a home/office NAT, so the
local agent instead connects *out* to the backend over WebSocket
(/ws/pc-agent) and stays connected. Sending it a command means writing a
JSON message down that existing connection and waiting for a correlated
response, not making an outbound request the way JiraTool/DriveTool/
CalendarTool do.

Known limitation: the connection registry below is an in-process dict.
It works for a single backend instance, which matches where this project
actually is (no horizontal scaling has been introduced yet -- Phase 5's
Redis is for caching, not this). Running more than one backend instance
would need this registry moved to something shared (Redis pub/sub
keyed by tenant_id) so a command issued against an instance the agent
isn't connected to can still be routed to the right one.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import WebSocket

from config import PC_AGENT_COMMAND_TIMEOUT_SECONDS

log = structlog.get_logger()


class PCAgentError(Exception):
    """Raised when a command can't be delivered to or executed by a local
    PC agent (not connected, the agent rejected the capability, the agent
    reported a failure, or the command timed out)."""


@dataclass
class _Connection:
    websocket: WebSocket
    capabilities: dict[str, bool]
    pending: dict[str, asyncio.Future] = field(default_factory=dict)


class PCAgentBridge:
    """One instance is shared process-wide (see the module-level
    singleton below). Holds at most one active connection per tenant --
    a second local agent connecting for the same tenant replaces the
    first, since v1 doesn't support routing to a specific one of several
    paired devices."""

    def __init__(self) -> None:
        self._connections: dict[str, _Connection] = {}

    async def register(
        self, tenant_id: str, websocket: WebSocket, capabilities: dict[str, bool]
    ) -> None:
        existing = self._connections.get(tenant_id)
        if existing is not None:
            log.info("pc_agent_replaced", tenant_id=tenant_id)
            for future in existing.pending.values():
                if not future.done():
                    future.set_exception(PCAgentError("Replaced by a new PC agent connection"))
        self._connections[tenant_id] = _Connection(websocket=websocket, capabilities=capabilities)
        log.info("pc_agent_connected", tenant_id=tenant_id, capabilities=capabilities)

    def unregister(self, tenant_id: str) -> None:
        connection = self._connections.pop(tenant_id, None)
        if connection is None:
            return
        for future in connection.pending.values():
            if not future.done():
                future.set_exception(PCAgentError("PC agent disconnected"))
        log.info("pc_agent_disconnected", tenant_id=tenant_id)

    def is_connected(self, tenant_id: str) -> bool:
        return tenant_id in self._connections

    def get_capabilities(self, tenant_id: str) -> dict[str, bool] | None:
        connection = self._connections.get(tenant_id)
        return connection.capabilities if connection else None

    async def handle_result(self, tenant_id: str, message: dict[str, Any]) -> None:
        """Called by the /ws/pc-agent receive loop for every message
        arriving from the local agent. Resolves the future send_command
        is awaiting, matched by request_id."""
        connection = self._connections.get(tenant_id)
        if connection is None:
            return
        request_id = message.get("request_id")
        if not isinstance(request_id, str):
            return
        future = connection.pending.pop(request_id, None)
        if future is None or future.done():
            return
        future.set_result(message)

    async def send_command(
        self,
        tenant_id: str,
        capability: str,
        action: str,
        params: dict[str, Any],
        timeout: float = PC_AGENT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Sends ``action``/``params`` to the tenant's connected PC agent
        and waits for its result.

        Raises:
            PCAgentError: If no agent is connected, the agent hasn't
                enabled ``capability`` locally, the agent reports a
                failure, or no response arrives within ``timeout``
                seconds.
        """
        connection = self._connections.get(tenant_id)
        if connection is None:
            raise PCAgentError(
                "No PC agent is currently connected. Start the local agent on your "
                "machine and try again."
            )
        if not connection.capabilities.get(capability, False):
            raise PCAgentError(
                f"The connected PC agent has '{capability}' disabled locally. Enable it "
                "in the agent's config to allow this action."
            )

        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        connection.pending[request_id] = future

        try:
            await connection.websocket.send_json(
                {
                    "type": "command",
                    "request_id": request_id,
                    "capability": capability,
                    "action": action,
                    "params": params,
                }
            )
        except Exception as exc:  # noqa: BLE001
            connection.pending.pop(request_id, None)
            raise PCAgentError(f"Failed to send command to PC agent: {exc}") from exc

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            connection.pending.pop(request_id, None)
            raise PCAgentError(
                f"The PC agent did not respond within {timeout:.0f}s."
            ) from exc

        if not result.get("success", False):
            raise PCAgentError(result.get("error", "The PC agent reported an unspecified failure."))

        return result.get("details", {})


# Shared process-wide registry -- see the class docstring for the
# single-instance limitation.
bridge = PCAgentBridge()
