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
import math
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi import WebSocket

from config import PC_AGENT_COMMAND_TIMEOUT_SECONDS
from services.auth.devices import DeviceRegistry, PairedDevice

log = structlog.get_logger()


class PCAgentError(Exception):
    """Raised when a command can't be delivered to or executed by a local
    PC agent (not connected, the agent rejected the capability, the agent
    reported a failure, or the command timed out)."""


@dataclass
class _Connection:
    websocket: WebSocket
    capabilities: dict[str, bool]
    device: PairedDevice | None = None
    pending: dict[str, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PCAgentBridge:
    """One instance is shared process-wide (see the module-level
    singleton below). Holds at most one active connection per tenant --
    a second local agent connecting for the same tenant replaces the
    first, since v1 doesn't support routing to a specific one of several
    paired devices."""

    def __init__(
        self,
        max_pending_commands: int = 16,
        *,
        check_device: Callable[[PairedDevice], Awaitable[None]] | None = None,
    ) -> None:
        if max_pending_commands <= 0:
            raise ValueError("max_pending_commands must be positive")
        self._connections: dict[str, _Connection] = {}
        self._max_pending_commands = max_pending_commands
        self._check_device = check_device
        self._closing = False

    async def register(
        self,
        tenant_id: str,
        websocket: WebSocket,
        capabilities: dict[str, bool],
        *,
        device: PairedDevice | None = None,
    ) -> None:
        if self._closing:
            raise PCAgentError("PC agent bridge is shutting down")
        if self._check_device is not None:
            if device is None or device.tenant_id != tenant_id:
                raise ValueError("A paired device binding is required")
            try:
                await self._check_device(device)
            except Exception as exc:
                raise PCAgentError("Paired device verification failed") from exc
        if self._closing:
            raise PCAgentError("PC agent bridge is shutting down")
        if not isinstance(capabilities, dict) or any(
            key not in {"open_apps", "file_rw", "shell_exec", "mouse_keyboard_control"}
            or type(value) is not bool
            for key, value in capabilities.items()
        ):
            raise ValueError("Invalid PC agent capabilities")
        existing = self._connections.get(tenant_id)
        if existing is not None:
            log.info("pc_agent_replaced", tenant_id=tenant_id)
            for future in existing.pending.values():
                if not future.done():
                    future.set_exception(PCAgentError("Replaced by a new PC agent connection"))
        self._connections[tenant_id] = _Connection(
            websocket=websocket, capabilities=dict(capabilities), device=device
        )
        log.info("pc_agent_connected", tenant_id=tenant_id, capabilities=capabilities)
        if existing is not None:
            try:
                await asyncio.wait_for(
                    existing.websocket.close(code=1000, reason="Connection replaced"),
                    timeout=1,
                )
            except Exception as exc:
                log.warning("pc_agent_old_socket_close_failed", error_type=type(exc).__name__)

    def unregister(self, tenant_id: str, websocket: WebSocket) -> None:
        """Remove only the session belonging to this exact source socket."""
        connection = self._connections.get(tenant_id)
        if connection is None or connection.websocket is not websocket:
            return
        del self._connections[tenant_id]
        for future in connection.pending.values():
            if not future.done():
                future.set_exception(PCAgentError("PC agent disconnected"))
        log.info("pc_agent_disconnected", tenant_id=tenant_id)

    def is_connected(self, tenant_id: str) -> bool:
        return tenant_id in self._connections

    async def close(self) -> None:
        """Reject new work, fail pending commands and close every owned socket."""
        if self._closing:
            return
        self._closing = True
        connections = list(self._connections.values())
        self._connections.clear()
        for connection in connections:
            for future in connection.pending.values():
                if not future.done():
                    future.set_exception(PCAgentError("PC agent bridge is shutting down"))

        async def close_socket(connection: _Connection) -> None:
            try:
                await asyncio.wait_for(
                    connection.websocket.close(code=1012, reason="Service shutting down"),
                    timeout=1,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("pc_agent_shutdown_close_failed", error_type=type(exc).__name__)

        await asyncio.gather(*(close_socket(item) for item in connections))

    def get_capabilities(self, tenant_id: str) -> dict[str, bool] | None:
        connection = self._connections.get(tenant_id)
        return dict(connection.capabilities) if connection else None

    async def handle_result(
        self, tenant_id: str, message: dict[str, Any], websocket: WebSocket
    ) -> None:
        """Called by the /ws/pc-agent receive loop for every message
        arriving from the local agent. Resolves the future send_command
        is awaiting, matched by request_id."""
        connection = self._connections.get(tenant_id)
        if connection is None or connection.websocket is not websocket:
            return
        request_id = message.get("request_id")
        if not isinstance(request_id, str):
            return
        future = connection.pending.get(request_id)
        if future is None or future.done():
            return
        if self._check_device is not None and connection.device is not None:
            try:
                await self._check_device(connection.device)
            except Exception:
                if not future.done():
                    future.set_exception(PCAgentError("Paired device verification failed"))
                return
            # Verification yielded: replacement/cancellation may have happened.
            if future.done():
                return
            if self._connections.get(tenant_id) is not connection:
                future.set_exception(PCAgentError("PC agent connection changed before result"))
                return
        connection.pending.pop(request_id, None)
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
        if self._closing:
            raise PCAgentError("PC agent bridge is shutting down")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        connection = self._connections.get(tenant_id)
        if connection is None:
            raise PCAgentError(
                "No PC agent is currently connected. Start the local agent on your "
                "machine and try again."
            )
        if connection.capabilities.get(capability) is not True:
            raise PCAgentError(
                f"The connected PC agent has '{capability}' disabled locally. Enable it "
                "in the agent's config to allow this action."
            )

        if len(connection.pending) >= self._max_pending_commands:
            raise PCAgentError("The PC agent has too many pending commands. Try again later.")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = future
        expires_at = time.time() + timeout

        try:
            # The deadline covers queueing, socket send and response together.
            async with asyncio.timeout(timeout):
                async with connection.send_lock:
                    if self._check_device is not None and connection.device is not None:
                        await self._check_device(connection.device)
                    if self._connections.get(tenant_id) is not connection:
                        raise PCAgentError("PC agent connection changed before command delivery")
                    await connection.websocket.send_json(
                        {
                            "type": "command",
                            "request_id": request_id,
                            "capability": capability,
                            "action": action,
                            "params": params,
                            "expires_at": expires_at,
                        }
                    )
                result = await future
        except TimeoutError as exc:
            raise PCAgentError(f"The PC agent did not respond within {timeout:.0f}s.") from exc
        except PCAgentError:
            raise
        except Exception as exc:
            raise PCAgentError("Failed to communicate with the PC agent.") from exc
        finally:
            connection.pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # Retrieve a replacement/disconnect exception even when send failed.
                future.exception()

        if result.get("success") is not True:
            raise PCAgentError(
                "The PC agent declined or failed, or was interrupted. Effects may be partial; inspect the device before retrying."
            )

        details = result.get("details", {})
        if not isinstance(details, dict):
            raise PCAgentError("The PC agent returned an invalid result.")
        return details


# Shared process-wide registry -- see the class docstring for the
# single-instance limitation.
async def _check_paired_device(device: PairedDevice) -> None:
    await asyncio.to_thread(DeviceRegistry().require_active, device)


bridge = PCAgentBridge(check_device=_check_paired_device)
