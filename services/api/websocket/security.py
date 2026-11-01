"""Authenticated, bounded WebSocket sessions owned by the API lifespan."""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import psycopg2
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from services.auth.dependencies import authenticate_key
from services.auth.principal import AuthorizationDenied, Principal
from services.auth.settings import SecuritySettings
from services.auth.tickets import AUDIENCE_SCOPE, Audience, TicketStore


class ProtocolViolation(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code


class SocketSession:
    def __init__(
        self, websocket: WebSocket, principal: Principal, settings: SecuritySettings
    ) -> None:
        self.websocket, self.principal, self.settings = websocket, principal, settings

    async def receive_text(self) -> str:
        try:
            message = await asyncio.wait_for(
                self.websocket.receive(), timeout=self.settings.idle_timeout_seconds
            )
        except TimeoutError as exc:
            raise ProtocolViolation(1001, "Connection idle timeout") from exc
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        text = message.get("text")
        if not isinstance(text, str):
            raise ProtocolViolation(1003, "Text messages are required")
        if len(text.encode("utf-8")) > self.settings.message_max_bytes:
            raise ProtocolViolation(1009, "Message exceeds size limit")
        return text

    async def receive_json(self) -> dict[str, Any]:
        try:
            value = json.loads(await self.receive_text())
        except json.JSONDecodeError as exc:
            raise ProtocolViolation(1003, "Invalid JSON message") from exc
        if not isinstance(value, dict):
            raise ProtocolViolation(1003, "JSON object required")
        return value

    async def send_json(self, value: object) -> None:
        try:
            await asyncio.wait_for(
                self.websocket.send_json(value), self.settings.send_timeout_seconds
            )
        except TimeoutError as exc:
            raise ProtocolViolation(1013, "Client send timeout") from exc


class WebSocketSecurity:
    def __init__(self, settings: SecuritySettings, tickets: TicketStore) -> None:
        self.settings, self.tickets = settings, tickets
        self._active: dict[WebSocket, str] = {}
        self._closing = False
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._auth_inflight = 0

    def _allow_attempt(self, host: str) -> bool:
        now = time.monotonic()
        while self._attempts and next(iter(self._attempts.values()))[-1] <= now - 60:
            self._attempts.popitem(last=False)
        attempts = self._attempts.get(host)
        if attempts is None:
            if len(self._attempts) >= self.settings.max_auth_clients:
                return False
            attempts = self._attempts[host] = deque()
        while attempts and attempts[0] <= now - 60:
            attempts.popleft()
        if len(attempts) >= self.settings.auth_attempts_per_minute:
            return False
        attempts.append(now)
        self._attempts.move_to_end(host)
        return True

    async def authenticate(self, websocket: WebSocket, audience: Audience) -> Principal | None:
        host = websocket.client.host if websocket.client is not None else "unknown"
        if not self._allow_attempt(host) or self._auth_inflight >= self.settings.max_auth_inflight:
            raise ProtocolViolation(1013, "Authentication rate limit reached")
        self._auth_inflight += 1
        try:
            return await self._authenticate(websocket, audience)
        finally:
            self._auth_inflight -= 1

    async def _authenticate(self, websocket: WebSocket, audience: Audience) -> Principal | None:
        origin = websocket.headers.get("origin")
        if self.settings.require_tls and websocket.url.scheme != "wss":
            raise ProtocolViolation(1008, "TLS is required")
        if origin is not None and origin not in self.settings.allowed_origins:
            raise ProtocolViolation(1008, "Origin is not allowed")
        ticket = websocket.query_params.get("ticket")
        header_key = websocket.headers.get("x-api-key")
        legacy_key = websocket.query_params.get("api_key")
        if sum(value is not None for value in (ticket, header_key, legacy_key)) > 1:
            raise ProtocolViolation(1008, "Use exactly one authentication method")
        if legacy_key is not None and not self.settings.allow_query_api_keys:
            raise ProtocolViolation(1008, "Query-string API keys are disabled")
        try:
            if ticket is not None:
                principal = await run_in_threadpool(self.tickets.consume, ticket, audience)
            else:
                principal = await authenticate_key(header_key or legacy_key)
            if principal is not None:
                principal.require(AUDIENCE_SCOPE[audience])
            return principal
        except (AuthorizationDenied, ValueError) as exc:
            raise ProtocolViolation(1008, "Credential is not permitted for this socket") from exc
        except (HTTPException, psycopg2.Error) as exc:
            raise ProtocolViolation(1013, "Authentication is temporarily unavailable") from exc

    @asynccontextmanager
    async def session(
        self, websocket: WebSocket, audience: Audience
    ) -> AsyncIterator[SocketSession | None]:
        yielded = False
        lifetime = asyncio.timeout(self.settings.session_max_seconds)
        try:
            principal = await self.authenticate(websocket, audience)
            if principal is None:
                raise ProtocolViolation(1008, "Missing or invalid credential")
            tenant_count = sum(tenant == principal.tenant_id for tenant in self._active.values())
            if (
                self._closing
                or len(self._active) >= self.settings.max_connections
                or tenant_count >= self.settings.max_connections_per_tenant
            ):
                raise ProtocolViolation(1013, "Connection limit reached")
            self._active[websocket] = principal.tenant_id
            await websocket.accept()
            async with lifetime:
                yielded = True
                yield SocketSession(websocket, principal, self.settings)
        except ProtocolViolation as exc:
            await self._close(websocket, exc.code, str(exc))
            if not yielded:
                yield None
        except TimeoutError:
            if not lifetime.expired():
                raise
            await self._close(websocket, 1008, "Session expired; reconnect to authenticate")
        except WebSocketDisconnect:
            if not yielded:
                yield None
        finally:
            self._active.pop(websocket, None)
            if yielded:
                await self._close(websocket, 1000, "Session ended")

    async def _close(self, websocket: WebSocket, code: int, reason: str) -> None:
        try:
            await asyncio.wait_for(
                websocket.close(code=code, reason=reason), self.settings.send_timeout_seconds
            )
        except (RuntimeError, OSError, TimeoutError, WebSocketDisconnect):
            pass

    async def close(self) -> None:
        self._closing = True
        await asyncio.gather(
            *(self._close(ws, 1001, "Server shutting down") for ws in list(self._active))
        )


def get_websocket_security(websocket: WebSocket) -> WebSocketSecurity:
    return websocket.app.state.websocket_security
