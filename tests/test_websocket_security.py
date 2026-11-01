"""Transport tests with fake authentication; live ticket atomicity is tested separately."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import WebSocket
from pydantic import ValidationError

from services.api.websocket.security import WebSocketSecurity
from services.auth.principal import DEFAULT_SCOPES, Principal
from services.auth.settings import SecuritySettings, load_security_settings

CHAT = Principal("tenant-a", "credential-a", DEFAULT_SCOPES)


def socket(*messages, headers=None, query=b"", scheme="ws", host="client-a"):
    events = [{"type": "websocket.connect"}, *messages]
    sent = []

    async def receive():
        if events:
            return events.pop(0)
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    ws = WebSocket(
        {
            "type": "websocket",
            "scheme": scheme,
            "path": "/ws/chat",
            "query_string": query,
            "headers": headers or [],
            "client": (host, 1234),
            "server": ("server", 8000),
        },
        receive,
        send,
    )
    return ws, sent


class SecuritySettingsTest(unittest.TestCase):
    def test_production_defaults_require_tls_and_disable_query_keys(self):
        settings = load_security_settings({"ALLY_ENV": "production"})
        self.assertTrue(settings.require_tls)
        self.assertFalse(settings.allow_query_api_keys)

    def test_unsafe_production_or_invalid_limits_fail_fast(self):
        for fields in (
            {"ALLY_ENV": "production", "WS_REQUIRE_TLS": "false"},
            {"ALLY_ENV": "production", "WS_ALLOW_QUERY_API_KEYS": "true"},
            {"WS_MAX_CONNECTIONS": "0"},
            {"WS_IDLE_TIMEOUT_SECONDS": "nan"},
            {"WS_ALLOWED_ORIGINS": "*"},
            {"WS_ALLOWED_ORIGINS": "https://user:pass@example.com"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                load_security_settings(fields)


class SocketSecurityTest(unittest.IsolatedAsyncioTestCase):
    async def test_backend_timeout_is_not_misreported_as_session_expiry(self):
        security = self.security()
        ws, sent = socket(query=b"ticket=x")
        with self.assertRaises(TimeoutError):
            async with security.session(ws, "chat"):
                raise TimeoutError("upstream timeout")
        self.assertFalse(security._active)
        self.assertNotIn("Session expired", str(sent))

    async def test_malformed_json_and_non_object_messages_are_rejected(self):
        for text in ("not-json", "[]"):
            security = self.security()
            ws, sent = socket({"type": "websocket.receive", "text": text}, query=b"ticket=x")
            async with security.session(ws, "chat") as session:
                await session.receive_json()
            self.assertEqual(sent[-1]["code"], 1003)

    async def test_slow_client_send_has_a_deadline_and_releases_capacity(self):
        security = self.security(send_timeout_seconds=0.01)
        ws, sent = socket(query=b"ticket=x")

        async def blocked(_):
            await asyncio.Event().wait()

        with patch.object(ws, "send_json", new=AsyncMock(side_effect=blocked)):
            async with security.session(ws, "chat") as session:
                await session.send_json({"type": "token", "content": "text"})
        self.assertEqual(sent[-1]["code"], 1013)
        self.assertFalse(security._active)

    def security(self, **settings):
        tickets = Mock(consume=Mock(return_value=CHAT))
        return WebSocketSecurity(SecuritySettings(**settings), tickets)

    async def test_native_header_authentication_never_needs_a_query_key(self):
        security = self.security()
        ws, sent = socket(headers=[(b"x-api-key", b"header-credential")])
        with patch(
            "services.api.websocket.security.authenticate_key", new=AsyncMock(return_value=CHAT)
        ) as auth:
            async with security.session(ws, "chat") as session:
                self.assertEqual(session.principal, CHAT)
        auth.assert_awaited_once_with("header-credential")
        self.assertEqual(sent[0]["type"], "websocket.accept")
        self.assertFalse(security._active)

    async def test_disallowed_origin_cannot_consume_a_ticket(self):
        security = self.security(allowed_origins=("https://app.example",))
        ws, sent = socket(query=b"ticket=ephemeral", headers=[(b"origin", b"https://evil.example")])
        async with security.session(ws, "chat") as session:
            self.assertIsNone(session)
        security.tickets.consume.assert_not_called()
        self.assertEqual(sent[-1]["code"], 1008)

    async def test_browser_ticket_passes_exact_audience_to_store(self):
        security = self.security(allowed_origins=("https://app.example",))
        ws, sent = socket(query=b"ticket=ephemeral", headers=[(b"origin", b"https://app.example")])
        async with security.session(ws, "chat") as session:
            self.assertEqual(session.principal.tenant_id, CHAT.tenant_id)
        security.tickets.consume.assert_called_once_with("ephemeral", "chat")

    async def test_production_rejects_plaintext_or_query_api_keys(self):
        security = self.security(
            environment="production", require_tls=True, allow_query_api_keys=False
        )
        for scheme, query in (("ws", b""), ("wss", b"api_key=private")):
            ws, sent = socket(scheme=scheme, query=query)
            async with security.session(ws, "chat") as session:
                self.assertIsNone(session)
            self.assertEqual(sent[-1]["code"], 1008)

    async def test_ambiguous_credentials_are_rejected(self):
        security = self.security()
        ws, sent = socket(query=b"ticket=x", headers=[(b"x-api-key", b"private")])
        async with security.session(ws, "chat") as session:
            self.assertIsNone(session)
        security.tickets.consume.assert_not_called()

    async def test_chat_key_cannot_register_a_pc_agent(self):
        security = self.security()
        ws, sent = socket(query=b"ticket=x")
        async with security.session(ws, "pc-agent") as session:
            self.assertIsNone(session)
        self.assertEqual(sent[-1]["code"], 1008)

    async def test_limit_releases_capacity_on_disconnect_or_cancellation(self):
        security = self.security(max_connections=1)
        first, _ = socket(query=b"ticket=one")
        second, sent = socket(query=b"ticket=two")
        async with security.session(first, "chat"):
            async with security.session(second, "chat") as session:
                self.assertIsNone(session)
        self.assertEqual(sent[-1]["code"], 1013)
        third, _ = socket(query=b"ticket=three")
        with self.assertRaises(asyncio.CancelledError):
            async with security.session(third, "chat") as session:
                self.assertIsNotNone(session)
                raise asyncio.CancelledError
        self.assertFalse(security._active)

    async def test_unicode_byte_limit_and_binary_messages_are_rejected(self):
        for message, code in (
            ({"type": "websocket.receive", "text": "é" * 600}, 1009),
            ({"type": "websocket.receive", "bytes": b"binary"}, 1003),
        ):
            security = self.security(message_max_bytes=1024)
            ws, sent = socket(message, query=b"ticket=x")
            async with security.session(ws, "chat") as session:
                await session.receive_text()
            self.assertEqual(sent[-1]["code"], code)
            self.assertFalse(security._active)

    async def test_idle_and_total_lifetime_are_bounded(self):
        for settings, receive, code in (
            ({"idle_timeout_seconds": 0.01}, True, 1001),
            ({"session_max_seconds": 0.01}, False, 1008),
        ):
            security = self.security(**settings)
            ws, sent = socket(query=b"ticket=x")
            async with security.session(ws, "chat") as session:
                if receive:
                    await session.receive_text()
                else:
                    await asyncio.Event().wait()
            self.assertEqual(sent[-1]["code"], code)
            self.assertFalse(security._active)

    async def test_auth_attempts_and_client_tracking_are_bounded(self):
        security = self.security(auth_attempts_per_minute=1, max_auth_clients=1)
        first, _ = socket(query=b"ticket=one")
        async with security.session(first, "chat"):
            pass
        for host in ("client-a", "client-b"):
            ws, sent = socket(query=b"ticket=two", host=host)
            async with security.session(ws, "chat") as session:
                self.assertIsNone(session)
            self.assertEqual(sent[-1]["code"], 1013)
        self.assertEqual(len(security._attempts), 1)


if __name__ == "__main__":
    unittest.main()
