"""Real FastAPI boundaries with injected authentication and ticket storage."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
import psycopg2

from services.api import main
from services.api.routes.authentication import router
from services.api.websocket.security import WebSocketSecurity
from services.auth.dependencies import get_current_principal
from services.auth.principal import DEFAULT_SCOPES, Principal
from services.auth.settings import SecuritySettings
from services.auth.tickets import TicketCapacityExceeded

CHAT = Principal("tenant-a", "credential-a", DEFAULT_SCOPES)


class TicketHTTPTest(unittest.TestCase):
    def client(self, principal=CHAT, settings=None):
        app = FastAPI()
        app.include_router(router)
        tickets = Mock(issue=Mock(return_value="ephemeral-ticket"))
        app.state.websocket_security = WebSocketSecurity(settings or SecuritySettings(), tickets)
        app.dependency_overrides[get_current_principal] = lambda: principal
        return TestClient(app), tickets

    def test_ticket_contract_and_cache_prevention(self):
        client, tickets = self.client()
        with client:
            response = client.post("/auth/ws-tickets", json={"audience": "chat"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"ticket": "ephemeral-ticket", "expires_in": 30, "audience": "chat"}
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        tickets.issue.assert_called_once_with(CHAT, "chat")

    def test_scope_and_request_validation_happen_before_storage(self):
        client, tickets = self.client()
        with client:
            self.assertEqual(
                client.post("/auth/ws-tickets", json={"audience": "pc-agent"}).status_code, 403
            )
            self.assertEqual(
                client.post(
                    "/auth/ws-tickets", json={"audience": "chat", "tenant_id": "victim"}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post("/auth/ws-tickets", json={"audience": "admin"}).status_code, 422
            )
        tickets.issue.assert_not_called()

    def test_plaintext_production_request_is_rejected(self):
        client, tickets = self.client(
            settings=SecuritySettings(
                environment="production", require_tls=True, allow_query_api_keys=False
            )
        )
        with client:
            response = client.post("/auth/ws-tickets", json={"audience": "chat"})
        self.assertEqual(response.status_code, 400)
        tickets.issue.assert_not_called()

    def test_quota_and_outage_have_distinct_sanitized_status_codes(self):
        for failure, status in (
            (TicketCapacityExceeded("secret-sentinel"), 429),
            (psycopg2.OperationalError("secret-sentinel"), 503),
        ):
            client, tickets = self.client()
            tickets.issue.side_effect = failure
            with client:
                response = client.post("/auth/ws-tickets", json={"audience": "chat"})
            self.assertEqual(response.status_code, status)
            self.assertNotIn("secret-sentinel", response.text)


class ChatScopeTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_use_does_not_grant_document_or_history_read(self):
        session = Mock(
            principal=Principal("a", "key-a", frozenset({"chat:use"})),
            receive_text=AsyncMock(side_effect=["question", WebSocketDisconnect()]),
            send_json=AsyncMock(),
        )
        ws = Mock(query_params={})
        with (
            patch.object(main, "route_message", new=AsyncMock(return_value={"route": "rag"})),
            patch.object(ws.app.state.history_store, "get_history") as history,
            patch.object(main, "retrieve_async", new=AsyncMock()) as retrieve,
        ):
            await main._chat_session(ws, session)
        history.assert_not_called()
        retrieve.assert_not_awaited()
        self.assertTrue(
            any(
                call.args[0].get("detail") == "Document retrieval is not permitted."
                for call in session.send_json.call_args_list
            )
        )

    async def test_history_resume_requires_history_scope(self):
        session = Mock(principal=Principal("a", "key-a", frozenset({"chat:use"})))
        ws = Mock(query_params={"conversation_id": "other-conversation"}, close=AsyncMock())
        await main._chat_session(ws, session)
        ws.close.assert_awaited_once_with(
            code=1008, reason="Conversation history access is not permitted"
        )


if __name__ == "__main__":
    unittest.main()
