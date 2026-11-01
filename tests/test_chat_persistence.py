"""Completion reflects acknowledged exchange storage without hiding tool effects."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from services.api import main
from services.agent.classifier import Intent
from services.agent.tools.base import ToolResult
from services.auth.principal import Principal, LEGACY_SCOPES
from services.chat.ports import HistoryUnavailable


def conversation():
    store = Mock(get_history=Mock(return_value=[]), append_exchange=Mock())
    ws = Mock(query_params={"conversation_id": "conversation"})
    ws.app.state.history_store = store
    ws.app.state.qdrant_store = Mock()
    ws.app.state.runtime_settings = main.runtime_settings
    session = Mock(
        principal=Principal("tenant-a", "key-a", LEGACY_SCOPES),
        receive_text=AsyncMock(side_effect=["question", WebSocketDisconnect()]),
        send_json=AsyncMock(),
    )
    return store, ws, session


class ChatPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_done_follows_one_acknowledged_pair_append(self):
        store, ws, session = conversation()
        order = []
        session.send_json.side_effect = lambda event: order.append(event["type"])
        store.append_exchange.side_effect = lambda *args: order.append("persisted")
        with (
            patch.object(main, "route_message", new=AsyncMock(return_value={"route": "rag"})),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=[])),
            patch.object(main, "condense_query", new=AsyncMock(return_value="question")),
        ):
            await main._chat_session(ws, session)
        self.assertEqual(order, ["ready", "sources", "token", "persisted", "done"])
        store.append_exchange.assert_called_once_with(
            "tenant-a", "conversation", "question", main.NO_CONTEXT_MESSAGE
        )

    async def test_ambiguous_save_emits_error_without_done_or_raw_exception(self):
        for route in ("rag", "conversational"):
            store, ws, session = conversation()
            store.append_exchange.side_effect = HistoryUnavailable("private-password-and-message")

            async def stream(*args):
                yield "answer"

            with (
                self.subTest(route=route),
                patch.object(main, "route_message", new=AsyncMock(return_value={"route": route})),
                patch.object(main, "retrieve_async", new=AsyncMock(return_value=[])),
                patch.object(main, "condense_query", new=AsyncMock(return_value="question")),
                patch.object(main, "stream_chat_completion", side_effect=stream),
                patch.object(main, "log") as log,
            ):
                await main._chat_session(ws, session)
            events = [call.args[0] for call in session.send_json.call_args_list]
            self.assertNotIn("done", [event["type"] for event in events])
            self.assertEqual(events[-1]["code"], "history_write_unconfirmed")
            self.assertIn("may already be stored", events[-1]["detail"])
            self.assertNotIn("private", str(events))
            self.assertNotIn("private", str(log.mock_calls))
            store.append_exchange.assert_called_once()

    async def test_history_outage_stops_before_classifier_or_side_effect(self):
        store, ws, session = conversation()
        store.get_history.side_effect = HistoryUnavailable("private-prompt")
        with patch.object(main, "route_message", new=AsyncMock()) as route:
            await main._chat_session(ws, session)
        route.assert_not_awaited()
        store.append_exchange.assert_not_called()
        self.assertEqual(
            [call.args[0]["type"] for call in session.send_json.call_args_list], ["ready", "error"]
        )

    async def test_failed_generation_is_not_recorded_as_a_completed_pair(self):
        store, ws, session = conversation()

        async def stream(*args):
            yield "partial answer"
            raise main.LLMGatewayError("Generation interrupted")

        with (
            patch.object(
                main, "route_message", new=AsyncMock(return_value={"route": "conversational"})
            ),
            patch.object(main, "stream_chat_completion", side_effect=stream),
        ):
            await main._chat_session(ws, session)
        store.append_exchange.assert_not_called()
        self.assertEqual(session.send_json.call_args.args[0]["type"], "error")

    async def test_websocket_disconnect_closes_the_active_llm_stream(self):
        store, ws, session = conversation()
        closed = []

        def send_event(event):
            if event.get("type") == "token":
                raise WebSocketDisconnect()

        session.send_json.side_effect = send_event

        async def stream(*args):
            try:
                yield "partial"
                yield "unreachable"
            finally:
                closed.append(True)

        with (
            patch.object(
                main, "route_message", new=AsyncMock(return_value={"route": "conversational"})
            ),
            patch.object(main, "stream_chat_completion", side_effect=stream),
        ):
            await main._chat_session(ws, session)

        self.assertEqual(closed, [True])
        store.append_exchange.assert_not_called()

    async def test_tool_effect_result_is_preserved_when_history_is_unconfirmed(self):
        store, ws, session = conversation()
        store.append_exchange.side_effect = HistoryUnavailable()
        routed = {
            "route": "tool",
            "intent": Intent(kind="tool", tool_name="local_pc", action="write_file"),
            "tool_result": ToolResult(success=True, summary="Wrote the file"),
        }
        with patch.object(main, "route_message", new=AsyncMock(return_value=routed)) as route:
            await main._chat_session(ws, session)
        result = session.send_json.call_args.args[0]
        self.assertEqual(result["type"], "tool_result")
        self.assertTrue(result["success"])
        self.assertEqual(result["history_status"], "unconfirmed")
        self.assertIn("do not repeat", result["summary"])
        route.assert_awaited_once()
        store.append_exchange.assert_called_once_with(
            "tenant-a", "conversation", "question", "Wrote the file"
        )

    async def test_invalid_conversation_is_rejected_before_history_access(self):
        store, ws, session = conversation()
        ws.query_params = {"conversation_id": "x" * 129}
        ws.close = AsyncMock()
        await main._chat_session(ws, session)
        store.get_history.assert_not_called()
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid conversation identity")

    async def test_startup_failure_closes_the_new_owner(self):
        store = Mock()
        qdrant = Mock()
        app = FastAPI()
        with (
            patch.object(main, "QdrantStore", return_value=qdrant),
            patch.object(main, "MongoHistoryStore", return_value=store),
            patch.object(
                main,
                "_start_optional_services",
                new=AsyncMock(side_effect=RuntimeError("startup failure")),
            ),
        ):
            with self.assertRaises(RuntimeError):
                async with main.lifespan(app):
                    self.fail("Startup should have failed")
        store.close.assert_called_once()
        qdrant.close.assert_called_once()
        self.assertTrue(app.state.websocket_security._closing)


class HistoryLifespanTest(unittest.TestCase):
    def test_each_lifespan_has_its_own_lazy_owner_and_closes_it(self):
        with patch.object(main, "_start_optional_services", new=AsyncMock()):
            with TestClient(main.app):
                first = main.app.state.history_store
                self.assertIsNone(first._client)
            with self.assertRaises(HistoryUnavailable):
                first.get_history("a", "conversation")
            with TestClient(main.app):
                second = main.app.state.history_store
                self.assertIsNot(first, second)
            with self.assertRaises(HistoryUnavailable):
                second.get_history("a", "conversation")


if __name__ == "__main__":
    unittest.main()
