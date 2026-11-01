"""Public errors and structured logs must not repeat dependency secrets."""

from __future__ import annotations

import unittest
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from pathlib import Path

from services.auth.principal import Principal, LEGACY_SCOPES
from services.agent.classifier import Intent
from services.agent.graph import tool_node
from services.api import main
from services.llm_gateway import router

SENTINEL = "credential-must-not-appear"


class _Breaker:
    async def call_async(self, function, *args, **kwargs):
        return await function(*args, **kwargs)


class ErrorRedactionTest(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_failure_does_not_expose_dependency_error(self):
        temporal = Mock(start_workflow=AsyncMock(side_effect=RuntimeError(SENTINEL)))
        with (
            TemporaryDirectory() as directory,
            patch.object(
                main.app.state,
                "runtime_settings",
                main.app.state.runtime_settings.model_copy(
                    update={
                        "application": main.app.state.runtime_settings.application.model_copy(
                            update={"upload_dir": Path(directory)}
                        )
                    }
                ),
            ),
            patch.object(main.app.state, "temporal_client", temporal, create=True),
            patch.object(main, "log") as log,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.upload_document(
                    Request({"type": "http", "app": main.app}),
                    UploadFile(BytesIO(b"text"), filename="safe.txt"),
                    "tenant-a",
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn(SENTINEL, raised.exception.detail)
        self.assertNotIn(SENTINEL, str(log.mock_calls))

    async def test_gateway_request_error_is_redacted(self):
        with (
            patch.object(router, "_get_breaker", return_value=_Breaker()),
            patch.object(
                router, "_start_stream", new=AsyncMock(side_effect=RuntimeError(SENTINEL))
            ),
            patch.object(router, "log") as log,
        ):
            with self.assertRaises(router.LLMGatewayError) as raised:
                async for _ in router.stream_chat_completion([]):
                    pass
        self.assertNotIn(SENTINEL, str(raised.exception))
        self.assertNotIn(SENTINEL, str(log.mock_calls))

    async def test_mid_stream_error_preserves_tokens_and_redacts_failure(self):
        async def broken_stream():
            yield Mock(choices=[Mock(delta=Mock(content="partial"))])
            raise RuntimeError(SENTINEL)

        received = []
        with (
            patch.object(router, "_get_breaker", return_value=_Breaker()),
            patch.object(router, "_start_stream", new=AsyncMock(return_value=broken_stream())),
            patch.object(router, "log") as log,
        ):
            with self.assertRaises(router.LLMGatewayError) as raised:
                async for delta in router.stream_chat_completion([]):
                    received.append(delta)
        self.assertEqual(received, ["partial"])
        self.assertNotIn(SENTINEL, str(raised.exception))
        self.assertNotIn(SENTINEL, str(log.mock_calls))

    async def test_unexpected_tool_error_is_redacted(self):
        tool = Mock()
        tool.name = "jira"
        tool.execute = AsyncMock(side_effect=RuntimeError(SENTINEL))
        registry = Mock()
        registry.get.return_value = tool
        with (
            patch("services.agent.graph.get_registry", return_value=registry),
            patch("services.agent.graph.log") as log,
        ):
            result = await tool_node(
                {
                    "tenant_id": "a",
                    "principal": Principal("a", "test-credential", LEGACY_SCOPES),
                    "query": "create ticket",
                    "intent": Intent(kind="tool", tool_name="jira", action="create_ticket"),
                }
            )
        self.assertNotIn(SENTINEL, result["tool_error"])
        self.assertNotIn(SENTINEL, str(log.mock_calls))


if __name__ == "__main__":
    unittest.main()
