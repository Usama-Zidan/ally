import unittest
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.api import main
from services.auth import dependencies as auth_dependencies
from services.auth.principal import Principal, LEGACY_SCOPES

TENANT_ID = "11111111-1111-1111-1111-111111111111"
VALID_KEY = "ally_test-key"
AUTH_HEADERS = {"X-API-Key": VALID_KEY}


@contextmanager
def patch_runtime(**overrides):
    """Override the injected API snapshot; merge nested test contexts on entry."""
    snapshot = main.app.state.runtime_settings
    application = snapshot.application.model_validate(
        {**snapshot.application.model_dump(), **overrides}
    )
    with patch.object(
        main.app.state, "runtime_settings", snapshot.model_copy(update={"application": application})
    ):
        yield


def fake_resolve_tenant(raw_key):
    """Stand-in for the Postgres-backed API key lookup."""
    return Principal(TENANT_ID, "test-credential", LEGACY_SCOPES) if raw_key == VALID_KEY else None


def patch_auth():
    """Patch the api_keys lookup that both auth dependencies call."""
    return patch.object(auth_dependencies, "resolve_principal", side_effect=fake_resolve_tenant)


def patch_rag_route():
    """Keep chat endpoint tests focused on retrieval/streaming, not classification."""
    return patch.object(main, "route_message", new=AsyncMock(return_value={"route": "rag"}))


class TemporalProbe:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.start_workflow = AsyncMock()

    async def list_workflows(self):
        if self.error:
            raise self.error
        yield None


class HealthEndpointTest(unittest.TestCase):
    def test_health_endpoint_is_ok(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            TestClient(main.app) as client,
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertFalse(main.app.state.temporal_connected)

    def test_health_dependencies_degrades_cleanly_without_temporal(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            TestClient(main.app) as client,
        ):
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_health_dependencies_reports_connected_temporal(self):
        temporal = TemporalProbe()
        with (
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            TestClient(main.app) as client,
        ):
            response = client.get("/health/dependencies")
            self.assertTrue(main.app.state.temporal_connected)

        self.assertEqual(response.json(), {"temporal": "ok"})

    def test_health_dependencies_degrades_when_probe_fails(self):
        temporal = TemporalProbe(error=RuntimeError("probe failed"))
        with (
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            TestClient(main.app) as client,
        ):
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_reranker_warmup_failure_does_not_block_startup(self):
        # A cold model host must not stop the app from serving traffic --
        # the reranker is warmed opportunistically, not as a hard gate.
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker", side_effect=RuntimeError("model host down")),
            TestClient(main.app) as client,
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_reranker_is_warmed_during_startup(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker") as warm,
            TestClient(main.app),
        ):
            pass

        warm.assert_called_once_with(main.app.state.runtime_settings.application.reranker_model)


class UploadAuthTest(unittest.TestCase):
    def test_upload_without_api_key_is_rejected(self):
        temporal = TemporalProbe()
        with (
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello"), "text/plain")},
            )

        self.assertEqual(response.status_code, 401)
        temporal.start_workflow.assert_not_awaited()

    def test_upload_with_invalid_api_key_is_rejected(self):
        temporal = TemporalProbe()
        with (
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello"), "text/plain")},
                headers={"X-API-Key": "ally_wrong-key"},
            )

        self.assertEqual(response.status_code, 401)
        temporal.start_workflow.assert_not_awaited()


class UploadHardeningTest(unittest.TestCase):
    def test_path_traversal_filename_cannot_escape_upload_dir(self):
        # basename() must strip the traversal components so the file lands
        # inside UPLOAD_DIR no matter what filename the client sent.
        temporal = TemporalProbe()
        with (
            TemporaryDirectory() as directory,
            patch_runtime(upload_dir=Path(directory)),
            patch.object(main.uuid, "uuid4", return_value="document-id"),
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "../../../../tmp/evil.txt",
                        BytesIO(b"payload"),
                        "text/plain",
                    )
                },
                headers=AUTH_HEADERS,
            )

            self.assertEqual(response.status_code, 200)
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ally-document:{TENANT_ID}:evil.txt"))
            stored_path = Path(directory) / f"{document_id}_document-id_evil.txt"
            self.assertTrue(stored_path.exists())
            # Nothing was written outside the upload directory.
            self.assertEqual(
                sorted(item.name for item in Path(directory).iterdir()),
                [stored_path.name],
            )

    def test_disallowed_extension_is_rejected_before_starting_workflow(self):
        temporal = TemporalProbe()
        with (
            TemporaryDirectory() as directory,
            patch_runtime(upload_dir=Path(directory)),
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("payload.exe", BytesIO(b"MZ"), "application/octet-stream")},
                headers=AUTH_HEADERS,
            )

            self.assertEqual(list(Path(directory).iterdir()), [])

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])
        temporal.start_workflow.assert_not_awaited()

    def test_oversized_upload_is_rejected_and_partial_file_removed(self):
        temporal = TemporalProbe()
        with (
            TemporaryDirectory() as directory,
            patch_runtime(upload_dir=Path(directory)),
            patch_runtime(max_upload_bytes=10),
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("big.txt", BytesIO(b"x" * 5000), "text/plain")},
                headers=AUTH_HEADERS,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("upload limit", response.json()["detail"])
            # The partial write must be cleaned up, not left behind on disk.
            self.assertEqual(list(Path(directory).iterdir()), [])

        temporal.start_workflow.assert_not_awaited()


class UploadWorkflowTest(unittest.TestCase):
    def test_upload_degrades_cleanly_without_temporal(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal service unavailable", response.json()["detail"])

    def test_upload_persists_file_and_passes_tenant_to_workflow(self):
        temporal = TemporalProbe()
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ally-document:{TENANT_ID}:sample.txt"))
        with (
            TemporaryDirectory() as directory,
            patch_runtime(upload_dir=Path(directory)),
            patch.object(main.uuid, "uuid4", return_value="document-id"),
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
                headers=AUTH_HEADERS,
            )
            stored_path = Path(directory) / f"{document_id}_document-id_sample.txt"
            self.assertEqual(stored_path.read_bytes(), b"hello world")

        self.assertEqual(
            response.json(),
            {
                "doc_id": document_id,
                "workflow_id": f"ingest-{document_id}-document-id",
                "status": "queued",
            },
        )
        # The tenant resolved from the API key must reach the workflow, or
        # the chunks it writes would not be scoped to anyone.
        temporal.start_workflow.assert_awaited_once_with(
            "IngestDocumentWorkflow",
            args=[
                str(stored_path),
                "sample.txt",
                TENANT_ID,
                document_id,
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
                "document-id",
                "document-id",
            ],
            id=f"ingest-{document_id}-document-id",
            task_queue="ingestion-task-queue",
        )

    def test_upload_reports_workflow_start_failure(self):
        temporal = TemporalProbe()
        temporal.start_workflow.side_effect = RuntimeError("queue unavailable")
        with (
            TemporaryDirectory() as directory,
            patch_runtime(upload_dir=Path(directory)),
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal workflow could not start", response.json()["detail"])

    def test_upload_reports_storage_failure_without_starting_workflow(self):
        temporal = TemporalProbe()
        with (
            patch.object(main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch("builtins.open", side_effect=OSError("disk full")),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 400)
        # Error text is intentionally sanitized; raw storage errors can contain credentials.
        self.assertIn("Upload failed", response.json()["detail"])
        self.assertNotIn("disk full", response.json()["detail"])
        temporal.start_workflow.assert_not_awaited()


class ChatWebSocketTest(unittest.TestCase):
    def test_chat_websocket_rejects_missing_api_key(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/chat") as websocket:
                    websocket.receive_json()

    def test_chat_websocket_rejects_invalid_api_key(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/chat?api_key=ally_wrong-key") as websocket:
                    websocket.receive_json()

    def _connected_chat_client(self, client, conversation_id=None):
        url = f"/ws/chat?api_key={VALID_KEY}"
        if conversation_id:
            url += f"&conversation_id={conversation_id}"
        return client.websocket_connect(url)

    def test_chat_websocket_sends_ready_with_conversation_id_on_connect(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                ready = websocket.receive_json()

        self.assertEqual(ready["type"], "ready")
        self.assertTrue(ready["conversation_id"])

    def test_chat_websocket_reuses_conversation_id_when_provided(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client, "existing-convo") as websocket:
                ready = websocket.receive_json()

        self.assertEqual(ready["conversation_id"], "existing-convo")

    def test_chat_websocket_sends_no_context_message_without_calling_llm(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=[])),
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch.object(main.MongoHistoryStore, "append_exchange") as append_exchange,
            patch.object(main, "stream_chat_completion") as stream_chat,
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("What is the meaning of life?")

                sources_msg = websocket.receive_json()
                token_msg = websocket.receive_json()
                done_msg = websocket.receive_json()

        self.assertEqual(sources_msg, {"type": "sources", "sources": []})
        self.assertEqual(token_msg["type"], "token")
        self.assertIn("couldn't find", token_msg["content"])
        self.assertEqual(done_msg, {"type": "done", "citations": []})
        # No sources means no LLM call at all.
        stream_chat.assert_not_called()
        self.assertEqual(append_exchange.call_count, 1)

    def test_chat_websocket_streams_tokens_and_emits_page_level_citations(self):
        sources = [
            {
                "text": "Employees receive 21 days of leave.",
                "filename": "hr_policy.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "score": 0.91,
            }
        ]

        async def fake_stream(messages):
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("[1]", messages[-1]["content"])
            for token in ["Employees ", "receive 21 days ", "of leave [1]."]:
                yield token

        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=sources)),
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch.object(main.MongoHistoryStore, "append_exchange") as append_exchange,
            patch.object(main, "stream_chat_completion", side_effect=fake_stream),
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("How much leave do employees get?")

                sources_msg = websocket.receive_json()
                tokens = [websocket.receive_json() for _ in range(3)]
                done_msg = websocket.receive_json()

        self.assertEqual(
            sources_msg,
            {
                "type": "sources",
                "sources": [
                    {
                        "index": 1,
                        "filename": "hr_policy.pdf",
                        "page_number": 3,
                        "score": 0.91,
                        # The cited passage travels with its own citation
                        # so a client can render the evidence, not just the
                        # pointer to it.
                        "text": "Employees receive 21 days of leave.",
                    }
                ],
            },
        )
        self.assertEqual(
            [t["content"] for t in tokens], ["Employees ", "receive 21 days ", "of leave [1]."]
        )
        self.assertEqual(done_msg["citations"], sources_msg["sources"])

        # Both the user turn and the full assembled answer are persisted.
        self.assertEqual(append_exchange.call_count, 1)
        self.assertEqual(
            append_exchange.call_args.args[2:],
            ("How much leave do employees get?", "Employees receive 21 days of leave [1]."),
        )

    def test_chat_websocket_reports_retrieval_failure_and_keeps_socket_open(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(
                main, "retrieve_async", new=AsyncMock(side_effect=RuntimeError("qdrant down"))
            ),
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("anything")
                error_msg = websocket.receive_json()

                # Socket must still be usable for the next message.
                websocket.send_text("ping")

        self.assertEqual(error_msg["type"], "error")

    def test_chat_websocket_reports_llm_gateway_failure_and_keeps_socket_open(self):
        sources = [
            {
                "text": "context",
                "filename": "a.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "score": 0.5,
            }
        ]

        async def failing_stream(messages):
            raise main.LLMGatewayError("model host unavailable")
            yield  # pragma: no cover - makes this an async generator

        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=sources)),
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch.object(main.MongoHistoryStore, "append_exchange") as append_exchange,
            patch.object(main, "stream_chat_completion", side_effect=failing_stream),
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("question")
                websocket.receive_json()  # sources
                error_msg = websocket.receive_json()

        self.assertEqual(error_msg, {"type": "error", "detail": "model host unavailable"})
        # A failed generation is not persisted as a completed turn.
        append_exchange.assert_not_called()

    def test_chat_websocket_truncates_source_preview(self):
        long_text = "Employees receive " + ("x" * 5000)
        sources = [
            {
                "text": long_text,
                "filename": "hr_policy.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "score": 0.9,
            }
        ]

        async def fake_stream(messages):
            del messages
            yield "Answer [1]."

        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=sources)),
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch.object(main.MongoHistoryStore, "append_exchange"),
            patch.object(main, "stream_chat_completion", side_effect=fake_stream),
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client) as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("question")
                sources_msg = websocket.receive_json()

        preview = sources_msg["sources"][0]["text"]
        self.assertEqual(len(preview), main.SOURCE_PREVIEW_CHARS)
        self.assertTrue(long_text.startswith(preview))

    def test_chat_websocket_scopes_retrieval_and_history_to_resolved_tenant(self):
        with (
            patch.object(
                main.TemporalClient,
                "connect",
                new=AsyncMock(side_effect=ConnectionError("offline")),
            ),
            patch.object(main, "warm_reranker"),
            patch_auth(),
            patch.object(main, "retrieve_async", new=AsyncMock(return_value=[])) as retrieve,
            patch.object(main.MongoHistoryStore, "get_history", return_value=[]),
            patch.object(main.MongoHistoryStore, "append_exchange") as append_exchange,
            patch_rag_route(),
            TestClient(main.app) as client,
        ):
            with self._connected_chat_client(client, "convo-1") as websocket:
                websocket.receive_json()  # ready
                websocket.send_text("hello")
                websocket.receive_json()  # sources
                websocket.receive_json()  # token
                websocket.receive_json()  # done

        retrieve.assert_awaited_once_with(
            "hello",
            TENANT_ID,
            top_k=main.runtime_settings.application.chat_retrieval_top_k,
            qdrant_store=main.app.state.qdrant_store,
            retrieval_models=main.app.state.retrieval_models,
        )
        for call in append_exchange.call_args_list:
            self.assertEqual(call.args[0], TENANT_ID)
            self.assertEqual(call.args[1], "convo-1")


if __name__ == "__main__":
    unittest.main()
