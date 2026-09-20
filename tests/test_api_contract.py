import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.api import main
from services.auth import dependencies as auth_dependencies

TENANT_ID = "11111111-1111-1111-1111-111111111111"
VALID_KEY = "ally_test-key"
AUTH_HEADERS = {"X-API-Key": VALID_KEY}


def fake_resolve_tenant(raw_key):
    """Stand-in for the Postgres-backed API key lookup."""
    return TENANT_ID if raw_key == VALID_KEY else None


def patch_auth():
    """Patch the api_keys lookup that both auth dependencies call."""
    return patch.object(
        auth_dependencies, "resolve_tenant", side_effect=fake_resolve_tenant
    )


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
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), TestClient(main.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertFalse(main.app.state.temporal_connected)

    def test_health_dependencies_degrades_cleanly_without_temporal(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), TestClient(main.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_health_dependencies_reports_connected_temporal(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), TestClient(main.app) as client:
            response = client.get("/health/dependencies")
            self.assertTrue(main.app.state.temporal_connected)

        self.assertEqual(response.json(), {"temporal": "ok"})

    def test_health_dependencies_degrades_when_probe_fails(self):
        temporal = TemporalProbe(error=RuntimeError("probe failed"))
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), TestClient(main.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_reranker_warmup_failure_does_not_block_startup(self):
        # A cold model host must not stop the app from serving traffic --
        # the reranker is warmed opportunistically, not as a hard gate.
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(
            main, "warm_reranker", side_effect=RuntimeError("model host down")
        ), TestClient(main.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_reranker_is_warmed_during_startup(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker") as warm, TestClient(main.app):
            pass

        warm.assert_called_once_with()


class UploadAuthTest(unittest.TestCase):
    def test_upload_without_api_key_is_rejected(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello"), "text/plain")},
            )

        self.assertEqual(response.status_code, 401)
        temporal.start_workflow.assert_not_awaited()

    def test_upload_with_invalid_api_key_is_rejected(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
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
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(
            main.uuid, "uuid4", return_value="document-id"
        ), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
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
            stored_path = Path(directory) / "document-id_evil.txt"
            self.assertTrue(stored_path.exists())
            # Nothing was written outside the upload directory.
            self.assertEqual(
                sorted(item.name for item in Path(directory).iterdir()),
                ["document-id_evil.txt"],
            )

    def test_disallowed_extension_is_rejected_before_starting_workflow(self):
        temporal = TemporalProbe()
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            response = client.post(
                "/documents/upload",
                files={
                    "file": ("payload.exe", BytesIO(b"MZ"), "application/octet-stream")
                },
                headers=AUTH_HEADERS,
            )

            self.assertEqual(list(Path(directory).iterdir()), [])

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])
        temporal.start_workflow.assert_not_awaited()

    def test_oversized_upload_is_rejected_and_partial_file_removed(self):
        temporal = TemporalProbe()
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(main, "MAX_UPLOAD_BYTES", 10), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
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
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal service unavailable", response.json()["detail"])

    def test_upload_persists_file_and_passes_tenant_to_workflow(self):
        temporal = TemporalProbe()
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(
            main.uuid, "uuid4", return_value="document-id"
        ), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
                headers=AUTH_HEADERS,
            )
            stored_path = Path(directory) / "document-id_sample.txt"
            self.assertEqual(stored_path.read_bytes(), b"hello world")

        self.assertEqual(
            response.json(),
            {
                "doc_id": "document-id",
                "workflow_id": "ingest-document-id",
                "status": "queued",
            },
        )
        # The tenant resolved from the API key must reach the workflow, or
        # the chunks it writes would not be scoped to anyone.
        temporal.start_workflow.assert_awaited_once_with(
            "IngestDocumentWorkflow",
            args=[str(stored_path), "sample.txt", TENANT_ID],
            id="ingest-document-id",
            task_queue="ingestion-task-queue",
        )

    def test_upload_reports_workflow_start_failure(self):
        temporal = TemporalProbe()
        temporal.start_workflow.side_effect = RuntimeError("queue unavailable")
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal workflow could not start", response.json()["detail"])

    def test_upload_reports_storage_failure_without_starting_workflow(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch.object(main, "warm_reranker"), patch_auth(), patch(
            "builtins.open", side_effect=OSError("disk full")
        ), TestClient(main.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload failed: disk full", response.json()["detail"])
        temporal.start_workflow.assert_not_awaited()


class ChatWebSocketTest(unittest.TestCase):
    def test_chat_websocket_rejects_missing_api_key(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/chat") as websocket:
                    websocket.receive_json()

    def test_chat_websocket_rejects_invalid_api_key(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            with self.assertRaises(Exception):
                with client.websocket_connect(
                    "/ws/chat?api_key=ally_wrong-key"
                ) as websocket:
                    websocket.receive_json()

    def test_chat_websocket_echoes_token_then_done_with_valid_key(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), patch.object(main, "warm_reranker"), patch_auth(), TestClient(
            main.app
        ) as client:
            with client.websocket_connect(
                f"/ws/chat?api_key={VALID_KEY}"
            ) as websocket:
                websocket.send_text("hello")
                self.assertEqual(
                    websocket.receive_json(),
                    {"type": "token", "content": "(stub) you said: hello"},
                )
                self.assertEqual(
                    websocket.receive_json(),
                    {"type": "done", "citations": []},
                )


if __name__ == "__main__":
    unittest.main()
