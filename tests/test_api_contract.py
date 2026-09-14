import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.api import main as api


class ApiContractTest(unittest.TestCase):
    def test_health_endpoint_is_ok(self):
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(api.app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_health_dependencies_degrades_cleanly_without_temporal(self):
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(api.app) as client:
            response = client.get("/health/dependencies")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_upload_degrades_cleanly_without_temporal(self):
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(api.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
            )
            self.assertEqual(response.status_code, 503)
            payload = response.json()
            self.assertIn("Temporal service unavailable", payload["detail"])

    def test_health_dependencies_reports_available_temporal(self):
        temporal_client = unittest.mock.Mock()

        async def workflows():
            yield object()

        temporal_client.list_workflows = workflows
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(return_value=temporal_client),
        ), TestClient(api.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "ok"})

    def test_health_dependencies_degrades_when_probe_fails(self):
        temporal_client = unittest.mock.Mock()

        async def failing_workflows():
            raise ConnectionError("probe failed")
            yield  # pragma: no cover - makes this an async generator

        temporal_client.list_workflows = failing_workflows
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(return_value=temporal_client),
        ), TestClient(api.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_upload_persists_file_and_starts_expected_workflow(self):
        temporal_client = unittest.mock.Mock()
        temporal_client.start_workflow = AsyncMock()
        with TemporaryDirectory() as directory, patch.object(
            api, "UPLOAD_DIR", directory
        ), patch.object(
            api.uuid, "uuid4", return_value="document-id"
        ), patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(return_value=temporal_client),
        ), TestClient(api.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("policy.txt", BytesIO(b"policy text"), "text/plain")},
            )

            self.assertEqual(Path(directory, "document-id_policy.txt").read_bytes(), b"policy text")

        self.assertEqual(
            response.json(),
            {
                "doc_id": "document-id",
                "workflow_id": "ingest-document-id",
                "status": "queued",
            },
        )
        temporal_client.start_workflow.assert_awaited_once_with(
            "IngestDocumentWorkflow",
            args=[str(Path(directory, "document-id_policy.txt")), "policy.txt"],
            id="ingest-document-id",
            task_queue="ingestion-task-queue",
        )

    def test_upload_returns_503_when_workflow_start_fails(self):
        temporal_client = unittest.mock.Mock()
        temporal_client.start_workflow = AsyncMock(side_effect=RuntimeError("unavailable"))
        with TemporaryDirectory() as directory, patch.object(
            api, "UPLOAD_DIR", directory
        ), patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(return_value=temporal_client),
        ), TestClient(api.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("policy.txt", BytesIO(b"policy text"), "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal workflow could not start", response.json()["detail"])

    def test_chat_websocket_echoes_token_then_done(self):
        with patch.object(
            api.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(api.app) as client, client.websocket_connect("/ws/chat") as websocket:
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
