import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.api import main


class TemporalProbe:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.start_workflow = AsyncMock()

    async def list_workflows(self):
        if self.error:
            raise self.error
        yield None


class ApiContractTest(unittest.TestCase):
    def test_health_endpoint_is_ok(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(main.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertFalse(main.app.state.temporal_connected)

    def test_health_dependencies_degrades_cleanly_without_temporal(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(main.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_health_dependencies_reports_connected_temporal(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), TestClient(main.app) as client:
            response = client.get("/health/dependencies")
            self.assertTrue(main.app.state.temporal_connected)

        self.assertEqual(response.json(), {"temporal": "ok"})

    def test_health_dependencies_degrades_when_probe_fails(self):
        temporal = TemporalProbe(error=RuntimeError("probe failed"))
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), TestClient(main.app) as client:
            response = client.get("/health/dependencies")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"temporal": "unavailable"})

    def test_upload_degrades_cleanly_without_temporal(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(main.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal service unavailable", response.json()["detail"])

    def test_upload_persists_file_and_starts_expected_workflow(self):
        temporal = TemporalProbe()
        with TemporaryDirectory() as directory, patch.object(
            main, "UPLOAD_DIR", directory
        ), patch.object(
            main.uuid, "uuid4", return_value="document-id"
        ), patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), TestClient(main.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
            )
            stored_path = Path(directory) / "document-id_sample.txt"
            self.assertEqual(stored_path.read_bytes(), b"hello world")

        self.assertEqual(
            response.json(),
            {"doc_id": "document-id", "workflow_id": "ingest-document-id", "status": "queued"},
        )
        temporal.start_workflow.assert_awaited_once_with(
            "IngestDocumentWorkflow",
            args=[str(stored_path), "sample.txt"],
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
        ), TestClient(main.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Temporal workflow could not start", response.json()["detail"])

    def test_upload_reports_storage_failure_without_starting_workflow(self):
        temporal = TemporalProbe()
        with patch.object(
            main.TemporalClient, "connect", new=AsyncMock(return_value=temporal)
        ), patch("builtins.open", side_effect=OSError("disk full")), TestClient(main.app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"content"), "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload failed: disk full", response.json()["detail"])
        temporal.start_workflow.assert_not_awaited()

    def test_chat_websocket_echoes_token_then_done(self):
        with patch.object(
            main.TemporalClient,
            "connect",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ), TestClient(main.app) as client, client.websocket_connect("/ws/chat") as websocket:
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
