import hashlib
import asyncio
import threading
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from services.ingestion.upload import (
    LocalUploadStorage,
    S3UploadStorage,
    UploadFailure,
    UploadUseCase,
)


class ByteSource:
    filename = "source.txt"

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    async def read(self, size: int = -1) -> bytes:
        del size
        if self._read:
            return b""
        self._read = True
        return self._payload


class S3UploadStorageTest(unittest.IsolatedAsyncioTestCase):
    async def test_repeat_uploads_keep_document_identity_and_use_immutable_version_keys(self):
        workflow = Mock()
        workflow.start_workflow = AsyncMock()
        generated_ids = iter(["attempt-1", "version-1", "attempt-2", "version-2"])
        with tempfile.TemporaryDirectory() as directory:
            use_case = UploadUseCase(
                Path(directory),
                1024,
                LocalUploadStorage(),
                workflow,
                make_id=lambda: next(generated_ids),
            )
            first = await use_case.execute(ByteSource(b"first"), "tenant-a")
            second = await use_case.execute(ByteSource(b"second"), "tenant-a")

            expected_document_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, "ally-document:tenant-a:source.txt")
            )
            first_args = workflow.start_workflow.await_args_list[0].kwargs["args"]
            second_args = workflow.start_workflow.await_args_list[1].kwargs["args"]
            self.assertEqual(first["doc_id"], expected_document_id)
            self.assertEqual(second["doc_id"], expected_document_id)
            self.assertNotEqual(first_args[0], second_args[0])
            self.assertEqual(first_args[3], second_args[3])
            self.assertEqual(first_args[5:], ["attempt-1", "version-1"])
            self.assertEqual(second_args[5:], ["attempt-2", "version-2"])
            self.assertNotEqual(first["workflow_id"], second["workflow_id"])

    async def test_cancelled_local_stream_removes_partial_file(self):
        class CancelledSource(ByteSource):
            async def read(self, size: int = -1) -> bytes:
                del size
                raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as directory:
            use_case = UploadUseCase(
                Path(directory), 1024, LocalUploadStorage(), Mock(), make_id=lambda: "doc"
            )
            with self.assertRaises(asyncio.CancelledError):
                await use_case.execute(CancelledSource(b"ignored"), "tenant")
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_streams_to_object_store_and_returns_digest_reference(self):
        client = Mock()
        storage = S3UploadStorage("ally-ingestion", "us-east-1")
        payload = b"large enough to go through the disk spool"

        with patch("services.ingestion.upload._s3_client", return_value=client):
            stored = await storage.save(
                "tenants/acme/documents/document-1/source.txt",
                ByteSource(payload),
                max_bytes=1024,
            )

        self.assertEqual(
            stored.storage_ref,
            "s3://ally-ingestion/tenants/acme/documents/document-1/source.txt",
        )
        self.assertEqual(stored.size_bytes, len(payload))
        self.assertEqual(stored.content_sha256, hashlib.sha256(payload).hexdigest())
        staged_path, bucket, key = client.upload_file.call_args.args
        self.assertEqual(
            (bucket, key), ("ally-ingestion", "tenants/acme/documents/document-1/source.txt")
        )
        self.assertFalse(Path(staged_path).exists())

    async def test_oversized_stream_is_rejected_before_s3_upload(self):
        client = Mock()
        storage = S3UploadStorage("ally-ingestion", "us-east-1")
        with patch("services.ingestion.upload._s3_client", return_value=client):
            with self.assertRaises(UploadFailure):
                await storage.save("tenants/acme/document", ByteSource(b"too big"), max_bytes=2)
        client.upload_file.assert_not_called()

    async def test_cancelled_s3_upload_keeps_spool_until_transfer_then_deletes_object(self):
        started = threading.Event()
        release = threading.Event()
        client = Mock()

        def upload(path: str, bucket: str, key: str) -> None:
            self.assertTrue(Path(path).exists())
            started.set()
            release.wait(timeout=3)
            self.assertTrue(Path(path).exists())

        client.upload_file.side_effect = upload
        storage = S3UploadStorage("ally-ingestion", "us-east-1")
        with patch("services.ingestion.upload._s3_client", return_value=client):
            task = asyncio.create_task(
                storage.save("tenant/document", ByteSource(b"payload"), max_bytes=1024)
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        client.delete_object.assert_called_once_with(Bucket="ally-ingestion", Key="tenant/document")

    async def test_worker_materializes_s3_reference_and_removes_local_copy(self):
        from workflows.activities import _materialized_source

        client = Mock()

        def download(bucket: str, key: str, destination: str) -> None:
            self.assertEqual((bucket, key), ("ally-ingestion", "tenants/acme/source.txt"))
            Path(destination).write_bytes(b"source")

        client.download_file.side_effect = download
        with patch("boto3.client", return_value=client):
            async with _materialized_source(
                "s3://ally-ingestion/tenants/acme/source.txt"
            ) as local_path:
                materialized = Path(local_path)
                self.assertEqual(materialized.read_bytes(), b"source")
            self.assertFalse(materialized.exists())

    async def test_worker_rejects_traversal_in_s3_object_reference(self):
        from workflows.activities import _materialized_source

        with self.assertRaisesRegex(ValueError, "Invalid ingestion object"):
            async with _materialized_source("s3://ally-ingestion/tenant/../secret"):
                self.fail("invalid reference must not be downloaded")

    async def test_textract_start_uses_stable_client_token_for_safe_retry(self):
        from workflows import activities

        s3 = Mock()
        textract = Mock()
        textract.start_document_analysis.return_value = {"JobId": "job-1"}
        textract.get_document_analysis.side_effect = [
            {"JobStatus": "SUCCEEDED"},
            {"Blocks": []},
        ]
        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as source,
            patch("workflows.activities.TEXTRACT_S3_BUCKET", "textract-input"),
            patch("boto3.client", return_value=s3),
            patch.object(activities, "_get_textract_job_sync", return_value=None),
            patch.object(activities, "_save_textract_job_sync", return_value=True) as save_job,
            patch.object(activities, "_set_textract_job_status_sync", return_value=True),
        ):
            await activities._run_async_textract(textract, source.name, "f" * 64, "version-1")

        self.assertEqual(
            textract.start_document_analysis.call_args.kwargs["ClientRequestToken"], "f" * 64
        )
        textract.start_document_analysis.assert_called_once()
        save_job.assert_called_once_with("version-1", "f" * 64, "job-1", "started")

    async def test_existing_persisted_textract_job_is_reused_without_restart(self):
        from workflows import activities

        s3 = Mock()
        textract = Mock()
        textract.get_document_analysis.side_effect = [
            {"JobStatus": "SUCCEEDED"},
            {"Blocks": []},
        ]
        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as source,
            patch("workflows.activities.TEXTRACT_S3_BUCKET", "textract-input"),
            patch("boto3.client", return_value=s3),
            patch.object(activities, "_get_textract_job_sync", return_value="existing-job"),
            patch.object(activities, "_set_textract_job_status_sync", return_value=True),
        ):
            blocks = await activities._run_async_textract(
                textract, source.name, "f" * 64, "version-1"
            )

        self.assertEqual(blocks, [])
        textract.start_document_analysis.assert_not_called()
        self.assertEqual(
            textract.get_document_analysis.call_args_list[0].kwargs, {"JobId": "existing-job"}
        )

    async def test_blocking_activity_heartbeats_while_sync_operation_runs(self):
        from workflows import activities

        with patch("workflows.activities.activity.heartbeat") as heartbeat:
            await activities._run_sync_with_heartbeat(time.sleep, 0.03, heartbeat_interval=0.005)
        heartbeat.assert_called()
