"""Opt-in live Temporal cancellation and history replay checks for ingestion."""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid

from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Replayer, Worker

from workflows.ingestion_workflow import IngestDocumentWorkflow


@unittest.skipUnless(
    os.environ.get("RUN_TEMPORAL_INGESTION_TESTS") == "1",
    "set RUN_TEMPORAL_INGESTION_TESTS=1 for disposable Temporal workflow checks",
)
class TemporalIngestionRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_ingestion_records_status_and_history_replays(self) -> None:
        address = os.environ.get("TEMPORAL_TEST_ADDRESS", "127.0.0.1:57233")
        client = await Client.connect(address, namespace="default")
        extract_started = asyncio.Event()
        finish_called = asyncio.Event()
        finish_statuses: list[str] = []

        @activity.defn(name="register_ingestion")
        async def register_ingestion(
            document_id: str,
            attempt_id: str,
            version_id: str,
            storage_ref: str,
            content_sha256: str,
            filename: str,
            tenant_id: str,
            workflow_id: str,
        ) -> bool:
            return True

        @activity.defn(name="update_ingestion_stage")
        async def update_ingestion_stage(
            document_id: str, attempt_id: str, tenant_id: str, stage: str
        ) -> bool:
            return True

        @activity.defn(name="detect_document_type")
        async def detect_document_type(file_path: str, filename: str) -> str:
            return "native"

        @activity.defn(name="extract_with_unstructured")
        async def extract_with_unstructured(
            file_path: str, artifact_identity: dict[str, str] | None = None
        ) -> dict[str, object]:
            extract_started.set()
            await asyncio.Event().wait()
            return {}

        @activity.defn(name="finish_ingestion_attempt")
        async def finish_ingestion_attempt(
            document_id: str,
            attempt_id: str,
            version_id: str,
            tenant_id: str,
            status: str,
            error_code: str | None,
        ) -> bool:
            finish_statuses.append(status)
            finish_called.set()
            return True

        task_queue = f"ingestion-recovery-{uuid.uuid4()}"
        workflow_id = f"ingest-test-{uuid.uuid4()}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[IngestDocumentWorkflow],
            activities=[
                register_ingestion,
                update_ingestion_stage,
                detect_document_type,
                extract_with_unstructured,
                finish_ingestion_attempt,
            ],
        ):
            handle = await client.start_workflow(
                IngestDocumentWorkflow.run,
                args=[
                    "file:///tmp/ally-temporal-cancel.txt",
                    "cancel.txt",
                    "tenant-test",
                    "document-test",
                    "a" * 64,
                    "attempt-test",
                    "version-test",
                ],
                id=workflow_id,
                task_queue=task_queue,
            )
            await asyncio.wait_for(extract_started.wait(), timeout=20)
            await handle.cancel()
            with self.assertRaises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), timeout=20)
            await asyncio.wait_for(finish_called.wait(), timeout=10)

            self.assertEqual(finish_statuses, ["cancelled"])
            history = await handle.fetch_history()
            replay_result = await Replayer(workflows=[IngestDocumentWorkflow]).replay_workflow(
                history
            )
            self.assertIsNone(replay_result.replay_failure)


if __name__ == "__main__":
    unittest.main()
