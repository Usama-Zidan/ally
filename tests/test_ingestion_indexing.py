import unittest
import asyncio
import hashlib
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
from temporalio.exceptions import CancelledError as TemporalCancelledError

from services.retrieval import backfill_index
from services.retrieval.qdrant_store import Chunk, make_chunk_id
from workflows import activities, worker
from workflows.ingestion_workflow import IngestDocumentWorkflow
from services.runtime.task_drain import DrainableTasks


class PersistChunksTest(unittest.IsolatedAsyncioTestCase):
    async def test_versioned_chunks_are_written_to_staging_without_replacing_legacy_rows(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        identity = {
            "tenant_id": "tenant-a",
            "document_id": "document-a",
            "version_id": "version-a",
        }
        chunks = [{"page_number": 1, "text": "new content"}]
        descriptor = {"storage_ref": "s3://ally/chunks.json.gz", "stage": "chunks", "item_count": 1}

        with (
            patch("psycopg2.connect", return_value=connection),
            patch.object(activities, "_load_artifact", new=AsyncMock(return_value=chunks)),
        ):
            self.assertTrue(
                await activities.persist_chunks(descriptor, "policy.pdf", "tenant-a", identity)
            )

        sql = [item.args[0] for item in cursor.execute.call_args_list]
        self.assertIn("DELETE FROM document_version_chunks", sql[0])
        self.assertIn("INSERT INTO document_version_chunks", sql[1])
        self.assertIn("UPDATE document_versions SET chunk_count", sql[2])
        self.assertFalse(any("DELETE FROM document_chunks" in statement for statement in sql))
        connection.commit.assert_called_once_with()

    async def test_version_activation_commits_postgres_before_qdrant_flags(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            ("policy.pdf", None, "processing"),
            ("staged", 2, "attempt-a"),
        ]
        cursor.rowcount = 1
        with (
            patch("psycopg2.connect", return_value=connection),
            patch("services.retrieval.qdrant_store.activate_document_version") as activate,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres"),
        ):
            self.assertTrue(
                activities._activate_ingestion_version_sync("document-a", "version-a", "tenant-a")
            )

        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()
        activate.assert_called_once_with("tenant-a", "document-a", "version-a", "policy.pdf")
        self.assertTrue(
            any("active_version_id = %s" in item.args[0] for item in cursor.execute.call_args_list)
        )

    async def test_qdrant_activation_failure_propagates_after_postgres_pointer_commit(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            ("policy.pdf", "version-old", "processing"),
            ("staged", 2, "attempt-a"),
        ]
        cursor.rowcount = 1
        with (
            patch("psycopg2.connect", return_value=connection),
            patch(
                "services.retrieval.qdrant_store.activate_document_version",
                side_effect=RuntimeError("qdrant unavailable"),
            ),
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres"),
        ):
            with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
                activities._activate_ingestion_version_sync("document-a", "version-new", "tenant-a")

        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    async def test_tombstone_is_persisted_and_qdrant_cleanup_is_attempted(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = ("policy.pdf", "indexed")
        cursor.rowcount = 1
        with (
            patch("psycopg2.connect", return_value=connection),
            patch("services.retrieval.qdrant_store.reconcile_document_version") as reconcile,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres"),
        ):
            self.assertTrue(
                activities._tombstone_ingestion_document_sync(
                    "document-a", "tenant-a", "user_deleted"
                )
            )

        sql = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("INSERT INTO ingestion_tombstones" in statement for statement in sql))
        self.assertTrue(any("status = 'deleted'" in statement for statement in sql))
        self.assertTrue(any("DELETE FROM document_chunks" in statement for statement in sql))
        connection.commit.assert_called_once_with()
        reconcile.assert_called_once_with(
            "tenant-a", "document-a", None, "policy.pdf", deleted=True
        )

    async def test_reconciliation_reads_authoritative_active_pointer_and_tombstones(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("document-a", "tenant-a", "policy.pdf", "version-a", "indexed"),
            ("document-b", "tenant-b", "old.pdf", None, "deleted"),
        ]
        with (
            patch("psycopg2.connect", return_value=connection),
            patch("services.retrieval.qdrant_store.reconcile_document_version") as reconcile,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres"),
        ):
            self.assertEqual(activities._reconcile_ingestion_indexes_sync(), 2)

        self.assertEqual(reconcile.call_count, 2)
        self.assertEqual(
            reconcile.call_args_list[0].args,
            ("tenant-a", "document-a", "version-a", "policy.pdf"),
        )
        self.assertEqual(reconcile.call_args_list[1].kwargs["deleted"], True)

    async def test_persist_chunks_replaces_document_rows_scoped_to_tenant(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        chunks = [
            {"page_number": 1, "text": "first"},
            {"page_number": 3, "text": "second"},
        ]

        with patch("psycopg2.connect", return_value=connection) as connect:
            result = await activities.persist_chunks(chunks, "policy.pdf", "tenant-a")

        self.assertTrue(result)
        connect.assert_called_once_with(activities.POSTGRES_DSN)
        # The DELETE must filter on tenant_id as well as filename, or
        # re-ingesting "policy.pdf" for one tenant would wipe another
        # tenant's identically-named document.
        self.assertEqual(
            cursor.execute.call_args_list[0],
            call(
                "DELETE FROM document_chunks WHERE tenant_id = %s AND filename = %s;",
                ("tenant-a", "policy.pdf"),
            ),
        )
        self.assertEqual(
            cursor.execute.call_args_list[1].args[1], ("tenant-a", "policy.pdf", 1, 0, "first")
        )
        self.assertEqual(
            cursor.execute.call_args_list[2].args[1], ("tenant-a", "policy.pdf", 3, 1, "second")
        )
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    async def test_persist_chunks_closes_connection_when_database_write_fails(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = RuntimeError("database unavailable")

        with patch("psycopg2.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await activities.persist_chunks([], "policy.pdf", "tenant-a")

        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()


class IngestionActivityTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_index_activity_is_drained_before_worker_store_close(self):
        tracker = DrainableTasks()
        store = Mock()
        entered = threading.Event()
        release = threading.Event()
        rebuilt = threading.Event()

        def blocking_index(*args, **kwargs):
            entered.set()
            release.wait(timeout=3)

        def rebuild():
            rebuilt.set()

        activities.set_qdrant_store(store)
        activities.set_worker_work(tracker)
        try:
            with (
                patch("services.retrieval.qdrant_store.delete_by_filename"),
                patch("services.retrieval.qdrant_store.index_chunks", side_effect=blocking_index),
                patch("services.retrieval.bm25_index.rebuild_index_from_postgres", rebuild),
            ):
                activity_task = asyncio.create_task(
                    activities.embed_and_index(
                        [{"page_number": 1, "text": "chunk"}], "policy.pdf", "tenant-a"
                    )
                )
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                activity_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await activity_task

                shutdown = asyncio.create_task(tracker.close())
                await asyncio.sleep(0)
                self.assertFalse(shutdown.done())
                release.set()
                await asyncio.wait_for(shutdown, timeout=1)
        finally:
            release.set()
            activities.set_worker_work(None)
            activities.set_qdrant_store(None)

        self.assertTrue(rebuilt.is_set())

    async def test_extract_with_unstructured_falls_back_to_pypdf_on_import_error(self):
        pypdf_module = ModuleType("pypdf")
        setattr(
            pypdf_module,
            "PdfReader",
            lambda _: SimpleNamespace(
                pages=[
                    SimpleNamespace(extract_text=lambda: "  First page text.  "),
                    SimpleNamespace(extract_text=lambda: "Second page text."),
                ]
            ),
        )

        with patch.dict(
            "sys.modules",
            {"unstructured.partition.auto": None, "pypdf": pypdf_module},
        ):
            pages = await activities.extract_with_unstructured("policy.pdf")

        self.assertEqual(
            pages,
            [
                {"page_number": 1, "text": "First page text.", "category": "Text"},
                {"page_number": 2, "text": "Second page text.", "category": "Text"},
            ],
        )

    async def test_extract_with_unstructured_reraises_import_error_for_non_pdfs(self):
        with patch.dict("sys.modules", {"unstructured.partition.auto": None}):
            with self.assertRaises(ImportError):
                await activities.extract_with_unstructured("policy.docx")

    async def test_detect_document_type_routes_images_and_pdf_text_layers(self):
        self.assertEqual(await activities.detect_document_type("scan.png", "scan.PNG"), "scanned")

        with patch.object(activities, "_pdf_has_text_layer", return_value=True) as has_text:
            self.assertEqual(
                await activities.detect_document_type("policy.pdf", "policy.pdf"), "native"
            )
        has_text.assert_called_once_with("policy.pdf")

        with patch.object(activities, "_pdf_has_text_layer", return_value=False):
            self.assertEqual(
                await activities.detect_document_type("scan.pdf", "scan.pdf"), "scanned"
            )

    async def test_embed_and_index_maps_chunks_then_rebuilds_lexical_index(self):
        chunks = [
            {"text": "first", "page_number": 1},
            {"text": "second", "page_number": 3},
        ]
        with (
            patch("services.retrieval.qdrant_store.delete_by_filename") as delete,
            patch("services.retrieval.qdrant_store.index_chunks") as index_chunks,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres") as rebuild,
        ):
            result = await activities.embed_and_index(chunks, "policy.pdf", "tenant-a")

        self.assertTrue(result)
        delete.assert_called_once_with("tenant-a", "policy.pdf")
        indexed = index_chunks.call_args.args[0]
        self.assertEqual(
            indexed,
            [
                Chunk(
                    make_chunk_id("tenant-a", "policy.pdf", 1, 0),
                    "first",
                    "policy.pdf",
                    1,
                    0,
                    "tenant-a",
                ),
                Chunk(
                    make_chunk_id("tenant-a", "policy.pdf", 3, 1),
                    "second",
                    "policy.pdf",
                    3,
                    1,
                    "tenant-a",
                ),
            ],
        )
        rebuild.assert_called_once_with()

    async def test_embed_and_index_handles_empty_document_consistently(self):
        with (
            patch("services.retrieval.qdrant_store.delete_by_filename") as delete,
            patch("services.retrieval.qdrant_store.index_chunks") as index_chunks,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres") as rebuild,
        ):
            result = await activities.embed_and_index([], "empty.txt", "tenant-a")

        self.assertTrue(result)
        delete.assert_called_once_with("tenant-a", "empty.txt")
        index_chunks.assert_called_once_with([])
        rebuild.assert_called_once_with()

    async def test_versioned_indexing_keeps_existing_qdrant_points_until_activation(self):
        identity = {
            "tenant_id": "tenant-a",
            "document_id": "document-a",
            "version_id": "version-a",
        }
        descriptor = {"storage_ref": "s3://ally/chunks.json.gz", "stage": "chunks", "item_count": 1}
        chunks = [{"page_number": 1, "text": "new chunk"}]
        with (
            patch.object(activities, "_load_artifact", new=AsyncMock(return_value=chunks)),
            patch("services.retrieval.qdrant_store.delete_by_filename") as delete,
            patch("services.retrieval.qdrant_store.index_chunks") as index_chunks,
            patch("services.retrieval.bm25_index.rebuild_index_from_postgres"),
        ):
            self.assertTrue(
                await activities.embed_and_index(descriptor, "policy.pdf", "tenant-a", identity)
            )

        delete.assert_not_called()
        indexed = index_chunks.call_args.args[0][0]
        self.assertEqual(indexed.version_id, "version-a")
        self.assertEqual(indexed.document_id, "document-a")
        self.assertEqual(indexed.id, make_chunk_id("tenant-a", "policy.pdf", 1, 0, "version-a"))


class IngestionWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_cancellation_records_cancelled_attempt(self):
        execute_activity = AsyncMock(
            side_effect=[True, True, TemporalCancelledError("cancelled"), True]
        )
        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=True),
            patch(
                "workflows.ingestion_workflow.workflow.info",
                return_value=SimpleNamespace(workflow_id="ingest-document-1"),
            ),
        ):
            with self.assertRaises(TemporalCancelledError):
                await IngestDocumentWorkflow().run(
                    "source.txt",
                    "source.txt",
                    "tenant-a",
                    "document-1",
                    "a" * 64,
                    "attempt-1",
                    "version-1",
                )

        finish_call = execute_activity.await_args_list[-1]
        self.assertIs(finish_call.args[0], activities.finish_ingestion_attempt)
        self.assertEqual(finish_call.kwargs["args"][4:], ["cancelled", "workflow_cancelled"])

    async def test_new_lifecycle_records_each_stage_and_terminal_status(self):
        extracted = {
            "storage_ref": "s3://ally/pages.json.gz",
            "stage": "pages",
            "item_count": 1,
        }
        chunks = {
            "storage_ref": "s3://ally/chunks.json.gz",
            "stage": "chunks",
            "item_count": 1,
        }
        execute_activity = AsyncMock(
            side_effect=[
                True,
                True,
                "native",
                True,
                extracted,
                True,
                chunks,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        )

        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=True),
            patch(
                "workflows.ingestion_workflow.workflow.info",
                return_value=SimpleNamespace(workflow_id="ingest-document-1"),
            ),
        ):
            result = await IngestDocumentWorkflow().run(
                "source.txt",
                "source.txt",
                "tenant-a",
                "document-1",
                "a" * 64,
                "attempt-1",
                "version-1",
            )

        stage_calls = [
            item.kwargs["args"][-1]
            for item in execute_activity.await_args_list
            if item.args[0] is activities.update_ingestion_stage
        ]
        self.assertEqual(
            stage_calls, ["detecting", "extracting", "chunking", "persisting", "indexing"]
        )
        finish_call = next(
            item
            for item in execute_activity.await_args_list
            if item.args[0] is activities.finish_ingestion_attempt
        )
        self.assertEqual(finish_call.kwargs["args"][4:], ["indexed", None])
        artifact_identity = {
            "tenant_id": "tenant-a",
            "document_id": "document-1",
            "version_id": "version-1",
        }
        chunk_call = next(
            item
            for item in execute_activity.await_args_list
            if item.args[0] is activities.chunk_pages
        )
        self.assertEqual(chunk_call.kwargs["args"], [extracted, 1500, 200, artifact_identity])
        persist_call = next(
            item
            for item in execute_activity.await_args_list
            if item.args[0] is activities.persist_chunks
        )
        self.assertEqual(
            persist_call.kwargs["args"], [chunks, "source.txt", "tenant-a", artifact_identity]
        )
        cleanup_call = next(
            item
            for item in execute_activity.await_args_list
            if item.args[0] is activities.cleanup_ingestion_artifacts
        )
        self.assertEqual(cleanup_call.kwargs["args"], [artifact_identity])
        activation_call = next(
            item
            for item in execute_activity.await_args_list
            if item.args[0] is activities.activate_ingestion_version
        )
        self.assertEqual(activation_call.kwargs["args"], ["document-1", "version-1", "tenant-a"])
        self.assertEqual(result["document_id"], "document-1")

    async def test_scanned_document_uses_version_derived_textract_request_token(self):
        execute_activity = AsyncMock(
            side_effect=[True, "scanned", [{"page_number": 1}], [], True, True]
        )
        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=False),
            patch(
                "workflows.ingestion_workflow.workflow.info",
                return_value=SimpleNamespace(workflow_id="workflow-1"),
            ),
        ):
            await IngestDocumentWorkflow().run(
                "source.pdf",
                "source.pdf",
                "tenant-a",
                "document-1",
                "a" * 64,
                "attempt-1",
                "version-1",
            )

        textract_call = execute_activity.await_args_list[2]
        self.assertIs(textract_call.args[0], activities.extract_with_textract)
        self.assertEqual(
            textract_call.kwargs["args"],
            ["source.pdf", hashlib.sha256(b"version-1").hexdigest()],
        )

    async def test_new_ingestion_registers_document_attempt_and_version_first(self):
        extracted = [{"page_number": 1, "text": "content"}]
        chunks = [{"page_number": 1, "text": "chunk"}]
        execute_activity = AsyncMock(side_effect=[True, "native", extracted, chunks, True, True])

        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=False),
            patch(
                "workflows.ingestion_workflow.workflow.info",
                return_value=SimpleNamespace(workflow_id="ingest-document-1"),
            ),
        ):
            await IngestDocumentWorkflow().run(
                "s3://ally-ingestion/tenants/acme/documents/doc/source.txt",
                "source.txt",
                "acme",
                "document-1",
                "a" * 64,
                "attempt-1",
                "version-1",
            )

        calls = execute_activity.await_args_list
        self.assertIs(calls[0].args[0], activities.register_ingestion)
        self.assertEqual(
            calls[0].kwargs["args"],
            [
                "document-1",
                "attempt-1",
                "version-1",
                "s3://ally-ingestion/tenants/acme/documents/doc/source.txt",
                "a" * 64,
                "source.txt",
                "acme",
                "ingest-document-1",
            ],
        )
        self.assertIs(calls[1].args[0], activities.detect_document_type)

    async def test_native_document_runs_persist_then_index_pipeline(self):
        extracted = [{"page_number": 1, "text": "content"}]
        chunks = [{"page_number": 1, "text": "chunk"}]
        execute_activity = AsyncMock(side_effect=["native", extracted, chunks, True, True])

        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=False),
        ):
            result = await IngestDocumentWorkflow().run("/tmp/policy.txt", "policy.txt", "tenant-a")

        self.assertEqual(
            [invocation.args[0] for invocation in execute_activity.await_args_list],
            [
                activities.detect_document_type,
                activities.extract_with_unstructured,
                activities.chunk_pages,
                activities.persist_chunks,
                activities.embed_and_index,
            ],
        )
        self.assertEqual(
            execute_activity.await_args_list[3].kwargs["args"], [chunks, "policy.txt", "tenant-a"]
        )
        self.assertEqual(
            execute_activity.await_args_list[4].kwargs["args"], [chunks, "policy.txt", "tenant-a"]
        )
        self.assertEqual(
            result,
            {
                "filename": "policy.txt",
                "tenant_id": "tenant-a",
                "num_chunks": 1,
                "stored": True,
                "indexed": True,
            },
        )

    async def test_scanned_document_selects_textract_before_indexing(self):
        execute_activity = AsyncMock(side_effect=["scanned", [{"page_number": 1}], [], True, True])

        with (
            patch("workflows.ingestion_workflow.workflow.execute_activity", execute_activity),
            patch("workflows.ingestion_workflow.workflow.patched", return_value=False),
        ):
            result = await IngestDocumentWorkflow().run("/tmp/scan.pdf", "scan.pdf", "tenant-a")

        self.assertIs(execute_activity.await_args_list[1].args[0], activities.extract_with_textract)
        self.assertEqual(result["num_chunks"], 0)
        self.assertTrue(result["indexed"])


class IngestionLifecyclePersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_stage_update_is_tenant_bound_and_terminal_write_is_idempotent(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        with patch("psycopg2.connect", return_value=connection):
            self.assertTrue(
                await activities.update_ingestion_stage(
                    "document-1", "attempt-1", "tenant-1", "extracting"
                )
            )
            self.assertTrue(
                await activities.finish_ingestion_attempt(
                    "document-1", "attempt-1", "version-1", "tenant-1", "indexed"
                )
            )
            cursor.rowcount = 0
            cursor.fetchone.return_value = (1,)
            self.assertTrue(
                await activities.finish_ingestion_attempt(
                    "document-1", "attempt-1", "version-1", "tenant-1", "indexed"
                )
            )

        self.assertEqual(
            cursor.execute.call_args_list[0].args[1],
            ("extracting", "attempt-1", "document-1", "tenant-1"),
        )
        self.assertEqual(connection.commit.call_count, 3)


class WorkerConfigurationTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_registers_new_indexing_activity(self):
        client = object()
        worker_instance = Mock()
        worker_instance.run = AsyncMock()
        qdrant_owner = Mock()

        with (
            patch.object(worker.Client, "connect", new=AsyncMock(return_value=client)) as connect,
            patch.object(worker, "Worker", return_value=worker_instance) as worker_type,
            patch.object(worker, "QdrantStore", return_value=qdrant_owner),
            patch.object(worker.activities, "set_qdrant_store") as set_owner,
        ):
            await worker.main()

        connect.assert_awaited_once_with(worker.runtime_settings.application.temporal_address)
        self.assertEqual(worker_type.call_args.args[0], client)
        worker_configuration = worker_type.call_args.kwargs
        self.assertEqual(worker_configuration["task_queue"], "ingestion-task-queue")
        self.assertIn(activities.register_ingestion, worker_configuration["activities"])
        self.assertIn(activities.update_ingestion_stage, worker_configuration["activities"])
        self.assertIn(activities.finish_ingestion_attempt, worker_configuration["activities"])
        self.assertIn(activities.embed_and_index, worker_configuration["activities"])
        worker_instance.run.assert_awaited_once_with()
        self.assertEqual(set_owner.call_args_list, [call(qdrant_owner), call(None)])
        qdrant_owner.close.assert_called_once_with()

    async def test_worker_closes_qdrant_owner_when_run_fails(self):
        client = object()
        worker_instance = Mock()
        worker_instance.run = AsyncMock(side_effect=RuntimeError("worker stopped"))
        qdrant_owner = Mock()

        with (
            patch.object(worker.Client, "connect", new=AsyncMock(return_value=client)),
            patch.object(worker, "Worker", return_value=worker_instance),
            patch.object(worker, "QdrantStore", return_value=qdrant_owner),
            patch.object(worker.activities, "set_qdrant_store") as set_owner,
        ):
            with self.assertRaisesRegex(RuntimeError, "worker stopped"):
                await worker.main()

        self.assertEqual(set_owner.call_args_list, [call(qdrant_owner), call(None)])
        qdrant_owner.close.assert_called_once_with()


class BackfillIndexTest(unittest.TestCase):
    def test_load_chunks_maps_postgres_rows_and_closes_connection(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("tenant-a", "a.pdf", 1, 0, "first"),
            ("tenant-a", "a.pdf", 2, 1, "second"),
        ]

        with patch.object(backfill_index.psycopg2, "connect", return_value=connection) as connect:
            chunks = backfill_index.load_chunks_from_postgres()

        connect.assert_called_once_with(backfill_index.POSTGRES_DSN)
        self.assertEqual(
            chunks,
            [
                Chunk(make_chunk_id("tenant-a", "a.pdf", 1, 0), "first", "a.pdf", 1, 0, "tenant-a"),
                Chunk(
                    make_chunk_id("tenant-a", "a.pdf", 2, 1), "second", "a.pdf", 2, 1, "tenant-a"
                ),
            ],
        )
        connection.close.assert_called_once_with()

    def test_load_chunks_closes_connection_when_query_fails(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = RuntimeError("query failed")

        with patch.object(backfill_index.psycopg2, "connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                backfill_index.load_chunks_from_postgres()

        connection.close.assert_called_once_with()

    def test_main_indexes_loaded_chunks_and_rebuilds_bm25(self):
        chunks = [Chunk("one", "text", "a.pdf", 1, 0, "tenant-a")]
        with (
            patch.object(backfill_index, "load_chunks_from_postgres", return_value=chunks),
            patch.object(backfill_index, "index_chunks") as index_chunks,
            patch.object(backfill_index, "rebuild_index_from_postgres") as rebuild,
            patch("builtins.print") as print_message,
        ):
            backfill_index.main()

        index_chunks.assert_called_once_with(chunks, store=unittest.mock.ANY)
        rebuild.assert_called_once_with(force=True)
        print_message.assert_any_call("Indexed 1 chunks into Qdrant and rebuilt the BM25 index.")

    def test_main_still_rebuilds_bm25_when_postgres_has_no_chunks(self):
        # An empty corpus is a valid state (e.g. right after infra
        # bootstrap, before anything has been ingested) -- main() must
        # still call index_chunks([]) and force a rebuild rather than
        # short-circuiting, or a corpus that goes from N chunks to 0
        # (every document deleted) would leave stale entries behind.
        with (
            patch.object(backfill_index, "load_chunks_from_postgres", return_value=[]),
            patch.object(backfill_index, "index_chunks") as index_chunks,
            patch.object(backfill_index, "rebuild_index_from_postgres") as rebuild,
            patch("builtins.print") as print_message,
        ):
            backfill_index.main()

        index_chunks.assert_called_once_with([], store=unittest.mock.ANY)
        rebuild.assert_called_once_with(force=True)
        print_message.assert_any_call("Indexed 0 chunks into Qdrant and rebuilt the BM25 index.")


if __name__ == "__main__":
    unittest.main()
