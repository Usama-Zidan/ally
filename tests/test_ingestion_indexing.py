import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from services.retrieval import backfill_index
from services.retrieval.qdrant_store import Chunk, make_chunk_id
from workflows import activities, worker
from workflows.ingestion_workflow import IngestDocumentWorkflow


class PersistChunksTest(unittest.IsolatedAsyncioTestCase):
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
    async def test_detect_document_type_routes_images_and_pdf_text_layers(self):
        self.assertEqual(await activities.detect_document_type("scan.png", "scan.PNG"), "scanned")

        with patch.object(activities, "_pdf_has_text_layer", return_value=True) as has_text:
            self.assertEqual(await activities.detect_document_type("policy.pdf", "policy.pdf"), "native")
        has_text.assert_called_once_with("policy.pdf")

        with patch.object(activities, "_pdf_has_text_layer", return_value=False):
            self.assertEqual(await activities.detect_document_type("scan.pdf", "scan.pdf"), "scanned")

    async def test_embed_and_index_maps_chunks_then_rebuilds_lexical_index(self):
        chunks = [
            {"text": "first", "page_number": 1},
            {"text": "second", "page_number": 3},
        ]
        with patch(
            "services.retrieval.qdrant_store.delete_by_filename"
        ) as delete, patch(
            "services.retrieval.qdrant_store.index_chunks"
        ) as index_chunks, patch(
            "services.retrieval.bm25_index.rebuild_index_from_postgres"
        ) as rebuild:
            result = await activities.embed_and_index(chunks, "policy.pdf", "tenant-a")

        self.assertTrue(result)
        delete.assert_called_once_with("tenant-a", "policy.pdf")
        indexed = index_chunks.call_args.args[0]
        self.assertEqual(
            indexed,
            [
                Chunk(make_chunk_id("tenant-a", "policy.pdf", 1, 0), "first", "policy.pdf", 1, 0, "tenant-a"),
                Chunk(make_chunk_id("tenant-a", "policy.pdf", 3, 1), "second", "policy.pdf", 3, 1, "tenant-a"),
            ],
        )
        rebuild.assert_called_once_with()

    async def test_embed_and_index_handles_empty_document_consistently(self):
        with patch(
            "services.retrieval.qdrant_store.delete_by_filename"
        ) as delete, patch(
            "services.retrieval.qdrant_store.index_chunks"
        ) as index_chunks, patch(
            "services.retrieval.bm25_index.rebuild_index_from_postgres"
        ) as rebuild:
            result = await activities.embed_and_index([], "empty.txt", "tenant-a")

        self.assertTrue(result)
        delete.assert_called_once_with("tenant-a", "empty.txt")
        index_chunks.assert_called_once_with([])
        rebuild.assert_called_once_with()


class IngestionWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_document_runs_persist_then_index_pipeline(self):
        extracted = [{"page_number": 1, "text": "content"}]
        chunks = [{"page_number": 1, "text": "chunk"}]
        execute_activity = AsyncMock(side_effect=["native", extracted, chunks, True, True])

        with patch(
            "workflows.ingestion_workflow.workflow.execute_activity", execute_activity
        ):
            result = await IngestDocumentWorkflow().run(
                "/tmp/policy.txt", "policy.txt", "tenant-a"
            )

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
        self.assertEqual(execute_activity.await_args_list[3].kwargs["args"], [chunks, "policy.txt", "tenant-a"])
        self.assertEqual(execute_activity.await_args_list[4].kwargs["args"], [chunks, "policy.txt", "tenant-a"])
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
        execute_activity = AsyncMock(
            side_effect=["scanned", [{"page_number": 1}], [], True, True]
        )

        with patch(
            "workflows.ingestion_workflow.workflow.execute_activity", execute_activity
        ):
            result = await IngestDocumentWorkflow().run(
                "/tmp/scan.pdf", "scan.pdf", "tenant-a"
            )

        self.assertIs(execute_activity.await_args_list[1].args[0], activities.extract_with_textract)
        self.assertEqual(result["num_chunks"], 0)
        self.assertTrue(result["indexed"])


class WorkerConfigurationTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_registers_new_indexing_activity(self):
        client = object()
        worker_instance = Mock()
        worker_instance.run = AsyncMock()

        with patch.object(worker.Client, "connect", new=AsyncMock(return_value=client)) as connect, patch.object(
            worker, "Worker", return_value=worker_instance
        ) as worker_type:
            await worker.main()

        connect.assert_awaited_once_with(worker.TEMPORAL_ADDRESS)
        self.assertEqual(worker_type.call_args.args[0], client)
        worker_configuration = worker_type.call_args.kwargs
        self.assertEqual(worker_configuration["task_queue"], "ingestion-task-queue")
        self.assertIn(activities.embed_and_index, worker_configuration["activities"])
        worker_instance.run.assert_awaited_once_with()


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
                Chunk(make_chunk_id("tenant-a", "a.pdf", 2, 1), "second", "a.pdf", 2, 1, "tenant-a"),
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
        with patch.object(
            backfill_index, "load_chunks_from_postgres", return_value=chunks
        ), patch.object(backfill_index, "index_chunks") as index_chunks, patch.object(
            backfill_index, "rebuild_index_from_postgres"
        ) as rebuild, patch("builtins.print") as print_message:
            backfill_index.main()

        index_chunks.assert_called_once_with(chunks)
        rebuild.assert_called_once_with(force=True)
        print_message.assert_any_call(
            "Indexed 1 chunks into Qdrant and rebuilt the BM25 index."
        )


if __name__ == "__main__":
    unittest.main()
