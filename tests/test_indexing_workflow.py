import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from services.retrieval import backfill_index
from workflows import activities
from workflows import ingestion_workflow
from workflows import worker as worker_module


class IndexingActivityTest(unittest.IsolatedAsyncioTestCase):
    async def test_persist_chunks_replaces_document_rows_before_upserting(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        chunks = [
            {"page_number": 1, "text": "first"},
            {"page_number": 3, "text": "second"},
        ]

        with patch("psycopg2.connect", return_value=connection) as connect:
            result = await activities.persist_chunks(chunks, "policy.pdf")

        self.assertTrue(result)
        connect.assert_called_once_with(activities.POSTGRES_DSN)
        self.assertEqual(
            cursor.execute.call_args_list[0],
            call("DELETE FROM document_chunks WHERE filename = %s;", ("policy.pdf",)),
        )
        self.assertEqual(cursor.execute.call_args_list[1].args[1], ("policy.pdf", 1, 0, "first"))
        self.assertEqual(cursor.execute.call_args_list[2].args[1], ("policy.pdf", 3, 1, "second"))
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    async def test_persist_chunks_closes_connection_when_database_write_fails(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = RuntimeError("database unavailable")

        with patch("psycopg2.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await activities.persist_chunks([], "policy.pdf")

        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    async def test_embed_and_index_builds_deterministic_chunks_then_refreshes_bm25(self):
        chunks = [
            {"page_number": 1, "text": "first"},
            {"page_number": 2, "text": "second"},
        ]
        with patch(
            "services.retrieval.qdrant_store.make_chunk_id",
            side_effect=["id-one", "id-two"],
        ) as make_id, patch(
            "services.retrieval.qdrant_store.index_chunks"
        ) as index_chunks, patch(
            "services.retrieval.bm25_index.rebuild_index_from_postgres"
        ) as rebuild:
            result = await activities.embed_and_index(chunks, "policy.pdf")

        self.assertTrue(result)
        self.assertEqual(
            make_id.call_args_list,
            [call("policy.pdf", 1, 0), call("policy.pdf", 2, 1)],
        )
        indexed = index_chunks.call_args.args[0]
        self.assertEqual([chunk.id for chunk in indexed], ["id-one", "id-two"])
        self.assertEqual([chunk.text for chunk in indexed], ["first", "second"])
        self.assertEqual([chunk.chunk_index for chunk in indexed], [0, 1])
        rebuild.assert_called_once_with()


class BackfillTest(unittest.TestCase):
    def test_load_chunks_maps_ordered_postgres_rows_and_closes_connection(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("hr.pdf", 1, 0, "annual leave"),
            ("it.pdf", 2, 4, "security review"),
        ]

        with patch(
            "services.retrieval.backfill_index.psycopg2.connect",
            return_value=connection,
        ), patch(
            "services.retrieval.backfill_index.make_chunk_id",
            side_effect=["id-one", "id-two"],
        ):
            chunks = backfill_index.load_chunks_from_postgres()

        cursor.execute.assert_called_once_with(
            "SELECT filename, page_number, chunk_index, text "
            "FROM document_chunks ORDER BY filename, page_number, chunk_index;"
        )
        connection.close.assert_called_once_with()
        self.assertEqual([chunk.id for chunk in chunks], ["id-one", "id-two"])
        self.assertEqual(chunks[1].filename, "it.pdf")
        self.assertEqual(chunks[1].chunk_index, 4)

    def test_backfill_main_updates_dense_and_lexical_indexes_even_when_empty(self):
        with patch.object(backfill_index, "load_chunks_from_postgres", return_value=[]), patch.object(
            backfill_index, "index_chunks"
        ) as index_chunks, patch.object(
            backfill_index, "rebuild_index_from_postgres"
        ) as rebuild, patch("builtins.print") as output:
            backfill_index.main()

        index_chunks.assert_called_once_with([])
        rebuild.assert_called_once_with()
        output.assert_any_call("Indexed 0 chunks into Qdrant and rebuilt the BM25 index.")


class IngestionWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_indexes_only_after_chunks_are_persisted(self):
        extracted = [{"page_number": 1, "text": "document"}]
        chunks = [{"page_number": 1, "text": "chunk"}]
        execute_activity = AsyncMock(
            side_effect=["native", extracted, chunks, True, True]
        )

        with patch.object(ingestion_workflow.workflow, "execute_activity", execute_activity):
            result = await ingestion_workflow.IngestDocumentWorkflow().run(
                "/tmp/policy.pdf", "policy.pdf"
            )

        self.assertEqual(
            result,
            {"filename": "policy.pdf", "num_chunks": 1, "stored": True, "indexed": True},
        )
        calls = execute_activity.await_args_list
        self.assertEqual([item.args[0] for item in calls], [
            activities.detect_document_type,
            activities.extract_with_unstructured,
            activities.chunk_pages,
            activities.persist_chunks,
            activities.embed_and_index,
        ])
        self.assertEqual(calls[4].kwargs["args"], [chunks, "policy.pdf"])
        self.assertEqual(calls[4].kwargs["start_to_close_timeout"], timedelta(minutes=10))

    async def test_worker_registers_new_indexing_activity(self):
        client = object()
        worker = SimpleNamespace(run=AsyncMock())
        with patch.object(
            worker_module.Client, "connect", new=AsyncMock(return_value=client)
        ), patch.object(worker_module, "Worker", return_value=worker) as worker_factory:
            await worker_module.main()

        kwargs = worker_factory.call_args.kwargs
        self.assertIs(worker_factory.call_args.args[0], client)
        self.assertEqual(kwargs["task_queue"], "ingestion-task-queue")
        self.assertIn(activities.embed_and_index, kwargs["activities"])
        worker.run.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
