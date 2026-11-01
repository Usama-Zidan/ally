"""App factories build isolated graph, tool, bridge and storage owners."""

from __future__ import annotations

import asyncio
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import main
from services.ingestion.upload import S3UploadStorage
from services.runtime.settings import load_runtime_settings
from services.runtime.task_drain import DrainableTasks


class ApiCompositionTest(unittest.TestCase):
    def test_upload_composition_selects_configured_s3_adapter(self) -> None:
        settings = load_runtime_settings(
            {
                "INGESTION_STORAGE_BACKEND": "s3",
                "INGESTION_S3_BUCKET": "ally-ingestion",
            }
        )
        app = main.create_app(settings)
        app.state.temporal_client = Mock()

        use_case = main._build_upload_use_case(app)

        self.assertIsInstance(use_case.storage, S3UploadStorage)
        self.assertEqual(use_case.storage.bucket, "ally-ingestion")

    def test_partial_startup_failure_closes_every_acquired_owner(self) -> None:
        with TemporaryDirectory() as directory:
            settings = load_runtime_settings({"UPLOAD_DIR": f"{directory}/uploads"})
            app = main.create_app(settings)
            qdrant = Mock(close=Mock())
            history = Mock(close=Mock())
            security = Mock(close=AsyncMock())
            bridge = Mock(close=AsyncMock())
            with (
                patch.object(main, "QdrantStore", return_value=qdrant),
                patch.object(main, "MongoHistoryStore", return_value=history),
                patch.object(main, "WebSocketSecurity", return_value=security),
                patch.object(main, "PCAgentBridge", return_value=bridge),
                patch.object(
                    main,
                    "build_default_registry",
                    side_effect=RuntimeError("registry construction failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "registry construction failed"),
            ):
                with TestClient(app):
                    self.fail("startup should fail before serving requests")

            qdrant.close.assert_called_once_with()
            history.close.assert_called_once_with()
            security.close.assert_awaited_once_with()
            bridge.close.assert_awaited_once_with()

    def test_apps_own_independent_agent_and_storage_dependencies(self) -> None:
        with TemporaryDirectory() as directory:
            first_settings = load_runtime_settings(
                {"PROJECT_NAME": "First", "UPLOAD_DIR": f"{directory}/first"}
            )
            second_settings = load_runtime_settings(
                {
                    "PROJECT_NAME": "Second",
                    "UPLOAD_DIR": f"{directory}/second",
                    "QWEN_MODEL_NAME": "isolated-model",
                    "EMBEDDING_MODEL": "isolated-embedding",
                    "RERANKER_MODEL": "isolated-reranker",
                }
            )
            first = main.create_app(first_settings)
            second = main.create_app(second_settings)

            with patch.object(main, "_start_optional_services", new=AsyncMock()):
                with TestClient(first) as first_client, TestClient(second) as second_client:
                    first_state = first.state
                    second_state = second.state
                    self.assertEqual(first.title, "First")
                    self.assertEqual(second.title, "Second")
                    self.assertIsNot(first_state.agent_graph, second_state.agent_graph)
                    self.assertIsNot(first_state.tool_registry, second_state.tool_registry)
                    self.assertIsNot(first_state.pc_agent_bridge, second_state.pc_agent_bridge)
                    self.assertIsNot(first_state.qdrant_store, second_state.qdrant_store)
                    self.assertIsNot(first_state.llm_gateway, second_state.llm_gateway)
                    self.assertIsNot(first_state.retrieval_work, second_state.retrieval_work)
                    self.assertIsNot(
                        first_state.llm_gateway._get_breaker(),
                        second_state.llm_gateway._get_breaker(),
                    )
                    self.assertNotEqual(
                        first_state.llm_gateway._primary_model,
                        second_state.llm_gateway._primary_model,
                    )
                    self.assertNotEqual(
                        first_state.retrieval_models.embedding_model,
                        second_state.retrieval_models.embedding_model,
                    )
                    self.assertNotEqual(
                        first_state.retrieval_models.reranker_model,
                        second_state.retrieval_models.reranker_model,
                    )
                    self.assertIs(
                        first_state.tool_registry.get("local_pc").bridge,
                        first_state.pc_agent_bridge,
                    )
                    self.assertIs(
                        second_state.tool_registry.get("local_pc").bridge,
                        second_state.pc_agent_bridge,
                    )
                    self.assertEqual(first_client.get("/health").json(), {"status": "ok"})
                    self.assertEqual(second_client.get("/health").json(), {"status": "ok"})

            self.assertTrue(first_state.pc_agent_bridge._closing)
            self.assertTrue(second_state.pc_agent_bridge._closing)
            self.assertTrue(first_state.retrieval_work.closing)
            self.assertTrue(second_state.retrieval_work.closing)
            self.assertTrue(first_state.qdrant_store._closed)
            self.assertTrue(second_state.qdrant_store._closed)
            self.assertIsNone(first_state.qdrant_store._client)
            self.assertIsNone(second_state.qdrant_store._client)


class ApiRetrievalDrainTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_chat_waiter_leaves_search_owned_until_shutdown_drain(self) -> None:
        app = FastAPI()
        app.state.runtime_settings = load_runtime_settings({})
        app.state.history_store = Mock()
        app.state.retrieval_work = DrainableTasks()
        use_case = main._build_chat_use_case(app)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_retrieval(*args, **kwargs):
            entered.set()
            await release.wait()
            return []

        with patch.object(main, "retrieve_async", new=slow_retrieval):
            request = asyncio.create_task(use_case._dependencies.retrieve("query", "tenant-a"))
            await asyncio.wait_for(entered.wait(), timeout=1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request

            shutdown = asyncio.create_task(app.state.retrieval_work.close())
            await asyncio.sleep(0)
            self.assertFalse(shutdown.done())
            release.set()
            await asyncio.wait_for(shutdown, timeout=1)

        self.assertTrue(app.state.retrieval_work.closing)
        self.assertEqual(app.state.retrieval_work.active, 0)

    def test_configured_reranker_is_selected_and_warmed(self) -> None:
        with TemporaryDirectory() as directory:
            settings = load_runtime_settings(
                {
                    "UPLOAD_DIR": f"{directory}/uploads",
                    "EMBEDDING_MODEL": "app-embedding",
                    "RERANKER_MODEL": "app-reranker",
                }
            )
            app = main.create_app(settings)
            with (
                patch.object(
                    main.TemporalClient,
                    "connect",
                    new=AsyncMock(side_effect=ConnectionError("offline")),
                ),
                patch.object(main, "warm_reranker") as warm,
            ):
                with TestClient(app):
                    self.assertEqual(app.state.retrieval_models.embedding_model, "app-embedding")
                    self.assertEqual(app.state.retrieval_models.reranker_model, "app-reranker")

            warm.assert_called_once_with("app-reranker")


if __name__ == "__main__":
    unittest.main()
