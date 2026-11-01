import importlib
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from pydantic import SecretStr
from qdrant_client.models import Distance

from services.retrieval import (
    embeddings,
    mmr,
    qdrant_store,
    reranker,
    sentence_transformers_loader,
)
from services.retrieval.bm25_index import (
    BM25Index,
    _resolve_index_path,
    rebuild_index_from_postgres,
)
from services.retrieval.fusion import reciprocal_rank_fusion
from services.retrieval.qdrant_store import Chunk
from services.runtime.settings import load_runtime_settings


class _BlockedPyarrowFinder:
    """Import finder that reproduces the Windows Application Control block.

    A real policy stops the DLL at load time, so pyarrow raises ImportError
    from inside pyarrow/dataset.py. Raising from find_spec reproduces the
    same failure from the import machinery's point of view.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname in sentence_transformers_loader.STUBBED_MODULES:
            raise ImportError("DLL load failed while importing _dataset")
        return None


class BM25IndexTest(unittest.TestCase):
    def test_index_directory_path_gets_default_filename(self):
        self.assertEqual(_resolve_index_path("."), Path("bm25_index.pkl"))

    def test_search_is_case_insensitive_and_tokenizes_punctuation(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "match",
                    "text": "Annual-leave policy",
                    "filename": "hr.pdf",
                    "tenant_id": "tenant-a",
                },
                {
                    "id": "other",
                    "text": "Security handbook",
                    "filename": "it.pdf",
                    "tenant_id": "tenant-a",
                },
            ]
        )

        self.assertEqual(
            [item["id"] for item in index.search("ANNUAL, leave!", "tenant-a")], ["match"]
        )

    def test_search_uses_document_id_as_stable_tie_breaker(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "b",
                    "text": "same term",
                    "tenant_id": "tenant-a",
                },
                {
                    "id": "a",
                    "text": "same term",
                    "tenant_id": "tenant-a",
                },
            ]
        )

        self.assertEqual([item["id"] for item in index.search("term", "tenant-a")], ["a", "b"])

    def test_search_rejects_non_positive_limit_and_ignores_blank_query(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "one",
                    "text": "annual leave",
                    "tenant_id": "tenant-a",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "top_k"):
            index.search("annual", "tenant-a", top_k=0)
        self.assertEqual(index.search("   ", "tenant-a"), [])

    def test_search_applies_all_metadata_filter_fields(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "page-1",
                    "text": "annual leave",
                    "filename": "hr.pdf",
                    "page_number": 1,
                    "tenant_id": "tenant-a",
                },
                {
                    "id": "page-2",
                    "text": "annual leave",
                    "filename": "hr.pdf",
                    "page_number": 2,
                    "tenant_id": "tenant-a",
                },
            ]
        )

        results = index.search(
            "annual leave",
            "tenant-a",
            metadata_filter={"filename": "hr.pdf", "page_number": 2},
        )

        self.assertEqual([item["id"] for item in results], ["page-2"])

    def test_rebuild_index_reads_ordered_rows_and_closes_connection(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("tenant-a", "policy.pdf", 2, 3, "annual leave", None)]

        with (
            patch(
                "services.retrieval.bm25_index.psycopg2.connect",
                return_value=connection,
            ) as connect,
            patch.object(BM25Index, "save") as save,
            patch("services.retrieval.bm25_index.POSTGRES_DSN", "postgresql://override/test"),
            patch.dict("os.environ", {"POSTGRES_DSN": "postgresql://later-env/ignored"}),
        ):
            index = rebuild_index_from_postgres(force=True)

        connect.assert_called_once_with("postgresql://override/test")
        cursor.execute.assert_called_once()
        self.assertIn(
            "ORDER BY tenant_id, filename, page_number, chunk_index",
            cursor.execute.call_args.args[0],
        )
        if index is None:
            self.fail("rebuild_index_from_postgres() returned None")
        self.assertEqual(index.doc_metadata[0]["filename"], "policy.pdf")
        self.assertEqual(index.doc_metadata[0]["chunk_index"], 3)
        save.assert_called_once_with()
        connection.close.assert_called_once_with()


class FusionTest(unittest.TestCase):
    def test_fusion_accumulates_reciprocal_rank_scores_and_preserves_first_payload(self):
        fused = reciprocal_rank_fusion(
            [
                [
                    {
                        "id": "shared",
                        "text": "dense",
                        "tenant_id": "tenant-a",
                    },
                    {
                        "id": "dense",
                        "text": "only",
                        "tenant_id": "tenant-a",
                    },
                ],
                [
                    {
                        "id": "shared",
                        "text": "lexical",
                        "tenant_id": "tenant-a",
                    },
                ],
            ],
            k=10,
        )

        self.assertEqual([item["id"] for item in fused], ["shared", "dense"])
        self.assertEqual(fused[0]["text"], "dense")
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 11)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 12)

    def test_fusion_supports_custom_identifier_key_and_empty_lists(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        result = reciprocal_rank_fusion([[{"chunk": 7, "text": "value"}]], id_key="chunk")
        self.assertEqual(result[0]["chunk"], 7)


class EmbeddingTest(unittest.TestCase):
    def test_embed_texts_requests_normalized_batch_embeddings(self):
        model = Mock()
        model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

        with patch.object(embeddings, "_get_model", return_value=model):
            result = embeddings.embed_texts(["first", "second"], model_name="test-model")

        self.assertEqual(result, [[1.0, 0.0], [0.0, 1.0]])
        model.encode.assert_called_once_with(
            ["first", "second"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def test_embed_query_adds_bge_retrieval_instruction(self):
        model = Mock()
        model.encode.return_value = np.array([0.25, 0.75])

        with patch.object(embeddings, "_get_model", return_value=model):
            result = embeddings.embed_query("annual leave", model_name="test-model")

        self.assertEqual(result, [0.25, 0.75])
        model.encode.assert_called_once_with(
            "Represent this sentence for searching relevant passages: annual leave",
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def test_concurrent_cold_calls_construct_one_embedding_model(self):
        barrier = Barrier(2)
        model = Mock()
        constructor = Mock(side_effect=lambda _name: (time.sleep(0.05), model)[1])
        embeddings._load_model.cache_clear()

        def load_model():
            barrier.wait(timeout=2)
            return embeddings._get_model("concurrent-model")

        with (
            patch.object(
                embeddings,
                "load_sentence_transformers",
                return_value=SimpleNamespace(SentenceTransformer=constructor),
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _index: load_model(), range(2)))

        self.assertIs(results[0], model)
        self.assertIs(results[1], model)
        constructor.assert_called_once_with("concurrent-model")

    def test_concurrent_apps_with_different_models_keep_their_selected_weights(self):
        barrier = Barrier(2)
        models = {"app-a-model": Mock(), "app-b-model": Mock()}
        constructor = Mock(side_effect=models.__getitem__)
        embeddings._load_model.cache_clear()

        def load_for_app(model_name: str):
            barrier.wait(timeout=2)
            return embeddings._get_model(model_name)

        try:
            with (
                patch.object(
                    embeddings,
                    "load_sentence_transformers",
                    return_value=SimpleNamespace(SentenceTransformer=constructor),
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                app_a = executor.submit(load_for_app, "app-a-model")
                app_b = executor.submit(load_for_app, "app-b-model")

                self.assertIs(app_a.result(timeout=2), models["app-a-model"])
                self.assertIs(app_b.result(timeout=2), models["app-b-model"])

            self.assertCountEqual(
                [call.args[0] for call in constructor.call_args_list],
                ["app-a-model", "app-b-model"],
            )
            self.assertEqual(embeddings._load_model.cache_info().maxsize, 1)
        finally:
            embeddings._load_model.cache_clear()


class MMRTest(unittest.TestCase):
    def test_mmr_balances_relevance_with_diversity(self):
        candidates = [
            {
                "id": "best",
                "text": "best",
                "tenant_id": "tenant-a",
            },
            {
                "id": "duplicate",
                "text": "duplicate",
                "tenant_id": "tenant-a",
            },
            {
                "id": "diverse",
                "text": "diverse",
                "tenant_id": "tenant-a",
            },
        ]
        candidate_vectors = [[0.9, 0.1], [0.8, 0.2], [0.0, 1.0]]

        with patch.object(mmr, "embed_texts", return_value=candidate_vectors):
            selected = mmr.mmr_select([1.0, 0.0], candidates, top_k=2, lambda_param=0.2)

        self.assertEqual([item["id"] for item in selected], ["best", "diverse"])

    def test_mmr_returns_all_available_candidates_when_limit_is_larger(self):
        candidates = [
            {
                "id": "one",
                "text": "one",
                "tenant_id": "tenant-a",
            },
            {
                "id": "two",
                "text": "two",
                "tenant_id": "tenant-a",
            },
        ]
        with patch.object(mmr, "embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]):
            selected = mmr.mmr_select([1.0, 0.0], candidates, top_k=5)

        self.assertEqual(len(selected), 2)

    def test_mmr_rejects_mismatched_embedding_dimensions(self):
        with patch.object(mmr, "embed_texts", return_value=[[1.0, 0.0, 0.0]]):
            with self.assertRaisesRegex(ValueError, "matching dimensions"):
                mmr.mmr_select(
                    [1.0, 0.0],
                    [
                        {
                            "id": "one",
                            "text": "one",
                            "tenant_id": "tenant-a",
                        },
                    ],
                )

    def test_mmr_short_circuits_without_embedding_empty_candidates(self):
        with patch.object(mmr, "embed_texts") as embed:
            self.assertEqual(mmr.mmr_select([1.0], []), [])
            self.assertEqual(mmr.mmr_select([1.0], [{"text": "one"}], top_k=0), [])
        embed.assert_not_called()


class RerankerTest(unittest.TestCase):
    def test_rerank_scores_sorts_and_limits_candidates(self):
        model = Mock()
        model.predict.return_value = [0.1, 0.9, 0.5]
        candidates = [
            {
                "id": "low",
                "text": "low",
                "tenant_id": "tenant-a",
            },
            {
                "id": "high",
                "text": "high",
                "tenant_id": "tenant-a",
            },
            {
                "id": "middle",
                "text": "middle",
                "tenant_id": "tenant-a",
            },
        ]

        with patch.object(reranker, "_get_reranker", return_value=model):
            result = reranker.rerank("query", candidates, top_k=2, model_name="test-model")

        self.assertEqual([item["id"] for item in result], ["high", "middle"])
        self.assertEqual([item["rerank_score"] for item in result], [0.9, 0.5])
        model.predict.assert_called_once_with(
            [("query", "low"), ("query", "high"), ("query", "middle")]
        )

    def test_rerank_short_circuits_empty_candidates(self):
        with patch.object(reranker, "_get_reranker") as get_reranker:
            self.assertEqual(reranker.rerank("query", []), [])
        get_reranker.assert_not_called()

    def test_concurrent_cold_calls_construct_one_reranker(self):
        barrier = Barrier(2)
        model = Mock()
        constructor = Mock(side_effect=lambda _name: (time.sleep(0.05), model)[1])
        reranker._load_reranker.cache_clear()

        def load_model():
            barrier.wait(timeout=2)
            return reranker._get_reranker("concurrent-reranker")

        with (
            patch.object(
                reranker,
                "load_sentence_transformers",
                return_value=SimpleNamespace(CrossEncoder=constructor),
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _index: load_model(), range(2)))

        self.assertIs(results[0], model)
        self.assertIs(results[1], model)
        constructor.assert_called_once_with("concurrent-reranker")


class PyarrowPlaceholderTest(unittest.TestCase):
    """The shim that keeps sentence-transformers importable on hosts whose
    Application Control policy blocks pyarrow's native _dataset extension."""

    def test_healthy_host_uses_the_real_pyarrow_dataset_and_reuses_it(self):
        loaded_real_extension = sentence_transformers_loader.ensure_pyarrow_dataset_importable()
        module = importlib.import_module("pyarrow.dataset")

        self.assertIs(sys.modules["pyarrow.dataset"], module)
        self.assertEqual(
            sentence_transformers_loader.ensure_pyarrow_dataset_importable(),
            loaded_real_extension,
        )
        if loaded_real_extension:
            self.assertNotIsInstance(module, sentence_transformers_loader._PlaceholderModule)

    def test_blocked_extension_is_replaced_by_a_placeholder_that_satisfies_annotations(self):
        pyarrow = importlib.import_module("pyarrow")
        stubbed_names = sentence_transformers_loader.STUBBED_MODULES
        leaf_names = [name.rsplit(".", 1)[-1] for name in stubbed_names]
        saved_modules = {name: sys.modules.pop(name, None) for name in stubbed_names}
        saved_attributes = {name: getattr(pyarrow, name, None) for name in leaf_names}
        saved_had_attribute = {name: name in vars(pyarrow) for name in leaf_names}
        finder = _BlockedPyarrowFinder()
        sys.meta_path.insert(0, finder)
        try:
            with self.assertRaises(ImportError):
                importlib.import_module("pyarrow.dataset")

            installed_placeholder = sentence_transformers_loader.ensure_pyarrow_dataset_importable()

            dataset = importlib.import_module("pyarrow.dataset")
            self.assertFalse(installed_placeholder)
            self.assertIsInstance(dataset, sentence_transformers_loader._PlaceholderModule)
            self.assertIs(getattr(pyarrow, "dataset"), dataset)
            self.assertIs(getattr(pyarrow, "_dataset"), sys.modules["pyarrow._dataset"])
            # datasets annotates filters as Union[pds.Expression, list[tuple]],
            # so a placeholder attribute has to be a usable class.
            self.assertIsInstance(dataset.Expression, type)
            self.assertIsNotNone(Optional[Union[dataset.Expression, list[tuple]]])
            with self.assertRaises(AttributeError):
                dataset.__wrapped__
        finally:
            sys.meta_path.remove(finder)
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            for name, attribute in saved_attributes.items():
                if saved_had_attribute[name]:
                    setattr(pyarrow, name, attribute)
                else:
                    vars(pyarrow).pop(name, None)


class QdrantStoreTest(unittest.TestCase):
    def test_get_client_includes_api_key_only_when_configured(self):
        configured = qdrant_store.runtime_settings.application.model_copy(
            update={"qdrant_api_key": SecretStr("secret")}
        )
        with (
            patch.object(qdrant_store, "QdrantClient") as client_type,
            patch.object(qdrant_store, "runtime_settings", SimpleNamespace(application=configured)),
            patch.dict("os.environ", {"QDRANT_API_KEY": "later-env-ignored"}),
        ):
            qdrant_store.get_client()
        client_type.assert_called_once_with(url=configured.qdrant_url, api_key="secret", timeout=10)

        client_type.reset_mock()
        with patch.object(qdrant_store, "QdrantClient") as client_type:
            qdrant_store.get_client()
        client_type.assert_called_once_with(
            url=qdrant_store.runtime_settings.application.qdrant_url,
            api_key=None,
            timeout=10,
        )

    def test_ensure_collection_creates_missing_cosine_collection(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(collections=[])

        qdrant_store.ensure_collection(client)

        configuration = client.create_collection.call_args.kwargs
        self.assertEqual(configuration["collection_name"], qdrant_store.COLLECTION_NAME)
        self.assertEqual(configuration["vectors_config"].size, qdrant_store.VECTOR_SIZE)
        self.assertEqual(configuration["vectors_config"].distance, Distance.COSINE)

    def test_ensure_collection_reuses_matching_collection(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=qdrant_store.COLLECTION_NAME)]
        )
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=qdrant_store.VECTOR_SIZE))
            )
        )

        qdrant_store.ensure_collection(client)

        client.create_collection.assert_not_called()

    def test_index_chunks_embeds_and_upserts_payloads(self):
        client = Mock()
        chunks = [
            Chunk("one", "annual leave", "hr.pdf", 2, 4, "tenant-a"),
            Chunk("two", "security review", "it.pdf", 3, 1, "tenant-a"),
        ]
        vectors = [[0.0] * qdrant_store.VECTOR_SIZE, [1.0] * qdrant_store.VECTOR_SIZE]

        with (
            patch.object(qdrant_store, "ensure_collection") as ensure,
            patch.object(qdrant_store, "embed_texts", return_value=vectors) as embed,
        ):
            qdrant_store.index_chunks(chunks, client=client)

        ensure.assert_called_once_with(client)
        embed.assert_called_once_with(["annual leave", "security review"])
        points = client.upsert.call_args.kwargs["points"]
        self.assertEqual([point.id for point in points], ["one", "two"])
        self.assertEqual(points[0].payload["filename"], "hr.pdf")
        self.assertEqual(points[0].payload["page_number"], 2)
        self.assertEqual(points[0].payload["chunk_index"], 4)

    def test_index_chunks_short_circuits_empty_input(self):
        client = Mock()
        with (
            patch.object(qdrant_store, "ensure_collection") as ensure,
            patch.object(qdrant_store, "embed_texts") as embed,
        ):
            qdrant_store.index_chunks([], client=client)

        ensure.assert_not_called()
        embed.assert_not_called()
        client.upsert.assert_not_called()

    def test_dense_search_translates_filter_and_maps_qdrant_points(self):
        client = Mock()
        client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="chunk-id",
                    payload={"text": "annual leave", "filename": "hr.pdf", "page_number": 2},
                    score=0.91,
                )
            ]
        )
        with patch.object(qdrant_store, "embed_query", return_value=[0.1, 0.2]):
            results = qdrant_store.dense_search(
                "annual leave",
                "tenant-a",
                top_k=3,
                metadata_filter={"filename": "hr.pdf"},
                client=client,
            )

        self.assertEqual(
            results,
            [
                {
                    "id": "chunk-id",
                    "text": "annual leave",
                    "filename": "hr.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                    "score": 0.91,
                    "tenant_id": "tenant-a",
                },
            ],
        )
        query = client.query_points.call_args.kwargs
        self.assertEqual(query["collection_name"], qdrant_store.COLLECTION_NAME)
        self.assertEqual(query["query"], [0.1, 0.2])
        self.assertEqual(query["limit"], 3)
        self.assertEqual(query["query_filter"].must[0].key, "filename")
        self.assertEqual(query["query_filter"].must[0].match.value, "hr.pdf")
        active_filter = query["query_filter"].must[-1]
        self.assertEqual(active_filter.min_should.min_count, 1)
        self.assertEqual(len(active_filter.should), 2)

    def test_versioned_points_start_inactive_and_have_versioned_ids(self):
        client = Mock()
        chunk = Chunk(
            qdrant_store.make_chunk_id("tenant-a", "hr.pdf", 1, 0, "version-a"),
            "annual leave",
            "hr.pdf",
            1,
            0,
            "tenant-a",
            "document-a",
            "version-a",
        )
        with (
            patch.object(qdrant_store, "_ensure_collection"),
            patch.object(qdrant_store, "embed_texts", return_value=[[0.1] * 768]),
        ):
            qdrant_store.index_chunks([chunk], client=client)

        payload = client.upsert.call_args.kwargs["points"][0].payload
        self.assertEqual(payload["document_id"], "document-a")
        self.assertEqual(payload["version_id"], "version-a")
        self.assertFalse(payload["is_active"])
        self.assertNotEqual(
            chunk.id,
            qdrant_store.make_chunk_id("tenant-a", "hr.pdf", 1, 0, "version-b"),
        )

    def test_activation_exposes_target_and_retires_legacy_points(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=qdrant_store.COLLECTION_NAME)]
        )
        qdrant_store.activate_document_version(
            "tenant-a", "document-a", "version-a", "hr.pdf", client=client
        )

        self.assertEqual(client.set_payload.call_count, 2)
        activated_filter = client.set_payload.call_args_list[0].kwargs["points"]
        retired_filter = client.set_payload.call_args_list[1].kwargs["points"]
        self.assertEqual(activated_filter.must[1].key, "document_id")
        self.assertEqual(activated_filter.must[2].match.value, "version-a")
        self.assertEqual(retired_filter.must_not[0].match.value, "version-a")
        client.delete.assert_called_once()
        legacy_filter = client.delete.call_args.kwargs["points_selector"]
        self.assertEqual(legacy_filter.must[-1].is_empty, True)

    def test_owned_qdrant_store_selects_its_configured_embedding_for_search(self):
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        settings = load_runtime_settings({"EMBEDDING_MODEL": "app-search-model"}).application
        store = qdrant_store.QdrantStore(settings, client_factory=lambda **_: client)
        with patch.object(qdrant_store, "embed_query", return_value=[0.1, 0.2]) as embed:
            qdrant_store.dense_search("question", "tenant-a", store=store)

        embed.assert_called_once_with("question", model_name="app-search-model")
        store.close()

    def test_owned_qdrant_store_selects_its_configured_embedding_for_indexing(self):
        client = Mock()
        settings = load_runtime_settings({"EMBEDDING_MODEL": "app-index-model"}).application
        store = qdrant_store.QdrantStore(settings, client_factory=lambda **_: client)
        chunk = Chunk("id", "text", "file.txt", 1, 0, "tenant-a")
        with patch.object(qdrant_store, "embed_texts", return_value=[[0.1] * 768]) as embed:
            with patch.object(qdrant_store, "_ensure_collection"):
                qdrant_store.index_chunks([chunk], store=store)

        embed.assert_called_once_with(["text"], model_name="app-index-model")
        client.upsert.assert_called_once()
        store.close()

    def test_dense_search_validates_query_and_limit_before_external_calls(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "top_k"):
            qdrant_store.dense_search("query", "tenant-a", top_k=0, client=client)
        with self.assertRaisesRegex(ValueError, "query"):
            qdrant_store.dense_search("   ", "tenant-a", client=client)
        client.query_points.assert_not_called()

    def test_dense_search_raises_when_response_payload_is_invalid(self):
        client = Mock()
        client.query_points.return_value = SimpleNamespace(
            points=[SimpleNamespace(id="bad", payload=None, score=0.5)]
        )
        with patch.object(qdrant_store, "embed_query", return_value=[0.1]):
            with self.assertRaises(RuntimeError):
                qdrant_store.dense_search("query", "tenant-a", client=client)


if __name__ == "__main__":
    unittest.main()
