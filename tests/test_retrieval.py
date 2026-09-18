import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np

from eval.run_ragas_eval import recall_at_k, run_recall_comparison, run_ragas_metrics
from qdrant_client import QdrantClient
from services.retrieval import pipeline
from services.retrieval.bm25_index import BM25Index, rebuild_index_from_postgres
from services.retrieval.embeddings import embed_query, embed_texts
from services.retrieval.fusion import reciprocal_rank_fusion
from services.retrieval.mmr import mmr_select
from services.retrieval.pipeline import retrieve
from services.retrieval.qdrant_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    Chunk,
    dense_search,
    ensure_collection,
    index_chunks,
    make_chunk_id,
)
from services.retrieval.reranker import rerank


class RetrievalUnitTest(unittest.TestCase):
    def test_chunk_id_is_deterministic(self):
        first = make_chunk_id("policy.pdf", 2, 3)
        second = make_chunk_id("policy.pdf", 2, 3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, make_chunk_id("policy.pdf", 2, 4))

    def test_bm25_returns_matching_documents_and_applies_filter(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "one",
                    "text": "employees receive annual leave",
                    "filename": "hr.pdf",
                    "page_number": 1,
                },
                {
                    "id": "two",
                    "text": "servers require a security review",
                    "filename": "it.pdf",
                    "page_number": 2,
                },
            ]
        )

        results = index.search("annual leave", metadata_filter={"filename": "hr.pdf"})
        self.assertEqual([result["id"] for result in results], ["one"])
        self.assertEqual(
            index.search("annual leave", metadata_filter={"filename": "it.pdf"}),
            [],
        )

    def test_bm25_applies_metadata_filter_before_top_k(self):
        index = BM25Index()
        index.build(
            [
                {
                    "id": "other",
                    "text": "annual leave policy",
                    "filename": "other.pdf",
                },
                {
                    "id": "target",
                    "text": "annual leave policy",
                    "filename": "hr.pdf",
                },
            ]
        )
        results = index.search(
            "annual leave",
            top_k=1,
            metadata_filter={"filename": "hr.pdf"},
        )
        self.assertEqual([result["id"] for result in results], ["target"])

    def test_bm25_ties_are_deterministic_and_top_k_is_validated(self):
        index = BM25Index()
        index.build(
            [
                {"id": "z", "text": "policy handbook"},
                {"id": "a", "text": "policy handbook"},
            ]
        )

        self.assertEqual(
            [result["id"] for result in index.search("policy")],
            ["a", "z"],
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            index.search("policy", top_k=0)

    def test_empty_bm25_index_is_searchable(self):
        index = BM25Index()
        index.build([])
        self.assertEqual(index.search("anything"), [])

    def test_bm25_index_round_trips(self):
        index = BM25Index()
        index.build([{"id": "one", "text": "annual leave", "filename": "hr.pdf"}])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bm25.pkl"
            index.save(path)
            loaded = BM25Index.load(path)
            self.assertEqual([item["id"] for item in loaded.search("annual leave")], ["one"])

    def test_rebuild_bm25_index_reads_ordered_rows_and_closes_connection(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("policy.pdf", 2, 3, "annual leave")]

        with patch(
            "services.retrieval.bm25_index.psycopg2.connect",
            return_value=connection,
        ), patch(
            "services.retrieval.bm25_index.BM25Index.save"
        ) as save:
            with patch.dict(os.environ, {"POSTGRES_DSN": "postgresql://override"}):
                index = rebuild_index_from_postgres()

        cursor.execute.assert_called_once_with(
            "SELECT filename, page_number, chunk_index, text "
            "FROM document_chunks ORDER BY filename, page_number, chunk_index;"
        )
        connection.close.assert_called_once_with()
        save.assert_called_once_with()
        results = index.search("annual leave")
        self.assertEqual(results[0]["filename"], "policy.pdf")
        self.assertEqual(results[0]["chunk_index"], 3)

    def test_rrf_deduplicates_results(self):
        fused = reciprocal_rank_fusion(
            [
                [{"id": "one", "text": "a"}, {"id": "two", "text": "b"}],
                [{"id": "two", "text": "b"}, {"id": "three", "text": "c"}],
            ]
        )
        self.assertEqual({result["id"] for result in fused}, {"one", "two", "three"})
        self.assertEqual(len(fused), 3)

    def test_rrf_accumulates_rank_scores_and_uses_stable_tie_breaking(self):
        fused = reciprocal_rank_fusion(
            [
                [{"id": "b", "text": "first"}, {"id": "shared", "text": "old"}],
                [{"id": "a", "text": "second"}, {"id": "shared", "text": "new"}],
            ],
            k=10,
        )

        self.assertEqual([item["id"] for item in fused], ["shared", "a", "b"])
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 12)
        self.assertEqual(fused[0]["text"], "old")

    def test_rrf_rejects_invalid_results_and_configuration(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[{"text": "missing id"}]])

    def test_mmr_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            mmr_select([1.0], [{"id": "one", "text": "a"}], lambda_param=1.1)
        with self.assertRaises(ValueError):
            mmr_select([1.0], [{"id": "one", "text": "a"}], top_k=-1)

    def test_mmr_balances_relevance_and_diversity_without_mutating_candidates(self):
        candidates = [
            {"id": "best", "text": "best"},
            {"id": "duplicate", "text": "duplicate"},
            {"id": "diverse", "text": "diverse"},
        ]
        with patch(
            "services.retrieval.mmr.embed_texts",
            return_value=[[1.0, 0.0], [0.99, 0.01], [0.8, 0.6]],
        ):
            selected = mmr_select([1.0, 0.0], candidates, top_k=2, lambda_param=0.4)

        self.assertEqual([item["id"] for item in selected], ["best", "diverse"])
        self.assertEqual(candidates[0], {"id": "best", "text": "best"})

    def test_mmr_rejects_mismatched_embedding_dimensions(self):
        with patch("services.retrieval.mmr.embed_texts", return_value=[[1.0, 0.0]]):
            with self.assertRaisesRegex(ValueError, "matching dimensions"):
                mmr_select([1.0], [{"id": "one", "text": "a"}])

    def test_embedding_wrappers_forward_model_options_and_prefix_queries(self):
        model = Mock()
        model.encode.side_effect = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 6.0]),
        ]
        with patch("services.retrieval.embeddings._get_model", return_value=model):
            text_vectors = embed_texts(["one", "two"], model_name="test-model")
            query_vector = embed_query("annual leave", model_name="test-model")

        self.assertEqual(text_vectors, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(query_vector, [5.0, 6.0])
        self.assertEqual(
            model.encode.call_args_list,
            [
                call(["one", "two"], normalize_embeddings=True, show_progress_bar=False),
                call(
                    "Represent this sentence for searching relevant passages: annual leave",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            ],
        )

    def test_reranker_scores_sorts_and_limits_candidates(self):
        candidates = [
            {"id": "low", "text": "weak match"},
            {"id": "high", "text": "strong match"},
        ]
        model = Mock()
        model.predict.return_value = np.array([0.1, 0.9])
        with patch("services.retrieval.reranker._get_reranker", return_value=model):
            results = rerank("query", candidates, top_k=1, model_name="test-model")

        model.predict.assert_called_once_with(
            [("query", "weak match"), ("query", "strong match")]
        )
        self.assertEqual(results, [{"id": "high", "text": "strong match", "rerank_score": 0.9}])

    def test_reranker_does_not_load_model_for_empty_candidates(self):
        with patch("services.retrieval.reranker._get_reranker") as get_reranker:
            self.assertEqual(rerank("query", []), [])
        get_reranker.assert_not_called()

    def test_pipeline_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            retrieve("")
        with self.assertRaises(ValueError):
            retrieve("annual leave", top_k=0)
        with self.assertRaises(ValueError):
            retrieve("annual leave", top_k=5, candidate_pool_size=4)

    def test_pipeline_runs_all_hybrid_stages_with_requested_filter(self):
        metadata_filter = {"filename": "policy.pdf"}
        dense = [{"id": "dense", "text": "dense"}]
        lexical = [{"id": "lexical", "text": "lexical"}]
        fused = dense + lexical
        reranked = list(reversed(fused))
        bm25 = Mock()
        bm25.search.return_value = lexical

        with patch("services.retrieval.pipeline.dense_search", return_value=dense) as dense_search_mock, patch(
            "services.retrieval.pipeline._get_bm25_index", return_value=bm25
        ), patch(
            "services.retrieval.pipeline.reciprocal_rank_fusion", return_value=fused
        ) as fusion, patch(
            "services.retrieval.pipeline.rerank", return_value=reranked
        ) as rerank_mock, patch(
            "services.retrieval.pipeline.embed_query", return_value=[1.0]
        ) as embed_query_mock, patch(
            "services.retrieval.pipeline.mmr_select", return_value=[reranked[0]]
        ) as mmr:
            results = retrieve(
                "annual leave",
                top_k=1,
                candidate_pool_size=2,
                metadata_filter=metadata_filter,
            )

        self.assertEqual(results, [reranked[0]])
        dense_search_mock.assert_called_once_with(
            "annual leave", top_k=2, metadata_filter=metadata_filter
        )
        bm25.search.assert_called_once_with(
            "annual leave", top_k=2, metadata_filter=metadata_filter
        )
        fusion.assert_called_once_with([dense, lexical])
        rerank_mock.assert_called_once_with("annual leave", fused, top_k=2)
        embed_query_mock.assert_called_once_with("annual leave")
        mmr.assert_called_once_with([1.0], reranked, top_k=1)

    def test_pipeline_falls_back_independently_when_optional_stages_fail(self):
        dense = [
            {"id": "one", "text": "one"},
            {"id": "two", "text": "two"},
        ]
        with patch("services.retrieval.pipeline.dense_search", return_value=dense), patch(
            "services.retrieval.pipeline._get_bm25_index", side_effect=RuntimeError("missing")
        ), patch(
            "services.retrieval.pipeline.rerank", side_effect=RuntimeError("model offline")
        ), patch(
            "services.retrieval.pipeline.embed_query", side_effect=RuntimeError("model offline")
        ):
            results = retrieve("query", top_k=1, candidate_pool_size=2)

        self.assertEqual(results, [dense[0]])

    def test_pipeline_can_return_lexical_results_when_dense_search_is_empty(self):
        lexical = [{"id": "lexical", "text": "exact phrase"}]
        bm25 = Mock()
        bm25.search.return_value = lexical
        with patch("services.retrieval.pipeline.dense_search", return_value=[]), patch(
            "services.retrieval.pipeline._get_bm25_index", return_value=bm25
        ), patch("services.retrieval.pipeline.reciprocal_rank_fusion") as fusion:
            results = retrieve(
                "exact phrase",
                top_k=1,
                use_rerank=False,
                use_mmr=False,
            )

        self.assertEqual(results, lexical)
        fusion.assert_not_called()

    def test_bm25_cache_reloads_only_when_index_mtime_changes(self):
        first = BM25Index()
        second = BM25Index()
        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.pkl"
            index_path.write_bytes(b"placeholder")
            os.utime(index_path, ns=(1_000_000_000, 1_000_000_000))
            with patch.object(pipeline, "INDEX_PATH", index_path), patch.object(
                pipeline, "_bm25_index_cache", None
            ), patch.object(
                pipeline.BM25Index, "load", side_effect=[first, second]
            ) as load:
                self.assertIs(pipeline._get_bm25_index(), first)
                self.assertIs(pipeline._get_bm25_index(), first)
                os.utime(index_path, ns=(2_000_000_000, 2_000_000_000))
                self.assertIs(pipeline._get_bm25_index(), second)

        self.assertEqual(load.call_count, 2)

    def test_recall_accepts_exact_ids_and_page_labels(self):
        retrieved = [
            {
                "id": "chunk-id",
                "filename": "policy.pdf",
                "page_number": 4,
            }
        ]
        self.assertEqual(recall_at_k(retrieved, ["chunk-id"], 1), 1.0)
        self.assertEqual(recall_at_k(retrieved, ["policy.pdf:p4"], 1), 1.0)

    def test_recall_respects_k_and_handles_no_relevant_chunks(self):
        retrieved = [
            {"id": "first", "filename": "one.pdf", "page_number": 1},
            {"id": "second", "filename": "two.pdf", "page_number": 2},
        ]
        self.assertEqual(recall_at_k(retrieved, ["second"], 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, [], 2), 0.0)

    def test_recall_comparison_runs_every_pipeline_configuration(self):
        eval_queries = [
            {"query": "leave", "relevant_chunks": ["hit"]},
            {"query": "security", "relevant_chunks": ["missing"]},
        ]
        retrieved = [{"id": "hit", "filename": "policy.pdf", "page_number": 1}]
        with patch("eval.run_ragas_eval.retrieve", return_value=retrieved) as retrieve_mock:
            averages, per_query = run_recall_comparison(eval_queries, k=1)

        self.assertEqual(averages, {"vector_only": 0.5, "hybrid": 0.5, "hybrid_reranked": 0.5})
        self.assertEqual(len(per_query), 6)
        self.assertEqual(retrieve_mock.call_count, 6)

    def test_ragas_evaluation_skips_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "eval.run_ragas_eval.retrieve"
        ) as retrieve_mock:
            self.assertIsNone(run_ragas_metrics([], k=3))
        retrieve_mock.assert_not_called()

    def test_dense_search_raises_when_qdrant_is_unavailable(self):
        # dense_search intentionally does NOT swallow this into an empty
        # list: a connection failure and "no relevant documents" must stay
        # distinguishable, especially for the eval harness (see
        # qdrant_store.dense_search's docstring/comment for the rationale).
        with patch("services.retrieval.qdrant_store.get_client", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                dense_search("annual leave")

    def test_dense_search_translates_filters_and_maps_payloads(self):
        point = SimpleNamespace(
            id="point-id",
            payload={"text": "annual leave", "filename": "hr.pdf", "page_number": 2},
            score=0.75,
        )
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[point])
        with patch("services.retrieval.qdrant_store.embed_query", return_value=[0.1, 0.2]):
            results = dense_search(
                "annual leave",
                top_k=3,
                metadata_filter={"filename": "hr.pdf"},
                client=client,
            )

        self.assertEqual(
            results,
            [
                {
                    "id": "point-id",
                    "text": "annual leave",
                    "filename": "hr.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                    "score": 0.75,
                }
            ],
        )
        kwargs = client.query_points.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], COLLECTION_NAME)
        self.assertEqual(kwargs["query"], [0.1, 0.2])
        self.assertEqual(kwargs["limit"], 3)
        self.assertEqual(kwargs["query_filter"].must[0].key, "filename")
        self.assertEqual(kwargs["query_filter"].must[0].match.value, "hr.pdf")

    def test_dense_search_validates_input_before_contacting_qdrant(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "top_k"):
            dense_search("query", top_k=0, client=client)
        with self.assertRaisesRegex(ValueError, "query"):
            dense_search("  ", client=client)
        client.query_points.assert_not_called()

    def test_index_chunks_rejects_incomplete_embeddings(self):
        chunk = Chunk("one", "annual leave", "hr.pdf", 1, 0)
        client = cast(QdrantClient, Mock(spec=QdrantClient))
        with patch(
            "services.retrieval.qdrant_store.embed_texts",
            return_value=[],
        ), patch("services.retrieval.qdrant_store.ensure_collection"):
            with self.assertRaisesRegex(RuntimeError, "returned 0 vectors"):
                index_chunks([chunk], client=client)

    def test_index_chunks_rejects_wrong_embedding_dimension(self):
        chunk = Chunk("one", "annual leave", "hr.pdf", 1, 0)
        client = cast(QdrantClient, Mock(spec=QdrantClient))
        with patch(
            "services.retrieval.qdrant_store.embed_texts",
            return_value=[[0.0] * (VECTOR_SIZE - 1)],
        ), patch("services.retrieval.qdrant_store.ensure_collection"):
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                index_chunks([chunk], client=client)

    def test_index_chunks_upserts_embedded_chunks_with_citation_payload(self):
        chunks = [
            Chunk("one", "annual leave", "hr.pdf", 1, 0),
            Chunk("two", "security review", "it.pdf", 2, 4),
        ]
        client = Mock()
        vectors = [[0.0] * VECTOR_SIZE, [1.0] * VECTOR_SIZE]
        with patch("services.retrieval.qdrant_store.ensure_collection") as ensure, patch(
            "services.retrieval.qdrant_store.embed_texts", return_value=vectors
        ) as embed:
            index_chunks(chunks, client=client)

        ensure.assert_called_once_with(client)
        embed.assert_called_once_with(["annual leave", "security review"])
        kwargs = client.upsert.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], COLLECTION_NAME)
        self.assertEqual([point.id for point in kwargs["points"]], ["one", "two"])
        self.assertEqual(kwargs["points"][1].payload["chunk_index"], 4)

    def test_index_chunks_is_a_noop_for_an_empty_batch(self):
        client = Mock()
        with patch("services.retrieval.qdrant_store.ensure_collection") as ensure, patch(
            "services.retrieval.qdrant_store.embed_texts"
        ) as embed:
            index_chunks([], client=client)
        ensure.assert_not_called()
        embed.assert_not_called()
        client.upsert.assert_not_called()

    def test_qdrant_collection_is_created_with_expected_vector_configuration(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(collections=[])

        ensure_collection(client)

        kwargs = client.create_collection.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], COLLECTION_NAME)
        self.assertEqual(kwargs["vectors_config"].size, VECTOR_SIZE)
        self.assertEqual(kwargs["vectors_config"].distance.value, "Cosine")

    def test_qdrant_collection_dimension_is_validated(self):
        collection = type(
            "Collection",
            (),
            {"config": type("Config", (), {"params": type("Params", (), {"vectors": type("Vectors", (), {"size": VECTOR_SIZE - 1})()})()})()},
        )()
        client = cast(
            QdrantClient,
            type(
                "Client",
                (),
                {
                    "get_collections": lambda self: type(
                        "Collections",
                        (),
                        {"collections": [type("Named", (), {"name": COLLECTION_NAME})()]},
                    )(),
                    "get_collection": lambda self, name: collection,
                },
            )(),
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            ensure_collection(client)


if __name__ == "__main__":
    unittest.main()
