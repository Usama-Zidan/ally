import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from qdrant_client.models import Distance

from services.retrieval import embeddings, mmr, qdrant_store, reranker
from services.retrieval.bm25_index import BM25Index, rebuild_index_from_postgres
from services.retrieval.fusion import reciprocal_rank_fusion
from services.retrieval.qdrant_store import Chunk


class BM25IndexTest(unittest.TestCase):
    def test_search_is_case_insensitive_and_tokenizes_punctuation(self):
        index = BM25Index()
        index.build(
            [
                {"id": "match", "text": "Annual-leave policy", "filename": "hr.pdf"},
                {"id": "other", "text": "Security handbook", "filename": "it.pdf"},
            ]
        )

        self.assertEqual([item["id"] for item in index.search("ANNUAL, leave!")], ["match"])

    def test_search_uses_document_id_as_stable_tie_breaker(self):
        index = BM25Index()
        index.build(
            [
                {"id": "b", "text": "same term"},
                {"id": "a", "text": "same term"},
            ]
        )

        self.assertEqual([item["id"] for item in index.search("term")], ["a", "b"])

    def test_search_rejects_non_positive_limit_and_ignores_blank_query(self):
        index = BM25Index()
        index.build([{"id": "one", "text": "annual leave"}])

        with self.assertRaisesRegex(ValueError, "top_k"):
            index.search("annual", top_k=0)
        self.assertEqual(index.search("   "), [])

    def test_search_applies_all_metadata_filter_fields(self):
        index = BM25Index()
        index.build(
            [
                {"id": "page-1", "text": "annual leave", "filename": "hr.pdf", "page_number": 1},
                {"id": "page-2", "text": "annual leave", "filename": "hr.pdf", "page_number": 2},
            ]
        )

        results = index.search(
            "annual leave",
            metadata_filter={"filename": "hr.pdf", "page_number": 2},
        )

        self.assertEqual([item["id"] for item in results], ["page-2"])

    def test_rebuild_index_reads_ordered_rows_and_closes_connection(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("policy.pdf", 2, 3, "annual leave")]

        with patch(
            "services.retrieval.bm25_index.psycopg2.connect",
            return_value=connection,
        ) as connect, patch.object(BM25Index, "save") as save, patch.dict(
            "os.environ", {"POSTGRES_DSN": "postgresql://override/test"}
        ):
            index = rebuild_index_from_postgres()

        connect.assert_called_once_with("postgresql://override/test")
        cursor.execute.assert_called_once()
        self.assertIn("ORDER BY filename, page_number, chunk_index", cursor.execute.call_args.args[0])
        self.assertEqual(index.doc_metadata[0]["filename"], "policy.pdf")
        self.assertEqual(index.doc_metadata[0]["chunk_index"], 3)
        save.assert_called_once_with()
        connection.close.assert_called_once_with()


class FusionTest(unittest.TestCase):
    def test_fusion_accumulates_reciprocal_rank_scores_and_preserves_first_payload(self):
        fused = reciprocal_rank_fusion(
            [
                [{"id": "shared", "text": "dense"}, {"id": "dense", "text": "only"}],
                [{"id": "shared", "text": "lexical"}],
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


class MMRTest(unittest.TestCase):
    def test_mmr_balances_relevance_with_diversity(self):
        candidates = [
            {"id": "best", "text": "best"},
            {"id": "duplicate", "text": "duplicate"},
            {"id": "diverse", "text": "diverse"},
        ]
        candidate_vectors = [[0.9, 0.1], [0.8, 0.2], [0.0, 1.0]]

        with patch.object(mmr, "embed_texts", return_value=candidate_vectors):
            selected = mmr.mmr_select([1.0, 0.0], candidates, top_k=2, lambda_param=0.2)

        self.assertEqual([item["id"] for item in selected], ["best", "diverse"])

    def test_mmr_returns_all_available_candidates_when_limit_is_larger(self):
        candidates = [{"id": "one", "text": "one"}, {"id": "two", "text": "two"}]
        with patch.object(mmr, "embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]):
            selected = mmr.mmr_select([1.0, 0.0], candidates, top_k=5)

        self.assertEqual(len(selected), 2)

    def test_mmr_rejects_mismatched_embedding_dimensions(self):
        with patch.object(mmr, "embed_texts", return_value=[[1.0, 0.0, 0.0]]):
            with self.assertRaisesRegex(ValueError, "matching dimensions"):
                mmr.mmr_select([1.0, 0.0], [{"id": "one", "text": "one"}])

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
            {"id": "low", "text": "low"},
            {"id": "high", "text": "high"},
            {"id": "middle", "text": "middle"},
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


class QdrantStoreTest(unittest.TestCase):
    def test_get_client_includes_api_key_only_when_configured(self):
        with patch.object(qdrant_store, "QdrantClient") as client_type, patch.dict(
            qdrant_store.os.environ, {"QDRANT_API_KEY": "secret"}
        ):
            qdrant_store.get_client()
        client_type.assert_called_once_with(url=qdrant_store.QDRANT_URL, api_key="secret")

        client_type.reset_mock()
        with patch.object(qdrant_store, "QdrantClient") as client_type, patch.dict(
            qdrant_store.os.environ, {"QDRANT_API_KEY": ""}
        ):
            qdrant_store.get_client()
        client_type.assert_called_once_with(url=qdrant_store.QDRANT_URL)

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
            Chunk("one", "annual leave", "hr.pdf", 2, 4),
            Chunk("two", "security review", "it.pdf", 3, 1),
        ]
        vectors = [[0.0] * qdrant_store.VECTOR_SIZE, [1.0] * qdrant_store.VECTOR_SIZE]

        with patch.object(qdrant_store, "ensure_collection") as ensure, patch.object(
            qdrant_store, "embed_texts", return_value=vectors
        ) as embed:
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
        with patch.object(qdrant_store, "ensure_collection") as ensure, patch.object(
            qdrant_store, "embed_texts"
        ) as embed:
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
                }
            ],
        )
        query = client.query_points.call_args.kwargs
        self.assertEqual(query["collection_name"], qdrant_store.COLLECTION_NAME)
        self.assertEqual(query["query"], [0.1, 0.2])
        self.assertEqual(query["limit"], 3)
        self.assertEqual(query["query_filter"].must[0].key, "filename")
        self.assertEqual(query["query_filter"].must[0].match.value, "hr.pdf")

    def test_dense_search_validates_query_and_limit_before_external_calls(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "top_k"):
            qdrant_store.dense_search("query", top_k=0, client=client)
        with self.assertRaisesRegex(ValueError, "query"):
            qdrant_store.dense_search("   ", client=client)
        client.query_points.assert_not_called()

    def test_dense_search_degrades_when_response_payload_is_invalid(self):
        client = Mock()
        client.query_points.return_value = SimpleNamespace(
            points=[SimpleNamespace(id="bad", payload=None, score=0.5)]
        )
        with patch.object(qdrant_store, "embed_query", return_value=[0.1]):
            self.assertEqual(qdrant_store.dense_search("query", client=client), [])


if __name__ == "__main__":
    unittest.main()
