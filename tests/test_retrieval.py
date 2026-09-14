import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import Mock, patch

from eval.run_ragas_eval import recall_at_k
from qdrant_client import QdrantClient
from services.retrieval.bm25_index import BM25Index
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

    def test_rrf_deduplicates_results(self):
        fused = reciprocal_rank_fusion(
            [
                [{"id": "one", "text": "a"}, {"id": "two", "text": "b"}],
                [{"id": "two", "text": "b"}, {"id": "three", "text": "c"}],
            ]
        )
        self.assertEqual({result["id"] for result in fused}, {"one", "two", "three"})
        self.assertEqual(len(fused), 3)

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

    def test_pipeline_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            retrieve("")
        with self.assertRaises(ValueError):
            retrieve("annual leave", top_k=0)
        with self.assertRaises(ValueError):
            retrieve("annual leave", top_k=5, candidate_pool_size=4)

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

    def test_dense_search_returns_empty_when_qdrant_is_unavailable(self):
        with patch("services.retrieval.qdrant_store.get_client", side_effect=RuntimeError("offline")):
            self.assertEqual(dense_search("annual leave"), [])

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
