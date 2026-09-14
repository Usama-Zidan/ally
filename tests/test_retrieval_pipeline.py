import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from services.retrieval import pipeline
from services.retrieval.bm25_index import BM25Index


class RetrievalPipelineTest(unittest.TestCase):
    def setUp(self):
        pipeline._bm25_index_cache = None

    def tearDown(self):
        pipeline._bm25_index_cache = None

    def test_full_pipeline_passes_results_through_each_enabled_stage(self):
        dense = [{"id": "dense", "text": "dense"}]
        lexical = [{"id": "lexical", "text": "lexical"}]
        fused = [{"id": "fused", "text": "fused"}]
        reranked = [{"id": "reranked", "text": "reranked"}]
        final = [{"id": "final", "text": "final"}]
        bm25_index = Mock()
        bm25_index.search.return_value = lexical
        metadata_filter = {"filename": "policy.pdf"}

        with patch.object(pipeline, "dense_search", return_value=dense) as dense_search, patch.object(
            pipeline, "_get_bm25_index", return_value=bm25_index
        ), patch.object(
            pipeline, "reciprocal_rank_fusion", return_value=fused
        ) as fusion, patch.object(
            pipeline, "rerank", return_value=reranked
        ) as rerank, patch.object(
            pipeline, "embed_query", return_value=[0.25, 0.75]
        ) as embed_query, patch.object(
            pipeline, "mmr_select", return_value=final
        ) as mmr_select:
            result = pipeline.retrieve(
                "annual leave",
                top_k=2,
                candidate_pool_size=4,
                metadata_filter=metadata_filter,
            )

        self.assertEqual(result, final)
        dense_search.assert_called_once_with(
            "annual leave", top_k=4, metadata_filter=metadata_filter
        )
        bm25_index.search.assert_called_once_with(
            "annual leave", top_k=4, metadata_filter=metadata_filter
        )
        fusion.assert_called_once_with([dense, lexical])
        rerank.assert_called_once_with("annual leave", fused, top_k=1)
        embed_query.assert_called_once_with("annual leave")
        mmr_select.assert_called_once_with([0.25, 0.75], reranked, top_k=2)

    def test_disabled_optional_stages_return_dense_results_directly(self):
        dense = [
            {"id": "one", "text": "one"},
            {"id": "two", "text": "two"},
            {"id": "three", "text": "three"},
        ]
        with patch.object(pipeline, "dense_search", return_value=dense), patch.object(
            pipeline, "_get_bm25_index"
        ) as get_bm25, patch.object(pipeline, "rerank") as rerank, patch.object(
            pipeline, "embed_query"
        ) as embed_query, patch.object(pipeline, "mmr_select") as mmr_select:
            result = pipeline.retrieve(
                "query",
                top_k=2,
                candidate_pool_size=3,
                use_bm25=False,
                use_rerank=False,
                use_mmr=False,
            )

        self.assertEqual(result, dense[:2])
        get_bm25.assert_not_called()
        rerank.assert_not_called()
        embed_query.assert_not_called()
        mmr_select.assert_not_called()

    def test_bm25_failure_falls_back_to_dense_results(self):
        dense = [{"id": "one", "text": "one"}, {"id": "two", "text": "two"}]
        with patch.object(pipeline, "dense_search", return_value=dense), patch.object(
            pipeline, "_get_bm25_index", side_effect=OSError("missing")
        ), patch.object(pipeline, "reciprocal_rank_fusion") as fusion:
            result = pipeline.retrieve(
                "query", top_k=1, use_rerank=False, use_mmr=False
            )

        self.assertEqual(result, dense[:1])
        fusion.assert_not_called()

    def test_lexical_results_survive_when_dense_search_is_empty(self):
        lexical = [{"id": "lexical", "text": "lexical"}]
        bm25_index = Mock()
        bm25_index.search.return_value = lexical
        with patch.object(pipeline, "dense_search", return_value=[]), patch.object(
            pipeline, "_get_bm25_index", return_value=bm25_index
        ), patch.object(pipeline, "reciprocal_rank_fusion") as fusion:
            result = pipeline.retrieve(
                "query", top_k=1, use_rerank=False, use_mmr=False
            )

        self.assertEqual(result, lexical)
        fusion.assert_not_called()

    def test_reranker_failure_falls_back_to_fused_order(self):
        fused = [
            {"id": "one", "text": "one"},
            {"id": "two", "text": "two"},
            {"id": "three", "text": "three"},
        ]
        with patch.object(pipeline, "dense_search", return_value=fused), patch.object(
            pipeline, "rerank", side_effect=RuntimeError("model unavailable")
        ):
            result = pipeline.retrieve(
                "query",
                top_k=2,
                candidate_pool_size=3,
                use_bm25=False,
                use_mmr=False,
            )

        self.assertEqual(result, fused[:2])

    def test_mmr_failure_falls_back_to_reranked_order(self):
        dense = [{"id": "dense", "text": "dense"}]
        reranked = [{"id": "reranked", "text": "reranked"}]
        with patch.object(pipeline, "dense_search", return_value=dense), patch.object(
            pipeline, "rerank", return_value=reranked
        ), patch.object(pipeline, "embed_query", side_effect=RuntimeError("model unavailable")), patch.object(
            pipeline, "mmr_select"
        ) as mmr_select:
            result = pipeline.retrieve("query", top_k=1, use_bm25=False)

        self.assertEqual(result, reranked)
        mmr_select.assert_not_called()

    def test_bm25_cache_reuses_unchanged_index_and_reloads_new_mtime(self):
        first = BM25Index()
        second = BM25Index()
        with TemporaryDirectory() as directory:
            index_path = Path(directory, "bm25.pkl")
            index_path.write_bytes(b"placeholder")
            initial_mtime = index_path.stat().st_mtime_ns
            with patch.object(pipeline, "INDEX_PATH", index_path), patch.object(
                pipeline.BM25Index, "load", side_effect=[first, second]
            ) as load:
                self.assertIs(pipeline._get_bm25_index(), first)
                self.assertIs(pipeline._get_bm25_index(), first)
                os.utime(index_path, ns=(initial_mtime + 1_000_000, initial_mtime + 1_000_000))
                self.assertIs(pipeline._get_bm25_index(), second)

        self.assertEqual(load.call_count, 2)

    def test_bm25_cache_reports_actionable_error_when_index_is_missing(self):
        with TemporaryDirectory() as directory, patch.object(
            pipeline, "INDEX_PATH", Path(directory, "missing.pkl")
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild_index_from_postgres"):
                pipeline._get_bm25_index()


if __name__ == "__main__":
    unittest.main()
