"""
Hybrid retrieval pipeline — the single entrypoint the RAG/agent layer
(Phase 3/4) calls to go from a user query to a final, ordered list of
context chunks.

Flow: dense search (Qdrant) + BM25 (lexical) run in parallel candidate
pools -> reciprocal rank fusion merges them -> cross-encoder reranks the
fused top-N -> MMR re-orders the final top-K for diversity.

Each stage is independently swappable/toggleable, which is what makes it
possible to run the vector-only vs. hybrid vs. hybrid+rerank comparison
in eval/run_ragas_eval.py.
"""
from __future__ import annotations

import structlog

from services.retrieval.bm25_index import BM25Index, INDEX_PATH
from services.retrieval.embeddings import embed_query
from services.retrieval.fusion import reciprocal_rank_fusion
from services.retrieval.mmr import mmr_select
from services.retrieval.qdrant_store import dense_search
from services.retrieval.reranker import rerank

log = structlog.get_logger()

_bm25_index_cache: BM25Index | None = None


def _get_bm25_index() -> BM25Index:
    """Return the cached BM25 index, reloading it when the index file changes.

    Raises:
        RuntimeError: If the index file does not exist.
    """
    global _bm25_index_cache
    index_mtime = INDEX_PATH.stat().st_mtime_ns if INDEX_PATH.exists() else None
    cached_mtime = _bm25_index_cache.source_mtime_ns if _bm25_index_cache else None
    if _bm25_index_cache is None or index_mtime != cached_mtime:
        if not INDEX_PATH.exists():
            raise RuntimeError(
                "BM25 index not found — run "
                "services.retrieval.bm25_index.rebuild_index_from_postgres() first"
            )
        _bm25_index_cache = BM25Index.load()
        _bm25_index_cache.source_mtime_ns = index_mtime
    return _bm25_index_cache


def retrieve(
    query: str,
    top_k: int = 5,
    candidate_pool_size: int = 20,
    use_bm25: bool = True,
    use_rerank: bool = True,
    use_mmr: bool = True,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Return ranked chunks from the enabled dense, lexical, reranking, and MMR stages.

    ``candidate_pool_size`` limits intermediate candidates, while ``top_k``
    limits the final result. Exact ``metadata_filter`` matches are passed to
    dense and lexical search. Unavailable optional stages fall back to the
    preceding stage.

    Raises:
        ValueError: If the query is empty, limits are nonpositive, or the
            candidate pool is smaller than ``top_k``.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be greater than or equal to top_k")

    dense_results = dense_search(query, top_k=candidate_pool_size, metadata_filter=metadata_filter)
    log.info("dense_search_done", num_results=len(dense_results))

    if use_bm25:
        try:
            bm25_results = _get_bm25_index().search(
                query,
                top_k=candidate_pool_size,
                metadata_filter=metadata_filter,
            )
            log.info("bm25_search_done", num_results=len(bm25_results))
        except Exception as exc:  # noqa: BLE001
            log.warning("bm25_unavailable_falling_back_to_dense", error=str(exc))
            bm25_results = []

        if dense_results and bm25_results:
            fused = reciprocal_rank_fusion([dense_results, bm25_results])
        elif dense_results:
            fused = dense_results
        elif bm25_results:
            fused = bm25_results
        else:
            fused = []
    else:
        fused = dense_results

    if use_rerank and fused:
        try:
            reranked = rerank(query, fused, top_k=min(candidate_pool_size, len(fused)))
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank_unavailable_falling_back_to_fused", error=str(exc))
            reranked = fused[:candidate_pool_size]
    else:
        reranked = fused[:candidate_pool_size]

    if use_mmr and reranked:
        try:
            query_embedding = embed_query(query)
            final = mmr_select(query_embedding, reranked, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("mmr_unavailable_falling_back_to_reranked", error=str(exc))
            final = reranked[:top_k]
    else:
        final = reranked[:top_k]

    return final
