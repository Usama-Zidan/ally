"""
Cross-encoder reranking — takes the fused (dense + BM25) candidate list
and re-scores each (query, chunk) pair jointly, which is far more accurate
than the bi-encoder similarity used for initial retrieval. Applied only to
the top ~20-30 fused candidates since cross-encoders are too slow to run
over an entire corpus.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """Return a cached cross-encoder for ``model_name``."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def warm_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> None:
    """Loads the cross-encoder into the process-level cache eagerly.

    bge-reranker-v2-m3 is ~568M params; the first call to rerank() after
    a cold start blocks for tens of seconds while it loads. Calling this
    once during app startup (see services.api.main's lifespan) means that
    cost is paid before the app starts accepting traffic, not on some
    unlucky user's first query.
    """
    _get_reranker(model_name)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 8,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict]:
    """Score and return the most relevant candidates according to a cross-encoder.

    The input candidates are mutated by adding ``rerank_score`` and sorting
    them in descending score order.
    """
    if not candidates:
        return []

    reranker = _get_reranker(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, rerank_scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
