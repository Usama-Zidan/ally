"""
Reciprocal Rank Fusion (RRF) — combines the dense (Qdrant) and lexical
(BM25) result lists into one ranked list without needing to normalize or
compare their raw scores directly, which live on different scales (cosine
similarity vs. BM25 term-weighting) and are not directly comparable.

RRF score for a document = sum over each ranking list of 1 / (k + rank),
where rank is 1-indexed position in that list. k=60 is the standard
default from the original RRF paper and works well without tuning.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "id",
) -> list[dict]:
    """ranked_lists: e.g. [dense_results, bm25_results], each already
    sorted best-first. Returns a fused, deduplicated, re-sorted list where
    each item keeps its original payload plus an added "rrf_score"."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, item in enumerate(result_list, start=1):
            if id_key not in item:
                raise ValueError(f"retrieval result is missing required key: {id_key}")
            item_id = str(item[id_key])
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(item_id, item)

    fused = [
        {**payloads[item_id], "rrf_score": score}
        for item_id, score in scores.items()
    ]
    fused.sort(key=lambda x: (-x["rrf_score"], str(x[id_key])))
    return fused
