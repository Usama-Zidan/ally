"""
Maximal Marginal Relevance (MMR) — re-orders the reranked candidates to
balance relevance against diversity, so the final context passed to the
LLM isn't 5 near-duplicate chunks from the same paragraph. Applied as the
last step, after reranking has already established relevance quality.
"""
from __future__ import annotations

import numpy as np

from services.retrieval.embeddings import embed_texts


def mmr_select(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select up to ``top_k`` candidates by relevance and embedding diversity.

    ``lambda_param`` controls the balance: values closer to 1 favor relevance,
    while values closer to 0 favor diversity.

    Raises:
        ValueError: If ``top_k`` is negative, ``lambda_param`` is outside
            ``[0, 1]``, or query and candidate embedding dimensions differ.
    """
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be between 0 and 1")
    if not candidates or top_k == 0:
        return []

    candidate_embeddings = np.array(embed_texts([c["text"] for c in candidates]))
    query_vec = np.asarray(query_embedding, dtype=float)
    if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[1] != query_vec.shape[0]:
        raise ValueError("query and candidate embeddings must have matching dimensions")

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    relevance_scores = candidate_embeddings @ query_vec

    while remaining and len(selected) < top_k:
        if not selected:
            best_idx = int(np.argmax(relevance_scores[remaining]))
            selected.append(remaining.pop(best_idx))
            continue

        mmr_scores = []
        for idx in remaining:
            relevance = relevance_scores[idx]
            diversity_penalty = max(
                candidate_embeddings[idx] @ candidate_embeddings[s] for s in selected
            )
            mmr_scores.append(lambda_param * relevance - (1 - lambda_param) * diversity_penalty)

        best_local_idx = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best_local_idx))

    return [candidates[i] for i in selected]
