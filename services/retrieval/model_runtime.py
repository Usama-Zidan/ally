"""Application-scoped retrieval model selection over process-owned weights."""

from __future__ import annotations

from dataclasses import dataclass

from services.retrieval.embeddings import DEFAULT_MODEL_NAME, embed_query
from services.retrieval.reranker import DEFAULT_RERANKER_MODEL, rerank, warm_reranker


@dataclass(frozen=True)
class RetrievalModels:
    """Select configured models for retrieval without owning global model weights.

    Sentence-transformer weights are cached and shared within the process;
    this small immutable adapter belongs to one app and keeps model selection
    out of retrieval globals. Native model teardown is not exposed by the
    libraries, so process shutdown remains the weights' cleanup boundary.
    """

    embedding_model: str = DEFAULT_MODEL_NAME
    reranker_model: str = DEFAULT_RERANKER_MODEL

    def embed_query(self, query: str) -> list[float]:
        return embed_query(query, model_name=self.embedding_model)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return rerank(query, candidates, top_k=top_k, model_name=self.reranker_model)

    def warm_reranker(self) -> None:
        warm_reranker(self.reranker_model)
