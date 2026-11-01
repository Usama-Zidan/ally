"""
Dense embedding wrapper. Wraps sentence-transformers so the rest of the
retrieval service doesn't care which model is behind it — swapping models
(e.g. bge-small -> bge-large, or an OpenAI embedding) means changing one
place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from config import EMBEDDING_MODEL
from services.retrieval.sentence_transformers_loader import load_sentence_transformers
from services.retrieval.model_loading import MODEL_INFERENCE_GATE, MODEL_LOAD_LOCK

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = EMBEDDING_MODEL
EMBEDDING_DIMENSION = 768


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Return a cached sentence-transformer model for ``model_name``."""
    return load_sentence_transformers().SentenceTransformer(model_name)


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    # lru_cache is mapping-thread-safe but does not guarantee a single load
    # while multiple threads miss the same key at once.
    with MODEL_LOAD_LOCK:
        return _load_model(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    """Return normalized embeddings for texts without adding a query prefix."""
    model = _get_model(model_name)
    with MODEL_INFERENCE_GATE.inference():
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    """Return a normalized embedding after adding the BGE retrieval instruction."""
    model = _get_model(model_name)
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    with MODEL_INFERENCE_GATE.inference():
        vector = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
