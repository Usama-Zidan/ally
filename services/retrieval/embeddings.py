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

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = EMBEDDING_MODEL
EMBEDDING_DIMENSION = 768


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the configured sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    """Return normalized embeddings for texts without adding a query prefix."""
    model = _get_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    """Return a normalized embedding after adding the BGE retrieval instruction."""
    model = _get_model(model_name)
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
