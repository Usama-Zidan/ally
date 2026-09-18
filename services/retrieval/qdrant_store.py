"""
Qdrant client wrapper — dense vector side of hybrid retrieval.

The BM25 (sparse/lexical) side lives in bm25_index.py and is fused with
these results in fusion.py via reciprocal rank fusion. Keeping BM25 as a
separate lightweight index (rank_bm25) instead of Qdrant's native sparse
vectors keeps the stack simpler while still giving a real hybrid signal —
swap in Qdrant sparse vectors later if lexical recall needs to scale past
what fits in memory.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from services.retrieval.embeddings import (
    EMBEDDING_DIMENSION,
    embed_query,
    embed_texts,
)

log = structlog.get_logger()

COLLECTION_NAME = QDRANT_COLLECTION
VECTOR_SIZE = EMBEDDING_DIMENSION


@dataclass
class Chunk:
    id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int


def get_client() -> QdrantClient:
    """Create a Qdrant client with the configured URL and optional API key."""
    url = QDRANT_URL
    api_key = os.getenv("QDRANT_API_KEY", QDRANT_API_KEY)
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the configured collection or validate its vector dimension.

    Raises:
        RuntimeError: If an existing collection has a different vector size.
    """
    client = client or get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME)
        vector_config = collection.config.params.vectors
        configured_size = getattr(vector_config, "size", None)
        if configured_size != VECTOR_SIZE:
            raise RuntimeError(
                f"Qdrant collection {COLLECTION_NAME!r} has vector size "
                f"{configured_size}, expected {VECTOR_SIZE}"
            )
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def delete_by_filename(filename: str, client: QdrantClient | None = None) -> None:
    """Deletes every point belonging to ``filename`` from the collection.

    Must be called before re-indexing a document that has already been
    ingested once. Without this, re-ingesting a document that now produces
    fewer chunks leaves the old, higher-index points behind in Qdrant —
    they stay retrievable and citable even though the corresponding text
    no longer exists in Postgres or the source document.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = client or get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return  # nothing to delete yet

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
    )


def index_chunks(chunks: list[Chunk], client: QdrantClient | None = None) -> None:
    """Embed and upsert chunks into the configured Qdrant collection.

    Empty batches are ignored. Does NOT delete stale points from a prior
    version of the same document — call delete_by_filename() first when
    re-indexing (see embed_and_index in workflows/activities.py).

    Raises:
        RuntimeError: If the embedding count or vector dimension is invalid.
    """
    client = client or get_client()
    if not chunks:
        return
    ensure_collection(client)

    vectors = embed_texts([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    if any(len(vector) != VECTOR_SIZE for vector in vectors):
        raise RuntimeError(f"embedding dimension must be {VECTOR_SIZE}")

    points = [
        PointStruct(
            id=chunk.id,
            vector=vector,
            payload={
                "text": chunk.text,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def dense_search(
    query: str,
    top_k: int = 20,
    metadata_filter: dict | None = None,
    client: QdrantClient | None = None,
) -> list[dict]:
    """Return dense-vector matches using optional exact metadata filters.

    Qdrant and embedding errors are intentionally propagated so callers can
    distinguish search failures from valid empty results and explicitly
    implement fallback behavior.

    Raises:
        ValueError: If query is empty or top_k is not positive.
        RuntimeError: If a Qdrant point has an invalid or incomplete payload.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not query.strip():
        raise ValueError("query must not be empty")

    client = client or get_client()
    query_vector = embed_query(query)

    qdrant_filter = None
    if metadata_filter:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in metadata_filter.items()
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
    )

    results: list[dict] = []
    for point in response.points:
        payload = point.payload
        if not isinstance(payload, dict):
            raise RuntimeError(f"Qdrant point {point.id} has no payload")
        try:
            results.append(
                {
                    "id": point.id,
                    "text": payload["text"],
                    "filename": payload["filename"],
                    "page_number": payload["page_number"],
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": point.score,
                }
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Qdrant point {point.id} is missing required field: {exc}"
            ) from exc
    return results


def make_chunk_id(filename: str, page_number: int, chunk_index: int) -> str:
    """Deterministic id so re-indexing the same chunk upserts rather than
    duplicates — mirrors the (filename, page_number, chunk_index) unique
    key used in Postgres from Phase 1."""
    namespace = uuid.NAMESPACE_URL
    return str(uuid.uuid5(namespace, f"{filename}:{page_number}:{chunk_index}"))
