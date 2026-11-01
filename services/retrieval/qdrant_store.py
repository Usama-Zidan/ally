"""Owned Qdrant connection and dense-vector retrieval operations."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import runtime_settings
from services.runtime.settings import ApplicationSettings
from services.retrieval.embeddings import EMBEDDING_DIMENSION, embed_query, embed_texts

log = structlog.get_logger()
COLLECTION_NAME = runtime_settings.application.qdrant_collection
VECTOR_SIZE = EMBEDDING_DIMENSION


@dataclass
class Chunk:
    id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int
    tenant_id: str
    document_id: str | None = None
    version_id: str | None = None


class QdrantStore:
    """Lazy process/lifespan owner for one Qdrant client.

    A connection lease borrows the client for one operation. Closing waits for
    active leases, closes the SDK client once, and prevents new operations.
    """

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        *,
        client_factory: Callable[..., QdrantClient] | None = None,
    ) -> None:
        self.settings = settings or runtime_settings.application
        self._client_factory = client_factory or QdrantClient
        self._condition = threading.Condition()
        self._client: QdrantClient | None = None
        self._active = 0
        self._closed = False

    def _new_client(self) -> QdrantClient:
        api_key = self.settings.qdrant_api_key.get_secret_value()
        return self._client_factory(
            url=self.settings.qdrant_url,
            api_key=api_key or None,
            timeout=self.settings.qdrant_operation_timeout_seconds,
        )

    @contextmanager
    def connection(self) -> Iterator[QdrantClient]:
        with self._condition:
            if self._closed:
                raise RuntimeError("Qdrant store is closed")
            if self._client is None:
                self._client = self._new_client()
            client = self._client
            self._active += 1
        try:
            yield client
        finally:
            with self._condition:
                self._active -= 1
                if self._active == 0:
                    self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            while self._active:
                self._condition.wait()
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def get_client() -> QdrantClient:
    """Create a low-level client; callers own and must close the result."""
    return QdrantStore()._new_client()


@contextmanager
def _using_client(
    client: QdrantClient | None, store: QdrantStore | None
) -> Iterator[tuple[QdrantClient, str]]:
    if client is not None and store is not None:
        raise ValueError("Pass either a borrowed client or an owned store")
    if client is not None:
        yield client, COLLECTION_NAME
    elif store is not None:
        with store.connection() as owned_client:
            yield owned_client, store.settings.qdrant_collection
    else:
        temporary = get_client()
        try:
            yield temporary, COLLECTION_NAME
        finally:
            temporary.close()


def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        collection = client.get_collection(collection_name)
        vector_config = collection.config.params.vectors
        if isinstance(vector_config, dict):
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} uses named vector configs "
                f"{sorted(vector_config)!r}"
            )
        configured_size = getattr(vector_config, "size", None)
        if configured_size != VECTOR_SIZE:
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} has vector size "
                f"{configured_size}, expected {VECTOR_SIZE}"
            )
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    from qdrant_client.models import PayloadSchemaType

    client.create_payload_index(
        collection_name=collection_name,
        field_name="tenant_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def ensure_collection(
    client: QdrantClient | None = None, *, store: QdrantStore | None = None
) -> None:
    """Create the configured collection or validate its vector dimension."""
    with _using_client(client, store) as (active_client, collection_name):
        _ensure_collection(active_client, collection_name)


def delete_by_filename(
    tenant_id: str,
    filename: str,
    client: QdrantClient | None = None,
    *,
    store: QdrantStore | None = None,
) -> None:
    """Delete a document's points within its tenant only."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    with _using_client(client, store) as (active_client, collection_name):
        existing = [c.name for c in active_client.get_collections().collections]
        if collection_name not in existing:
            return
        active_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    FieldCondition(key="filename", match=MatchValue(value=filename)),
                ]
            ),
        )


def index_chunks(
    chunks: list[Chunk],
    client: QdrantClient | None = None,
    *,
    store: QdrantStore | None = None,
) -> None:
    """Embed and upsert chunks; explicit clients are borrowed, stores own clients."""
    if not chunks:
        return
    texts = [chunk.text for chunk in chunks]
    vectors = (
        embed_texts(texts)
        if store is None
        else embed_texts(texts, model_name=store.settings.embedding_model)
    )
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
                "tenant_id": chunk.tenant_id,
                **(
                    {
                        "document_id": chunk.document_id,
                        "version_id": chunk.version_id,
                        "is_active": False,
                    }
                    if chunk.document_id is not None and chunk.version_id is not None
                    else {}
                ),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    with _using_client(client, store) as (active_client, collection_name):
        if store is None:
            ensure_collection(active_client)
        else:
            _ensure_collection(active_client, collection_name)
        active_client.upsert(collection_name=collection_name, points=points)


def dense_search(
    query: str,
    tenant_id: str,
    top_k: int = 20,
    metadata_filter: dict | None = None,
    client: QdrantClient | None = None,
    *,
    store: QdrantStore | None = None,
    embedding_model: str | None = None,
) -> list[dict]:
    """Search dense vectors with an enforced tenant filter."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not query.strip():
        raise ValueError("query must not be empty")
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")

    selected_embedding_model = embedding_model
    if selected_embedding_model is None and store is not None:
        selected_embedding_model = store.settings.embedding_model
    query_vector = (
        embed_query(query)
        if selected_embedding_model is None
        else embed_query(query, model_name=selected_embedding_model)
    )
    from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

    conditions: list[Condition] = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in (metadata_filter or {}).items()
    ]
    conditions.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))
    active_version_conditions = [
        FieldCondition(key="version_id", is_empty=True),
        FieldCondition(key="is_active", match=MatchValue(value=True)),
    ]
    conditions.append(
        Filter(
            should=active_version_conditions,
            min_should={"conditions": active_version_conditions, "min_count": 1},
        )
    )
    qdrant_filter = Filter(must=conditions)

    with _using_client(client, store) as (active_client, collection_name):
        response = active_client.query_points(
            collection_name=collection_name,
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
            result = {
                "id": point.id,
                "text": payload["text"],
                "filename": payload["filename"],
                "page_number": payload["page_number"],
                "chunk_index": payload.get("chunk_index", 0),
                "tenant_id": payload.get("tenant_id", tenant_id),
                "score": point.score,
            }
            for optional_key in ("document_id", "version_id"):
                if optional_key in payload:
                    result[optional_key] = payload[optional_key]
            results.append(result)
        except KeyError as exc:
            raise RuntimeError(f"Qdrant point {point.id} is missing required field: {exc}") from exc
        if results[-1]["tenant_id"] != tenant_id:
            raise RuntimeError(f"Qdrant point {point.id} belongs to a different tenant")
    return results


def make_chunk_id(
    tenant_id: str,
    filename: str,
    page_number: int,
    chunk_index: int,
    version_id: str | None = None,
) -> str:
    """Make deterministic tenant-scoped IDs for idempotent chunk upserts."""
    identity = f"{tenant_id}:{filename}:{page_number}:{chunk_index}"
    if version_id is not None:
        identity = f"{identity}:{version_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def activate_document_version(
    tenant_id: str,
    document_id: str,
    version_id: str,
    filename: str,
    client: QdrantClient | None = None,
    *,
    store: QdrantStore | None = None,
) -> None:
    """Expose a staged version and retire its previous versioned/legacy points."""
    reconcile_document_version(
        tenant_id,
        document_id,
        version_id,
        filename,
        delete_legacy=True,
        client=client,
        store=store,
    )


def reconcile_document_version(
    tenant_id: str,
    document_id: str,
    active_version_id: str | None,
    filename: str,
    *,
    deleted: bool = False,
    delete_legacy: bool = False,
    client: QdrantClient | None = None,
    store: QdrantStore | None = None,
) -> None:
    """Make Qdrant visibility match PostgreSQL's active pointer or tombstone."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    with _using_client(client, store) as (active_client, collection_name):
        if collection_name not in {
            item.name for item in active_client.get_collections().collections
        }:
            return
        document_conditions = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ]
        if deleted:
            active_client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=document_conditions),
                wait=True,
            )
        else:
            if active_version_id is not None:
                active_client.set_payload(
                    collection_name=collection_name,
                    payload={"is_active": True},
                    points=Filter(
                        must=[
                            *document_conditions,
                            FieldCondition(
                                key="version_id", match=MatchValue(value=active_version_id)
                            ),
                        ]
                    ),
                    wait=True,
                )
            deactivate = Filter(
                must=document_conditions,
                must_not=(
                    [FieldCondition(key="version_id", match=MatchValue(value=active_version_id))]
                    if active_version_id is not None
                    else None
                ),
            )
            active_client.set_payload(
                collection_name=collection_name,
                payload={"is_active": False},
                points=deactivate,
                wait=True,
            )
        if deleted or delete_legacy:
            active_client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                        FieldCondition(key="filename", match=MatchValue(value=filename)),
                        FieldCondition(key="version_id", is_empty=True),
                    ]
                ),
                wait=True,
            )
