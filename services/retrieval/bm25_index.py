"""Persisted BM25 lexical index used alongside Qdrant dense retrieval."""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import psycopg2
from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH, BM25_REBUILD_MIN_INTERVAL_SECONDS, POSTGRES_DSN

INDEX_PATH = Path(BM25_INDEX_PATH)
# Rebuild bookkeeping lives next to the index itself: METADATA_PATH records
# when the corpus was last actually rebuilt, DIRTY_MARKER_PATH records that
# new chunks have been ingested since then but a rebuild was skipped by the
# debounce window below (see rebuild_index_from_postgres).
METADATA_PATH = INDEX_PATH.with_suffix(".meta.json")
DIRTY_MARKER_PATH = INDEX_PATH.with_suffix(".dirty")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from ``text``."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    doc_ids: list[str] = field(default_factory=list)
    doc_metadata: list[dict] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    source_mtime_ns: int | None = field(default=None, init=False, repr=False)

    def build(self, docs: list[dict]) -> None:
        """Replace this index's corpus with ``docs``."""
        self.doc_ids = [doc["id"] for doc in docs]
        self.doc_metadata = docs
        if not docs:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in docs])

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Return lexical matches scoped to ``tenant_id``, applying any
        additional exact metadata filters before limiting.

        tenant_id is required (not an optional metadata_filter key) for
        the same reason as qdrant_store.dense_search's tenant_id argument:
        it must not be possible for a caller to accidentally omit tenant
        scoping. It's merged into the filter alongside metadata_filter,
        never overridable by it.

        Each returned document includes its BM25 ``score``.

        Raises:
            ValueError: If ``top_k`` is not positive or ``tenant_id`` is empty.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not query.strip() or self.bm25 is None:
            return []

        # tenant_id is merged in last so it always wins even if a caller's
        # metadata_filter happens to also contain a "tenant_id" key.
        effective_filter = {**(metadata_filter or {}), "tenant_id": tenant_id}

        query_tokens = _tokenize(query)
        eligible_indexes = [
            index
            for index, document in enumerate(self.doc_metadata)
            if all(document.get(key) == value for key, value in effective_filter.items())
        ]
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            eligible_indexes,
            key=lambda index: (-scores[index], self.doc_ids[index]),
        )
        return [
            {**self.doc_metadata[index], "score": float(scores[index])}
            for index in ranked[:top_k]
            if scores[index] > 0
            or set(query_tokens) & set(_tokenize(self.doc_metadata[index]["text"]))
        ]

    def save(self, path: Path = INDEX_PATH) -> None:
        """Atomically serialize this index to ``path``, creating its parent directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("wb") as file:
            pickle.dump(self, file)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)

    @staticmethod
    def load(path: Path = INDEX_PATH) -> "BM25Index":
        """Load and return a serialized index from ``path``."""
        with path.open("rb") as file:
            return pickle.load(file)


def _last_rebuild_at() -> float:
    """Returns the epoch timestamp of the last full rebuild, or 0.0 if
    the index has never been built."""
    try:
        return json.loads(METADATA_PATH.read_text())["last_rebuild_at"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return 0.0


def _record_rebuild(timestamp: float) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps({"last_rebuild_at": timestamp}))
    DIRTY_MARKER_PATH.unlink(missing_ok=True)


def _mark_dirty() -> None:
    DIRTY_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIRTY_MARKER_PATH.touch(exist_ok=True)


def is_dirty() -> bool:
    """True if chunks have been ingested since the last full rebuild but
    a rebuild was skipped by the debounce window (see below)."""
    return DIRTY_MARKER_PATH.exists()


def _rebuild_now() -> BM25Index:
    """Unconditionally rebuilds and persists the index from every chunk
    currently in Postgres. O(corpus size) — this is the expensive call
    that rebuild_index_from_postgres() exists to debounce."""
    from services.retrieval.qdrant_store import make_chunk_id

    conn = psycopg2.connect(os.getenv("POSTGRES_DSN", POSTGRES_DSN))
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id, filename, page_number, chunk_index, text "
                "FROM document_chunks ORDER BY tenant_id, filename, page_number, chunk_index;"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    docs = [
        {
            "id": make_chunk_id(tenant_id, filename, page_number, chunk_index),
            "text": text,
            "filename": filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "tenant_id": tenant_id,
        }
        for tenant_id, filename, page_number, chunk_index, text in rows
    ]
    index = BM25Index()
    index.build(docs)
    index.save()
    _record_rebuild(time.time())
    return index


def rebuild_index_from_postgres(force: bool = False) -> BM25Index | None:
    """Rebuild the BM25 index from PostgreSQL, debounced to at most one
    full rebuild per BM25_REBUILD_MIN_INTERVAL_SECONDS.

    Rebuilding is O(corpus size): it re-tokenizes and re-pickles every
    chunk in the table. Calling this unconditionally from every single
    document's embed_and_index activity meant ingesting N documents did
    O(N * corpus_size) work — ingesting 30K documents each rebuilding a
    30K-document corpus. Callers that don't need up-to-the-second lexical
    freshness (i.e. every embed_and_index call) should call this with
    force=False (the default): if a rebuild happened more recently than
    the debounce window, this returns None immediately and just marks the
    index dirty, so a caller that periodically flushes dirty state (see
    flush_if_dirty) eventually catches up.

    Callers that need the index to reflect Postgres *right now* — a
    one-off backfill, an admin "reindex" action — should pass force=True.

    Returns:
        The rebuilt BM25Index, or None if the rebuild was skipped.
    """
    if not force and time.time() - _last_rebuild_at() < BM25_REBUILD_MIN_INTERVAL_SECONDS:
        _mark_dirty()
        return None
    return _rebuild_now()


def flush_if_dirty() -> BM25Index | None:
    """Forces a rebuild if chunks have been ingested since the last one,
    otherwise does nothing. Intended to be called periodically (a cron
    job, a Temporal Schedule, or the loop in bm25_rebuild_worker.py) so a
    burst of ingestions that each skipped their own rebuild (via the
    debounce in rebuild_index_from_postgres) still converges to an
    up-to-date index shortly afterward, rather than only updating on the
    next ingestion that happens to land outside the debounce window.
    """
    if is_dirty():
        return _rebuild_now()
    return None
