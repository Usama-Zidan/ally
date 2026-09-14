"""Persisted BM25 lexical index used alongside Qdrant dense retrieval."""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg2
from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH, POSTGRES_DSN

INDEX_PATH = Path(BM25_INDEX_PATH)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    doc_ids: list[str] = field(default_factory=list)
    doc_metadata: list[dict] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    source_mtime_ns: int | None = field(default=None, init=False, repr=False)

    def build(self, docs: list[dict]) -> None:
        self.doc_ids = [doc["id"] for doc in docs]
        self.doc_metadata = docs
        if not docs:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in docs])

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip() or self.bm25 is None:
            return []

        query_tokens = _tokenize(query)
        eligible_indexes = range(len(self.doc_metadata))
        if metadata_filter:
            eligible_indexes = [
                index
                for index, document in enumerate(self.doc_metadata)
                if all(document.get(key) == value for key, value in metadata_filter.items())
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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("wb") as file:
            pickle.dump(self, file)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)

    @staticmethod
    def load(path: Path = INDEX_PATH) -> "BM25Index":
        with path.open("rb") as file:
            return pickle.load(file)


def rebuild_index_from_postgres() -> BM25Index:
    """Rebuilds BM25 from the authoritative Phase 1 Postgres chunk table."""
    from services.retrieval.qdrant_store import make_chunk_id

    conn = psycopg2.connect(os.getenv("POSTGRES_DSN", POSTGRES_DSN))
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, page_number, chunk_index, text "
                "FROM document_chunks ORDER BY filename, page_number, chunk_index;"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    docs = [
        {
            "id": make_chunk_id(filename, page_number, chunk_index),
            "text": text,
            "filename": filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
        }
        for filename, page_number, chunk_index, text in rows
    ]
    index = BM25Index()
    index.build(docs)
    index.save()
    return index
