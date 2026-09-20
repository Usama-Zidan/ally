"""
Backfill Phase 1 Postgres chunks into the Phase 2 dense and lexical indexes.

Run this once for existing data or after an interrupted ingestion migration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import structlog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import POSTGRES_DSN
from services.retrieval.bm25_index import rebuild_index_from_postgres
from services.retrieval.qdrant_store import Chunk, index_chunks, make_chunk_id

log = structlog.get_logger()


def load_chunks_from_postgres() -> list[Chunk]:
    """Load ordered PostgreSQL chunk rows with deterministic retrieval IDs.

    Loads every tenant's chunks — this is an operator-run backfill, not a
    request path, so it is deliberately not tenant-scoped. Each resulting
    Chunk still carries its own tenant_id, so the vectors it writes to
    Qdrant remain correctly scoped per tenant.
    """
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id, filename, page_number, chunk_index, text "
                "FROM document_chunks ORDER BY tenant_id, filename, page_number, chunk_index;"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        Chunk(
            id=make_chunk_id(str(tenant_id), filename, page_number, chunk_index),
            text=text,
            filename=filename,
            page_number=page_number,
            chunk_index=chunk_index,
            tenant_id=str(tenant_id),
        )
        for tenant_id, filename, page_number, chunk_index, text in rows
    ]


def main() -> None:
    """Backfill Qdrant and rebuild BM25 from the current PostgreSQL chunks."""
    chunks = load_chunks_from_postgres()
    log.info("chunks_loaded_from_postgres", count=len(chunks))
    index_chunks(chunks)
    # force=True: this is an explicit operator-initiated backfill, so it
    # must actually rebuild rather than be skipped by the debounce window
    # that throttles the per-document rebuilds in embed_and_index.
    rebuild_index_from_postgres(force=True)
    log.info("retrieval_indexes_rebuilt", count=len(chunks))
    print(f"Indexed {len(chunks)} chunks into Qdrant and rebuilt the BM25 index.")


if __name__ == "__main__":
    main()
