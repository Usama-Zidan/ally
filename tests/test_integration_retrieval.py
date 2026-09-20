"""
End-to-end integration test: real Postgres + real Qdrant + real embedding
and reranker models, no mocks in the path under test.

Every other test file in this suite mocks the storage and model layers,
so they verify wiring but cannot catch the failures that actually matter
in retrieval: a chunk stored under the wrong page number, an embedding
space mismatch, a tenant filter that does not filter, or a query that
simply fails to retrieve text that is demonstrably in the corpus. This
test ingests real text, indexes it for real, and asserts that a natural
language query returns the correct chunk with the correct page number.

Skipped automatically unless RUN_INTEGRATION_TESTS=1 and the backing
services are reachable, so `python -m unittest discover` stays fast and
green on a laptop or in CI without docker compose running.

To run it:

    cd infra && docker compose up -d postgres qdrant
    RUN_INTEGRATION_TESTS=1 python -m unittest tests.test_integration_retrieval
"""
from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS") == "1"


def _services_available() -> tuple[bool, str]:
    """Check Postgres and Qdrant are actually reachable before running."""
    try:
        import psycopg2

        from config import POSTGRES_DSN

        conn = psycopg2.connect(POSTGRES_DSN, connect_timeout=3)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        return False, f"PostgreSQL unavailable: {exc}"

    try:
        from services.retrieval.qdrant_store import get_client

        get_client().get_collections()
    except Exception as exc:  # noqa: BLE001
        return False, f"Qdrant unavailable: {exc}"

    return True, ""


@unittest.skipUnless(RUN_INTEGRATION, "set RUN_INTEGRATION_TESTS=1 to run")
class EndToEndRetrievalTest(unittest.TestCase):
    """Ingest -> chunk -> persist -> index -> retrieve, against live services."""

    # Page 1 and page 2 carry deliberately distinct facts so the assertions
    # below can prove the retrieved chunk came from the right page rather
    # than merely from the right document.
    PAGE_ONE = (
        "The annual leave entitlement for full-time employees is twenty-one "
        "paid days per calendar year. Leave requests should be submitted "
        "through the internal HR portal at least five working days before "
        "the intended start date. Unused leave does not carry over."
    )
    PAGE_TWO = (
        "The standard probation period for a new employee is three months. "
        "A probation period may be extended once by up to three additional "
        "months, subject to written approval from both the reporting manager "
        "and the human resources department."
    )

    @classmethod
    def setUpClass(cls):
        available, reason = _services_available()
        if not available:
            raise unittest.SkipTest(reason)

        import psycopg2

        from config import POSTGRES_DSN

        cls.postgres_dsn = POSTGRES_DSN
        # A unique tenant + filename per run keeps repeated local runs
        # from colliding with each other or with real data.
        cls.tenant_id = str(uuid.uuid4())
        cls.other_tenant_id = str(uuid.uuid4())
        cls.filename = f"integration_{uuid.uuid4().hex[:8]}.txt"

        conn = psycopg2.connect(cls.postgres_dsn)
        try:
            with conn.cursor() as cur:
                # document_chunks.tenant_id is a FK to tenants(id), so both
                # test tenants must exist before any chunk references them.
                cur.execute(
                    "INSERT INTO tenants (id, name) VALUES (%s, %s), (%s, %s) "
                    "ON CONFLICT (id) DO NOTHING;",
                    (
                        cls.tenant_id,
                        f"integration-{cls.tenant_id[:8]}",
                        cls.other_tenant_id,
                        f"integration-other-{cls.other_tenant_id[:8]}",
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def tearDownClass(cls):
        import psycopg2

        from services.retrieval.qdrant_store import delete_by_filename

        try:
            delete_by_filename(cls.tenant_id, cls.filename)
        except Exception:  # noqa: BLE001
            pass

        conn = psycopg2.connect(cls.postgres_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE tenant_id IN (%s, %s);",
                    (cls.tenant_id, cls.other_tenant_id),
                )
                cur.execute(
                    "DELETE FROM tenants WHERE id IN (%s, %s);",
                    (cls.tenant_id, cls.other_tenant_id),
                )
            conn.commit()
        finally:
            conn.close()

    def _ingest(self):
        """Run the real chunk -> persist -> index path for a two-page doc."""
        from workflows.activities import chunk_pages, embed_and_index, persist_chunks

        pages = [
            {"page_number": 1, "text": self.PAGE_ONE, "category": "Text"},
            {"page_number": 2, "text": self.PAGE_TWO, "category": "Text"},
        ]
        chunks = asyncio.run(chunk_pages(pages))
        self.assertGreater(len(chunks), 0, "chunking produced no chunks")
        asyncio.run(persist_chunks(chunks, self.filename, self.tenant_id))
        asyncio.run(embed_and_index(chunks, self.filename, self.tenant_id))
        return chunks

    def test_ingested_document_is_retrievable_with_correct_page_citation(self):
        from services.retrieval.pipeline import retrieve

        self._ingest()

        # A natural-language query that shares almost no literal vocabulary
        # with the source text, so a passing result means dense retrieval
        # genuinely worked rather than BM25 matching on exact keywords.
        results = retrieve(
            "How much paid holiday do staff get each year?",
            self.tenant_id,
            top_k=3,
            candidate_pool_size=10,
        )

        self.assertTrue(results, "retrieval returned no results for an ingested document")
        top = results[0]
        self.assertEqual(top["filename"], self.filename)
        self.assertIn("twenty-one", top["text"].lower())
        # The leave entitlement is stated on page 1 only -- a citation
        # pointing anywhere else is the page-attribution bug class this
        # test exists to catch.
        self.assertEqual(top["page_number"], 1)

    def test_retrieval_distinguishes_between_pages_of_the_same_document(self):
        from services.retrieval.pipeline import retrieve

        self._ingest()

        results = retrieve(
            "How long is the trial period before a role becomes permanent?",
            self.tenant_id,
            top_k=3,
            candidate_pool_size=10,
        )

        self.assertTrue(results)
        top = results[0]
        self.assertIn("probation", top["text"].lower())
        self.assertEqual(top["page_number"], 2)

    def test_retrieval_never_crosses_tenant_boundaries(self):
        """The security-critical assertion: a different tenant must not be
        able to retrieve this tenant's chunks, even with a query that is a
        near-verbatim match for their content."""
        from services.retrieval.pipeline import retrieve

        self._ingest()

        results = retrieve(
            "annual leave entitlement twenty-one paid days",
            self.other_tenant_id,
            top_k=5,
            candidate_pool_size=10,
        )

        leaked = [r for r in results if r["filename"] == self.filename]
        self.assertEqual(
            leaked,
            [],
            "tenant isolation failure: another tenant retrieved this tenant's chunks",
        )

    def test_reingestion_does_not_leave_orphaned_vectors(self):
        """Re-ingesting a shorter version of the same document must not
        leave the longer version's extra chunks retrievable."""
        from services.retrieval.pipeline import retrieve
        from workflows.activities import chunk_pages, embed_and_index, persist_chunks

        self._ingest()

        # Re-ingest with page 2 removed entirely.
        shorter_pages = [{"page_number": 1, "text": self.PAGE_ONE, "category": "Text"}]
        shorter_chunks = asyncio.run(chunk_pages(shorter_pages))
        asyncio.run(persist_chunks(shorter_chunks, self.filename, self.tenant_id))
        asyncio.run(embed_and_index(shorter_chunks, self.filename, self.tenant_id))

        results = retrieve(
            "How long is the probation period for new employees?",
            self.tenant_id,
            top_k=5,
            candidate_pool_size=10,
        )

        stale = [
            r
            for r in results
            if r["filename"] == self.filename and r["page_number"] == 2
        ]
        self.assertEqual(
            stale,
            [],
            "orphaned vectors: page 2 is still retrievable after being removed",
        )


if __name__ == "__main__":
    unittest.main()
