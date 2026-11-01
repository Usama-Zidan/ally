"""Real Qdrant owner/filter checks against the isolated test service."""

from __future__ import annotations

import os
import unittest
import uuid
from unittest.mock import patch

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from services.retrieval import qdrant_store
from services.retrieval.qdrant_store import Chunk

RUN_INTEGRATION = os.getenv("RUN_QDRANT_OWNER_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(
    RUN_INTEGRATION, "set RUN_QDRANT_OWNER_INTEGRATION_TESTS=1 for disposable Qdrant checks"
)
class QdrantOwnerIntegrationTest(unittest.TestCase):
    def test_staged_versions_remain_hidden_until_activation_and_old_points_retire(self):
        url = os.getenv("QDRANT_OWNER_TEST_URL", "http://127.0.0.1:56333")
        collection = f"ally_version_test_{uuid.uuid4().hex}"
        settings = qdrant_store.runtime_settings.application.model_copy(
            update={"qdrant_url": url, "qdrant_collection": collection}
        )
        owner = qdrant_store.QdrantStore(settings)
        vector = [0.0] * qdrant_store.VECTOR_SIZE
        vector[0] = 1.0
        legacy = Chunk(str(uuid.uuid4()), "legacy contents", "policy.pdf", 1, 0, "tenant-a")
        version_a = Chunk(
            qdrant_store.make_chunk_id("tenant-a", "policy.pdf", 1, 0, "version-a"),
            "version a contents",
            "policy.pdf",
            1,
            0,
            "tenant-a",
            "document-a",
            "version-a",
        )
        version_b = Chunk(
            qdrant_store.make_chunk_id("tenant-a", "policy.pdf", 1, 0, "version-b"),
            "version b contents",
            "policy.pdf",
            1,
            0,
            "tenant-a",
            "document-a",
            "version-b",
        )
        try:
            with (
                patch.object(qdrant_store, "embed_texts", side_effect=lambda texts, **_: [vector]),
                patch.object(qdrant_store, "embed_query", return_value=vector),
            ):
                qdrant_store.index_chunks([legacy], store=owner)
                qdrant_store.index_chunks([version_a], store=owner)
                results = qdrant_store.dense_search("query", "tenant-a", store=owner)
                self.assertEqual([item["text"] for item in results], ["legacy contents"])

                qdrant_store.activate_document_version(
                    "tenant-a", "document-a", "version-a", "policy.pdf", store=owner
                )
                results = qdrant_store.dense_search("query", "tenant-a", store=owner)
                self.assertEqual([item["text"] for item in results], ["version a contents"])

                qdrant_store.index_chunks([version_b], store=owner)
                results = qdrant_store.dense_search("query", "tenant-a", store=owner)
                self.assertEqual([item["text"] for item in results], ["version a contents"])

                qdrant_store.activate_document_version(
                    "tenant-a", "document-a", "version-b", "policy.pdf", store=owner
                )
                results = qdrant_store.dense_search("query", "tenant-a", store=owner)
                self.assertEqual([item["text"] for item in results], ["version b contents"])
        finally:
            owner.close()
            admin = QdrantClient(url=url, timeout=5)
            try:
                admin.delete_collection(collection)
            finally:
                admin.close()

    def test_owner_timeout_tenant_search_and_shutdown_against_real_service(self):
        url = os.getenv("QDRANT_OWNER_TEST_URL", "http://127.0.0.1:56333")
        collection = f"ally_owner_test_{uuid.uuid4().hex}"
        settings = qdrant_store.runtime_settings.application.model_copy(
            update={
                "qdrant_url": url,
                "qdrant_collection": collection,
                "qdrant_operation_timeout_seconds": 5,
            }
        )
        owner = qdrant_store.QdrantStore(settings)
        vector = [0.0] * qdrant_store.VECTOR_SIZE
        vector[0] = 1.0
        try:
            qdrant_store.ensure_collection(store=owner)
            with owner.connection() as client:
                client.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "text": "tenant-owned record",
                                "filename": "policy.pdf",
                                "page_number": 1,
                                "chunk_index": 0,
                                "tenant_id": "tenant-a",
                            },
                        ),
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "text": "other tenant record",
                                "filename": "policy.pdf",
                                "page_number": 1,
                                "chunk_index": 0,
                                "tenant_id": "tenant-b",
                            },
                        ),
                    ],
                    wait=True,
                )

            with patch.object(qdrant_store, "embed_query", return_value=vector):
                results = qdrant_store.dense_search("query", "tenant-a", store=owner)
            self.assertEqual([item["text"] for item in results], ["tenant-owned record"])
        finally:
            owner.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            with owner.connection():
                self.fail("A closed Qdrant owner must reject future leases")

        # Clean up only the uniquely named test collection created above.
        admin = QdrantClient(url=url, timeout=5)
        try:
            admin.delete_collection(collection)
        finally:
            admin.close()


if __name__ == "__main__":
    unittest.main()
