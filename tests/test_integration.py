"""Opt-in checks against the actual Docker PostgreSQL and Qdrant services."""
from __future__ import annotations

import os
import unittest
import uuid


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS") == "1",
    "set RUN_INTEGRATION_TESTS=1 after `docker compose -f infra/docker-compose.yml up -d`",
)
class DockerIntegrationTest(unittest.TestCase):
    def test_postgres_and_qdrant_enforce_tenant_boundaries(self):
        """Exercise live stores with two tenants sharing a filename."""
        import psycopg2
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

        from config import POSTGRES_DSN, QDRANT_URL

        tenant_a, tenant_b = str(uuid.uuid4()), str(uuid.uuid4())
        tenant_name_a = f"integration-a-{uuid.uuid4()}"
        tenant_name_b = f"integration-b-{uuid.uuid4()}"
        filename = f"integration-{uuid.uuid4()}.txt"
        collection = f"integration_{uuid.uuid4().hex}"
        connection = psycopg2.connect(POSTGRES_DSN)
        client = QdrantClient(url=QDRANT_URL)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO tenants (id, name) VALUES (%s, %s), (%s, %s)",
                    (tenant_a, tenant_name_a, tenant_b, tenant_name_b),
                )
                cursor.execute(
                    "INSERT INTO document_chunks "
                    "(tenant_id, filename, page_number, chunk_index, text) "
                    "VALUES (%s, %s, 1, 0, %s), (%s, %s, 1, 0, %s)",
                    (tenant_a, filename, "a-only", tenant_b, filename, "b-only"),
                )
                cursor.execute(
                    "SELECT tenant_id, text FROM document_chunks "
                    "WHERE filename = %s AND tenant_id = %s",
                    (filename, tenant_a),
                )
                self.assertEqual(cursor.fetchall(), [(tenant_a, "a-only")])
                cursor.execute("DELETE FROM document_chunks WHERE filename = %s", (filename,))
                cursor.execute("DELETE FROM tenants WHERE id IN (%s, %s)", (tenant_a, tenant_b))
            connection.commit()

            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=2, distance=Distance.COSINE),
            )
            client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(id=1, vector=[1.0, 0.0], payload={"tenant_id": tenant_a}),
                    PointStruct(id=2, vector=[1.0, 0.0], payload={"tenant_id": tenant_b}),
                ],
            )
            result = client.query_points(
                collection_name=collection,
                query=[1.0, 0.0],
                query_filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_a))]
                ),
                limit=10,
            )
            tenant_ids = []
            for point in result.points:
                payload = point.payload
                if payload is None:
                    self.fail("Qdrant returned a point without a payload")
                tenant_ids.append(payload["tenant_id"])
            self.assertEqual(tenant_ids, [tenant_a])
        finally:
            connection.close()
            try:
                client.delete_collection(collection)
            except Exception:  # cleanup must not hide the assertion failure
                pass
