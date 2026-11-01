"""Ownership and lifecycle contracts for Qdrant client use."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch
from pydantic import SecretStr

from services.retrieval import qdrant_store


class QdrantStoreOwnerTest(unittest.TestCase):
    def test_owner_is_lazy_reuses_one_client_and_closes_idempotently(self):
        client = Mock()
        factory = Mock(return_value=client)
        owner = qdrant_store.QdrantStore(client_factory=factory)
        factory.assert_not_called()

        with owner.connection() as first, owner.connection() as second:
            self.assertIs(first, client)
            self.assertIs(second, client)
        factory.assert_called_once()
        owner.close()
        owner.close()
        client.close.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with owner.connection():
                self.fail("A closed owner must reject a new connection lease")

    def test_factory_uses_immutable_endpoint_credentials_and_timeout(self):
        client = Mock()
        settings = qdrant_store.runtime_settings.application.model_copy(
            update={
                "qdrant_url": "https://qdrant.example.test:6333",
                "qdrant_api_key": SecretStr("sensitive-test-value"),
                "qdrant_operation_timeout_seconds": 17,
            }
        )
        factory = Mock(return_value=client)
        owner = qdrant_store.QdrantStore(settings, client_factory=factory)
        with owner.connection():
            pass
        factory.assert_called_once_with(
            url="https://qdrant.example.test:6333",
            api_key="sensitive-test-value",
            timeout=17,
        )
        owner.close()

    def test_close_waits_for_active_leases_before_closing_client(self):
        client = Mock()
        owner = qdrant_store.QdrantStore(client_factory=Mock(return_value=client))
        entered = threading.Event()
        release = threading.Event()

        def hold_lease():
            with owner.connection():
                entered.set()
                release.wait(timeout=5)

        lease_thread = threading.Thread(target=hold_lease)
        lease_thread.start()
        self.assertTrue(entered.wait(timeout=2))
        close_thread = threading.Thread(target=owner.close)
        close_thread.start()
        self.assertTrue(close_thread.is_alive())
        client.close.assert_not_called()
        release.set()
        lease_thread.join(timeout=2)
        close_thread.join(timeout=2)
        self.assertFalse(lease_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        client.close.assert_called_once_with()

    def test_implicit_helper_client_is_closed_after_operation(self):
        client = Mock()
        client.get_collections.return_value.collections = []
        with patch.object(qdrant_store, "get_client", return_value=client):
            qdrant_store.delete_by_filename("tenant-a", "missing.pdf")
        client.close.assert_called_once_with()

    def test_borrowed_client_is_never_closed_by_helper(self):
        client = Mock()
        client.query_points.return_value.points = []
        with patch.object(qdrant_store, "embed_query", return_value=[0.0] * 768):
            qdrant_store.dense_search("question", "tenant-a", client=client)
        client.close.assert_not_called()

    def test_implicit_helper_closes_client_when_operation_raises(self):
        client = Mock()
        client.get_collections.side_effect = OSError("offline")
        with patch.object(qdrant_store, "get_client", return_value=client):
            with self.assertRaises(OSError):
                qdrant_store.delete_by_filename("tenant-a", "policy.pdf")
        client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
