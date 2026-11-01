"""Tests for services.chat.history."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.chat import history
from services.chat import mongo_history


class ChatHistoryTest(unittest.TestCase):
    def setUp(self):
        history.close_history()
        self.addCleanup(history.close_history)

    def _mock_collection(self):
        client = MagicMock()
        collection = MagicMock()
        collection.find_one.return_value = None
        client.__getitem__.return_value.get_collection.return_value = collection
        return client, collection

    def test_get_history_returns_empty_list_for_unknown_conversation(self):
        client, collection = self._mock_collection()
        collection.find_one.return_value = None

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client):
            result = history.get_history("tenant-a", "convo-1")

        self.assertEqual(result, [])

    def test_get_history_returns_role_and_content_only(self):
        client, collection = self._mock_collection()
        collection.find_one.return_value = {
            "messages": [
                {"role": "user", "content": "hi", "extra_field": "ignored"},
                {"role": "assistant", "content": "hello"},
            ]
        }

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client):
            result = history.get_history("tenant-a", "convo-1")

        self.assertEqual(
            result,
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        )

    def test_get_history_scopes_query_to_tenant_and_conversation(self):
        client, collection = self._mock_collection()
        collection.find_one.return_value = None

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client):
            history.get_history("tenant-a", "convo-1", max_messages=10)

        query, projection = collection.find_one.call_args.args
        self.assertEqual(query, {"tenant_id": "tenant-a", "conversation_id": "convo-1"})
        self.assertEqual(projection, {"messages": {"$slice": -10}})

    def test_append_turn_uses_atomic_push_with_slice(self):
        client, collection = self._mock_collection()

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client):
            history.append_turn("tenant-a", "convo-1", "user", "hello", max_messages=5)

        collection.update_one.assert_called_once_with(
            {"tenant_id": "tenant-a", "conversation_id": "convo-1"},
            {
                "$push": {
                    "messages": {
                        "$each": [{"role": "user", "content": "hello"}],
                        "$slice": -5,
                    }
                }
            },
            upsert=True,
        )

    def test_clear_history_deletes_scoped_document(self):
        client, collection = self._mock_collection()

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client):
            history.clear_history("tenant-a", "convo-1")

        collection.delete_one.assert_called_once_with(
            {"tenant_id": "tenant-a", "conversation_id": "convo-1"}
        )

    def test_mongo_client_is_created_once_and_reused(self):
        client, _ = self._mock_collection()

        with patch.object(mongo_history.pymongo, "MongoClient", return_value=client) as ctor:
            history.get_history("tenant-a", "convo-1")
            history.get_history("tenant-a", "convo-2")

        ctor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
