"""Owned history initialization, failure, concurrency and privacy contracts."""

from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, MagicMock, patch

from pymongo.errors import DuplicateKeyError, NetworkTimeout, OperationFailure
from pymongo.results import UpdateResult

from services.chat.mongo_history import MongoHistoryStore
from services.chat.ports import HistoryInputError, HistoryUnavailable
from services.chat.settings import HistorySettings, load_history_settings


def candidate():
    client, collection = MagicMock(), MagicMock()
    client.__getitem__.return_value.get_collection.return_value = collection
    collection.find_one.return_value = None
    collection.update_one.return_value = UpdateResult({"n": 1, "nModified": 1}, True)
    return client, collection


class HistoryOwnerTest(unittest.TestCase):
    def setUp(self):
        self.client, self.collection = candidate()
        self.factory = Mock(return_value=self.client)
        self.store = MongoHistoryStore(HistorySettings(), client_factory=self.factory)
        self.addCleanup(self.store.close)

    def test_construction_and_unused_shutdown_do_not_connect(self):
        self.store.close()
        self.store.close()
        self.factory.assert_not_called()
        with self.assertRaises(HistoryUnavailable):
            self.store.get_history("a", "conversation")
        self.factory.assert_not_called()

    def test_failed_index_initialization_closes_candidate_and_can_retry_safely(self):
        second, collection = candidate()
        self.factory.side_effect = [self.client, second]
        self.collection.create_index.side_effect = OperationFailure("private-uri-and-password")
        with self.assertRaises(HistoryUnavailable) as context:
            self.store.get_history("a", "conversation")
        self.assertNotIn("private", str(context.exception))
        self.assertIsNone(self.store._client)
        self.client.close.assert_called_once()
        self.assertEqual(self.store.get_history("a", "conversation"), [])
        self.assertIs(self.store._client, second)
        collection.create_index.assert_called_once()

    def test_concurrent_first_use_publishes_one_initialized_client(self):
        entered, release = threading.Event(), threading.Event()

        def initialize(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test index gate not released")

        self.collection.create_index.side_effect = initialize
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.store.get_history, "a", str(index)) for index in range(4)
            ]
            try:
                self.assertTrue(entered.wait(timeout=2))
                self.assertIsNone(self.store._client)
            finally:
                release.set()
            self.assertEqual([future.result(timeout=3) for future in futures], [[], [], [], []])
        self.factory.assert_called_once()
        self.collection.create_index.assert_called_once()

    def test_close_drains_admitted_operations_and_rejects_new_work(self):
        entered, release = threading.Event(), threading.Event()

        def read(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test read gate not released")
            return None

        self.collection.find_one.side_effect = read
        with ThreadPoolExecutor(max_workers=2) as executor:
            reading = executor.submit(self.store.get_history, "a", "conversation")
            self.assertTrue(entered.wait(timeout=2))
            closing = executor.submit(self.store.close)
            try:
                with self.store._condition:
                    self.store._condition.wait_for(lambda: self.store._closed, timeout=0.2)
                with self.assertRaises(HistoryUnavailable):
                    self.store.get_history("a", "new-conversation")
                self.client.close.assert_not_called()
            finally:
                release.set()
            self.assertEqual(reading.result(timeout=3), [])
            closing.result(timeout=3)
        self.store.close()
        self.client.close.assert_called_once()

    def test_pool_and_timeout_options_override_uri_defaults(self):
        self.store.get_history("a", "conversation")
        kwargs = self.factory.call_args.kwargs
        self.assertFalse(kwargs["connect"])
        self.assertFalse(kwargs["retryWrites"])
        self.assertEqual(kwargs["maxPoolSize"], 16)
        self.assertEqual(kwargs["timeoutMS"], 3000)
        options = self.client.__getitem__.return_value.get_collection.call_args.kwargs
        self.assertEqual(
            options["write_concern"].document, {"w": "majority", "j": True, "wtimeout": 3000}
        )

    def test_exchange_is_one_scoped_atomic_pair_update(self):
        self.store.append_exchange("a", "conversation", "question", "answer", max_messages=6)
        self.collection.update_one.assert_called_once_with(
            {"tenant_id": "a", "conversation_id": "conversation"},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                        "$slice": -6,
                    }
                }
            },
            upsert=True,
        )

    def test_known_rejected_upsert_collision_has_one_scoped_non_upsert_attempt(self):
        self.collection.update_one.side_effect = [
            DuplicateKeyError("collision"),
            UpdateResult({"n": 1}, True),
        ]
        self.store.append_exchange("a", "conversation", "question", "answer")
        first, second = self.collection.update_one.call_args_list
        self.assertEqual(first.args, second.args)
        self.assertEqual(second.kwargs, {"upsert": False})

    def test_ambiguous_network_timeout_is_not_retried_or_exposed(self):
        self.collection.update_one.side_effect = NetworkTimeout("private-message-and-password")
        with self.assertRaises(HistoryUnavailable) as context:
            self.store.append_exchange("a", "conversation", "question", "answer")
        self.assertNotIn("private", str(context.exception))
        self.collection.update_one.assert_called_once()

    def test_unacknowledged_write_cannot_claim_completion(self):
        self.collection.update_one.return_value = UpdateResult({}, False)
        with self.assertRaises(HistoryUnavailable):
            self.store.append_exchange("a", "conversation", "question", "answer")

    def test_invalid_identity_or_budget_never_opens_a_client(self):
        for args in (
            ({"$ne": "a"}, "conversation", "q", "a"),
            ("a", "x" * 129, "q", "a"),
            ("a", "conversation", "x" * 65537, "a"),
            ("a", "conversation", "\ud800", "a"),
        ):
            with self.subTest(args=args), self.assertRaises(HistoryInputError):
                self.store.append_exchange(*args)
        with self.assertRaises(HistoryInputError):
            self.store.append_exchange("a", "conversation", "q", "a", max_messages=5)
        self.factory.assert_not_called()

    def test_capacity_rejection_does_not_enter_database_or_queue_more_work(self):
        store = MongoHistoryStore(HistorySettings(max_pool_size=1), client_factory=self.factory)
        self.addCleanup(store.close)
        entered, release = threading.Barrier(3), threading.Event()

        def read(*args, **kwargs):
            entered.wait(timeout=3)
            if not release.wait(timeout=3):
                raise RuntimeError("Test capacity gate not released")
            return None

        self.collection.find_one.side_effect = read
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(store.get_history, "a", str(index)) for index in range(2)]
            try:
                entered.wait(timeout=3)
                with self.assertRaisesRegex(HistoryUnavailable, "capacity"):
                    store.get_history("a", "excess")
                self.assertEqual(self.collection.find_one.call_count, 2)
            finally:
                release.set()
            self.assertEqual([future.result(timeout=3) for future in futures], [[], []])

    def test_malformed_legacy_roles_do_not_become_system_prompts(self):
        self.collection.find_one.return_value = {
            "messages": [{"role": "system", "content": "private prompt"}]
        }
        with self.assertRaises(HistoryUnavailable) as context:
            self.store.get_history("a", "conversation")
        self.assertNotIn("private", str(context.exception))


class HistorySettingsTest(unittest.TestCase):
    def test_loader_is_bounded_and_redacts_invalid_secret_configuration(self):
        with patch.dict(
            os.environ,
            {
                "MONGO_URI": "mongodb://user:private-sentinel@localhost",
                "CHAT_HISTORY_MAX_MESSAGES": "5",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError) as context:
                load_history_settings()
        self.assertNotIn("private-sentinel", str(context.exception))
        for kwargs in (
            {"max_messages": 5},
            {"max_messages": 1000},
            {"database": "bad/name"},
            {"max_pool_size": 0},
            {"operation_timeout_ms": 0},
            {"uri": "https://private-host"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HistorySettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
