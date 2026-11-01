"""Opt-in real history checks against an explicitly disposable MongoDB server.

Failpoints affect the whole test server; never point this at application stores.
The fixture requires enableTestCommands and deletes only generated databases.
"""

from __future__ import annotations

import os
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import AsyncMock, patch

import pymongo
from fastapi.testclient import TestClient
from pymongo.errors import InvalidOperation

from services.auth.principal import Principal
from services.chat.mongo_history import COLLECTION_NAME, MongoHistoryStore
from services.chat.ports import HistoryUnavailable
from services.chat.settings import HistorySettings

URI = os.getenv("HISTORY_TEST_MONGO_URI", "")


@unittest.skipUnless(
    os.getenv("RUN_HISTORY_INTEGRATION_TESTS") == "1" and URI,
    "requires RUN_HISTORY_INTEGRATION_TESTS=1 and HISTORY_TEST_MONGO_URI for a disposable MongoDB server",
)
class HistoryMongoIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.admin = pymongo.MongoClient(
            URI,
            connectTimeoutMS=2000,
            serverSelectionTimeoutMS=2000,
            timeoutMS=3000,
            retryWrites=False,
        )
        self.addCleanup(self.admin.close)
        # This marker is enabled only in the disposable test Compose file.
        self.assertTrue(
            self.admin.admin.command({"getParameter": 1, "enableTestCommands": 1})[
                "enableTestCommands"
            ]
        )
        self.database = "ally_history_test_" + uuid.uuid4().hex
        self.stores = []

    def tearDown(self):
        self.admin.admin.command({"configureFailPoint": "failCommand", "mode": "off"})
        for store in self.stores:
            store.close()
        assert self.database.startswith("ally_history_test_")
        assert self.database.removeprefix("ally_history_test_").isalnum()
        self.admin.drop_database(self.database)

    def store(self, **kwargs):
        settings = HistorySettings(uri=URI, database=self.database, **kwargs)
        store = MongoHistoryStore(settings)
        self.stores.append(store)
        return store

    def fail(self, commands, **data):
        self.admin.admin.command(
            {
                "configureFailPoint": "failCommand",
                "mode": {"times": 1},
                "data": {"failCommands": commands, **data},
            }
        )

    def test_concurrent_first_upserts_preserve_every_complete_pair(self):
        stores = [self.store(max_messages=100) for _ in range(4)]
        barrier = Barrier(8)

        def write(worker):
            barrier.wait(timeout=5)
            for index in range(4):
                token = f"{worker}:{index}"
                stores[worker % 4].append_exchange(
                    "tenant-a", "conversation", "q:" + token, "a:" + token
                )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(write, worker) for worker in range(8)]
            for future in futures:
                future.result(timeout=10)
        messages = stores[0].get_history("tenant-a", "conversation")
        self.assertEqual(len(messages), 64)
        users = set()
        for index in range(0, len(messages), 2):
            user, assistant = messages[index : index + 2]
            self.assertEqual((user["role"], assistant["role"]), ("user", "assistant"))
            self.assertEqual(user["content"][2:], assistant["content"][2:])
            users.add(user["content"])
        self.assertEqual(len(users), 32)
        self.assertEqual(self.admin[self.database][COLLECTION_NAME].count_documents({}), 1)

    def test_tenant_scope_applies_to_read_pagination_and_clear(self):
        store = self.store()
        store.append_exchange("tenant-a", "same", "q-a", "a-a")
        store.append_exchange("tenant-b", "same", "q-b", "a-b")
        self.assertEqual(store.get_history("tenant-a", "same")[0]["content"], "q-a")
        self.assertEqual(
            store.get_history_paginated("tenant-b", "same", 2, 1),
            [{"role": "assistant", "content": "a-b"}],
        )
        store.clear_history("tenant-a", "same")
        self.assertEqual(store.get_history("tenant-a", "same"), [])
        self.assertEqual(len(store.get_history("tenant-b", "same")), 2)

    def test_retention_keeps_complete_new_pairs_and_preserves_legacy_metadata(self):
        store = self.store(max_messages=6)
        self.admin[self.database][COLLECTION_NAME].insert_one(
            {
                "tenant_id": "a",
                "conversation_id": "c",
                "messages": [{"role": "user", "content": "legacy"}],
                "legacy_metadata": "preserved",
            }
        )
        for index in range(8):
            store.append_exchange("a", "c", f"q{index}", f"a{index}")
        self.assertEqual(
            [message["content"] for message in store.get_history("a", "c")],
            ["q5", "a5", "q6", "a6", "q7", "a7"],
        )
        self.assertEqual(
            self.admin[self.database][COLLECTION_NAME].find_one({})["legacy_metadata"], "preserved"
        )

    def test_real_failed_index_candidate_is_closed_before_retry(self):
        clients = []

        def factory(*args, **kwargs):
            client = pymongo.MongoClient(*args, **kwargs)
            clients.append(client)
            return client

        store = MongoHistoryStore(
            HistorySettings(uri=URI, database=self.database), client_factory=factory
        )
        self.stores.append(store)
        self.fail(["createIndexes"], errorCode=13)
        with self.assertRaises(HistoryUnavailable):
            store.get_history("a", "c")
        self.assertIsNone(store._client)
        with self.assertRaises(InvalidOperation):
            clients[0].admin.command("ping")
        self.assertEqual(store.get_history("a", "c"), [])
        self.assertEqual(len(clients), 2)
        indexes = self.admin[self.database][COLLECTION_NAME].index_information()
        self.assertTrue(indexes["tenant_id_1_conversation_id_1"]["unique"])

    def test_real_operation_timeout_is_bounded_and_sanitized(self):
        store = self.store(operation_timeout_ms=500)
        store.get_history("a", "c")  # Establish the client/index before injecting delay.
        self.fail(["find"], blockConnection=True, blockTimeMS=2000)
        began = time.monotonic()
        with self.assertRaises(HistoryUnavailable) as context:
            store.get_history("a", "c")
        self.assertLess(time.monotonic() - began, 1.5)
        self.assertEqual(str(context.exception), "History operation was not confirmed")

    def test_real_close_releases_client_and_rejects_future_work(self):
        store = self.store()
        store.append_exchange("a", "c", "q", "a")
        client = store._client
        store.close()
        with self.assertRaises(HistoryUnavailable):
            store.get_history("a", "c")
        with self.assertRaises(InvalidOperation):
            client.admin.command("ping")

    def test_http_socket_reports_unconfirmed_write_even_when_pair_was_stored(self):
        from services.api import main

        store = self.store()
        store.get_history("tenant-a", "conversation")
        self.fail(
            ["update"],
            writeConcernError={
                "code": 64,
                "errmsg": "injected-test-wtimeout",
                "errInfo": {"wtimeout": True},
            },
        )

        async def stream(*args):
            yield "answer"

        with (
            patch.object(main, "MongoHistoryStore", return_value=store),
            patch.object(main, "_start_optional_services", new=AsyncMock()),
            patch(
                "services.api.websocket.security.authenticate_key",
                new=AsyncMock(
                    return_value=Principal(
                        "tenant-a", "test-key", frozenset({"chat:use", "chat:history"})
                    )
                ),
            ),
            patch.object(
                main, "route_message", new=AsyncMock(return_value={"route": "conversational"})
            ) as route,
            patch.object(main, "stream_chat_completion", side_effect=stream),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect(
                "/ws/chat?conversation_id=conversation", headers={"X-API-Key": "test-key"}
            ) as ws:
                self.assertEqual(ws.receive_json()["type"], "ready")
                ws.send_text("question")
                self.assertEqual(ws.receive_json()["type"], "sources")
                self.assertEqual(ws.receive_json()["type"], "token")
                event = ws.receive_json()
                self.assertEqual(event["type"], "error")
                self.assertEqual(event["code"], "history_write_unconfirmed")
            route.assert_awaited_once()
        document = self.admin[self.database][COLLECTION_NAME].find_one(
            {"tenant_id": "tenant-a", "conversation_id": "conversation"}
        )
        self.assertEqual(
            document["messages"],
            [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        )


if __name__ == "__main__":
    unittest.main()
