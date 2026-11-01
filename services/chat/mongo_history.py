"""Owned synchronous MongoDB history adapter; API callers offload operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pymongo
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern

from services.chat.ports import (
    ChatMessage,
    HistoryInputError,
    HistoryUnavailable,
    validate_identity,
)
from services.chat.settings import HistorySettings, load_history_settings

COLLECTION_NAME = "chat_history"
Document = dict[str, Any]


class MongoHistoryStore:
    def __init__(
        self,
        settings: HistorySettings | None = None,
        *,
        client_factory: Callable[..., pymongo.MongoClient[Document]] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_history_settings()
        self._factory = client_factory
        self._client: pymongo.MongoClient[Document] | None = None
        self._collection: Collection[Document] | None = None
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False

    def _initialize(self) -> Collection[Document]:
        # Called with the condition held; publish only after unique-index success.
        if self._collection is not None:
            return self._collection
        factory = self._factory if self._factory is not None else pymongo.MongoClient
        client = factory(
            self.settings.uri.get_secret_value(),
            connect=False,
            connectTimeoutMS=self.settings.connect_timeout_ms,
            serverSelectionTimeoutMS=self.settings.connect_timeout_ms,
            socketTimeoutMS=self.settings.operation_timeout_ms,
            timeoutMS=self.settings.operation_timeout_ms,
            waitQueueTimeoutMS=self.settings.operation_timeout_ms,
            maxPoolSize=self.settings.max_pool_size,
            minPoolSize=0,
            maxConnecting=2,
            retryWrites=False,
            retryReads=False,
        )
        try:
            collection = client[self.settings.database].get_collection(
                COLLECTION_NAME,
                write_concern=WriteConcern(
                    w="majority", j=True, wtimeout=self.settings.operation_timeout_ms
                ),
                read_concern=ReadConcern("majority"),
                read_preference=ReadPreference.PRIMARY,
            )
            # Keep the existing auto-generated name so legacy indexes are adopted.
            collection.create_index(
                [("tenant_id", 1), ("conversation_id", 1)],
                unique=True,
                maxTimeMS=self.settings.operation_timeout_ms,
            )
        except BaseException:
            client.close()
            raise
        self._client, self._collection = client, collection
        return collection

    @contextmanager
    def _operation(self) -> Iterator[Collection[Document]]:
        try:
            # Includes initialization, pool checkout and the database operation.
            with pymongo.timeout(self.settings.operation_timeout_ms / 1000):
                with self._condition:
                    if self._closed:
                        raise HistoryUnavailable("History store is closed")
                    if self._active >= self.settings.max_pool_size * 2:
                        raise HistoryUnavailable("History capacity is exhausted")
                    collection = self._initialize()
                    self._active += 1
                try:
                    yield collection
                finally:
                    with self._condition:
                        self._active -= 1
                        self._condition.notify_all()
        except PyMongoError:
            raise HistoryUnavailable("History operation was not confirmed") from None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            deadline = time.monotonic() + self.settings.operation_timeout_ms / 1000 + 1
            while self._active and time.monotonic() < deadline:
                self._condition.wait(max(0, deadline - time.monotonic()))
            client = self._client
            self._client, self._collection = None, None
        if client is not None:
            client.close()

    def _limit(self, value: int | None, *, pair: bool = False) -> int:
        limit = self.settings.max_messages if value is None else value
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise HistoryInputError("Invalid history message limit")
        if pair and (
            limit < 2 or limit % 2 or limit * self.settings.message_max_bytes > 8 * 1024 * 1024
        ):
            raise HistoryInputError(
                "Exchange retention requires a bounded number of complete pairs"
            )
        return limit

    def _content(self, value: object) -> None:
        if not isinstance(value, str) or len(value) > self.settings.message_max_bytes:
            raise HistoryInputError("History message exceeds its content budget")
        try:
            encoded_size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise HistoryInputError("History message contains invalid Unicode") from None
        if encoded_size > self.settings.message_max_bytes:
            raise HistoryInputError("History message exceeds its content budget")

    def _messages(self, document: Document | None) -> list[ChatMessage]:
        if document is None:
            return []
        raw = document.get("messages", [])
        if not isinstance(raw, list):
            raise HistoryUnavailable("Stored history has an invalid message shape")
        messages: list[ChatMessage] = []
        for message in raw:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message["role"] not in {"user", "assistant"}
            ):
                raise HistoryUnavailable("Stored history has an invalid message role")
            try:
                self._content(message.get("content"))
            except HistoryInputError:
                raise HistoryUnavailable("Stored history exceeds its content contract") from None
            messages.append({"role": message["role"], "content": message["content"]})
        return messages

    def get_history(
        self, tenant_id: str, conversation_id: str, max_messages: int | None = None
    ) -> list[ChatMessage]:
        validate_identity(tenant_id, conversation_id)
        limit = self._limit(max_messages)
        with self._operation() as collection:
            document = collection.find_one(
                {"tenant_id": tenant_id, "conversation_id": conversation_id},
                {"messages": {"$slice": -limit}},
            )
        return self._messages(document)

    def append_exchange(
        self,
        tenant_id: str,
        conversation_id: str,
        user: str,
        assistant: str,
        max_messages: int | None = None,
    ) -> None:
        validate_identity(tenant_id, conversation_id)
        self._content(user)
        self._content(assistant)
        limit = self._limit(max_messages, pair=True)
        self._append(
            tenant_id,
            conversation_id,
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            limit,
        )

    def _append(
        self, tenant_id: str, conversation_id: str, messages: list[ChatMessage], limit: int
    ) -> None:
        if limit * self.settings.message_max_bytes > 8 * 1024 * 1024:
            raise HistoryInputError("History retention exceeds its content budget")
        query = {"tenant_id": tenant_id, "conversation_id": conversation_id}
        update = {"$push": {"messages": {"$each": messages, "$slice": -limit}}}
        with self._operation() as collection:
            try:
                result = collection.update_one(query, update, upsert=True)
            except DuplicateKeyError:
                # This rejected insert did not append. One scoped non-upsert
                # attempt can update the winner. Never retrying ambiguous failures.
                result = collection.update_one(query, update, upsert=False)
                if not result.acknowledged or result.matched_count != 1:
                    raise HistoryUnavailable(
                        "Concurrent history creation was not confirmed"
                    ) from None
            if not result.acknowledged:
                raise HistoryUnavailable("History write acknowledgement is required")

    def append_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        role: str,
        content: str,
        max_messages: int | None = None,
    ) -> None:
        """Deprecated single-message compatibility; the API uses append_exchange."""
        validate_identity(tenant_id, conversation_id)
        if role not in {"user", "assistant"}:
            raise HistoryInputError("Invalid history message role")
        self._content(content)
        self._append(
            tenant_id,
            conversation_id,
            [{"role": role, "content": content}],
            self._limit(max_messages),
        )

    def get_history_paginated(
        self, tenant_id: str, conversation_id: str, page: int = 1, page_size: int = 10
    ) -> list[ChatMessage]:
        validate_identity(tenant_id, conversation_id)
        limit = self._limit(page_size)
        if type(page) is not int or page < 1 or (page - 1) * limit > 2**31 - 1:
            raise HistoryInputError("Invalid history page")
        with self._operation() as collection:
            document = collection.find_one(
                {"tenant_id": tenant_id, "conversation_id": conversation_id},
                {"messages": {"$slice": [(page - 1) * limit, limit]}},
            )
        return self._messages(document)

    def clear_history(self, tenant_id: str, conversation_id: str) -> None:
        validate_identity(tenant_id, conversation_id)
        with self._operation() as collection:
            collection.delete_one({"tenant_id": tenant_id, "conversation_id": conversation_id})
