"""Legacy history facade; the API injects a separate lifespan-owned store."""

from __future__ import annotations

import atexit
import threading

from services.chat.mongo_history import COLLECTION_NAME as COLLECTION_NAME, MongoHistoryStore
from services.chat.ports import ChatMessage

_legacy_store: MongoHistoryStore | None = None
_legacy_lock = threading.Lock()


def _default_store() -> MongoHistoryStore:
    global _legacy_store
    with _legacy_lock:
        if _legacy_store is None:
            _legacy_store = MongoHistoryStore()
        return _legacy_store


def close_history() -> None:
    global _legacy_store
    with _legacy_lock:
        store, _legacy_store = _legacy_store, None
    if store is not None:
        store.close()


def get_history(
    tenant_id: str, conversation_id: str, max_messages: int | None = None
) -> list[ChatMessage]:
    return _default_store().get_history(tenant_id, conversation_id, max_messages)


def append_turn(
    tenant_id: str, conversation_id: str, role: str, content: str, max_messages: int | None = None
) -> None:
    _default_store().append_turn(tenant_id, conversation_id, role, content, max_messages)


def append_exchange(
    tenant_id: str, conversation_id: str, user: str, assistant: str, max_messages: int | None = None
) -> None:
    _default_store().append_exchange(tenant_id, conversation_id, user, assistant, max_messages)


def get_history_paginated(
    tenant_id: str, conversation_id: str, page: int = 1, page_size: int = 10
) -> list[ChatMessage]:
    return _default_store().get_history_paginated(tenant_id, conversation_id, page, page_size)


def clear_history(tenant_id: str, conversation_id: str) -> None:
    _default_store().clear_history(tenant_id, conversation_id)


atexit.register(close_history)
