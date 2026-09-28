"""
Chat history, backed by MongoDB (per the stack table in README.md) rather
than Redis — this module owns durable conversation turns; Redis (Phase 5)
is for caching, a different concern.

History is scoped by (tenant_id, conversation_id) so one tenant's chat
history is never visible to another, and one browser tab's conversation
doesn't bleed into a different tab's.
"""
from __future__ import annotations

from typing import TypedDict

import pymongo
from pymongo.collection import Collection
import structlog

from config import CHAT_HISTORY_MAX_MESSAGES, MONGO_DB_NAME, MONGO_URI

COLLECTION_NAME = "chat_history"


class ChatMessage(TypedDict):
    role: str  # "user" | "assistant"
    content: str


_client: pymongo.MongoClient | None = None

log = structlog.get_logger()

def _get_collection() -> Collection:
    global _client
    if _client is None:
        try:
            _client = pymongo.MongoClient(MONGO_URI)
            _client[MONGO_DB_NAME][COLLECTION_NAME].create_index(
                [("tenant_id", 1), ("conversation_id", 1)], unique=True
            )
        except Exception as e:
            log.error(f"Error occurred while connecting to MongoDB: {e}")
            raise
    return _client[MONGO_DB_NAME][COLLECTION_NAME]


def get_history(
    tenant_id: str, conversation_id: str, max_messages: int = CHAT_HISTORY_MAX_MESSAGES
) -> list[ChatMessage]:
    """Returns up to the most recent ``max_messages`` turns for this
    conversation, oldest first (the order an LLM prompt expects).

    Returns an empty list for a conversation that doesn't exist yet.
    """
    try:
        document = _get_collection().find_one(
            {"tenant_id": tenant_id, "conversation_id": conversation_id},
            {"messages": {"$slice": -max_messages}},
        )
    except Exception as e:
        log.error(f"Error occurred while fetching chat history: {e}")
        raise

    if document is None:
        return []
    return [{"role": m["role"], "content": m["content"]} for m in document.get("messages", [])]

def get_history_paginated(
    tenant_id: str,
    conversation_id: str,
    page: int = 1,
    page_size: int = 10,
) -> list[ChatMessage]:
    """Returns a paginated list of chat messages for the specified conversation.

    Args:
        tenant_id (str): The tenant ID.
        conversation_id (str): The conversation ID.
        page (int): The page number (1-based).
        page_size (int): The number of messages per page.

    Returns:
        list[ChatMessage]: A list of chat messages for the specified page.
    """
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    
    skip_count = (page - 1) * page_size

    try:
        document = _get_collection().find_one(
            {"tenant_id": tenant_id, "conversation_id": conversation_id},
            {"messages": {"$slice": [skip_count, page_size]}},
        )
    except Exception as e:
        log.error(f"Error occurred while fetching paginated chat history: {e}")
        raise

    if document is None:
        return []
    return [{"role": m["role"], "content": m["content"]} for m in document.get("messages", [])]

def append_turn(
    tenant_id: str,
    conversation_id: str,
    role: str,
    content: str,
    max_messages: int = CHAT_HISTORY_MAX_MESSAGES,
) -> None:
    """Appends one message and trims history to the last ``max_messages``.

    Uses MongoDB's $push with $slice in a single atomic update rather than
    reading the array, appending in Python, and writing it back — that
    read-modify-write pattern would race under concurrent turns on the
    same conversation (unlikely for one browser tab, but the atomic form
    costs nothing and removes the failure mode entirely).
    """
    try:
        _get_collection().update_one(
            {"tenant_id": tenant_id, "conversation_id": conversation_id},
            {
                "$push": {
                    "messages": {
                        "$each": [{"role": role, "content": content}],
                        "$slice": -max_messages,
                    }
                }
            },
            upsert=True,
        )
    except Exception as e:
        log.error(f"Error occurred while appending chat turn: {e}")
        raise


def clear_history(tenant_id: str, conversation_id: str) -> None:
    """Deletes a conversation's history entirely."""
    try:
        _get_collection().delete_one({"tenant_id": tenant_id, "conversation_id": conversation_id})
    except Exception as e:
        log.error(f"Error occurred while clearing chat history: {e}")
        raise
