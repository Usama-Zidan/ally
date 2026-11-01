"""Framework-free conversation history boundary."""

from __future__ import annotations

from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class HistoryUnavailable(Exception):
    """History access or acknowledgement is unavailable; effects may be unknown."""


class HistoryInputError(ValueError):
    """An internal caller supplied invalid or excessive history data."""


def validate_identity(tenant_id: str, conversation_id: str) -> None:
    for value in (tenant_id, conversation_id):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 128
            or any(ord(character) < 32 for character in value)
        ):
            raise HistoryInputError("Invalid conversation identity")


class HistoryStore(Protocol):
    def get_history(
        self, tenant_id: str, conversation_id: str, max_messages: int | None = None
    ) -> list[ChatMessage]: ...

    def append_exchange(
        self,
        tenant_id: str,
        conversation_id: str,
        user: str,
        assistant: str,
        max_messages: int | None = None,
    ) -> None: ...

    def close(self) -> None: ...
