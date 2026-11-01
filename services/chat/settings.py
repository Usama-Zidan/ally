"""Validated history resource budgets; loading creates no clients."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class HistorySettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    uri: SecretStr = SecretStr("mongodb://localhost:27017")
    database: str = "ally"
    max_messages: int = Field(default=20, ge=2, le=1000)
    message_max_bytes: int = Field(default=65536, ge=1024, le=262144)
    connect_timeout_ms: int = Field(default=2000, ge=100, le=30000)
    operation_timeout_ms: int = Field(default=3000, ge=100, le=60000)
    max_pool_size: int = Field(default=16, ge=1, le=128)

    @field_validator("uri")
    @classmethod
    def valid_uri(cls, value: SecretStr) -> SecretStr:
        try:
            raw = value.get_secret_value()
            parsed = urlsplit(raw)
            if (
                len(raw) > 8192
                or parsed.scheme not in {"mongodb", "mongodb+srv"}
                or not parsed.netloc
                or parsed.fragment
            ):
                raise ValueError()
        except ValueError as exc:
            raise ValueError("Invalid MongoDB URI") from exc
        return value

    @field_validator("database")
    @classmethod
    def valid_database(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,63}", value):
            raise ValueError("Invalid MongoDB database name")
        return value

    @model_validator(mode="after")
    def bounded_retention(self) -> Self:
        if self.max_messages % 2:
            raise ValueError("History retention must contain complete message pairs")
        if self.max_messages * self.message_max_bytes > 8 * 1024 * 1024:
            raise ValueError("History retention exceeds the document budget")
        return self


def load_history_settings(environ: Mapping[str, str] | None = None) -> HistorySettings:
    values = os.environ if environ is None else environ
    fields = {
        "uri": "MONGO_URI",
        "database": "MONGO_DB_NAME",
        "max_messages": "CHAT_HISTORY_MAX_MESSAGES",
        "message_max_bytes": "CHAT_HISTORY_MESSAGE_MAX_BYTES",
        "connect_timeout_ms": "MONGO_CONNECT_TIMEOUT_MS",
        "operation_timeout_ms": "MONGO_OPERATION_TIMEOUT_MS",
        "max_pool_size": "MONGO_MAX_POOL_SIZE",
    }
    try:
        return HistorySettings.model_validate(
            {field: values[env] for field, env in fields.items() if env in values}
        )
    except ValueError:
        raise ValueError(
            "Invalid history configuration; check MongoDB settings and resource budgets"
        ) from None
