"""Validated authentication/transport settings; no clients or I/O at import."""

from __future__ import annotations

import os
from typing import Literal, Mapping, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SecuritySettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", hide_input_in_errors=True, allow_inf_nan=False
    )

    environment: Literal["development", "test", "production"] = "development"
    allow_query_api_keys: bool = True
    require_tls: bool = False
    allowed_origins: tuple[str, ...] = ()
    ticket_ttl_seconds: int = Field(default=30, ge=1, le=120)
    max_tickets: int = Field(default=4096, ge=1, le=100000)
    max_tickets_per_credential: int = Field(default=8, ge=1, le=100)
    max_connections: int = Field(default=256, ge=1, le=10000)
    max_connections_per_tenant: int = Field(default=8, ge=1, le=100)
    message_max_bytes: int = Field(default=65536, ge=1024, le=1048576)
    idle_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    session_max_seconds: float = Field(default=900, gt=0, le=3600)
    send_timeout_seconds: float = Field(default=10, gt=0, le=60)
    auth_attempts_per_minute: int = Field(default=60, ge=1, le=10000)
    max_auth_clients: int = Field(default=4096, ge=1, le=100000)
    max_auth_inflight: int = Field(default=32, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_transport(self) -> Self:
        if self.environment == "production" and (self.allow_query_api_keys or not self.require_tls):
            raise ValueError("Production requires TLS and forbids query-string API keys")
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path
                or parsed.query
                or parsed.fragment
                or "*" in origin
                or self.environment == "production"
                and parsed.scheme != "https"
            ):
                raise ValueError(
                    "Allowed origins must be exact HTTP(S) origins without credentials or paths"
                )
        return self


def load_security_settings(environ: Mapping[str, str] | None = None) -> SecuritySettings:
    values = os.environ if environ is None else environ
    environment = values.get("ALLY_ENV", "development")
    fields: dict[str, object] = {"environment": environment}
    for field in SecuritySettings.model_fields:
        name = "WS_" + field.upper()
        if name in values and field not in {"allowed_origins", "environment"}:
            fields[field] = values[name]
    fields.setdefault("allow_query_api_keys", environment != "production")
    fields.setdefault("require_tls", environment == "production")
    fields["allowed_origins"] = tuple(
        item.strip() for item in values.get("WS_ALLOWED_ORIGINS", "").split(",") if item.strip()
    )
    return SecuritySettings.model_validate(fields)
