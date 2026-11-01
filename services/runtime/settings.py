"""Immutable process snapshots. Loading validates values without external I/O."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from services.auth.settings import SecuritySettings, load_security_settings
from services.chat.settings import HistorySettings, load_history_settings


class ConfigurationError(ValueError):
    """Safe startup error; never includes environment values or validation input."""


def _temp_root(values: Mapping[str, str]) -> Path:
    # tempfile.gettempdir() probes directories by writing files on first use.
    # Choose a path without probing; the runtime owner checks actual writability.
    for name in ("TMPDIR", "TEMP", "TMP"):
        if values.get(name):
            return Path(values[name])
    if os.name != "nt":
        return Path("/tmp")
    profile = values.get("USERPROFILE")
    if not profile and values.get("HOMEDRIVE") and values.get("HOMEPATH"):
        profile = values["HOMEDRIVE"] + values["HOMEPATH"]
    if profile:
        return Path(profile) / "AppData" / "Local" / "Temp"
    return Path(values.get("SYSTEMROOT", values.get("SystemRoot", "C:/Windows"))) / "Temp"


class ApplicationSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    project_name: str = "Ally"
    temporal_address: str = "localhost:7233"
    temporal_connect_timeout_seconds: float = Field(default=10, gt=0, le=120)
    temporal_probe_timeout_seconds: float = Field(default=5, gt=0, le=60)
    upload_dir: Path = Field(default_factory=lambda: _temp_root(os.environ) / "ally_uploads")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)
    ingestion_storage_backend: str = "local"
    ingestion_s3_bucket: str = ""
    ingestion_artifact_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024, le=4 * 1024**3)
    aws_region: str = "us-east-1"
    textract_s3_bucket: str = ""
    postgres_dsn: SecretStr = SecretStr(
        "postgresql://assistant:assistant@localhost:5432/assistant_db"
    )
    bm25_index_path: Path = Field(default_factory=lambda: _temp_root(os.environ) / "bm25_index.pkl")
    bm25_rebuild_min_interval_seconds: float = Field(default=60, ge=0, le=86400)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_operation_timeout_seconds: int = Field(default=10, ge=1, le=300)
    qdrant_collection: str = "document_chunks_bge_base_en_v1_5"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    redis_url: SecretStr = SecretStr("redis://localhost:6379")
    qwen_vllm_base_url: str = "http://localhost:8002/v1"
    qwen_model_name: str = "Qwen/Qwen3-7B"
    openai_api_key: SecretStr = SecretStr("")
    llm_fallback_model: str = "gpt-4o-mini"
    cohere_api_key: SecretStr = SecretStr("")
    deepseek_api_key: SecretStr = SecretStr("")
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_request_timeout_seconds: float = Field(default=30, gt=0, le=600)
    circuit_breaker_fail_max: int = Field(default=5, ge=1, le=1000)
    circuit_breaker_reset_timeout_seconds: int = Field(default=30, ge=1, le=3600)
    # The current retrieval candidate pool is 20. Do not silently change ranking.
    chat_retrieval_top_k: int = Field(default=5, ge=1, le=20)
    model_inference_concurrency: int = Field(default=1, ge=1, le=64)
    model_inference_max_waiters: int = Field(default=8, ge=0, le=1024)
    model_inference_wait_timeout_seconds: float = Field(default=30, gt=0, le=600)
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "http://localhost:3001"
    jira_base_url: str = ""
    jira_api_token: SecretStr = SecretStr("")
    jira_email: SecretStr = SecretStr("")
    google_calendar_credentials_json: SecretStr = SecretStr("")
    connected_accounts_encryption_key: SecretStr = SecretStr("")
    intent_classifier: str = "llm"
    pc_agent_command_timeout_seconds: float = Field(default=30, gt=0, le=300)
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")

    @field_validator("ingestion_storage_backend")
    @classmethod
    def storage_backend(cls, value: str) -> str:
        if value not in {"local", "s3"}:
            raise ValueError("Unsupported ingestion storage backend")
        return value

    @field_validator("ingestion_s3_bucket")
    @classmethod
    def bucket_name(cls, value: str) -> str:
        if len(value) > 255 or any(ord(char) < 32 for char in value):
            raise ValueError("Invalid ingestion object-store bucket")
        return value.strip()

    @model_validator(mode="after")
    def configured_ingestion_storage(self) -> "ApplicationSettings":
        if self.ingestion_storage_backend == "s3" and not self.ingestion_s3_bucket:
            raise ValueError("S3 ingestion storage requires an ingestion bucket")
        return self

    @field_validator(
        "project_name",
        "aws_region",
        "qdrant_collection",
        "embedding_model",
        "reranker_model",
        "qwen_model_name",
        "llm_fallback_model",
        "intent_classifier",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip() or len(value) > 512 or any(ord(char) < 32 for char in value):
            raise ValueError("Expected bounded nonempty configuration text")
        return value

    @field_validator("temporal_address")
    @classmethod
    def temporal_target(cls, value: str) -> str:
        # Preserve SDK-supported legacy HTTP(S) targets without changing TLS.
        legacy_url = "://" in value
        if not value or len(value) > 512 or any(char.isspace() for char in value):
            raise ValueError("Invalid Temporal address")
        try:
            parsed = urlsplit(value if legacy_url else "//" + value)
            if (
                not parsed.hostname
                or legacy_url
                and parsed.scheme not in {"http", "https"}
                or not legacy_url
                and parsed.port is None
                or parsed.port is not None
                and not 1 <= parsed.port <= 65535
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid Temporal address") from None
        return value

    @field_validator("qdrant_url", "qwen_vllm_base_url", "langfuse_host", "jira_base_url")
    @classmethod
    def http_endpoint(cls, value: str) -> str:
        if value == "":  # Jira can be disabled; required endpoints checked below.
            return value
        try:
            parsed = urlsplit(value)
            if (
                len(value) > 8192
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or any(char.isspace() for char in value)
                or parsed.port is not None
                and not 1 <= parsed.port <= 65535
            ):
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid HTTP endpoint; use separate credentials") from None
        return value

    @field_validator("qdrant_url", "qwen_vllm_base_url", "langfuse_host")
    @classmethod
    def required_endpoint(cls, value: str) -> str:
        if not value:
            raise ValueError("Required HTTP endpoint is empty")
        return value

    @field_validator("postgres_dsn", "redis_url")
    @classmethod
    def storage_uri(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        raw = value.get_secret_value()
        schemes = (
            {"postgres", "postgresql"} if info.field_name == "postgres_dsn" else {"redis", "rediss"}
        )
        try:
            parsed = urlsplit(raw)
            if (
                len(raw) > 8192
                or parsed.scheme not in schemes
                or not parsed.hostname
                or parsed.fragment
                or any(ord(char) < 32 for char in raw)
                or parsed.port is not None
                and not 1 <= parsed.port <= 65535
            ):
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid storage URI") from None
        return value

    @field_validator("upload_dir", "bm25_index_path", mode="before")
    @classmethod
    def filesystem_path(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name == "bm25_index_path" and value == "":
            # Preserve the existing blank .env template's Path("") semantics.
            return Path(".")
        if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
            raise ValueError("Invalid filesystem path")
        return value

    @field_validator("connected_accounts_encryption_key")
    @classmethod
    def optional_vault_key(cls, value: SecretStr) -> SecretStr:
        if value.get_secret_value():
            from cryptography.fernet import Fernet

            try:
                Fernet(value.get_secret_value().encode("ascii"))
            except (ValueError, UnicodeError):
                raise ValueError("Invalid connected-account encryption key") from None
        return value

    @field_validator("intent_classifier")
    @classmethod
    def classifier_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("Invalid classifier identifier")
        return value


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    application: ApplicationSettings = Field(default_factory=ApplicationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    history: HistorySettings = Field(default_factory=HistorySettings)


def load_runtime_settings(environ: Mapping[str, str] | None = None) -> RuntimeSettings:
    # Copy once so every group sees the same source, even if the caller mutates it.
    values = dict(os.environ if environ is None else environ)
    try:
        application = ApplicationSettings.model_validate(
            {
                "upload_dir": _temp_root(values) / "ally_uploads",
                "bm25_index_path": _temp_root(values) / "bm25_index.pkl",
                **{
                    field: values[field.upper()]
                    for field in ApplicationSettings.model_fields
                    if field.upper() in values
                },
            }
        )
        if (
            values.get("ALLY_ENV", "development").strip().lower() == "production"
            and application.ingestion_storage_backend != "s3"
        ):
            raise ValueError("Production ingestion requires shared S3 object storage")
        return RuntimeSettings(
            application=application,
            security=load_security_settings(values),
            history=load_history_settings(values),
        )
    except (ValueError, TypeError, OverflowError):
        raise ConfigurationError(
            "Invalid runtime configuration; check settings and resource budgets"
        ) from None
