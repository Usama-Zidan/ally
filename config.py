"""Compatibility exports backed by a validated, immutable process snapshot.

New composition code receives RuntimeSettings explicitly. Legacy imports remain
strings/numbers; they are a startup snapshot, not a live environment interface.
"""

from __future__ import annotations

import os  # Public compatibility for existing operator/test environment overrides.
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from services.chat.settings import HistorySettings
from services.runtime.settings import (
    ApplicationSettings,
    ConfigurationError,
    RuntimeSettings,
    load_runtime_settings,
)


@dataclass(frozen=True)
class Settings:
    """Legacy root view. New code uses the complete RuntimeSettings snapshot."""

    project_name: str
    temporal_address: str
    upload_dir: Path
    max_upload_bytes: int
    aws_region: str
    postgres_dsn: str = field(repr=False)
    bm25_index_path: Path
    bm25_rebuild_min_interval_seconds: float
    qdrant_url: str
    qdrant_api_key: str = field(repr=False)
    qdrant_collection: str
    embedding_model: str
    mongo_uri: str = field(repr=False)
    redis_url: str = field(repr=False)
    textract_s3_bucket: str = ""
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    def validate(self) -> None:
        try:
            values = asdict(self)
            mongo_uri = values.pop("mongo_uri")
            ApplicationSettings.model_validate(values)
            HistorySettings(uri=mongo_uri)
        except (ValueError, TypeError, OverflowError):
            raise ConfigurationError("Invalid application configuration") from None


def _legacy_settings(snapshot: RuntimeSettings) -> Settings:
    values: dict[str, Any] = {}
    for name in Settings.__dataclass_fields__:
        value = snapshot.history.uri if name == "mongo_uri" else getattr(snapshot.application, name)
        values[name] = value.get_secret_value() if isinstance(value, SecretStr) else value
    return Settings(**values)


def _load_settings() -> Settings:
    return _legacy_settings(load_runtime_settings(os.environ))


runtime_settings = load_runtime_settings()
settings = _legacy_settings(runtime_settings)

PROJECT_NAME = runtime_settings.application.project_name
TEMPORAL_ADDRESS = runtime_settings.application.temporal_address
TEMPORAL_CONNECT_TIMEOUT_SECONDS = runtime_settings.application.temporal_connect_timeout_seconds
TEMPORAL_PROBE_TIMEOUT_SECONDS = runtime_settings.application.temporal_probe_timeout_seconds
UPLOAD_DIR = str(runtime_settings.application.upload_dir)
MAX_UPLOAD_BYTES = runtime_settings.application.max_upload_bytes
INGESTION_STORAGE_BACKEND = runtime_settings.application.ingestion_storage_backend
INGESTION_S3_BUCKET = runtime_settings.application.ingestion_s3_bucket
INGESTION_ARTIFACT_MAX_BYTES = runtime_settings.application.ingestion_artifact_max_bytes
AWS_REGION = runtime_settings.application.aws_region
TEXTRACT_S3_BUCKET = runtime_settings.application.textract_s3_bucket
POSTGRES_DSN = runtime_settings.application.postgres_dsn.get_secret_value()
BM25_INDEX_PATH = str(runtime_settings.application.bm25_index_path)
BM25_REBUILD_MIN_INTERVAL_SECONDS = runtime_settings.application.bm25_rebuild_min_interval_seconds
QDRANT_URL = runtime_settings.application.qdrant_url
QDRANT_API_KEY = runtime_settings.application.qdrant_api_key.get_secret_value()
QDRANT_OPERATION_TIMEOUT_SECONDS = runtime_settings.application.qdrant_operation_timeout_seconds
QDRANT_COLLECTION = runtime_settings.application.qdrant_collection
EMBEDDING_MODEL = runtime_settings.application.embedding_model
RERANKER_MODEL = runtime_settings.application.reranker_model
MONGO_URI = runtime_settings.history.uri.get_secret_value()
REDIS_URL = runtime_settings.application.redis_url.get_secret_value()
QWEN_VLLM_BASE_URL = runtime_settings.application.qwen_vllm_base_url
QWEN_MODEL_NAME = runtime_settings.application.qwen_model_name
OPENAI_API_KEY = runtime_settings.application.openai_api_key.get_secret_value()
LLM_FALLBACK_MODEL = runtime_settings.application.llm_fallback_model
COHERE_API_KEY = runtime_settings.application.cohere_api_key.get_secret_value()
DEEPSEEK_API_KEY = runtime_settings.application.deepseek_api_key.get_secret_value()
LLM_MAX_RETRIES = runtime_settings.application.llm_max_retries
LLM_REQUEST_TIMEOUT_SECONDS = runtime_settings.application.llm_request_timeout_seconds
CIRCUIT_BREAKER_FAIL_MAX = runtime_settings.application.circuit_breaker_fail_max
CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS = (
    runtime_settings.application.circuit_breaker_reset_timeout_seconds
)
CHAT_RETRIEVAL_TOP_K = runtime_settings.application.chat_retrieval_top_k
MODEL_INFERENCE_CONCURRENCY = runtime_settings.application.model_inference_concurrency
MODEL_INFERENCE_MAX_WAITERS = runtime_settings.application.model_inference_max_waiters
MODEL_INFERENCE_WAIT_TIMEOUT_SECONDS = (
    runtime_settings.application.model_inference_wait_timeout_seconds
)
CHAT_HISTORY_MAX_MESSAGES = runtime_settings.history.max_messages
MONGO_DB_NAME = runtime_settings.history.database
LANGFUSE_PUBLIC_KEY = runtime_settings.application.langfuse_public_key.get_secret_value()
LANGFUSE_SECRET_KEY = runtime_settings.application.langfuse_secret_key.get_secret_value()
LANGFUSE_HOST = runtime_settings.application.langfuse_host
JIRA_BASE_URL = runtime_settings.application.jira_base_url
JIRA_API_TOKEN = runtime_settings.application.jira_api_token.get_secret_value()
JIRA_EMAIL = runtime_settings.application.jira_email.get_secret_value()
GOOGLE_CALENDAR_CREDENTIALS_JSON = (
    runtime_settings.application.google_calendar_credentials_json.get_secret_value()
)
CONNECTED_ACCOUNTS_ENCRYPTION_KEY = (
    runtime_settings.application.connected_accounts_encryption_key.get_secret_value()
)
INTENT_CLASSIFIER = runtime_settings.application.intent_classifier
PC_AGENT_COMMAND_TIMEOUT_SECONDS = runtime_settings.application.pc_agent_command_timeout_seconds
AWS_ACCESS_KEY_ID = runtime_settings.application.aws_access_key_id.get_secret_value()
AWS_SECRET_ACCESS_KEY = runtime_settings.application.aws_secret_access_key.get_secret_value()
