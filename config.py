"""application configuration shared by all services."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    project_name: str
    temporal_address: str
    upload_dir: Path
    aws_region: str
    postgres_dsn: str
    bm25_index_path: Path
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    embedding_model: str
    mongo_uri: str
    redis_url: str

    def validate(self) -> None:
        if not self.project_name.strip():
            raise ValueError("PROJECT_NAME must not be empty")
        if not self.temporal_address.strip():
            raise ValueError("TEMPORAL_ADDRESS must not be empty")
        if urlparse(self.postgres_dsn).scheme not in {"postgres", "postgresql"}:
            raise ValueError("POSTGRES_DSN must be a PostgreSQL URL")
        if urlparse(self.qdrant_url).scheme not in {"http", "https"}:
            raise ValueError("QDRANT_URL must be an HTTP(S) URL")
        if not self.qdrant_collection.strip():
            raise ValueError("QDRANT_COLLECTION must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty")


def _load_settings() -> Settings:
    settings = Settings(
        project_name=os.getenv("PROJECT_NAME", "Ally"),
        temporal_address=os.getenv("TEMPORAL_ADDRESS", "localhost:7233"),
        upload_dir=Path(os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "ally_uploads"))),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        postgres_dsn=os.getenv(
            "POSTGRES_DSN",
            "postgresql://assistant:assistant@localhost:5432/assistant_db",
        ),
        bm25_index_path=Path(
            os.getenv("BM25_INDEX_PATH", os.path.join(tempfile.gettempdir(), "bm25_index.pkl"))
        ),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "document_chunks_bge_base_en_v1_5"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )
    settings.validate()
    return settings


settings = _load_settings()

PROJECT_NAME = settings.project_name
TEMPORAL_ADDRESS = settings.temporal_address
UPLOAD_DIR = str(settings.upload_dir)
AWS_REGION = settings.aws_region
POSTGRES_DSN = settings.postgres_dsn
BM25_INDEX_PATH = str(settings.bm25_index_path)
QDRANT_URL = settings.qdrant_url
QDRANT_API_KEY = settings.qdrant_api_key
QDRANT_COLLECTION = settings.qdrant_collection
EMBEDDING_MODEL = settings.embedding_model
MONGO_URI = settings.mongo_uri
REDIS_URL = settings.redis_url

# LLM/provider/observability/integration routes.
QWEN_VLLM_BASE_URL = os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:8001/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
GOOGLE_CALENDAR_CREDENTIALS_JSON = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON", "")

# Keep AWS credential names available in the same import surface.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
