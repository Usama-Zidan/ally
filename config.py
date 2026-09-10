"""Application configuration shared by all services."""

import os


PROJECT_NAME = os.getenv("PROJECT_NAME", "Ally")
TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://assistant:assistant@localhost:5432/assistant_db")

# Planned / future service endpoints kept in one place for consistency.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

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
