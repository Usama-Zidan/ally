import unittest
from pathlib import Path
from unittest.mock import patch

import config
from config import (
    EMBEDDING_MODEL,
    PROJECT_NAME,
    TEMPORAL_ADDRESS,
    UPLOAD_DIR,
    AWS_REGION,
    POSTGRES_DSN,
    QDRANT_URL,
    MONGO_URI,
    REDIS_URL,
    QDRANT_COLLECTION,
    MAX_UPLOAD_BYTES,
    BM25_REBUILD_MIN_INTERVAL_SECONDS,
    TEXTRACT_S3_BUCKET,
    Settings,
)


class ConfigContractTest(unittest.TestCase):
    def test_shared_config_exposes_wired_env_settings(self):
        self.assertIsInstance(PROJECT_NAME, str)
        self.assertIsInstance(TEMPORAL_ADDRESS, str)
        self.assertIsInstance(UPLOAD_DIR, str)
        self.assertIsInstance(AWS_REGION, str)
        self.assertIsInstance(POSTGRES_DSN, str)
        self.assertIsInstance(QDRANT_URL, str)
        self.assertIsInstance(MONGO_URI, str)
        self.assertIsInstance(REDIS_URL, str)
        self.assertTrue(POSTGRES_DSN.startswith(("postgres://", "postgresql://")))
        self.assertIsInstance(QDRANT_COLLECTION, str)
        self.assertIsInstance(EMBEDDING_MODEL, str)
        self.assertIsInstance(MAX_UPLOAD_BYTES, int)
        self.assertGreater(MAX_UPLOAD_BYTES, 0)
        self.assertIsInstance(BM25_REBUILD_MIN_INTERVAL_SECONDS, float)
        self.assertIsInstance(TEXTRACT_S3_BUCKET, str)

    def test_settings_accept_valid_supported_urls(self):
        settings = self._settings(
            postgres_dsn="postgres://user:pass@db/name",
            qdrant_url="https://qdrant.example.test",
        )

        self.assertIsNone(settings.validate())

    def test_settings_reject_each_required_invalid_value(self):
        invalid_values = {
            "project_name": "  ",
            "temporal_address": "",
            "postgres_dsn": "mysql://localhost/db",
            "qdrant_url": "ftp://localhost",
            "qdrant_collection": " ",
            "embedding_model": "",
            "max_upload_bytes": 0,
        }

        for field, value in invalid_values.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._settings(**{field: value}).validate()

    def test_load_settings_reads_retrieval_environment_overrides(self):
        overrides = {
            "PROJECT_NAME": "Test Ally",
            "TEMPORAL_ADDRESS": "temporal.test:7233",
            "UPLOAD_DIR": "/tmp/test-uploads",
            "POSTGRES_DSN": "postgresql://user:pass@db/test",
            "BM25_INDEX_PATH": "/tmp/test-index.pkl",
            "QDRANT_URL": "https://qdrant.test",
            "QDRANT_API_KEY": "local-key",
            "QDRANT_COLLECTION": "test_chunks",
            "EMBEDDING_MODEL": "test-embedding",
            "MAX_UPLOAD_BYTES": "1234",
            "BM25_REBUILD_MIN_INTERVAL_SECONDS": "90",
            "TEXTRACT_S3_BUCKET": "test-bucket",
        }
        with patch.dict(config.os.environ, overrides, clear=True):
            loaded = config._load_settings()

        self.assertEqual(loaded.project_name, "Test Ally")
        self.assertEqual(loaded.temporal_address, "temporal.test:7233")
        self.assertEqual(loaded.upload_dir, Path("/tmp/test-uploads"))
        self.assertEqual(loaded.bm25_index_path, Path("/tmp/test-index.pkl"))
        self.assertEqual(loaded.qdrant_url, "https://qdrant.test")
        self.assertEqual(loaded.qdrant_api_key, "local-key")
        self.assertEqual(loaded.qdrant_collection, "test_chunks")
        self.assertEqual(loaded.embedding_model, "test-embedding")
        self.assertEqual(loaded.max_upload_bytes, 1234)
        self.assertEqual(loaded.bm25_rebuild_min_interval_seconds, 90.0)
        self.assertEqual(loaded.textract_s3_bucket, "test-bucket")

    @staticmethod
    def _settings(**overrides):
        values = {
            "project_name": "Ally",
            "temporal_address": "localhost:7233",
            "upload_dir": Path("/tmp/uploads"),
            "max_upload_bytes": 50 * 1024 * 1024,
            "aws_region": "us-east-1",
            "textract_s3_bucket": "",
            "postgres_dsn": "postgresql://user:pass@localhost/db",
            "bm25_index_path": Path("/tmp/index.pkl"),
            "bm25_rebuild_min_interval_seconds": 60.0,
            "qdrant_url": "http://localhost:6333",
            "qdrant_api_key": "",
            "qdrant_collection": "chunks",
            "embedding_model": "embedding-model",
            "mongo_uri": "mongodb://localhost:27017",
            "redis_url": "redis://localhost:6379",
        }
        values.update(overrides)
        return Settings(**values)


if __name__ == "__main__":
    unittest.main()
