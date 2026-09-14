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

    def test_load_settings_uses_environment_overrides(self):
        environment = {
            "PROJECT_NAME": "Test Ally",
            "TEMPORAL_ADDRESS": "temporal.example:7233",
            "UPLOAD_DIR": "/tmp/custom-uploads",
            "POSTGRES_DSN": "postgres://user:pass@db/test",
            "BM25_INDEX_PATH": "/tmp/custom-bm25.pkl",
            "QDRANT_URL": "https://qdrant.example",
            "QDRANT_COLLECTION": "test_chunks",
            "EMBEDDING_MODEL": "test/model",
        }
        with patch.dict(config.os.environ, environment, clear=True):
            loaded = config._load_settings()

        self.assertEqual(loaded.project_name, "Test Ally")
        self.assertEqual(loaded.temporal_address, "temporal.example:7233")
        self.assertEqual(loaded.upload_dir, Path("/tmp/custom-uploads"))
        self.assertEqual(loaded.bm25_index_path, Path("/tmp/custom-bm25.pkl"))
        self.assertEqual(loaded.qdrant_url, "https://qdrant.example")
        self.assertEqual(loaded.qdrant_collection, "test_chunks")
        self.assertEqual(loaded.embedding_model, "test/model")

    def test_settings_reject_invalid_required_values(self):
        valid = {
            "project_name": "Ally",
            "temporal_address": "localhost:7233",
            "upload_dir": Path("/tmp/uploads"),
            "aws_region": "us-east-1",
            "postgres_dsn": "postgresql://localhost/ally",
            "bm25_index_path": Path("/tmp/bm25.pkl"),
            "qdrant_url": "http://localhost:6333",
            "qdrant_api_key": "",
            "qdrant_collection": "chunks",
            "embedding_model": "test/model",
            "mongo_uri": "mongodb://localhost:27017",
            "redis_url": "redis://localhost:6379",
        }
        invalid_cases = {
            "project_name": " ",
            "temporal_address": "",
            "postgres_dsn": "mysql://localhost/ally",
            "qdrant_url": "grpc://localhost:6333",
            "qdrant_collection": " ",
            "embedding_model": "",
        }

        for field, value in invalid_cases.items():
            with self.subTest(field=field):
                candidate = Settings(**{**valid, field: value})
                with self.assertRaises(ValueError):
                    candidate.validate()


if __name__ == "__main__":
    unittest.main()
