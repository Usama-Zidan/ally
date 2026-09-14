import unittest

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


if __name__ == "__main__":
    unittest.main()
