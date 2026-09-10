import unittest

from config import (
    PROJECT_NAME,
    TEMPORAL_ADDRESS,
    UPLOAD_DIR,
    AWS_REGION,
    POSTGRES_DSN,
    QDRANT_URL,
    MONGO_URI,
    REDIS_URL,
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


if __name__ == "__main__":
    unittest.main()
