import unittest
from io import BytesIO

from fastapi.testclient import TestClient

from services.api.main import app


class ApiContractTest(unittest.TestCase):
    def test_health_endpoint_is_ok(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_health_dependencies_degrades_cleanly_without_temporal(self):
        with TestClient(app) as client:
            response = client.get("/health/dependencies")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["temporal"], "unavailable")

    def test_upload_degrades_cleanly_without_temporal(self):
        with TestClient(app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.txt", BytesIO(b"hello world"), "text/plain")},
            )
            self.assertEqual(response.status_code, 503)
            payload = response.json()
            self.assertIn("Temporal service unavailable", payload["detail"])


if __name__ == "__main__":
    unittest.main()
