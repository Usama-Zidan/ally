"""Enrollment bounds, native-only policy, error redaction and secret transport."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import psycopg2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.routes.devices import router
from services.auth.admission import PairingAdmission
from services.auth.devices import DeviceEnrollment
from services.auth.principal import AuthorizationDenied
from services.auth.settings import SecuritySettings


class EnrollmentTransportTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.configure(SecuritySettings())
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.token = "t" * 43

    def configure(self, settings):
        self.registry = Mock(redeem=Mock(return_value=DeviceEnrollment("device-id", "test-key")))
        self.app.state.device_registry = self.registry
        self.app.state.pairing_admission = PairingAdmission(settings)
        self.app.state.websocket_security = Mock(settings=settings)

    def test_redemption_is_not_cached_and_takes_no_tenant_or_device_claim(self):
        response = self.client.post("/auth/devices/redeem", json={"token": self.token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json(), {"device_id": "device-id", "api_key": "test-key"})
        self.registry.redeem.assert_called_once_with(self.token)

    def test_invalid_bodies_and_client_identity_claims_fail_before_db(self):
        for payload in (
            {"token": "short"},
            {"token": self.token, "tenant_id": "victim"},
            {"token": self.token, "device_id": "spoof"},
            {"token": 123},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(
                    self.client.post("/auth/devices/redeem", json=payload).status_code, 422
                )
        self.assertEqual(
            self.client.post("/auth/devices/redeem", content="x" * 1025).status_code, 413
        )
        self.registry.redeem.assert_not_called()

    def test_browser_origin_and_insecure_production_request_fail_before_db(self):
        response = self.client.post(
            "/auth/devices/redeem",
            json={"token": self.token},
            headers={"Origin": "https://ally.example"},
        )
        self.assertEqual(response.status_code, 403)
        self.configure(
            SecuritySettings(environment="production", require_tls=True, allow_query_api_keys=False)
        )
        self.assertEqual(
            self.client.post("/auth/devices/redeem", json={"token": self.token}).status_code, 400
        )
        self.registry.redeem.assert_not_called()

    def test_rate_limit_cannot_be_bypassed_with_another_token(self):
        self.configure(SecuritySettings(auth_attempts_per_minute=1))
        self.registry.redeem.side_effect = AuthorizationDenied("sensitive SQL")
        self.assertEqual(
            self.client.post("/auth/devices/redeem", json={"token": self.token}).status_code, 403
        )
        self.assertEqual(
            self.client.post("/auth/devices/redeem", json={"token": "q" * 43}).status_code, 429
        )
        self.assertEqual(self.registry.redeem.call_count, 1)

    def test_database_errors_never_expose_driver_or_credential_details(self):
        self.registry.redeem.side_effect = psycopg2.OperationalError("password=private")
        response = self.client.post("/auth/devices/redeem", json={"token": self.token})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()
