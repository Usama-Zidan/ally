"""Companion storage fails closed and binds a credential to its server."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

LOCAL = str(Path(__file__).resolve().parents[1] / "local_agent")
if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)

import device_credentials as storage  # noqa: E402
import pair as pairing  # noqa: E402

DEVICE_ID = "550e8400-e29b-41d4-a716-446655440000"


class CredentialPolicyTest(unittest.TestCase):
    def test_non_loopback_cleartext_embedded_secrets_and_wrong_endpoints_rejected(self):
        for url in (
            "ws://ally.example/ws/pc-agent",
            "wss://user:password@ally.example/ws/pc-agent",
            "wss://ally.example/ws/pc-agent?token=secret",
            "wss://ally.example/ws/pc-agent#secret",
            "https://ally.example/ws/pc-agent",
            "wss://ally.example/other",
            "wss://ally.example:99999/ws/pc-agent",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                storage.validate_server_url(url)
        self.assertEqual(
            storage.validate_server_url("ws://localhost:8000/ws/pc-agent"),
            "ws://localhost:8000/ws/pc-agent",
        )

    def test_enrollment_never_redirects_a_secret_to_another_origin(self):
        self.assertIsNone(
            pairing._NoRedirect().redirect_request(
                None, None, 307, "redirect", {}, "https://attacker.example"
            )
        )

    def test_redemption_sends_token_in_http_body_not_url(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock()
        response.read.return_value = (
            '{"device_id":"' + DEVICE_ID + '","api_key":"test-key"}'
        ).encode()
        opener = Mock(open=Mock(return_value=response))
        token = "t" * 43
        with patch.object(pairing, "build_opener", return_value=opener):
            credentials = pairing.redeem("wss://ally.example/ws/pc-agent", token)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://ally.example/auth/devices/redeem")
        self.assertIn(token.encode(), request.data)
        self.assertNotIn("test-key", repr(credentials))

    def test_unsupported_os_has_no_plaintext_fallback(self):
        with patch.object(storage.os, "name", "posix"), self.assertRaises(RuntimeError):
            storage._protect(b"secret")

    def test_noninteractive_enrollment_is_denied_before_reading_a_token(self):
        with (
            patch.object(pairing.sys, "argv", ["pair.py", "unused-config.json"]),
            patch.object(pairing.sys.stdin, "isatty", return_value=False),
            patch.object(pairing.getpass, "getpass") as prompt,
            patch.object(pairing, "redeem") as redeem,
            patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as context:
                pairing.main()
        self.assertEqual(context.exception.code, 1)
        prompt.assert_not_called()
        redeem.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires real Windows current-user DPAPI")
    def test_real_dpapi_roundtrip_server_binding_and_explicit_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credentials.dpapi"
            credentials = storage.DeviceCredentials(
                "wss://ally.example/ws/pc-agent", DEVICE_ID, "test-secret-value"
            )
            storage.ensure_storage_available()
            storage.save_credentials(path, credentials)
            self.assertNotIn(credentials.api_key.encode(), path.read_bytes())
            self.assertEqual(storage.load_credentials(path, credentials.server_url), credentials)
            with self.assertRaises(ValueError):
                storage.load_credentials(path, "wss://another.example/ws/pc-agent")
            with self.assertRaises(FileExistsError):
                storage.save_credentials(path, credentials)
            replacement = storage.DeviceCredentials(
                credentials.server_url, DEVICE_ID, "replacement-key"
            )
            storage.save_credentials(path, replacement, replace=True)
            self.assertEqual(storage.load_credentials(path, credentials.server_url), replacement)
            path.write_bytes(b"corrupted")
            with self.assertRaises(OSError):
                storage.load_credentials(path, credentials.server_url)


if __name__ == "__main__":
    unittest.main()
