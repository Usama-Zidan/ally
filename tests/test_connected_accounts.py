"""Tests for services.agent.connected_accounts — the encrypted per-tenant
OAuth token vault. Uses the real `cryptography` library (not a stub) so
the encrypt/decrypt round-trip is genuinely exercised."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from services.agent import connected_accounts as vault


def _mock_connection():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    return conn, cursor


class VaultNotConfiguredTest(unittest.TestCase):
    def test_save_raises_without_an_encryption_key(self):
        with patch.object(vault, "CONNECTED_ACCOUNTS_ENCRYPTION_KEY", ""):
            with self.assertRaises(vault.VaultNotConfigured):
                vault.save_connected_account("tenant-a", "jira", "token")

    def test_get_raises_without_an_encryption_key(self):
        with patch.object(vault, "CONNECTED_ACCOUNTS_ENCRYPTION_KEY", ""):
            with self.assertRaises(vault.VaultNotConfigured):
                vault.get_connected_account("tenant-a", "jira")


class SaveConnectedAccountTest(unittest.TestCase):
    def setUp(self):
        self.key = Fernet.generate_key().decode()
        self._patch = patch.object(vault, "CONNECTED_ACCOUNTS_ENCRYPTION_KEY", self.key)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_rejects_unsupported_provider(self):
        with self.assertRaises(ValueError):
            vault.save_connected_account("tenant-a", "slack", "token")

    def test_access_token_is_encrypted_not_stored_as_plaintext(self):
        conn, cursor = _mock_connection()
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            vault.save_connected_account("tenant-a", "jira", "super-secret-token")

        params = cursor.execute.call_args.args[1]
        encrypted_access_token = params[2]
        self.assertNotIn("super-secret-token", encrypted_access_token)
        # And it must actually decrypt back correctly with the same key.
        fernet = Fernet(self.key.encode())
        self.assertEqual(
            fernet.decrypt(encrypted_access_token.encode()).decode(), "super-secret-token"
        )

    def test_refresh_token_is_optional(self):
        conn, cursor = _mock_connection()
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            vault.save_connected_account("tenant-a", "jira", "access-only")

        params = cursor.execute.call_args.args[1]
        self.assertIsNone(params[3])  # encrypted_refresh_token

    def test_commits_and_closes_the_connection(self):
        conn, _ = _mock_connection()
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            vault.save_connected_account("tenant-a", "jira", "token")

        conn.commit.assert_called_once_with()
        conn.close.assert_called_once_with()


class GetConnectedAccountTest(unittest.TestCase):
    def setUp(self):
        self.key = Fernet.generate_key().decode()
        self._patch = patch.object(vault, "CONNECTED_ACCOUNTS_ENCRYPTION_KEY", self.key)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.fernet = Fernet(self.key.encode())

    def test_returns_none_when_no_account_is_connected(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = None
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            result = vault.get_connected_account("tenant-a", "jira")
        self.assertIsNone(result)

    def test_decrypts_access_and_refresh_tokens(self):
        conn, cursor = _mock_connection()
        encrypted_access = self.fernet.encrypt(b"my-access-token").decode()
        encrypted_refresh = self.fernet.encrypt(b"my-refresh-token").decode()
        cursor.fetchone.return_value = (encrypted_access, encrypted_refresh, None, ["read"])

        with patch.object(vault.psycopg2, "connect", return_value=conn):
            account = vault.get_connected_account("tenant-a", "jira")

        assert account is not None
        self.assertEqual(account.access_token, "my-access-token")
        self.assertEqual(account.refresh_token, "my-refresh-token")
        self.assertEqual(account.scopes, ["read"])

    def test_handles_a_missing_refresh_token(self):
        conn, cursor = _mock_connection()
        encrypted_access = self.fernet.encrypt(b"my-access-token").decode()
        cursor.fetchone.return_value = (encrypted_access, None, None, [])

        with patch.object(vault.psycopg2, "connect", return_value=conn):
            account = vault.get_connected_account("tenant-a", "jira")

        assert account is not None
        self.assertIsNone(account.refresh_token)

    def test_save_then_get_round_trips_through_a_shared_fake_table(self):
        """Full round trip against an in-memory fake table, proving the
        two functions agree on ciphertext shape end to end."""
        store: dict = {}

        conn, cursor = _mock_connection()

        def fake_execute(sql, params=None):
            assert params is not None
            normalized = " ".join(sql.split())
            if normalized.startswith("INSERT INTO connected_accounts"):
                tenant_id, provider, enc_access, enc_refresh, expires_at, scopes = params
                store[(tenant_id, provider)] = (enc_access, enc_refresh, expires_at, scopes)
            elif normalized.startswith("SELECT encrypted_access_token"):
                cursor.fetchone.return_value = store.get(params)

        cursor.execute.side_effect = fake_execute

        with patch.object(vault.psycopg2, "connect", return_value=conn):
            vault.save_connected_account(
                "tenant-a", "google_drive", "drive-access", refresh_token="drive-refresh"
            )
            account = vault.get_connected_account("tenant-a", "google_drive")

        assert account is not None
        self.assertEqual(account.access_token, "drive-access")
        self.assertEqual(account.refresh_token, "drive-refresh")


class DisconnectAndListTest(unittest.TestCase):
    def test_disconnect_returns_true_when_a_row_was_deleted(self):
        conn, cursor = _mock_connection()
        cursor.rowcount = 1
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            result = vault.disconnect_account("tenant-a", "jira")
        self.assertTrue(result)
        conn.commit.assert_called_once_with()

    def test_disconnect_returns_false_when_nothing_was_connected(self):
        conn, cursor = _mock_connection()
        cursor.rowcount = 0
        with patch.object(vault.psycopg2, "connect", return_value=conn):
            result = vault.disconnect_account("tenant-a", "jira")
        self.assertFalse(result)

    def test_list_connected_providers_does_not_require_the_encryption_key(self):
        # Listing which providers are connected shouldn't need to decrypt
        # anything, so it must work even if the key isn't configured.
        conn, cursor = _mock_connection()
        cursor.fetchall.return_value = [("jira",), ("google_drive",)]
        with patch.object(vault, "CONNECTED_ACCOUNTS_ENCRYPTION_KEY", ""), patch.object(
            vault.psycopg2, "connect", return_value=conn
        ):
            providers = vault.list_connected_providers("tenant-a")
        self.assertEqual(providers, ["jira", "google_drive"])


if __name__ == "__main__":
    unittest.main()
