"""Tests for services.auth.api_keys — this module is the entire
authentication/tenant-resolution boundary, so it gets direct coverage of
its own rather than only being exercised indirectly through main.py."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock, patch

import psycopg2

from services.auth import api_keys


def _mock_connection():
    """Builds a MagicMock connection whose `with conn.cursor() as cur`
    yields a controllable fake cursor."""
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    return conn, cursor


class HashKeyTest(unittest.TestCase):
    def test_hash_is_sha256_hex_digest(self):
        raw = "ally_example-key"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self.assertEqual(api_keys.hash_key(raw), expected)

    def test_hash_never_reproduces_the_raw_key(self):
        digest = api_keys.hash_key("ally_secret")
        self.assertNotIn("secret", digest)

    def test_different_keys_hash_differently(self):
        self.assertNotEqual(api_keys.hash_key("ally_one"), api_keys.hash_key("ally_two"))


class GenerateApiKeyTest(unittest.TestCase):
    def test_generated_key_carries_the_expected_prefix(self):
        raw_key, _ = api_keys.generate_api_key()
        self.assertTrue(raw_key.startswith(api_keys.API_KEY_PREFIX))

    def test_returned_hash_matches_the_raw_key(self):
        raw_key, key_hash = api_keys.generate_api_key()
        self.assertEqual(key_hash, api_keys.hash_key(raw_key))

    def test_successive_keys_are_not_identical(self):
        first, _ = api_keys.generate_api_key()
        second, _ = api_keys.generate_api_key()
        self.assertNotEqual(first, second)


class ResolveTenantTest(unittest.TestCase):
    def test_missing_key_resolves_to_none_without_querying_the_database(self):
        with patch.object(psycopg2, "connect") as connect:
            result = api_keys.resolve_tenant(None)

        self.assertIsNone(result)
        connect.assert_not_called()

    def test_empty_key_resolves_to_none_without_querying_the_database(self):
        with patch.object(psycopg2, "connect") as connect:
            result = api_keys.resolve_tenant("")

        self.assertIsNone(result)
        connect.assert_not_called()

    def test_known_active_key_resolves_to_its_tenant(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = ("tenant-123",)

        with patch.object(psycopg2, "connect", return_value=conn):
            result = api_keys.resolve_tenant("ally_valid-key")

        self.assertEqual(result, "tenant-123")
        query, params = cursor.execute.call_args.args
        self.assertIn("revoked_at IS NULL", query)
        self.assertEqual(params, (api_keys.hash_key("ally_valid-key"),))

    def test_unknown_key_resolves_to_none(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = None

        with patch.object(psycopg2, "connect", return_value=conn):
            result = api_keys.resolve_tenant("ally_unknown-key")

        self.assertIsNone(result)

    def test_connection_is_always_closed_even_on_query_failure(self):
        conn, cursor = _mock_connection()
        cursor.execute.side_effect = RuntimeError("query failed")

        with patch.object(psycopg2, "connect", return_value=conn):
            with self.assertRaises(RuntimeError):
                api_keys.resolve_tenant("ally_some-key")

        conn.close.assert_called_once_with()

    def test_revoked_key_is_filtered_out_by_the_query_not_python(self):
        # revoked_at IS NULL is part of the SQL itself, not a post-filter
        # in Python -- a revoked key must never reach fetchone() as a hit.
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = None  # simulates the row being excluded by SQL

        with patch.object(psycopg2, "connect", return_value=conn):
            result = api_keys.resolve_tenant("ally_revoked-key")

        self.assertIsNone(result)

    def test_transient_connection_failure_is_retried_then_succeeds(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = ("tenant-123",)

        attempts = {"n": 0}

        def flaky_connect(dsn):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise psycopg2.OperationalError("connection refused")
            return conn

        with patch.object(psycopg2, "connect", side_effect=flaky_connect):
            result = api_keys.resolve_tenant("ally_some-key")

        self.assertEqual(result, "tenant-123")
        self.assertEqual(attempts["n"], 3)

    def test_persistent_connection_failure_exhausts_retries_and_reraises(self):
        with patch.object(
            psycopg2, "connect", side_effect=psycopg2.OperationalError("still down")
        ) as connect:
            with self.assertRaises(psycopg2.OperationalError):
                api_keys.resolve_tenant("ally_some-key")

        # stop_after_attempt(3): exactly 3 attempts, not fewer, not more.
        self.assertEqual(connect.call_count, 3)

    def test_non_transient_error_is_not_retried(self):
        # Only OperationalError (connection-level, transient) is retried.
        # A different exception type must fail immediately.
        with patch.object(
            psycopg2, "connect", side_effect=RuntimeError("not a connection issue")
        ) as connect:
            with self.assertRaises(RuntimeError):
                api_keys.resolve_tenant("ally_some-key")

        connect.assert_called_once()


class CreateTenantWithKeyTest(unittest.TestCase):
    def test_creates_tenant_and_issues_a_key_in_one_transaction(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = ("tenant-abc",)

        with patch.object(psycopg2, "connect", return_value=conn):
            tenant_id, raw_key = api_keys.create_tenant_with_key("Acme Corp")

        self.assertEqual(tenant_id, "tenant-abc")
        self.assertTrue(raw_key.startswith(api_keys.API_KEY_PREFIX))
        conn.commit.assert_called_once_with()
        conn.close.assert_called_once_with()

        insert_tenant_call, insert_key_call = cursor.execute.call_args_list
        self.assertIn("INSERT INTO tenants", insert_tenant_call.args[0])
        self.assertIn("INSERT INTO api_keys", insert_key_call.args[0])
        self.assertEqual(insert_key_call.args[1][0], "tenant-abc")

    def test_raises_if_tenant_insert_returns_no_row(self):
        conn, cursor = _mock_connection()
        cursor.fetchone.return_value = None

        with patch.object(psycopg2, "connect", return_value=conn):
            with self.assertRaises(RuntimeError):
                api_keys.create_tenant_with_key("Acme Corp")

        conn.commit.assert_not_called()
        conn.close.assert_called_once_with()


class RevokeApiKeyTest(unittest.TestCase):
    def test_revoking_an_active_key_returns_true(self):
        conn, cursor = _mock_connection()
        cursor.rowcount = 1

        with patch.object(psycopg2, "connect", return_value=conn):
            result = api_keys.revoke_api_key("ally_active-key")

        self.assertTrue(result)
        conn.commit.assert_called_once_with()

    def test_revoking_an_unknown_or_already_revoked_key_returns_false(self):
        conn, cursor = _mock_connection()
        cursor.rowcount = 0

        with patch.object(psycopg2, "connect", return_value=conn):
            result = api_keys.revoke_api_key("ally_already-revoked")

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
