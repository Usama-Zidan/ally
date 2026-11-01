"""Ticket persistence unit tests; atomic replay behavior requires live PostgreSQL."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.auth.principal import DEFAULT_SCOPES, AuthorizationDenied, Principal
from services.auth.settings import SecuritySettings
from services.auth.tickets import PostgresTicketStore, TicketCapacityExceeded, ticket_hash

PRINCIPAL = Principal("tenant-a", "credential-a", DEFAULT_SCOPES)
ROW = ("credential-a", "tenant-a", sorted(DEFAULT_SCOPES))


class TicketStoreTest(unittest.TestCase):
    def connection(self):
        conn = MagicMock()
        return conn, conn.cursor.return_value.__enter__.return_value

    def test_only_hash_is_stored_and_successful_issue_commits(self):
        conn, cur = self.connection()
        cur.fetchone.side_effect = [ROW, (0, 0)]
        store = PostgresTicketStore(SecuritySettings())
        with patch("services.auth.tickets.connect_auth", return_value=conn):
            token = store.issue(PRINCIPAL, "chat")
        insert = cur.execute.call_args.args
        self.assertIn(ticket_hash(token), insert[1])
        self.assertNotIn(token, str(cur.execute.call_args_list))
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    def test_issue_rechecks_revocation_and_scope_at_database_boundary(self):
        for row in (None, ("credential-a", "tenant-a", [])):
            conn, cur = self.connection()
            cur.fetchone.return_value = row
            with (
                self.subTest(row=row),
                patch("services.auth.tickets.connect_auth", return_value=conn),
            ):
                with self.assertRaises(AuthorizationDenied):
                    PostgresTicketStore(SecuritySettings()).issue(PRINCIPAL, "chat")
            conn.commit.assert_not_called()
            conn.rollback.assert_called_once()
            conn.close.assert_called_once()

    def test_capacity_limit_rolls_back_and_returns_no_ticket(self):
        conn, cur = self.connection()
        cur.fetchone.side_effect = [ROW, (1, 1)]
        with patch("services.auth.tickets.connect_auth", return_value=conn):
            with self.assertRaises(TicketCapacityExceeded):
                PostgresTicketStore(SecuritySettings(max_tickets=1)).issue(PRINCIPAL, "chat")
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()

    def test_consume_uses_one_delete_returning_with_all_security_predicates(self):
        conn, cur = self.connection()
        cur.fetchone.return_value = ROW
        token = "test-ephemeral-token-" + "x" * 32
        with patch("services.auth.tickets.connect_auth", return_value=conn):
            actual = PostgresTicketStore(SecuritySettings()).consume(token, "chat")
        self.assertEqual(actual, PRINCIPAL)
        sql, params = cur.execute.call_args.args
        for predicate in (
            "DELETE FROM",
            "RETURNING",
            "expires_at > NOW()",
            "revoked_at IS NULL",
            "k.tenant_id = t.tenant_id",
            "t.audience = %s",
        ):
            self.assertIn(predicate, sql)
        self.assertEqual(params, (ticket_hash(token), "chat"))
        conn.commit.assert_called_once()

    def test_unknown_expired_or_replayed_ticket_is_none_without_success(self):
        conn, cur = self.connection()
        cur.fetchone.return_value = None
        with patch("services.auth.tickets.connect_auth", return_value=conn):
            self.assertIsNone(PostgresTicketStore(SecuritySettings()).consume("x" * 40, "chat"))

    def test_consumption_uses_current_scopes_not_ticket_issue_snapshot(self):
        conn, cur = self.connection()
        cur.fetchone.return_value = ("credential-a", "tenant-a", [])
        with patch("services.auth.tickets.connect_auth", return_value=conn):
            with self.assertRaises(AuthorizationDenied):
                PostgresTicketStore(SecuritySettings()).consume("x" * 40, "chat")


if __name__ == "__main__":
    unittest.main()
