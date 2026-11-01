"""Scope, tenant and authentication failure regression tests."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg2
from fastapi import HTTPException

from services.auth import api_keys, dependencies
from services.auth.principal import (
    DEFAULT_SCOPES,
    LEGACY_SCOPES,
    AuthorizationDenied,
    Principal,
    authorize_tool,
)


class PrincipalTest(unittest.TestCase):
    def test_classifier_description_never_probes_unauthorized_accounts(self):
        from services.agent.tools.base import ToolRegistry

        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "jira"
        tool.actions = {"get_ticket": "Read", "create_ticket": "Write"}
        tool.description = "Jira"
        registry.register(tool)
        self.assertNotIn(
            "jira:", registry.describe_for_principal(Principal("a", "key-a", DEFAULT_SCOPES))
        )
        tool.is_available.assert_not_called()
        description = registry.describe_for_principal(
            Principal("a", "key-a", frozenset({"tools:read"}))
        )
        self.assertIn("get_ticket", description)
        self.assertNotIn("create_ticket", description)

    def test_new_credentials_do_not_grant_pc_external_write_or_account_control(self):
        self.assertFalse(
            DEFAULT_SCOPES & {"pc:connect", "pc:execute", "tools:write", "accounts:manage"}
        )
        self.assertIn("pc:execute", LEGACY_SCOPES)

    def test_mutable_input_cannot_change_grants_after_authentication(self):
        scopes = {"documents:read"}
        principal = Principal("a", "key-a", scopes)
        scopes.add("pc:execute")
        with self.assertRaises(AuthorizationDenied):
            principal.require("pc:execute")

    def test_unknown_and_wildcard_grants_are_rejected(self):
        for scopes in ({"*"}, {"admin:all"}, "documents:read"):
            with self.subTest(scopes=scopes), self.assertRaises(ValueError):
                Principal("a", "key-a", scopes)

    def test_read_scope_never_authorizes_mutation_or_other_tenant(self):
        principal = Principal("a", "key-a", frozenset({"tools:read"}))
        authorize_tool(principal, "a", "jira", "get_ticket")
        for tenant, tool, action in (
            ("a", "jira", "create_ticket"),
            ("b", "jira", "get_ticket"),
            ("a", "unregistered", "read"),
        ):
            with self.subTest(tenant=tenant, action=action), self.assertRaises(AuthorizationDenied):
                authorize_tool(principal, tenant, tool, action)

    def test_missing_principal_cannot_execute_a_tool(self):
        with self.assertRaises(AuthorizationDenied):
            authorize_tool(None, "a", "local_pc", "read_file")

    def test_resolver_returns_credential_identity_and_database_grants(self):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = ("key-a", "a", ["documents:read"])
        with patch.object(api_keys, "connect_auth", return_value=conn):
            principal = api_keys.resolve_principal("ally_short-test-key")
        self.assertEqual(principal, Principal("a", "key-a", frozenset({"documents:read"})))
        self.assertIn("revoked_at IS NULL", cur.execute.call_args.args[0])
        conn.close.assert_called_once()


class PrincipalDependencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_ingestion_scope_is_required(self):
        principal = Principal("a", "key-a", frozenset({"documents:read"}))
        with patch.object(dependencies, "resolve_principal", return_value=principal):
            with self.assertRaises(HTTPException) as error:
                await dependencies.get_ingestion_tenant("ally_short-test-key")
        self.assertEqual(error.exception.status_code, 403)

    async def test_auth_outage_is_not_an_invalid_key_and_never_exposes_error(self):
        with patch.object(
            dependencies,
            "resolve_principal",
            side_effect=psycopg2.OperationalError("secret-sentinel"),
        ):
            with self.assertRaises(HTTPException) as error:
                await dependencies.get_current_principal("key")
        self.assertEqual(error.exception.status_code, 503)
        self.assertNotIn("secret-sentinel", error.exception.detail)

    async def test_explicit_pc_command_still_needs_authenticated_scope(self):
        from services.agent.classifier import Intent
        from services.agent.graph import tool_node

        tool = MagicMock()
        tool.name = "local_pc"
        tool.execute = AsyncMock()
        registry = MagicMock()
        registry.get.return_value = tool
        with patch("services.agent.graph.get_registry", return_value=registry):
            result = await tool_node(
                {
                    "tenant_id": "a",
                    "query": "/local_pc.read_file path=C:/private.txt",
                    "principal": Principal("a", "key-a", DEFAULT_SCOPES),
                    "intent": Intent(
                        kind="tool",
                        tool_name="local_pc",
                        action="read_file",
                        params={"path": "C:/private.txt"},
                    ),
                }
            )
        self.assertIn("not permitted", result["tool_error"])
        tool.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
