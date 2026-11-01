"""Framework-free credential identity and authorization rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

KNOWN_SCOPES = frozenset(
    {
        "documents:ingest",
        "documents:read",
        "chat:use",
        "chat:history",
        "tools:read",
        "tools:write",
        "pc:connect",
        "pc:execute",
        "accounts:manage",
    }
)
DEFAULT_SCOPES = frozenset({"documents:ingest", "documents:read", "chat:use", "chat:history"})
# Explicit compatibility grants for keys that existed before scoped credentials.
# No administrative identity or authority to provision credentials is implied.
LEGACY_SCOPES = KNOWN_SCOPES - {"accounts:manage"}


class AuthorizationDenied(Exception):
    """A credential lacks permission for the requested operation."""


def validate_scopes(scopes: Iterable[str]) -> frozenset[str]:
    """Reject unknown grants, including wildcard scopes."""
    if isinstance(scopes, str):
        raise ValueError("Scopes must be a collection")
    values = frozenset(scopes)
    if not values <= KNOWN_SCOPES:
        raise ValueError("Unknown credential scope")
    return values


@dataclass(frozen=True)
class Principal:
    """A tenant credential, deliberately not an individual user identity."""

    tenant_id: str
    credential_id: str
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.credential_id.strip():
            raise ValueError("Principal identity must not be empty")
        object.__setattr__(self, "scopes", validate_scopes(self.scopes))

    def require(self, *scopes: str) -> None:
        if not set(scopes) <= self.scopes:
            raise AuthorizationDenied("Credential is not permitted to perform this operation")


_TOOL_SCOPES = {
    "jira": {
        "create_ticket": "tools:write",
        "search_tickets": "tools:read",
        "get_ticket": "tools:read",
        "add_comment": "tools:write",
    },
    "drive": {"search_files": "tools:read", "get_file": "tools:read", "create_file": "tools:write"},
    "calendar": {
        "list_events": "tools:read",
        "create_event": "tools:write",
        "delete_event": "tools:write",
    },
    "local_pc": {
        action: "pc:execute"
        for action in (
            "open_app",
            "list_windows",
            "focus_window",
            "close_window",
            "read_file",
            "write_file",
            "list_directory",
            "run_command",
            "move_mouse",
            "move_mouse_to_corner",
            "click",
            "type_text",
            "press_key",
        )
    },
}


def authorize_tool(principal: Principal | None, tenant_id: str, tool: str, action: str) -> None:
    """Fail closed on absent identity, mismatched tenant or unknown action policy."""
    required = _TOOL_SCOPES.get(tool, {}).get(action)
    if principal is None or principal.tenant_id != tenant_id or required is None:
        raise AuthorizationDenied("Tool operation is not authorized")
    principal.require(required)
