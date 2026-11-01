"""
Explicit slash-command parsing: `/tool.action key=value key2="quoted value"`.

When present, this bypasses the intent classifier entirely -- an
explicit command from the user always wins over a model's guess about
what they meant. This is also the escape hatch for when the classifier
gets it wrong: the user can always force a specific tool/action directly.

Examples:
    /jira.create_ticket summary="Fix login bug" description="Users can't log in"
    /local_pc.open_app app_name=notepad
    /calendar.list_events start=2026-01-01 end=2026-01-31
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_COMMAND_RE = re.compile(
    r"^/(?P<tool>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<action>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?P<rest>.*)$",
    re.DOTALL,
)
_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


@dataclass
class ParsedCommand:
    tool_name: str
    action: str
    params: dict[str, str]


def parse_command(text: str) -> ParsedCommand | None:
    """Returns a ParsedCommand if ``text`` is an explicit command, else
    None -- meaning the caller should fall through to the classifier.

    Unmatched/malformed key=value fragments in the trailing text are
    silently ignored rather than raising, so a user's typo in one
    parameter doesn't prevent the recognized ones from going through;
    each concrete Tool validates required params itself and raises a
    clear ToolError if something it actually needs is missing.
    """
    match = _COMMAND_RE.match(text.strip())
    if not match:
        return None

    params: dict[str, str] = {}
    for key, value in _KV_RE.findall(match.group("rest")):
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"')
        params[key] = value

    return ParsedCommand(
        tool_name=match.group("tool").lower(),
        action=match.group("action").lower(),
        params=params,
    )
