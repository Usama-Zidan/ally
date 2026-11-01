"""Validate untrusted PC command arguments before requesting local approval."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

MAX_TEXT_CHARS = 65536

_REQUIRED = {
    "open_app": (),
    "list_windows": (),
    "focus_window": ("title",),
    "close_window": ("title",),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "list_directory": ("path",),
    "run_command": ("command",),
    "move_mouse": ("x", "y"),
    "move_mouse_to_corner": ("corner",),
    "click": (),
    "type_text": ("text",),
    "press_key": ("key",),
}
_OPTIONAL = {
    "open_app": (
        "app",
        "app_name",
        "name",
        "application",
        "target",
        "url",
        "program",
        "search_query",
    ),
    "run_command": ("cwd", "timeout_seconds"),
    "click": ("button",),
    "move_mouse_to_corner": (),
}


def validate_action(action: str, params: dict[str, Any]) -> None:
    """Reject unknown keys, unsafe sizes and incorrect types without side effects."""
    if action not in _REQUIRED or not isinstance(params, dict):
        raise ValueError("Invalid action or arguments")
    required = _REQUIRED[action]
    allowed = {*required, *_OPTIONAL.get(action, ())}
    if set(params) - allowed or any(key not in params for key in required):
        raise ValueError("Missing or unsupported action arguments")
    for key, value in params.items():
        if key in {"x", "y"}:
            if type(value) is not int or not -100000 <= value <= 100000:
                raise ValueError("Mouse coordinates must be bounded integers")
        elif key == "timeout_seconds":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError("Command timeout must be between 0 and 120 seconds")
        elif not isinstance(value, str) or len(value) > MAX_TEXT_CHARS:
            raise ValueError("Arguments must be bounded strings")
        elif not value.strip() and key not in {"content", "text"}:
            raise ValueError("Action argument must not be empty")
    if action == "open_app" and not any(params.values()):
        raise ValueError("open_app requires an app or website name")
    if action == "run_command" and (len(params["command"]) > 8192 or "\0" in params["command"]):
        raise ValueError("Shell command must be at most 8192 characters without NUL bytes")
    for key in ("path", "cwd"):
        if key in params and not Path(params[key]).is_absolute():
            raise ValueError("Filesystem paths must be absolute")
    if action == "click" and params.get("button", "left") not in {"left", "right", "middle"}:
        raise ValueError("Invalid mouse button")
    if action == "move_mouse_to_corner" and params.get("corner") not in {
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }:
        raise ValueError("Invalid screen corner")
