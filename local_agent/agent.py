"""
Local companion agent: runs on your own Windows PC, connects outbound to
the Ally backend over WebSocket, and executes commands the chat agent
sends it -- opening apps, reading/writing files, running shell commands,
or controlling the mouse/keyboard, depending on what YOU enable below.

THIS FILE IS THE ACTUAL SECURITY BOUNDARY, not the server. The backend's
`pc_agent_capabilities` table and the handshake capabilities this script
reports are a declaration, not an enforcement mechanism -- the server
cannot force your machine to do anything. Every capability check in this
script re-validates locally regardless of what the server believes is
enabled, because this script is the only thing that can actually stop an
action from happening.

Setup:
    1. pip install -r requirements.txt
    2. cp config.example.json config.json, then edit it:
       - server_url: your backend's wss:// address
       - device_credentials_file: Windows-protected credential file
       - capabilities: which of the four categories below to allow
       - allowed_folders: absolute paths file_rw is restricted to
    3. python pair.py config.json (use the operator's one-time enrollment token)
    4. python agent.py

Capabilities (each independently toggleable in config.json):
    open_apps               -- launch applications, list/focus/close windows
    file_rw                 -- read/write files, ONLY inside allowed_folders
    shell_exec               -- run arbitrary shell commands (HIGH RISK: see README)
    mouse_keyboard_control   -- move the mouse, click, type, press keys

Optional dependencies degrade gracefully: if pygetwindow/pyautogui aren't
installed, their capabilities report a clear error instead of crashing
the whole agent, so you can use (for example) just file_rw without
installing packages the other capabilities need.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import InvalidHandshake

import app_launcher
from action_policy import validate_action
from device_credentials import credentials_path, load_credentials, validate_server_url
from execution_audit import ExecutionAudit
from local_approval import approve_locally
from shell_runner import run_shell

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ally-local-agent")

# A dedicated, append-only audit log for the highest-risk capability.
# Always written when shell_exec is enabled, regardless of the general
# logging level above -- a record of every command this agent has ever
# run on your behalf should not be optional.
_AUDIT_LOG_PATH = Path(__file__).parent / "shell_exec_audit.log"
MAX_FILE_CHARS = 8192
MAX_LIST_ENTRIES = 200
MAX_LIST_CHARS = 8192
MAX_RESULT_BYTES = 65536


def _bounded_names(names: Iterable[str]) -> tuple[list[str], bool]:
    result: list[str] = []
    characters = 0
    for name in names:
        if len(result) >= MAX_LIST_ENTRIES or characters + len(name) > MAX_LIST_CHARS:
            return result, True
        result.append(name)
        characters += len(name)
    return result, False


def _encode_result(request_id: str, details: dict[str, Any]) -> str:
    message = json.dumps(
        {"type": "result", "request_id": request_id, "success": True, "details": details},
        ensure_ascii=False,
    )
    if len(message.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("Action result exceeds transport size limit")
    return message


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("capabilities", {})
    for capability in ("open_apps", "file_rw", "shell_exec", "mouse_keyboard_control"):
        config["capabilities"].setdefault(capability, False)
    config.setdefault("allowed_folders", [])
    config.setdefault("app_aliases", {})
    return config


def _resolve_allowed_path(raw_path: str, allowed_folders: list[str]) -> Path:
    """Resolves ``raw_path`` and verifies it falls inside one of
    ``allowed_folders``. Raises ValueError otherwise.

    Resolves symlinks and ".." components before checking (Path.resolve())
    specifically so "allowed_folder/../../../etc/passwd" can't escape the
    sandbox by construction -- checking the raw, unresolved string against
    the allowed prefix would not catch that.
    """
    resolved = Path(raw_path).resolve()
    for folder in allowed_folders:
        folder_resolved = Path(folder).resolve()
        if resolved == folder_resolved or folder_resolved in resolved.parents:
            return resolved
    raise ValueError(f"{raw_path!r} is not inside any allowed folder. Allowed: {allowed_folders}")


_APP_PARAM_KEYS = ("app", "app_name", "name", "application", "target", "url", "program")


class CapabilityDisabled(Exception):
    pass


class LocalAgent:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        approve: Callable[[str, dict[str, Any]], bool] | None = None,
        cancel: threading.Event | None = None,
        deadline: float | None = None,
    ):
        self.config = config
        self.capabilities: dict[str, bool] = config["capabilities"]
        self.allowed_folders: list[str] = config["allowed_folders"]
        self.app_aliases: dict[str, str] = config.get("app_aliases", {})
        self._approve = approve
        self._cancel = cancel if cancel is not None else threading.Event()
        self._deadline = deadline

    def _execution_allowed(self) -> bool:
        return not self._cancel.is_set() and (
            self._deadline is None or time.monotonic() < self._deadline
        )

    def _require(
        self,
        capability: str,
        action: str,
        params: dict[str, Any],
        *,
        approval_context: dict[str, Any] | None = None,
    ) -> None:
        if self.capabilities.get(capability) is not True:
            raise CapabilityDisabled(
                f"'{capability}' is disabled in this agent's local config.json."
            )
        validate_action(action, params)
        if not self._execution_allowed():
            raise PermissionError("Action expired or disconnected")
        approved = (
            self._approve(action, params)
            if self._approve is not None
            else approve_locally(
                action,
                params,
                cancel=self._cancel,
                deadline=self._deadline,
                approval_context=approval_context,
            )
        )
        if approved is not True or not self._execution_allowed():
            raise PermissionError("Action was not approved on this device")

    # ---- open_apps ----

    def open_app(self, params: dict[str, Any]) -> dict[str, Any]:
        # Small models don't reliably use the documented key, so accept
        # the names they actually produce.
        target = next((params[k] for k in _APP_PARAM_KEYS if params.get(k)), None)
        if not target:
            raise ValueError("open_app requires 'app' (an app or website name).")
        target = str(target)
        search_query = params.get("search_query")
        if search_query or app_launcher.needs_web_search(target, self.app_aliases):
            if self.capabilities.get("open_apps") is not True:
                raise CapabilityDisabled(
                    "'open_apps' is disabled in this agent's local config.json."
                )
            validate_action("open_app", params)
            if not self._execution_allowed():
                raise PermissionError("Action expired or disconnected")
            search_query = str(search_query or target)
            candidate = app_launcher.search_web(search_query)
            if candidate is None:
                raise ValueError("Local web search found no result to suggest.")
            self._require(
                "open_apps",
                "open_app",
                params,
                approval_context={"query": search_query, "candidate": candidate},
            )
            return app_launcher.open_search_result(search_query, candidate)
        self._require("open_apps", "open_app", params)
        return app_launcher.resolve_and_launch(target, self.app_aliases)

    def list_windows(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps", "list_windows", params)
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "list_windows requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        windows, truncated = _bounded_names(w.title for w in gw.getAllWindows() if w.title)
        return {"windows": windows, "truncated": truncated}

    def focus_window(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps", "focus_window", params)
        title = params.get("title")
        if not title:
            raise ValueError("focus_window requires 'title'.")
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "focus_window requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            raise ValueError(f"No window matching {title!r} found.")
        matches[0].activate()
        return {"focused": matches[0].title}

    def close_window(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps", "close_window", params)
        title = params.get("title")
        if not title:
            raise ValueError("close_window requires 'title'.")
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "close_window requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            raise ValueError(f"No window matching {title!r} found.")
        matches[0].close()
        return {"closed": matches[0].title}

    # ---- file_rw ----

    def read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw", "read_file", params)
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        with path.open("r", encoding="utf-8") as stream:
            content = stream.read(MAX_FILE_CHARS + 1)
        return {
            "path": str(path),
            "content": content[:MAX_FILE_CHARS],
            "truncated": len(content) > MAX_FILE_CHARS,
        }

    def write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw", "write_file", params)
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        content = params.get("content", "")
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}

    def list_directory(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw", "list_directory", params)
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        with os.scandir(path) as entries:
            names, truncated = _bounded_names(entry.name for entry in entries)
        return {"path": str(path), "entries": sorted(names), "truncated": truncated}

    # ---- shell_exec ----

    def run_command(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("shell_exec", "run_command", params)
        command = params.get("command")
        if not command:
            raise ValueError("run_command requires 'command'.")

        audit = ExecutionAudit(_AUDIT_LOG_PATH, command)
        audit.approved()
        try:
            result = run_shell(
                command,
                cwd=params.get("cwd"),
                timeout=params.get("timeout_seconds", 30),
                cancel=self._cancel,
                deadline=self._deadline,
            )
        except BaseException:
            try:
                audit.unknown()
            except OSError:
                log.error("shell terminal audit unavailable; execution status unknown")
            raise
        recorded = True
        try:
            audit.finished(result)
        except OSError:
            # The command may have run: report lost audit explicitly, never as
            # an invitation to retry an operation whose effects are unknown.
            recorded = False
            log.error("shell terminal audit unavailable; inspect device before retrying")
        return {
            **asdict(result),
            "execution_id": audit.execution_id,
            "effect_outcome": "not_verified" if result.started else "not_started",
            "audit_recorded": recorded,
        }

    # ---- mouse_keyboard_control ----

    def _pyautogui(self):
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "mouse/keyboard actions require the optional 'pyautogui' package "
                "(pip install pyautogui)."
            ) from exc
        return pyautogui

    def move_mouse(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control", "move_mouse", params)
        pyautogui = self._pyautogui()
        x, y = params.get("x"), params.get("y")
        if x is None or y is None:
            raise ValueError("move_mouse requires 'x' and 'y'.")
        pyautogui.moveTo(x, y)
        return {"x": x, "y": y}

    def move_mouse_to_corner(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control", "move_mouse_to_corner", params)
        pyautogui = self._pyautogui()
        corner = params["corner"]
        width, height = pyautogui.size()
        x = width - 1 if corner.endswith("right") else 0
        y = height - 1 if corner.startswith("bottom") else 0
        pyautogui.moveTo(x, y)
        return {"corner": corner, "x": x, "y": y}

    def click(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control", "click", params)
        pyautogui = self._pyautogui()
        pyautogui.click(button=params.get("button", "left"))
        return {"button": params.get("button", "left")}

    def type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control", "type_text", params)
        pyautogui = self._pyautogui()
        text = params.get("text", "")
        pyautogui.typewrite(text)
        return {"characters_typed": len(text)}

    def press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control", "press_key", params)
        pyautogui = self._pyautogui()
        key = params.get("key")
        if not key:
            raise ValueError("press_key requires 'key'.")
        pyautogui.press(key)
        return {"key": key}

    # ---- dispatch ----

    _HANDLERS = {
        "open_app": "open_app",
        "list_windows": "list_windows",
        "focus_window": "focus_window",
        "close_window": "close_window",
        "read_file": "read_file",
        "write_file": "write_file",
        "list_directory": "list_directory",
        "run_command": "run_command",
        "move_mouse": "move_mouse",
        "move_mouse_to_corner": "move_mouse_to_corner",
        "click": "click",
        "type_text": "type_text",
        "press_key": "press_key",
    }

    def handle_command(
        self,
        action: str,
        params: dict[str, Any],
        *,
        expires_at: float | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        handler_name = self._HANDLERS.get(action)
        if handler_name is None:
            raise ValueError(f"Unknown action {action!r}.")
        deadline = self._deadline
        if expires_at is not None:
            if type(expires_at) not in (int, float) or not math.isfinite(expires_at):
                raise ValueError("Invalid command expiration")
            deadline = time.monotonic() + min(120, expires_at - time.time())

        command_agent = LocalAgent(
            {
                **self.config,
                "capabilities": dict(self.capabilities),
                "allowed_folders": list(self.allowed_folders),
                "app_aliases": dict(self.app_aliases),
            },
            approve=self._approve,
            cancel=cancel if cancel is not None else self._cancel,
            deadline=deadline,
        )
        handler = getattr(command_agent, handler_name)
        return handler(params)


async def _run_with_disconnect(
    ws: Any, agent: LocalAgent, action: str, params: dict[str, Any], *, expires_at: float
) -> dict[str, Any] | None:
    cancel = threading.Event()
    execution = asyncio.create_task(
        asyncio.to_thread(
            agent.handle_command,
            action,
            params,
            expires_at=expires_at,
            cancel=cancel,
        )
    )
    disconnected = asyncio.create_task(ws.wait_closed())
    try:
        done, _ = await asyncio.wait((execution, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            cancel.set()
            await asyncio.gather(execution, return_exceptions=True)
            return None
        return execution.result()
    except asyncio.CancelledError:
        cancel.set()
        try:
            await asyncio.shield(execution)
        except Exception:
            pass
        raise
    finally:
        cancel.set()
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)


async def run(config_path: str) -> None:
    config = _load_config(config_path)
    agent = LocalAgent(config)

    server_url = validate_server_url(config["server_url"])

    backoff = 1
    while True:
        try:
            # Reload on reconnect so explicit re-pairing can replace the credential.
            credentials = load_credentials(credentials_path(config_path, config), server_url)
            async with websockets.connect(
                server_url,
                additional_headers={"X-API-Key": credentials.api_key},
                max_size=65536,
                max_queue=16,
            ) as ws:
                log.info("connected to %s", server_url)
                await ws.send(
                    json.dumps(
                        {
                            "type": "handshake",
                            "device_id": credentials.device_id,
                            "capabilities": agent.capabilities,
                        }
                    )
                )
                backoff = 1  # reset after a successful connection

                async for raw_message in ws:
                    message = json.loads(raw_message)
                    if message.get("type") != "command":
                        continue

                    request_id = message.get("request_id")
                    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
                        await ws.close(code=1003, reason="Invalid command identity")
                        break
                    action = message.get("action")
                    params = message.get("params", {})
                    log.info("received command: %s", action)

                    try:
                        if type(message.get("expires_at")) not in (int, float):
                            raise ValueError("Command expiration is required")
                        details = await _run_with_disconnect(
                            ws, agent, action, params, expires_at=message["expires_at"]
                        )
                        if details is None:
                            break
                        await ws.send(_encode_result(request_id, details))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("command failed: %s", type(exc).__name__)
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "result",
                                    "request_id": request_id,
                                    "success": False,
                                    "error": "Action declined, failed or interrupted. Effects may be partial; inspect the device before retrying.",
                                }
                            )
                        )
        except (websockets.ConnectionClosed, InvalidHandshake, OSError) as exc:
            log.warning("disconnected (%s); reconnecting in %ss", type(exc).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    config_file = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "config.json")
    )
    asyncio.run(run(config_file))
