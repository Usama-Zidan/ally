"""Interruptible local terminal consent. No input comes from the backend."""

from __future__ import annotations

import json
import os
import sys
import threading
import time


def approve_locally(
    action: str,
    params: dict,
    *,
    cancel: threading.Event,
    deadline: float | None,
    approval_context: dict | None = None,
) -> bool:
    if not sys.stdin.isatty():
        return False
    request = {"action": action, "params": params}
    if approval_context:
        request["approval_context"] = approval_context
        print("Ally found this web result. Check the title and URL before approving:")
        print(json.dumps(request, ensure_ascii=True))
        print("Type APPROVE to open this result: ", end="", flush=True)
    else:
        print("Ally requests a PC action: " + json.dumps(request, ensure_ascii=True))
        print("Type APPROVE to execute this action: ", end="", flush=True)
    value = ""
    try:
        while not cancel.is_set() and (deadline is None or time.monotonic() < deadline):
            if sys.platform == "win32" and os.name == "nt":
                import msvcrt

                if not msvcrt.kbhit():
                    cancel.wait(0.03)
                    continue
                key = msvcrt.getwch()
                if key in ("\r", "\n"):
                    return value == "APPROVE"
                if key == "\x03":  # Ctrl-C denies this action.
                    return False
                if key == "\b":
                    value = value[:-1]
                    print("\b \b", end="", flush=True)
                elif key in "APPROVE" and len(value) < 7:
                    value += key
                    print(key, end="", flush=True)
                else:
                    # Do not silently ignore extra or unsupported characters.
                    return False
            else:
                import select

                readable, _, _ = select.select([sys.stdin], [], [], 0.03)
                if readable:
                    return sys.stdin.readline(32).rstrip("\r\n") == "APPROVE"
    except (EOFError, OSError, ValueError):
        return False
    finally:
        print(flush=True)
    return False
