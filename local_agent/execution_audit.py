"""Durable attempt/outcome correlation without raw command, cwd or output."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from shell_runner import ShellResult


class ExecutionAudit:
    def __init__(self, path: Path, command: str) -> None:
        self.path = path
        self.execution_id = str(uuid.uuid4())
        self.command_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()

    def _append(self, fields: dict) -> None:
        event = {
            "event_id": str(uuid.uuid4()),
            "execution_id": self.execution_id,
            "timestamp": time.time(),
            "action": "run_command",
            "command_sha256": self.command_hash,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def approved(self) -> None:
        # Failure here aborts before process creation.
        self._append({"outcome": "approved_for_execution", "effect_outcome": "not_started"})

    def finished(self, result: ShellResult) -> None:
        self._append(
            {
                "outcome": result.outcome,
                "process_exit_code": result.returncode,
                "started": result.started,
                "cleanup_confirmed": result.cleanup_confirmed,
                "duration_ms": result.duration_ms,
                "truncated": result.truncated,
                "stdout_bytes_observed": result.stdout_bytes_observed,
                "stderr_bytes_observed": result.stderr_bytes_observed,
                "effect_outcome": "not_verified" if result.started else "not_started",
                "error_code": result.error_code,
            }
        )

    def unknown(self) -> None:
        # An unmatched approved record or an unexpected interruption proves no
        # terminal state. Never infer that effects were absent or rolled back.
        self._append(
            {
                "outcome": "execution_status_unknown",
                "effect_outcome": "unknown",
                "cleanup_confirmed": False,
            }
        )
