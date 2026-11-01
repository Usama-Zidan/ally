"""Truthful shell attempt/outcome audit and local approval failure boundaries."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

LOCAL = str(Path(__file__).resolve().parents[1] / "local_agent")
if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)
import agent  # noqa: E402
import local_approval  # noqa: E402
from shell_runner import ShellResult  # noqa: E402


class ShellAuditTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.audit_path = Path(temporary.name) / "audit.log"
        self.local = agent.LocalAgent(
            {"capabilities": {"shell_exec": True}, "allowed_folders": []},
            approve=lambda action, params: True,
        )

    def test_audit_write_failure_prevents_process_creation(self):
        with (
            patch.object(agent, "_AUDIT_LOG_PATH", self.audit_path),
            patch("execution_audit.os.fsync", side_effect=OSError("audit unavailable")),
            patch.object(agent, "run_shell") as execute,
        ):
            with self.assertRaises(OSError):
                self.local.run_command({"command": "never execute"})
        execute.assert_not_called()

    def test_terminal_audit_loss_is_explicit_and_does_not_retry_execution(self):
        with (
            patch.object(agent, "_AUDIT_LOG_PATH", self.audit_path),
            patch("execution_audit.ExecutionAudit.finished", side_effect=OSError("private-path")),
            patch.object(
                agent, "run_shell", return_value=ShellResult("exited", returncode=0, started=True)
            ) as execute,
        ):
            result = self.local.run_command({"command": "private-command"})
        execute.assert_called_once()
        self.assertFalse(result["audit_recorded"])
        self.assertEqual(result["effect_outcome"], "not_verified")
        self.assertEqual(len(self.audit_path.read_text().splitlines()), 1)

    def test_unexpected_abort_records_unknown_not_success_or_rollback(self):
        with (
            patch.object(agent, "_AUDIT_LOG_PATH", self.audit_path),
            patch.object(agent, "run_shell", side_effect=KeyboardInterrupt()),
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.local.run_command(
                    {"command": "private-command", "cwd": str(self.audit_path.parent)}
                )
        events = [json.loads(line) for line in self.audit_path.read_text().splitlines()]
        self.assertEqual(events[-1]["outcome"], "execution_status_unknown")
        self.assertEqual(events[-1]["effect_outcome"], "unknown")
        self.assertNotIn("private-command", self.audit_path.read_text())
        self.assertNotIn(str(self.audit_path.parent), self.audit_path.read_text())

    def test_timeout_records_observed_execution_and_unverified_effects_without_output(self):
        with (
            patch.object(agent, "_AUDIT_LOG_PATH", self.audit_path),
            patch.object(
                agent,
                "run_shell",
                return_value=ShellResult("timed_out", started=True, stdout="private-output"),
            ),
        ):
            result = self.local.run_command({"command": "private-command"})
        events = [json.loads(line) for line in self.audit_path.read_text().splitlines()]
        self.assertEqual(events[-1]["outcome"], "timed_out")
        self.assertEqual(events[-1]["effect_outcome"], "not_verified")
        self.assertEqual(events[0]["execution_id"], result["execution_id"])
        self.assertNotIn("private-output", self.audit_path.read_text())

    def test_disconnect_before_approval_cannot_start_or_audit_execution(self):
        cancel = threading.Event()
        cancel.set()
        approve = Mock(return_value=True)
        self.local._approve = approve
        with patch.object(agent, "run_shell") as execute:
            with self.assertRaises(PermissionError):
                self.local.handle_command(
                    "run_command", {"command": "never execute"}, cancel=cancel
                )
        approve.assert_not_called()
        execute.assert_not_called()

    def test_shell_budget_includes_time_spent_waiting_for_approval(self):
        with (
            patch.object(agent, "_AUDIT_LOG_PATH", self.audit_path),
            patch.object(agent.time, "time", return_value=100),
            patch.object(agent.time, "monotonic", side_effect=[10, 11, 14]),
            patch.object(agent, "run_shell", return_value=ShellResult("timed_out")) as execute,
        ):
            self.local.handle_command("run_command", {"command": "echo x"}, expires_at=105)
        self.assertEqual(execute.call_args.kwargs["deadline"], 15)


class InterruptibleApprovalTest(unittest.TestCase):
    def approve(self, keyboard, *, cancel=None):
        with (
            patch.object(local_approval.sys.stdin, "isatty", return_value=True),
            patch.object(local_approval.os, "name", "nt"),
            patch.object(local_approval.sys, "platform", "win32"),
            patch.dict(sys.modules, {"msvcrt": keyboard}),
            patch("builtins.print"),
        ):
            return local_approval.approve_locally(
                "run_command",
                {"command": "echo x"},
                cancel=cancel or threading.Event(),
                deadline=None,
            )

    def test_only_exact_local_text_approves(self):
        self.assertTrue(
            self.approve(
                Mock(kbhit=Mock(return_value=True), getwch=Mock(side_effect=list("APPROVE\r")))
            )
        )
        self.assertFalse(
            self.approve(
                Mock(kbhit=Mock(return_value=True), getwch=Mock(side_effect=list("APPROVEX\r")))
            )
        )

    def test_disconnect_interrupts_waiting_terminal_prompt(self):
        cancel = threading.Event()

        def no_input():
            cancel.set()
            return False

        keyboard = Mock(kbhit=Mock(side_effect=no_input), getwch=Mock())
        self.assertFalse(self.approve(keyboard, cancel=cancel))
        keyboard.getwch.assert_not_called()
