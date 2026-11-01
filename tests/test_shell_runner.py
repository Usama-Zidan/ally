"""Real Windows job, output, deadline and descendant ownership verification."""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes as w
from pathlib import Path
from unittest.mock import patch

LOCAL = str(Path(__file__).resolve().parents[1] / "local_agent")
if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)
import shell_runner as shell  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "shell_child.py"


def command(mode: str, directory: str) -> str:
    return subprocess.list2cmdline([sys.executable, str(FIXTURE), mode, directory])


def ready(directory: str) -> None:
    deadline = time.monotonic() + 5
    while not (Path(directory) / "child.pid").exists():
        if time.monotonic() >= deadline:
            raise AssertionError("Disposable child did not become ready")
        time.sleep(0.01)


class ShellPolicyTest(unittest.TestCase):
    def test_pre_cancelled_or_expired_command_never_loads_native_api(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(shell, "_kernel") as kernel:
            self.assertEqual(shell.run_shell("echo never", cancel=cancel).outcome, "cancelled")
            self.assertEqual(
                shell.run_shell("echo never", deadline=time.monotonic() - 1).outcome, "timed_out"
            )
        kernel.assert_not_called()

    def test_invalid_resource_contract_fails_before_process_creation(self):
        for kwargs in (
            {"command": "\0"},
            {"command": "x" * 8193},
            {"command": "echo x", "timeout": float("nan")},
            {"command": "echo x", "max_output_bytes": 0},
            {"command": "echo x", "cwd": "relative"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                shell.run_shell(**kwargs)

    def test_unsupported_platform_has_no_unowned_subprocess_fallback(self):
        with patch.object(shell.os, "name", "posix"), patch.object(shell, "_kernel") as kernel:
            result = shell.run_shell("echo never")
        self.assertEqual(result.outcome, "startup_failed")
        self.assertFalse(result.started)
        kernel.assert_not_called()


@unittest.skipUnless(os.name == "nt", "requires real Windows 10+ job objects")
class WindowsShellTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.WaitForSingleObject.restype = w.DWORD
        self.kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]

    def hold_child_handles(self):
        handles = []
        for name in ("parent.pid", "child.pid"):
            pid = int((Path(self.directory) / name).read_text())
            handle = self.kernel.OpenProcess(0x100000 | 0x1000 | 0x1, False, pid)
            self.assertTrue(handle, f"Test process {name} did not start")
            handles.append(handle)

            # These handles refer only to processes created by our fixture.
            def cleanup(owned=handle):
                if self.kernel.WaitForSingleObject(owned, 0) == 258:
                    self.kernel.TerminateProcess(owned, 1)
                    self.kernel.WaitForSingleObject(owned, 3000)
                self.kernel.CloseHandle(owned)

            self.addCleanup(cleanup)
        return handles

    def test_real_exit_code_and_both_output_streams(self):
        result = shell.run_shell(command("exit", self.directory), timeout=5)
        self.assertEqual(result.outcome, "exited", repr(result))
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "stdout-sentinel")
        self.assertEqual(result.stderr, "stderr-sentinel")
        self.assertTrue(result.cleanup_confirmed)

    def test_output_flood_is_terminated_with_bounded_retained_buffers(self):
        result = shell.run_shell(
            command("flood", self.directory), timeout=5, max_output_bytes=65536
        )
        self.assertEqual(result.outcome, "output_limit", repr(result))
        self.assertLessEqual(len(result.stdout.encode()), shell.TAIL_BYTES)
        self.assertLessEqual(len(result.stderr.encode()), shell.TAIL_BYTES)
        self.assertLessEqual(result.stdout_bytes_observed + result.stderr_bytes_observed, 65537)
        self.assertGreater(result.stdout_bytes_observed, 0)
        self.assertGreater(result.stderr_bytes_observed, 0)
        self.assertTrue(result.truncated)
        self.assertTrue(result.cleanup_confirmed)

    def test_timeout_cleans_the_entire_real_child_tree(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(shell.run_shell, command("fork", self.directory), timeout=2)
            ready(self.directory)
            handles = self.hold_child_handles()
            result = execution.result(timeout=6)
        self.assertEqual(result.outcome, "timed_out", repr(result))
        self.assertTrue(result.cleanup_confirmed)
        for handle in handles:
            self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 0)

    def test_cancel_signal_cleans_tree_before_returning(self):
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(
                shell.run_shell, command("fork", self.directory), timeout=8, cancel=cancel
            )
            ready(self.directory)
            handles = self.hold_child_handles()
            cancel.set()
            result = execution.result(timeout=4)
        self.assertEqual(result.outcome, "cancelled", repr(result))
        self.assertTrue(result.cleanup_confirmed)
        for handle in handles:
            self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 0)

    def test_normal_parent_exit_cannot_leave_a_background_descendant(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(
                shell.run_shell, command("background", self.directory), timeout=8
            )
            ready(self.directory)
            handles = self.hold_child_handles()
            (Path(self.directory) / "release").touch()
            result = execution.result(timeout=5)
        self.assertEqual(result.outcome, "exited", repr(result))
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.cleanup_confirmed)
        for handle in handles:
            self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 0)

    def test_abrupt_owner_death_closes_job_and_kills_descendants(self):
        owner = subprocess.Popen(
            [sys.executable, str(FIXTURE), "owner", self.directory],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.addCleanup(lambda: owner.poll() is None and owner.kill())
        ready(self.directory)
        handles = self.hold_child_handles()
        owner.kill()
        owner.wait(timeout=4)
        for handle in handles:
            self.assertEqual(self.kernel.WaitForSingleObject(handle, 3000), 0)

    def test_failed_native_setup_cannot_run_the_command(self):
        with patch.object(shell, "_attributes", side_effect=OSError(5, "injected setup failure")):
            result = shell.run_shell(command("fork", self.directory))
        self.assertEqual(result.outcome, "startup_failed")
        self.assertFalse(result.started)
        self.assertFalse((Path(self.directory) / "parent.pid").exists())

    def test_native_job_has_process_memory_and_kill_on_close_limits(self):
        original = shell._attributes
        observed = []

        def inspect_limits(stack, kernel, startup, inherited, jobs):
            limits = shell._ExtendedLimits()
            self.assertTrue(
                kernel.QueryInformationJobObject(
                    jobs[0], 9, ctypes.byref(limits), ctypes.sizeof(limits), None
                )
            )
            observed.append((limits.basic.flags, limits.basic.process_count, limits.job_memory))
            return original(stack, kernel, startup, inherited, jobs)

        with patch.object(shell, "_attributes", side_effect=inspect_limits):
            result = shell.run_shell("exit /b 0")
        self.assertEqual(result.outcome, "exited")
        self.assertEqual(observed, [(0x2000 | 0x8 | 0x200, 16, 512 * 1024 * 1024)])

    def test_repeated_commands_release_native_handles(self):
        self.kernel.GetCurrentProcess.restype = w.HANDLE
        self.kernel.GetProcessHandleCount.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        process = self.kernel.GetCurrentProcess()

        def count():
            value = w.DWORD()
            self.assertTrue(self.kernel.GetProcessHandleCount(process, ctypes.byref(value)))
            return value.value

        shell.run_shell("exit /b 0")  # Warm ctypes/runtime bookkeeping.
        before = count()
        for _ in range(5):
            self.assertTrue(shell.run_shell("exit /b 0").cleanup_confirmed)
        gc.collect()
        self.assertLessEqual(count(), before + 2)
