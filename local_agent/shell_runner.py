"""Bounded Windows command ownership; never an OS security sandbox.

Windows 10+ process attributes attach a kill-on-close job during creation. The
process remains suspended until all pipe/limit setup succeeds. Only its standard
handles are inherited. No command text, cwd or output is logged here.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from contextlib import ExitStack
from ctypes import wintypes as w
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Outcome = Literal[
    "exited", "timed_out", "cancelled", "output_limit", "startup_failed", "runtime_error"
]
MAX_COMMAND_CHARS = 8192
TAIL_BYTES = 4000
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_PROCESSES = 16
MAX_JOB_MEMORY = 512 * 1024 * 1024
POLL_SECONDS = 0.01
CLEANUP_SECONDS = 2.0


@dataclass(frozen=True)
class ShellResult:
    outcome: Outcome
    returncode: int | None = None
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)
    stdout_bytes_observed: int = 0
    stderr_bytes_observed: int = 0
    truncated: bool = False
    started: bool = False
    cleanup_confirmed: bool = True
    duration_ms: int = 0
    error_code: int | None = None


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("length", w.DWORD), ("descriptor", w.LPVOID), ("inherit", w.BOOL)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("reserved", w.LPWSTR),
        ("desktop", w.LPWSTR),
        ("title", w.LPWSTR),
        ("x", w.DWORD),
        ("y", w.DWORD),
        ("x_size", w.DWORD),
        ("y_size", w.DWORD),
        ("x_chars", w.DWORD),
        ("y_chars", w.DWORD),
        ("fill", w.DWORD),
        ("flags", w.DWORD),
        ("show", w.WORD),
        ("reserved_size", w.WORD),
        ("reserved_bytes", w.LPVOID),
        ("stdin", w.HANDLE),
        ("stdout", w.HANDLE),
        ("stderr", w.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("info", _StartupInfo), ("attributes", w.LPVOID)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", w.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("process_count", w.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", w.DWORD),
        ("scheduling", w.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_int64),
        ("kernel_time", ctypes.c_int64),
        ("period_user_time", ctypes.c_int64),
        ("period_kernel_time", ctypes.c_int64),
        ("faults", w.DWORD),
        ("total", w.DWORD),
        ("active", w.DWORD),
        ("terminated", w.DWORD),
    ]


class _ProcessIDs(ctypes.Structure):
    _fields_ = [("assigned", w.DWORD), ("count", w.DWORD), ("ids", ctypes.c_size_t * MAX_PROCESSES)]


def _kernel() -> Any:
    if sys.platform != "win32":
        raise OSError("Windows process ownership is unavailable")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    signatures = {
        "CreateJobObjectW": (w.HANDLE, [pointer, w.LPCWSTR]),
        "SetInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, pointer, w.DWORD]),
        "QueryInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, pointer, w.DWORD, pointer]),
        "TerminateJobObject": (w.BOOL, [w.HANDLE, w.UINT]),
        "CloseHandle": (w.BOOL, [w.HANDLE]),
        "CreatePipe": (w.BOOL, [pointer, pointer, pointer, w.DWORD]),
        "SetHandleInformation": (w.BOOL, [w.HANDLE, w.DWORD, w.DWORD]),
        "CreateFileW": (
            w.HANDLE,
            [w.LPCWSTR, w.DWORD, w.DWORD, pointer, w.DWORD, w.DWORD, w.HANDLE],
        ),
        "InitializeProcThreadAttributeList": (w.BOOL, [pointer, w.DWORD, w.DWORD, pointer]),
        "UpdateProcThreadAttribute": (
            w.BOOL,
            [pointer, w.DWORD, size, pointer, size, pointer, pointer],
        ),
        "DeleteProcThreadAttributeList": (None, [pointer]),
        "CreateProcessW": (
            w.BOOL,
            [
                w.LPCWSTR,
                w.LPWSTR,
                pointer,
                pointer,
                w.BOOL,
                w.DWORD,
                pointer,
                w.LPCWSTR,
                pointer,
                pointer,
            ],
        ),
        "ResumeThread": (w.DWORD, [w.HANDLE]),
        "WaitForSingleObject": (w.DWORD, [w.HANDLE, w.DWORD]),
        "GetExitCodeProcess": (w.BOOL, [w.HANDLE, pointer]),
        "OpenProcess": (w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
        "IsProcessInJob": (w.BOOL, [w.HANDLE, w.HANDLE, pointer]),
        "GetSystemDirectoryW": (w.UINT, [w.LPWSTR, w.UINT]),
        "PeekNamedPipe": (w.BOOL, [w.HANDLE, pointer, w.DWORD, pointer, pointer, pointer]),
        "ReadFile": (w.BOOL, [w.HANDLE, pointer, w.DWORD, pointer, pointer]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(kernel, name)
        function.restype, function.argtypes = restype, argtypes
    return kernel


def _last_error() -> int:
    if sys.platform == "win32":
        return ctypes.get_last_error()
    raise OSError("Windows process ownership is unavailable")


def _check(ok: Any) -> None:
    if not ok:
        # A numeric code is useful for diagnostics without command/path details.
        raise OSError(_last_error(), "Windows command ownership failed")


def _own(stack: ExitStack, kernel: Any, handle: int | None) -> int:
    if handle is None or handle == ctypes.c_void_p(-1).value:
        _check(False)
    assert handle is not None
    stack.callback(kernel.CloseHandle, handle)
    return handle


def _pipe(stack: ExitStack, kernel: Any, security: _SecurityAttributes) -> tuple[int, int]:
    read, write = w.HANDLE(), w.HANDLE()
    _check(
        kernel.CreatePipe(ctypes.byref(read), ctypes.byref(write), ctypes.byref(security), 65536)
    )
    reader = _own(stack, kernel, read.value)
    writer = _own(stack, kernel, write.value)
    _check(kernel.SetHandleInformation(reader, 1, 0))
    return reader, writer


def _attributes(
    stack: ExitStack, kernel: Any, startup: _StartupInfoEx, inherited: Any, jobs: Any
) -> Any:
    size = ctypes.c_size_t()
    # The sizing call is expected to fail with the required size populated.
    kernel.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
    if not size.value or size.value > 65536:
        raise OSError("Invalid Windows process attribute size")
    buffer = ctypes.create_string_buffer(size.value)
    pointer = ctypes.cast(buffer, w.LPVOID)
    _check(kernel.InitializeProcThreadAttributeList(pointer, 2, 0, ctypes.byref(size)))
    stack.callback(kernel.DeleteProcThreadAttributeList, pointer)
    _check(
        kernel.UpdateProcThreadAttribute(
            pointer, 0, 0x20002, inherited, ctypes.sizeof(inherited), None, None
        )
    )
    # PROC_THREAD_ATTRIBUTE_JOB_LIST assigns the job atomically during creation.
    _check(
        kernel.UpdateProcThreadAttribute(pointer, 0, 0x2000D, jobs, ctypes.sizeof(jobs), None, None)
    )
    startup.attributes = pointer
    return buffer  # Keep all attribute values/storage alive through CreateProcessW.


class _Tail:
    def __init__(self) -> None:
        self.data = bytearray()
        self.observed = 0

    def append(self, data: bytes) -> None:
        self.observed += len(data)
        self.data.extend(data)
        if len(self.data) > TAIL_BYTES:
            del self.data[:-TAIL_BYTES]

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def _drain(kernel: Any, handle: int, tail: _Tail, allowance: int) -> int:
    # One thread exclusively owns each read handle: no competing reader can
    # consume the bytes between PeekNamedPipe and ReadFile. Bound each pass so
    # an output flood cannot starve cancellation or the other stream.
    count = 0
    for _ in range(4):
        if count >= allowance:
            break
        available = w.DWORD()
        ok = kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None)
        if not ok and _last_error() == 109:  # ERROR_BROKEN_PIPE: EOF
            break
        _check(ok)
        if not available.value:
            break
        size = min(8192, available.value, allowance - count)
        buffer, read = ctypes.create_string_buffer(size), w.DWORD()
        _check(kernel.ReadFile(handle, buffer, size, ctypes.byref(read), None))
        tail.append(buffer.raw[: read.value])
        count += read.value
    return count


def _terminate_and_confirm(kernel: Any, job: int, process: int) -> bool:
    with ExitStack() as stack:
        members = _ProcessIDs()
        inspected = bool(
            kernel.QueryInformationJobObject(
                job, 3, ctypes.byref(members), ctypes.sizeof(members), None
            )
        )
        handles = []
        if inspected:
            for pid in members.ids[: min(members.count, MAX_PROCESSES)]:
                handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                if not handle:
                    # A member can exit during enumeration. Access/inspection
                    # failures other than a vanished PID make cleanup uncertain.
                    if _last_error() != 87:
                        inspected = False
                    continue
                stack.callback(kernel.CloseHandle, handle)
                member = w.BOOL()
                if not kernel.IsProcessInJob(handle, job, ctypes.byref(member)):
                    inspected = False
                elif member.value:
                    handles.append(handle)
        if not kernel.TerminateJobObject(job, 1):
            return False  # Closing the last job handle still requests kill-on-close.
        deadline = time.monotonic() + CLEANUP_SECONDS
        while time.monotonic() < deadline:
            accounting = _Accounting()
            if not kernel.QueryInformationJobObject(
                job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
            ):
                return False
            if (
                accounting.active == 0
                and kernel.WaitForSingleObject(process, 0) == 0
                and all(kernel.WaitForSingleObject(handle, 0) == 0 for handle in handles)
            ):
                return inspected
            time.sleep(POLL_SECONDS)
        return False


def run_shell(
    command: str,
    *,
    cwd: str | None = None,
    timeout: float = 30,
    cancel: threading.Event | None = None,
    deadline: float | None = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> ShellResult:
    """Execute an already approved command with bounded process/output ownership.

    `deadline` includes time spent obtaining approval. Cancellation cannot undo
    effects already performed. Unsupported platforms fail without executing.
    """
    if (
        not isinstance(command, str)
        or not command.strip()
        or len(command) > MAX_COMMAND_CHARS
        or "\0" in command
    ):
        raise ValueError("Invalid bounded shell command")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("Invalid shell timeout")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError("Invalid shell output limit")
    if deadline is not None and (not math.isfinite(deadline)):
        raise ValueError("Invalid shell deadline")
    if cwd is not None and (not isinstance(cwd, str) or not Path(cwd).is_absolute()):
        raise ValueError("Shell working directory must be absolute")
    cancel = cancel if cancel is not None else threading.Event()
    began = time.monotonic()
    end = min(began + timeout, deadline) if deadline is not None else began + timeout
    if cancel.is_set():
        return ShellResult("cancelled")
    if began >= end:
        return ShellResult("timed_out")
    if os.name != "nt":
        return ShellResult("startup_failed")

    stdout, stderr = _Tail(), _Tail()
    started, cleanup, returncode, error_code = False, True, None, None
    outcome: Outcome = "startup_failed"
    with ExitStack() as stack:
        try:
            kernel = _kernel()
            job = _own(stack, kernel, kernel.CreateJobObjectW(None, None))
            limits = _ExtendedLimits()
            # KILL_ON_JOB_CLOSE | ACTIVE_PROCESS | JOB_MEMORY; breakaway is forbidden.
            limits.basic.flags = 0x2000 | 0x8 | 0x200
            limits.basic.process_count, limits.job_memory = MAX_PROCESSES, MAX_JOB_MEMORY
            _check(
                kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            )
            security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, True)
            stdout_read, stdout_write = _pipe(stack, kernel, security)
            stderr_read, stderr_write = _pipe(stack, kernel, security)
            stdin = _own(
                stack,
                kernel,
                kernel.CreateFileW("NUL", 0x80000000, 3, ctypes.byref(security), 3, 0, None),
            )
            startup = _StartupInfoEx()
            startup.info.cb, startup.info.flags = ctypes.sizeof(startup), 0x100
            startup.info.stdin, startup.info.stdout, startup.info.stderr = (
                stdin,
                stdout_write,
                stderr_write,
            )
            inherited = (w.HANDLE * 3)(stdin, stdout_write, stderr_write)
            jobs = (w.HANDLE * 1)(job)
            attributes = _attributes(stack, kernel, startup, inherited, jobs)
            system_directory = ctypes.create_unicode_buffer(32768)
            length = kernel.GetSystemDirectoryW(system_directory, len(system_directory))
            _check(0 < length < len(system_directory))
            executable = str(Path(system_directory.value) / "cmd.exe")
            command_line = ctypes.create_unicode_buffer(f'"{executable}" /d /s /c "{command}"')
            process_info = _ProcessInfo()
            if cancel.is_set() or time.monotonic() >= end:
                outcome = "cancelled" if cancel.is_set() else "timed_out"
            else:
                # CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT.
                _check(
                    kernel.CreateProcessW(
                        executable,
                        command_line,
                        None,
                        None,
                        True,
                        0x4 | 0x08000000 | 0x80000,
                        None,
                        cwd,
                        ctypes.byref(startup),
                        ctypes.byref(process_info),
                    )
                )
                cleanup = False  # Only explicit observation can confirm termination.
                process = _own(stack, kernel, process_info.process)
                thread = _own(stack, kernel, process_info.thread)
                try:
                    if cancel.is_set() or time.monotonic() >= end:
                        outcome = "cancelled" if cancel.is_set() else "timed_out"
                    else:
                        _check(kernel.ResumeThread(thread) != 0xFFFFFFFF)
                        started = True
                        while True:
                            if cancel.is_set():
                                outcome = "cancelled"
                                break
                            if time.monotonic() >= end:
                                outcome = "timed_out"
                                break
                            for handle, tail in ((stdout_read, stdout), (stderr_read, stderr)):
                                remaining = max_output_bytes + 1 - stdout.observed - stderr.observed
                                _drain(kernel, handle, tail, remaining)
                            if stdout.observed + stderr.observed > max_output_bytes:
                                outcome = "output_limit"
                                break
                            state = kernel.WaitForSingleObject(process, 0)
                            if state == 0:
                                exit_code = w.DWORD()
                                _check(kernel.GetExitCodeProcess(process, ctypes.byref(exit_code)))
                                outcome, returncode = "exited", exit_code.value
                                break
                            _check(state == 258)  # WAIT_TIMEOUT; all other states are failures.
                            cancel.wait(POLL_SECONDS)
                finally:
                    # Always remove background descendants, including after a normal
                    # root exit. The command cannot leave an owned daemon running.
                    cleanup = _terminate_and_confirm(kernel, job, process)
                # Drain residual kernel-buffered bytes after terminating all writers.
                while True:
                    count = 0
                    for handle, tail in ((stdout_read, stdout), (stderr_read, stderr)):
                        remaining = max_output_bytes + 1 - stdout.observed - stderr.observed
                        count += _drain(kernel, handle, tail, remaining)
                    if stdout.observed + stderr.observed > max_output_bytes:
                        if outcome == "exited":
                            outcome = "output_limit"
                        break
                    if not count:
                        break
            # Keep attribute backing storage alive until process creation is finished.
            del attributes
        except OSError as exc:
            outcome = "runtime_error" if started else "startup_failed"
            error_code = exc.errno
    return ShellResult(
        outcome,
        returncode,
        stdout.text(),
        stderr.text(),
        stdout.observed,
        stderr.observed,
        stdout.observed > TAIL_BYTES or stderr.observed > TAIL_BYTES or outcome == "output_limit",
        started,
        cleanup,
        int((time.monotonic() - began) * 1000),
        error_code,
    )
