"""Disposable process tree for Windows shell ownership tests only."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "hold":
        (directory / "child.pid").write_text(str(os.getpid()))
        time.sleep(60)
    elif mode in {"fork", "background"}:
        subprocess.Popen([sys.executable, __file__, "hold", str(directory)])
        (directory / "parent.pid").write_text(str(os.getpid()))
        deadline = time.monotonic() + 5
        while not (directory / "child.pid").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Test child did not become ready")
            time.sleep(0.01)
        if mode == "fork":
            time.sleep(60)
        else:
            # Let the test acquire stable process handles before allowing root
            # exit; checking a PID only after cleanup can inspect a reused PID.
            while not (directory / "release").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Test did not release background parent")
                time.sleep(0.01)
    elif mode == "flood":
        for _ in range(10000):
            os.write(1, b"o" * 4096)
            os.write(2, b"e" * 4096)
    elif mode == "exit":
        os.write(1, b"stdout-sentinel")
        os.write(2, b"stderr-sentinel")
        raise SystemExit(7)
    elif mode == "owner":
        # This helper itself is terminated by the test. Its non-inherited job
        # handle must close and kill the owned descendants without Python finally.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_agent"))
        from shell_runner import run_shell

        command = subprocess.list2cmdline([sys.executable, __file__, "fork", str(directory)])
        run_shell(command, timeout=60)


if __name__ == "__main__":
    main()
