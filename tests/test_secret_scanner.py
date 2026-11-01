"""Verify sensitive values never occur in scanner findings or reports."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_secrets import scan_repository, scan_text


class SecretScannerTest(unittest.TestCase):
    def test_detects_key_on_correct_line_without_retaining_value(self):
        sentinel = "ally_" + "x" * 40
        findings = scan_text("settings.json", "{}\n" + sentinel)
        self.assertEqual([(f.line, f.category) for f in findings], [(2, "ally-api-key")])
        self.assertNotIn(sentinel, repr(findings))

    def test_local_paths_are_forbidden_even_without_recognizable_key(self):
        for name in (
            "local_agent/config.json",
            ".env.online",
            ".env.docker",
            "a/.env",
            "app.secret",
            "local_agent/device-credentials.dpapi",
        ):
            with self.subTest(name=name):
                self.assertTrue(scan_text(name, "arbitrary password"))

    def test_examples_and_short_test_placeholders_are_safe(self):
        self.assertEqual(scan_text("local_agent/config.example.json", "REPLACE_WITH_API_KEY"), [])
        self.assertEqual(scan_text(".env.example", "OPENAI_API_KEY="), [])
        self.assertEqual(scan_text("tests/test_auth.py", "ally_test-key"), [])

    def test_includes_tracked_ignored_files_and_nonignored_untracked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", tmp], check=True, capture_output=True)
            (root / ".env").write_text("arbitrary credential")
            sentinel = "ally_" + "s" * 40
            (root / "staged.txt").write_text(sentinel)
            subprocess.run(["git", "-C", tmp, "add", ".env"], check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "add", "staged.txt"], check=True, capture_output=True)
            (root / "staged.txt").write_text("cleaned worktree, secret still staged")
            (root / ".gitignore").write_text(".env\n")
            (root / "accidental.txt").write_text("ally_" + "z" * 40)
            paths = {f.path for f in scan_repository(root)}
            self.assertEqual(paths, {".env", "accidental.txt", "staged.txt"})


if __name__ == "__main__":
    unittest.main()
