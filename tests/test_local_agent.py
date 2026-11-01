"""
Tests for local_agent/agent.py -- the standalone client script that runs
on a user's own PC, not part of the `services` package. Imported here via
an explicit sys.path insert rather than being a proper installed
package, since it's a separate deployable artifact (see
local_agent/README.md).

These specifically target the two things that actually make this script
safe to run: path-traversal protection on file_rw (_resolve_allowed_path)
and that every action is gated behind its capability regardless of what
parameters are supplied (LocalAgent._require).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_LOCAL_AGENT_DIR = str(Path(__file__).resolve().parent.parent / "local_agent")
if _LOCAL_AGENT_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_AGENT_DIR)

import agent as local_agent  # noqa: E402

_resolve_allowed_path = getattr(local_agent, "_resolve_allowed_path")
_load_config = getattr(local_agent, "_load_config")
_CapabilityDisabled = getattr(local_agent, "CapabilityDisabled")


class ResolveAllowedPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.allowed = os.path.join(self._tmp.name, "sandbox")
        os.makedirs(self.allowed)
        Path(self.allowed, "file.txt").write_text("hello")
        self.outside = os.path.join(self._tmp.name, "outside.txt")
        Path(self.outside).write_text("secret")

    def test_a_file_inside_the_allowed_folder_resolves(self):
        resolved = _resolve_allowed_path(
            os.path.join(self.allowed, "file.txt"), [self.allowed]
        )
        self.assertEqual(resolved, (Path(self.allowed) / "file.txt").resolve())

    def test_the_allowed_folder_itself_resolves(self):
        resolved = _resolve_allowed_path(self.allowed, [self.allowed])
        self.assertEqual(resolved, Path(self.allowed).resolve())

    def test_traversal_via_dotdot_is_blocked(self):
        traversal = os.path.join(self.allowed, "..", "outside.txt")
        with self.assertRaises(ValueError):
            _resolve_allowed_path(traversal, [self.allowed])

    def test_an_absolute_path_outside_every_allowed_folder_is_blocked(self):
        with self.assertRaises(ValueError):
            _resolve_allowed_path(self.outside, [self.allowed])

    def test_a_sibling_folder_with_a_similar_prefix_is_not_treated_as_inside(self):
        # "sandbox-evil" must not be treated as inside "sandbox" just
        # because it shares a string prefix -- the check must be on
        # resolved path components, not a naive startswith().
        sibling = os.path.join(self._tmp.name, "sandbox-evil")
        os.makedirs(sibling)
        Path(sibling, "file.txt").write_text("not yours")
        with self.assertRaises(ValueError):
            _resolve_allowed_path(
                os.path.join(sibling, "file.txt"), [self.allowed]
            )

    def test_allowed_when_any_of_several_allowed_folders_matches(self):
        other_allowed = os.path.join(self._tmp.name, "other")
        os.makedirs(other_allowed)
        resolved = _resolve_allowed_path(
            os.path.join(self.allowed, "file.txt"), [other_allowed, self.allowed]
        )
        self.assertEqual(resolved, (Path(self.allowed) / "file.txt").resolve())


class LoadConfigTest(unittest.TestCase):
    def test_missing_capabilities_and_allowed_folders_default_safely(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"server_url": "wss://x", "api_key": "k"}, f)
            path = f.name
        self.addCleanup(os.unlink, path)

        config = _load_config(path)

        self.assertEqual(
            config["capabilities"],
            {
                "open_apps": False,
                "file_rw": False,
                "shell_exec": False,
                "mouse_keyboard_control": False,
            },
        )
        self.assertEqual(config["allowed_folders"], [])

    def test_explicit_values_are_preserved(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "server_url": "wss://x",
                    "api_key": "k",
                    "capabilities": {"shell_exec": True},
                    "allowed_folders": ["/tmp/sandbox"],
                },
                f,
            )
            path = f.name
        self.addCleanup(os.unlink, path)

        config = _load_config(path)

        self.assertTrue(config["capabilities"]["shell_exec"])
        self.assertFalse(config["capabilities"]["open_apps"])  # still defaulted
        self.assertEqual(config["allowed_folders"], ["/tmp/sandbox"])


class LocalAgentCapabilityGatingTest(unittest.TestCase):
    def _agent(self, **capabilities):
        config = {
            "capabilities": {
                "open_apps": False,
                "file_rw": False,
                "shell_exec": False,
                "mouse_keyboard_control": False,
                **capabilities,
            },
            "allowed_folders": [],
            "server_url": "wss://x",
            "api_key": "k",
        }
        # Trusted local approval fixture; production requires terminal consent.
        return getattr(local_agent, "LocalAgent")(config, approve=lambda action, params: True)

    def test_disabled_capability_blocks_the_action_regardless_of_params(self):
        agent = self._agent()  # everything disabled
        with self.assertRaises(_CapabilityDisabled):
            agent.read_file({"path": "/anything"})
        with self.assertRaises(_CapabilityDisabled):
            agent.run_command({"command": "echo hi"})
        with self.assertRaises(_CapabilityDisabled):
            agent.open_app({"app_name": "notepad"})

    def test_enabling_one_capability_does_not_enable_others(self):
        agent = self._agent(open_apps=True)
        with self.assertRaises(_CapabilityDisabled):
            agent.run_command({"command": "echo hi"})

    def test_enabled_file_rw_still_enforces_the_folder_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(file_rw=True)
            agent.allowed_folders = [tmp]
            with self.assertRaises(ValueError):
                agent.read_file({"path": "/etc/passwd"})

    def test_enabled_file_rw_actually_performs_the_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(file_rw=True)
            agent.allowed_folders = [tmp]
            target = os.path.join(tmp, "out.txt")

            agent.write_file({"path": target, "content": "written by agent"})
            result = agent.read_file({"path": target})

            self.assertEqual(result["content"], "written by agent")

    def test_handle_command_rejects_unknown_actions(self):
        agent = self._agent(open_apps=True, file_rw=True, shell_exec=True)
        with self.assertRaises(ValueError):
            agent.handle_command("delete_everything", {})

    def test_handle_command_dispatches_to_the_correct_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(file_rw=True)
            agent.allowed_folders = [tmp]
            Path(tmp, "a.txt").touch()
            Path(tmp, "b.txt").touch()

            result = agent.handle_command("list_directory", {"path": tmp})

            self.assertEqual(sorted(result["entries"]), ["a.txt", "b.txt"])


if __name__ == "__main__":
    unittest.main()
