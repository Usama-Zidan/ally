"""Tests for services.agent.commands."""
from __future__ import annotations

import unittest

from services.agent.commands import parse_command


class ParseCommandTest(unittest.TestCase):
    def test_returns_none_for_ordinary_chat_messages(self):
        self.assertIsNone(parse_command("What is our leave policy?"))
        self.assertIsNone(parse_command("hey there"))
        self.assertIsNone(parse_command(""))

    def test_parses_tool_action_and_simple_params(self):
        command = parse_command("/jira.create_ticket summary=Fix")
        self.assertEqual(command.tool_name, "jira")
        self.assertEqual(command.action, "create_ticket")
        self.assertEqual(command.params, {"summary": "Fix"})

    def test_parses_quoted_values_containing_spaces(self):
        command = parse_command(
            '/jira.create_ticket summary="Fix login bug" description="Users can\'t log in"'
        )
        self.assertEqual(command.params["summary"], "Fix login bug")
        self.assertEqual(command.params["description"], "Users can't log in")

    def test_tool_and_action_are_lowercased(self):
        command = parse_command("/Local_PC.Open_App app=notepad")
        self.assertEqual(command.tool_name, "local_pc")
        self.assertEqual(command.action, "open_app")

    def test_command_with_no_params_parses_to_empty_dict(self):
        command = parse_command("/local_pc.list_windows")
        self.assertEqual(command.params, {})

    def test_leading_and_trailing_whitespace_is_tolerated(self):
        command = parse_command("   /jira.search_tickets query=bug   ")
        self.assertEqual(command.tool_name, "jira")
        self.assertEqual(command.params, {"query": "bug"})

    def test_malformed_trailing_fragments_are_silently_ignored(self):
        # A typo in one parameter shouldn't prevent parsing the command
        # itself -- the concrete Tool validates required params and
        # raises a clear ToolError if something it needs is missing.
        command = parse_command("/jira.create_ticket summary=Fix this-is-not-kv")
        self.assertEqual(command.params, {"summary": "Fix"})

    def test_a_bare_slash_with_no_dot_is_not_a_command(self):
        self.assertIsNone(parse_command("/help"))

    def test_escaped_quotes_inside_a_quoted_value(self):
        command = parse_command(r'/jira.add_comment comment="she said \"hi\""')
        self.assertEqual(command.params["comment"], 'she said "hi"')


if __name__ == "__main__":
    unittest.main()
