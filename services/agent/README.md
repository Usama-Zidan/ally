# Agent service (Phase 4)

Classify -> route to {tool, rag, conversational}. Token-level streaming
for "rag"/"conversational" stays outside this graph, in
`services/chat/use_cases.py`'s `ChatUseCase` -- see `graph.py`'s module
docstring for why.

```
services/agent/
├── graph.py                 # LangGraph StateGraph: classify -> tool | END
├── local_pc_intent.py       # Natural-language fallbacks for local PC requests
├── classifier.py            # Pluggable intent classifier (LLM default, "JEV"-ready)
├── commands.py               # Explicit /tool.action slash-command parsing (overrides classifier)
├── registry.py                # Builds the default ToolRegistry
├── connected_accounts.py       # Encrypted per-tenant OAuth token vault (Fernet)
├── pc_agent_bridge.py           # WS connection registry + command/response correlation
└── tools/
    ├── base.py                  # Tool ABC, ToolResult, ToolRegistry
    ├── jira_tool.py               # Mocked; vault-ready
    ├── drive_tool.py              # Mocked; vault-ready
    ├── calendar_tool.py           # Mocked; vault-ready
    └── local_pc_tool.py           # Routes through pc_agent_bridge to local_agent/
```

## Adding a new tool

1. Subclass `Tool` in a new file under `tools/`, implement `execute()`.
2. Register an instance in `registry.build_default_registry()`.

That's the whole extension point — the classifier prompt and the
WebSocket protocol pick it up automatically via `ToolRegistry.describe_for_classifier()`.

## Explicit commands

`/tool_name.action key=value key2="quoted value"` bypasses the
classifier entirely. See `commands.py`.

## Local PC agent

See `local_agent/` at the repo root (a separate deployable, not part of
this package) and its README for setup and the capability model
(`open_apps`, `file_rw`, `shell_exec`, `mouse_keyboard_control` — each
independently toggleable, enforced on the local machine, not here).
