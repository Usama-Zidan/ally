# Ally local agent

Runs on your own Windows PC and lets the Ally chat agent act on it
directly: open applications, read/write files, run shell commands, or
control the mouse and keyboard — whichever of those you choose to allow.

## Why this exists as a separate program

The backend is cloud-hosted; it can't reach a machine sitting behind
your home or office router. This agent solves that the other way around:
it connects *out* to the backend over WebSocket and stays connected, so
no port forwarding or public IP is needed. When the chat agent decides to
act on your PC, the backend sends a command down that existing
connection and waits for this script to report a result.

## Setup

```bash
pip install -r requirements.txt
cp config.example.json config.json
# edit config.json: server_url, api_key, capabilities, allowed_folders
python agent.py
```

Get an API key with the backend's admin CLI:

```bash
python -m services.auth.provision_tenant create "My PC"
```

## Capabilities — read this before enabling anything

Each of the four categories below is an independent on/off switch in
`config.json`. **This file — not the server — is the actual security
boundary.** The backend's `pc_agent_capabilities` table and the
capabilities this agent reports at connect time are a declaration of
intent; they cannot force your machine to do anything. Every capability
check happens locally, in `agent.py`, regardless of what the server
believes is enabled.

| Capability | What it allows | Risk |
|---|---|---|
| `open_apps` | Launch applications; list, focus, close windows | Low — same as double-clicking a shortcut |
| `file_rw` | Read/write files, **restricted to `allowed_folders`** | Low, if `allowed_folders` is scoped tightly. Never includes your whole filesystem. |
| `shell_exec` | Run **any** shell command this agent's OS user can run | **High.** No sandbox, no command allowlist. Equivalent to giving the chat agent a terminal. Every command is appended to `shell_exec_audit.log` next to this script regardless of other settings — review it periodically. |
| `mouse_keyboard_control` | Move the mouse, click, type, press keys | **High.** Can interact with anything visible on screen, including things outside this agent's own intent (other open windows, password prompts, etc). |

Recommended posture: leave `shell_exec` and `mouse_keyboard_control`
**off** unless you have a specific reason to need them in the moment,
and turn them back off afterward. `open_apps` and `file_rw` (scoped to a
dedicated folder) are reasonable to leave on.

### `file_rw` is a folder sandbox, not a permission bit

`allowed_folders` must be absolute paths. A `read_file`/`write_file`/
`list_directory` request is resolved (symlinks and `..` included, via
`Path.resolve()`) and checked against this list *after* resolution — a
path like `allowed_folder/../../../Windows/System32` cannot escape the
sandbox by construction. If you want the agent to manage documents,
point `allowed_folders` at a dedicated folder you create for this
purpose, not your whole user directory.

### `shell_exec` has no command allowlist by design

Attempting to allowlist "safe" shell commands is not a real security
boundary — most shells make it trivial to chain, alias, or encode around
a blocklist, and a real allowlist narrow enough to be safe would also be
too narrow to be useful for an open-ended assistant. If you enable this,
you are trusting the chat agent (and, transitively, the LLM behind it)
with the same access you have in a terminal. The audit log is there so
you can review what ran, not to prevent anything from running.

## How `open_app` finds things

You say a name ("spotify", "edge", "youtube"); the agent works out what it
is. It never passes the name through a shell. In order: your
`app_aliases` from `config.json`, an explicit URL/domain, a path that
exists, an exact Start Menu app name (classic and Microsoft Store apps,
via PowerShell's `Get-StartApps`), a built-in website alias (`youtube`,
`gmail`, `github`, ...), an executable on PATH, then a fuzzy Start Menu
match. If nothing matches, the error lists the closest installed apps.

For anything it can't guess, add it yourself:

```json
"app_aliases": { "vscode": "C:\\path\\to\\Code.exe", "my music": "https://music.youtube.com" }
```

Check what it can see on your machine:

```bash
python -c "import app_launcher as a; print([x['Name'] for x in a.list_start_apps()][:20])"
python -c "import app_launcher as a; print(a.resolve_and_launch('youtube'))"
```

## Optional dependencies

`pygetwindow` (window management) and `pyautogui` (mouse/keyboard) are
only imported when their capability is actually used. If you only want
`file_rw`, you don't need to install either — the agent will tell you
clearly if a capability needs a package you haven't installed, rather
than failing to start at all.

## Not built (intentionally out of scope for v1)

Screenshots, clipboard access, and system info were deliberately left
out of this first version. Adding one is a new method on `LocalAgent`
plus an entry in `_HANDLERS` and `_ACTION_CAPABILITY` (the backend-side
counterpart, in `services/agent/tools/local_pc_tool.py`) — the same
pattern every existing action follows.
