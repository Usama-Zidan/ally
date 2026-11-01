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
# edit config.json: server_url, device_credentials_file, capabilities, allowed_folders
python pair.py config.json
python agent.py
```

First have the backend operator issue a one-time enrollment token for your
existing tenant, after migration `0003_devices`:

```bash
python -m services.auth.pair_device issue TENANT_UUID
```

Enter that token in the companion's interactive pairing prompt within 120
seconds. Pairing requires typing `PAIR` locally. The token is sent in an HTTPS
body, never a URL, and only its hash is stored server-side. Redirects are refused.
The server creates a dedicated `pc:connect` credential and a device ID; tenant
keys cannot claim a device ID. A tenant has one enrolled device. Redemption of
a replacement token revokes the previous device credential atomically.

The companion stores its server-bound credential using Windows current-user
DPAPI in `device_credentials_file` (a `.dpapi` file next to the config by default).
It has no plaintext storage fallback. This protects a file at rest from other
users; it does not protect against code running as the same Windows user or
provide hardware attestation. The config's old `api_key` field is ignored; do
not copy a chat/tenant key into this file. Existing configuration and keys are
not automatically changed. Pair the upgraded companion explicitly. Use
`python pair.py config.json --replace` to approve replacing an existing local
credential file, and `python -m services.auth.pair_device revoke TENANT_UUID`
on the backend to revoke device enrollment and pending tokens.

If token redemption succeeds but the response or local save fails, request a
new token and re-pair; the server cannot recover the raw credential. Keep the
`.dpapi` file out of source control, images and diagnostic attachments. A changed
server URL requires explicit re-pairing.

## Hardened action flow

Upgrade the backend and companion together. The companion sends its key in the
`X-API-Key` handshake header, never a newly constructed query URL. Use `wss://`
outside loopback development. The handshake includes the registered `device_id`;
the server verifies its tenant and credential binding and replies `ready`.

PC actions can be selected by the chat classifier from natural-language
requests, or named directly with a command such as
`/local_pc.open_app app_name=notepad`. Every action still requires approval on
this device's terminal, shows its exact escaped arguments, and requires typing
`APPROVE`. Non-interactive agents deny actions. Enabling a capability does not
approve future commands, and remote `confirmed` parameters grant no permission.
Opening executable/script paths and reading private files also require approval.

Remote commands must include the backend's numeric `expires_at`; the companion
checks expiry before and after approval, using a monotonic deadline locally.
Keep server/device clocks synchronized. The command timeout includes delivery
and human approval. Approval after expiry does not execute the action. A timeout
after execution starts cannot undo an OS effect: do not automatically retry.
The terminal prompt now observes cancellation and expiry while waiting for input.

File reads return at most 8192 characters and mark `truncated` if more remain.
Directory/window results include at most 200 names and 8192 name characters;
partial directory results sort the collected subset, not the entire directory.
The backend labels partial results in the tool summary. Final action responses
must fit 64 KiB of UTF-8 JSON. Shell output is separately bounded while running,
as described below.

The example starts with all capabilities disabled. Local config and audit logs
are ignored by Git and excluded from Docker context; they must never be committed.

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
| `open_apps` | Launch applications; list, focus, close windows | Can execute scripts/programs and close unsaved work; local approval required |
| `file_rw` | Read/write files, **restricted to `allowed_folders`** | Low, if `allowed_folders` is scoped tightly. Never includes your whole filesystem. |
| `shell_exec` | Run approved commands under a Windows job's process/memory/output limits | **High.** No OS security sandbox. Approved attempts and observed process outcomes are audited without raw command/output. OS effects remain unverified. |
| `mouse_keyboard_control` | Move the mouse, click, type, press keys | **High.** Can interact with anything visible on screen, including things outside this agent's own intent (other open windows, password prompts, etc). |

Recommended posture: leave `shell_exec` and `mouse_keyboard_control`
**off** unless you have a specific reason to need them in the moment,
and turn them back off afterward. `open_apps` and `file_rw` (scoped to a
dedicated folder) are reasonable to leave on.

### `file_rw` is a folder sandbox, not a permission bit

`allowed_folders` must be absolute paths. A `read_file`/`write_file`/
`list_directory` request is resolved (symlinks and `..` included, via
`Path.resolve()`) and checked against this list *after* resolution. A traversal
such as `allowed_folder/../../../Windows/System32` is rejected when its resolved
target is outside the allowed folders. Resolution does not seal a concurrent
filesystem/reparse change between checking and opening, and this is not an OS
sandbox. If you want the agent to manage documents,
point `allowed_folders` at a dedicated folder you create for this
purpose, not your whole user directory.

### `shell_exec` has no command allowlist by design

Attempting to allowlist "safe" shell commands is not a real security
boundary — most shells make it trivial to chain, alias, or encode around
a blocklist, and a real allowlist narrow enough to be safe would also be
too narrow to be useful for an open-ended assistant. If you enable this, each
exact command still needs independent local approval. It has the same OS access
as your terminal once approved. The digest audit supports correlation without
storing raw arguments; it does not prove effects or replace approval.

Shell requires Windows 10+ and uses the system `cmd.exe /d /s /c` with no child
console and no interactive child stdin. A kill-on-close job is attached during
process creation before the suspended process can execute. Limits: 16 processes,
512 MiB job memory, 1 MiB total observed output, 4000 retained bytes per stream and
at most 120 seconds within the remaining command deadline. Disconnect, timeout,
output limit or coroutine cancellation terminates the owned tree; normal command
exit also removes background descendants. Failed setup has no unrestricted fallback.

Audit writes are flushed before launch; terminal outcomes distinguish normal exit,
timeout, cancellation and unknown status. Missing terminal audit or unconfirmed
cleanup cannot be reported as shell success. A zero exit still leaves requested
effects unverified. These controls do not undo prior effects or contain work sent
through external OS brokers.

## How `open_app` finds things

You can ask to open an app, site, channel, or video by name. The companion
resolves known apps and sites locally. If it cannot resolve the target, it
searches DuckDuckGo from the companion PC, shows the top result's title and
URL in the local terminal, and opens it only if you type `APPROVE`. The search
query leaves the PC for DuckDuckGo; no search-provider API key is needed. The
agent never passes a site name through a shell. Resolution order: your
`app_aliases` from `config.json`, an explicit URL/domain, a path that
exists, an exact Start Menu app name (classic and Microsoft Store apps,
via PowerShell's `Get-StartApps`), a built-in website alias (`youtube`,
`gmail`, `github`, ...), an executable on PATH, then a fuzzy Start Menu
match. Web search runs only if these lookups do not find a target.

For a frequently used destination, you can still add a direct alias yourself:

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
