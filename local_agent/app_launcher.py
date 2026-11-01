"""
Resolves a human-friendly name ("youtube", "edge", "spotify", "notepad")
to something that can actually be launched, instead of handing the raw
string to cmd.exe.

Why not just `start <name>` / shell=True: cmd.exe only knows executables
on PATH and a few registered App Paths. "youtube" isn't a program at all,
it's a website, and Store/UWP apps and most Start Menu entries aren't on
PATH either -- so a shell lookup fails for exactly the requests users
actually make. Shelling out was also a hole: with shell=True, an
open_apps request could run arbitrary commands without needing the
separate shell_exec capability. Nothing here uses a shell.

Resolution order (first hit wins):
    1. user aliases from config.json ("app_aliases")
    2. an explicit URL / domain ("youtube.com", "https://...")
    3. a path that exists on disk
    4. an *exact* Start Menu app name (so "spotify" opens the Spotify app)
    5. a built-in website alias ("youtube" -> https://www.youtube.com)
    6. an executable on PATH ("cmd", "code")
    7. a fuzzy Start Menu match ("edge" -> "Microsoft Edge")
    8. the top DuckDuckGo result for an unresolved name, after local approval
Exact installed-app matches run before website aliases, and fuzzy ones run
after, so "youtube" opens the website rather than a "YouTube Music" app
that merely starts with the same word.

Start Menu apps (including Microsoft Store apps) come from PowerShell's
Get-StartApps and are launched through explorer.exe's shell:AppsFolder
handler, which works for both classic and Store apps.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

# Common website names; unresolved names use local web search after this map.
WEB_ALIASES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com",
    "google calendar": "https://calendar.google.com",
    "calendar": "https://calendar.google.com",
    "google maps": "https://maps.google.com",
    "github": "https://github.com",
    "linkedin": "https://www.linkedin.com",
    "reddit": "https://www.reddit.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "hugging face": "https://huggingface.co",
    "huggingface": "https://huggingface.co",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "netflix": "https://www.netflix.com",
}

# "notepad.exe" looks like a domain to a naive regex; these endings mean
# "file", not "website".
_FILE_EXTENSIONS = {
    "exe", "lnk", "bat", "cmd", "msc", "ps1", "msi", "txt", "md", "py",
    "json", "csv", "log", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "zip", "dll", "ini", "cfg", "yml", "yaml",
}

_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^(www\.)?[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}(/\S*)?$", re.IGNORECASE)

_NOT_AN_APP_RE = re.compile(
    r"\b(runtime|redistributable|uninstall|updater|update|setup|installer|helper|webview2?)\b"
)

_start_apps_cache: list[dict[str, str]] | None = None
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_SEARCH_RESPONSE_LIMIT = 1024 * 1024


class _SearchResults(HTMLParser):
    """Extract ordinary result links from DuckDuckGo's no-JavaScript page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        href = values.get("href")
        if "result__a" in classes and href:
            self._active_href = href
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._active_href is None:
            return
        title = " ".join("".join(self._active_text).split())
        url = _unwrap_search_url(self._active_href)
        if title and url:
            self.results.append({"title": title[:512], "url": url[:2048]})
        self._active_href = None
        self._active_text = []


def _unwrap_search_url(href: str) -> str | None:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlsplit(href)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com") and parsed.path == "/l/":
        href = parse_qs(parsed.query).get("uddg", [""])[0]
        parsed = urlsplit(href)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return href


def needs_web_search(target: str, aliases: dict[str, str] | None = None) -> bool:
    """True when the target is not a known app, alias, path, or web address."""
    name = (target or "").strip().strip('"').strip("'")
    if not name:
        return False
    key = name.casefold()
    if key in {alias.casefold() for alias in (aliases or {})}:
        return False
    if (
        looks_like_url(name)
        or os.path.exists(name)
        or key in WEB_ALIASES
        or shutil.which(name)
    ):
        return False
    apps = list_start_apps()
    return match_app(name, apps) is None


def search_web(target: str) -> dict[str, str] | None:
    """Search from the companion PC and return the top usable web result.

    Search is read-only. The caller must show this candidate and obtain local
    approval before opening it.
    """
    query = (target or "").strip()
    if not query or len(query) > 512:
        raise ValueError("Search query must be between 1 and 512 characters.")
    request = Request(
        _SEARCH_URL,
        data=urlencode({"q": query}).encode("utf-8"),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AllyLocalAgent/1.0",
            "Accept": "text/html",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            body = response.read(_SEARCH_RESPONSE_LIMIT + 1)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Local web search failed. Check this PC's internet connection.") from exc
    if len(body) > _SEARCH_RESPONSE_LIMIT:
        raise RuntimeError("Local web search response exceeded its size limit.")
    parser = _SearchResults()
    parser.feed(body.decode("utf-8", errors="replace"))
    return parser.results[0] if parser.results else None


def open_search_result(query: str, result: dict[str, str]) -> dict[str, Any]:
    url = _unwrap_search_url(result.get("url", ""))
    title = result.get("title", "").strip()
    if not title or not url:
        raise ValueError("The selected search result is invalid.")
    webbrowser.open(url)
    return {"launched": url, "title": title, "search_query": query, "method": "web_search"}


def _is_windows() -> bool:
    return sys.platform == "win32"


def looks_like_url(text: str) -> bool:
    if _SCHEME_RE.match(text):
        return True
    if " " in text or not _DOMAIN_RE.match(text):
        return False
    host = text.split("/", 1)[0]
    return host.rsplit(".", 1)[-1].lower() not in _FILE_EXTENSIONS


def normalize_url(text: str) -> str:
    return text if _SCHEME_RE.match(text) else f"https://{text}"


def list_start_apps(refresh: bool = False) -> list[dict[str, str]]:
    """Returns [{"Name": ..., "AppID": ...}, ...] for every Start Menu app,
    classic and Microsoft Store alike. Empty off Windows or if PowerShell
    isn't available. Cached for the life of the process; pass
    refresh=True after installing something new."""
    global _start_apps_cache
    if _start_apps_cache is not None and not refresh:
        return _start_apps_cache
    if not _is_windows():
        return []

    command = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "Get-StartApps | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            timeout=30,
        )
        text = result.stdout.decode("utf-8", errors="replace").strip()
        data = json.loads(text) if text else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []

    if isinstance(data, dict):  # PowerShell emits a bare object for one result
        data = [data]
    _start_apps_cache = [
        {"Name": str(item["Name"]), "AppID": str(item["AppID"])}
        for item in data
        if item.get("Name") and item.get("AppID")
    ]
    return _start_apps_cache


def match_app(
    query: str, apps: list[dict[str, str]], exact_only: bool = False
) -> dict[str, str] | None:
    """Best Start Menu match for ``query``: exact name, then name prefix,
    then whole-word, then substring, then close spelling. Ties go to the
    shortest name ("Edge" beats "Edge WebView2 Runtime")."""
    q = query.casefold().strip()
    if not q:
        return None

    best: tuple[int, int, dict[str, str]] | None = None
    for app in apps:
        name = app["Name"].casefold()
        if name == q:
            rank = 0
        elif exact_only:
            continue
        elif name.startswith(q) or re.search(rf"\b{re.escape(q)}\b", name):
            # Prefix and whole-word hits share a rank so the shortest name
            # decides: "edge" should find "Microsoft Edge", not the longer
            # "Edge WebView2 Runtime" just because that one starts with it.
            rank = 1
        elif q in name:
            rank = 3
        elif difflib.SequenceMatcher(None, q, name).ratio() >= 0.8:
            rank = 4
        else:
            continue
        # Background components show up in Get-StartApps too; they should
        # only ever win if nothing else matches (never over a real app).
        if rank and _NOT_AN_APP_RE.search(name):
            rank += 10
        candidate = (rank, len(name), app)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best else None


def _detached_flags() -> int:
    return getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )


def _open_path(path: str) -> None:
    if hasattr(os, "startfile"):
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def _launch_start_app(app: dict[str, str]) -> None:
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app['AppID']}"])


def _launch_executable(path: str) -> None:
    subprocess.Popen([path], creationflags=_detached_flags(), close_fds=True)


def _launch_alias_value(value: str) -> dict[str, Any]:
    if looks_like_url(value):
        webbrowser.open(normalize_url(value))
        return {"launched": normalize_url(value), "method": "user_alias_url"}
    if os.path.exists(value):
        _open_path(value)
        return {"launched": value, "method": "user_alias_path"}
    found = shutil.which(value)
    if found:
        _launch_executable(found)
        return {"launched": found, "method": "user_alias_executable"}
    raise ValueError(f"Your app_aliases entry points to {value!r}, which couldn't be launched.")


def resolve_and_launch(target: str, aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Finds and launches ``target``. Returns {"launched": ..., "method": ...}.

    Raises:
        ValueError: if nothing matches, with the closest installed apps
            and the config.json escape hatch in the message so the user
            (and the chat transcript) can see how to fix it.
    """
    name = (target or "").strip().strip('"').strip("'")
    if not name:
        raise ValueError("No app or website name was given.")
    key = name.casefold()

    user_aliases = {k.casefold(): v for k, v in (aliases or {}).items()}
    if key in user_aliases:
        return _launch_alias_value(user_aliases[key])

    if looks_like_url(name):
        webbrowser.open(normalize_url(name))
        return {"launched": normalize_url(name), "method": "url"}

    if os.path.exists(name):
        _open_path(name)
        return {"launched": name, "method": "path"}

    apps = list_start_apps()
    exact = match_app(name, apps, exact_only=True)
    if exact:
        _launch_start_app(exact)
        return {"launched": exact["Name"], "method": "start_menu_app"}

    if key in WEB_ALIASES:
        webbrowser.open(WEB_ALIASES[key])
        return {"launched": WEB_ALIASES[key], "method": "web_alias"}

    on_path = shutil.which(name)
    if on_path:
        _launch_executable(on_path)
        return {"launched": on_path, "method": "path_executable"}

    fuzzy = match_app(name, apps)
    if fuzzy:
        _launch_start_app(fuzzy)
        return {"launched": fuzzy["Name"], "method": "start_menu_app_fuzzy"}

    names = [a["Name"] for a in apps]
    close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
    hint = f" Closest installed apps: {', '.join(close)}." if close else ""
    raise ValueError(
        f"Couldn't find an app or website called {name!r}.{hint} "
        'You can map it yourself in config.json, e.g. "app_aliases": {"'
        f'{key}": "C:\\\\path\\\\to\\\\app.exe"}}.'
    )
