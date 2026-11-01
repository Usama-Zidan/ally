"""Natural-language fallbacks for clear local PC requests."""

from __future__ import annotations

import re
from typing import Any

from services.agent.classifier import Intent

_OPEN_REQUEST_FILLER = frozenset(
    {
        "about",
        "at",
        "browser",
        "can",
        "command",
        "for",
        "go",
        "in",
        "launch",
        "me",
        "open",
        "on",
        "page",
        "please",
        "pls",
        "run",
        "search",
        "shell",
        "show",
        "site",
        "start",
        "the",
        "to",
        "using",
        "visit",
        "website",
        "with",
        "you",
    }
)
_SPECIFIC_WEB_TARGETS = frozenset(
    {"channel", "clip", "episode", "movie", "playlist", "song", "video"}
)
_WEB_PLATFORMS = (
    "youtube",
    "tiktok",
    "twitch",
    "vimeo",
    "instagram",
    "facebook",
    "reddit",
    "google",
)


def web_search_phrase(query: str, target: str) -> str | None:
    """Keep content details when the classifier reduced a specific request to a site name."""
    query_words = re.findall(r"[\w@-]+", query.casefold())
    target_words = set(re.findall(r"[\w@-]+", target.casefold()))
    remaining = [
        word
        for word in query_words
        if word not in target_words and word not in _OPEN_REQUEST_FILLER
    ]
    if _SPECIFIC_WEB_TARGETS.intersection(remaining) or len(remaining) >= 2:
        return query.strip()
    return None


def infer_local_pc_intent(query: str) -> Intent | None:
    """Recover clear PC requests when the small intent model mistakes them for RAG."""
    normalized = query.casefold()
    if (
        re.search(r"\b(files?|folders?|directories)\b", normalized)
        and re.search(r"\b(can|could|able|access|see|read)\b", normalized)
        and re.search(r"\b(my\s+pc|my\s+computer|my\s+machine|this\s+pc)\b", normalized)
    ):
        return Intent(kind="conversational", reasoning="local PC access scope question")

    if re.search(r"\b(?:move|put|place)\b.*\b(?:mouse|cursor)\b", normalized):
        if re.search(r"\b(?:corner|right\s*most|left\s*most)\b", normalized):
            horizontal = "left" if re.search(r"\bleft\b", normalized) else "right"
            vertical = "bottom" if re.search(r"\b(?:bottom|lower)\b", normalized) else "top"
            return Intent(
                kind="tool",
                tool_name="local_pc",
                action="move_mouse_to_corner",
                params={"corner": f"{vertical}_{horizontal}"},
                reasoning="natural-language mouse corner request",
            )

    if not re.search(r"\b(?:open|launch|visit|go\s+to|play|watch|find|search)\b", normalized):
        return None
    has_web_target = bool(
        re.search(r"\b(?:site|website|browser|channel|video|clip|playlist)\b", normalized)
        or re.search(
            r"\b(?:youtube|tiktok|twitch|vimeo|instagram|facebook|reddit|google)\b",
            normalized,
        )
        or re.search(r"https?://|www\.|\b[\w-]+\.(?:com|org|net|tv|io)\b", normalized)
    )
    if not has_web_target:
        return None
    platform = next((name for name in _WEB_PLATFORMS if name in normalized), None)
    target = platform or query.strip()
    search_phrase = web_search_phrase(query, target)
    params: dict[str, Any] = {"app_name": target}
    if search_phrase:
        params["search_query"] = search_phrase
    return Intent(
        kind="tool",
        tool_name="local_pc",
        action="open_app",
        params=params,
        reasoning="natural-language website or media request",
    )
