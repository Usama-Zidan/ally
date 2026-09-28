"""
Query condensation: turns a follow-up message like "what about page 2?"
into a self-contained retrieval query using prior conversation turns.

This only changes the query sent to retrieval (Qdrant/BM25). The prompt
built in prompt.build_messages still receives the *original* user_query
plus the full history, so the model's own phrasing of the question (and
what gets stored in chat history) is unaffected — condensation exists
purely to fix what retrieval searches for.

The history is rendered as a plain-text transcript inside ONE user message
(not replayed as chat turns). Replaying cited RAG answers as assistant
turns makes the model continue the QA-with-citations pattern instead of
performing the rewrite task.
"""
from __future__ import annotations

import re

import structlog

from services.llm_gateway.router import LLMGatewayError, stream_chat_completion

log = structlog.get_logger()

_MAX_TURNS = 6            # last 3 exchanges is plenty for reference resolution
_ASSISTANT_CHARS = 150    # assistant turns only need to hint at the topic
_MAX_OUTPUT_CHARS = 300   # a real rewritten question is short

_CITATION_RE = re.compile(r"\s*\[\d+\]")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
# "u sure?", "are you sure about this?", "really?" — confirmations carry no
# topic of their own, so we resolve them from history in code instead of
# trusting a small model to do it.
_CONFIRM_RE = re.compile(
    r"^\W*(?:(?:are\s+)?(?:you|u|ya)\s+)?(?:really\s+)?(?:sure|certain)"
    r"(?:\s+(?:about\s+)?(?:this|that|it))?\W*$"
    r"|^\W*(?:really|seriously)\W*$",
    re.IGNORECASE,
)
_OUTPUT_PREFIX_RE = re.compile(r"^\s*output:\s*", re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"(do(es)? not (contain|mention|address)"
    r"|not (outlined|documented) in the (given|provided) sources)",
    re.IGNORECASE,
)

CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a user's latest message into a standalone search question. "
    "You are given a conversation transcript as DATA; you are not a participant "
    "in it. Never answer the question, never cite sources, never comment on "
    "whether information exists. Resolve references (\"this\", \"that\", "
    "\"the other one\") using the topic from the user's earlier messages, and "
    "fix obvious typos. If the latest message is a confirmation like "
    "\"u sure?\", output the earlier question it refers to. If it is already "
    "standalone, output it unchanged. Output ONLY the rewritten question on "
    "one line.\n\n"
    "Example 1:\n"
    "<conversation>\n"
    "User: What does the offer letter say about the notice period?\n"
    "Assistant: The notice period is 30 days.\n"
    "</conversation>\n"
    "<latest_message>\nWhat about garden leave?\n</latest_message>\n"
    "Output: What does the offer letter say about garden leave?\n\n"
    "Example 2:\n"
    "<conversation>\n"
    "User: what is the notice period\n"
    "Assistant: The notice period is 30 days for staff.\n"
    "</conversation>\n"
    "<latest_message>\nnothing else mentions this policy?\n</latest_message>\n"
    "Output: Do any other documents mention the notice period policy?\n\n"
    "Example 3:\n"
    "<conversation>\n"
    "User: what is the notice period\n"
    "Assistant: The notice period is 30 days for staff.\n"
    "</conversation>\n"
    "<latest_message>\nu sure?\n</latest_message>\n"
    "Output: What is the notice period?\n\n"
    "Example 4:\n"
    "<conversation>\n"
    "User: what is the notice period\n"
    "Assistant: The notice period is 30 days for staff.\n"
    "</conversation>\n"
    "<latest_message>\nhow bout the other one\n</latest_message>\n"
    "Output: What is the other notice period policy?"
)


def _render_transcript(history: list[dict]) -> str:
    """Flattens history into 'User: ...' / 'Assistant: ...' lines.

    User turns are kept whole (they carry the topic). Assistant turns are
    citation-stripped and truncated, and refusal boilerplate ("the provided
    sources do not contain...") is dropped entirely since it carries no
    topic and primes the model to refuse.
    """
    lines = []
    for msg in history[-_MAX_TURNS:]:
        role = msg.get("role")
        text = " ".join(_CITATION_RE.sub("", msg.get("content", "")).split())
        if not text:
            continue
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant" and not _REFUSAL_RE.search(text):
            lines.append(f"Assistant: {text[:_ASSISTANT_CHARS]}")
    return "\n".join(lines) or "(none)"


def _build_condense_messages(history: list[dict], user_query: str) -> list[dict]:
    return [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<conversation>\n{_render_transcript(history)}\n</conversation>\n"
                f"<latest_message>\n{user_query}\n</latest_message>\n"
                "Output:"
            ),
        },
    ]


def _clean_output(raw: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw)
    # Unmatched closing tag: the opening tag was consumed by the chat
    # template/server, so everything before </think> is reasoning residue.
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = _THINK_TAG_RE.sub("", text).strip()
    text = _OUTPUT_PREFIX_RE.sub("", text).strip()
    return text.strip("\"'").strip()


def _last_user_question(history: list[dict]) -> str | None:
    """Most recent user turn that isn't itself a confirmation."""
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        text = " ".join(msg.get("content", "").split())
        if text and not _CONFIRM_RE.match(text):
            return text
    return None


async def condense_query(history: list[dict], user_query: str) -> str:
    """Returns a history-aware, standalone rewrite of ``user_query`` for
    retrieval purposes.

    Skips the LLM call entirely on the first turn of a conversation
    (``history`` empty) since there is nothing to resolve against. Fails
    open on any gateway error or degenerate output: retrieval falls back
    to the raw ``user_query`` rather than the turn erroring out, since a
    condensation failure should degrade retrieval quality, not break the
    chat.
    """
    if not history:
        return user_query

    if _CONFIRM_RE.match(user_query):
        prior = _last_user_question(history)
        if prior:
            return prior

    messages = _build_condense_messages(history, user_query)
    try:
        parts = [
            delta
            async for delta in stream_chat_completion(
                messages, temperature=0.0, max_tokens=128
            )
        ]
    except LLMGatewayError as exc:
        log.warning("query_condensation_failed", error=str(exc))
        return user_query

    raw = "".join(parts)
    condensed = _clean_output(raw)

    if (
        not condensed
        or len(condensed) > _MAX_OUTPUT_CHARS
        or "\n" in condensed
        or _CITATION_RE.search(condensed)
    ):
        log.warning("query_condensation_degenerate", raw=raw[:200])
        return user_query

    return condensed