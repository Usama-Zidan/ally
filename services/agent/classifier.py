"""
Pluggable intent classification: decides whether a chat message should be
answered via RAG, routed to a tool action, or treated as plain
conversation.

INTENT_CLASSIFIER selects which implementation handles this (default:
"llm", a zero-shot classification call through the existing LLM
gateway). Registering a custom classifier -- e.g. the user's own
"JEV" model -- later means adding one entry to _CLASSIFIERS and setting
INTENT_CLASSIFIER=jev; nothing in services.agent.graph needs to change,
since every implementation returns the same Intent shape.

Not using Instructor here despite it being pinned in requirements.txt:
Instructor's documented integration patterns (from_litellm, or patching
an OpenAI-SDK-shaped client) assume direct access to litellm's
completion function or a client exposing .chat.completions.create().
This project's LLM gateway wraps litellm.Router, which exposes
.acompletion() instead -- adapting Instructor to that shape isn't a
drop-in, and doing it speculatively without a way to test it against a
live install risks shipping a broken integration. Deferred rather than
guessed at; the hand-rolled JSON parsing below has an explicit fallback
for exactly the failure mode Instructor would otherwise guard against
(malformed/unparseable model output).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from config import INTENT_CLASSIFIER
from services.llm_gateway.router import complete_chat

log = structlog.get_logger()

# Qwen3 (the project's default self-hosted model) opens every assistant
# turn with a <think>...</think> reasoning block unless the literal
# string "/no_think" appears in the user turn -- this is a documented
# soft-switch handled by Qwen3's own chat template, not something this
# codebase implements. Classification needs none of that reasoning: it's
# a single constrained JSON answer, and on a small model (the project's
# default is a 1.7B GGUF via llama-server) the reasoning phase can
# consume the entire max_tokens budget before any JSON is emitted at
# all, producing a completely empty response -- that empty-response
# failure mode is exactly what _parse_intent's fallback-to-"rag" exists
# to catch, but suppressing the reasoning phase up front avoids hitting
# it on every single message.
_NO_THINK_SUFFIX = "\n\n/no_think"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

# Generous relative to a typical JSON routing decision, but still bounded
# -- this is defense-in-depth for a backend that doesn't honor /no_think
# (e.g. a different self-hosted model behind QWEN_VLLM_BASE_URL), so a
# verbose reasoning phase still leaves room for the actual JSON to follow
# rather than silently truncating to nothing the way the previous
# default of 256 did in practice.
_CLASSIFIER_MAX_TOKENS = 1024


@dataclass
class Intent:
    kind: str  # "rag" | "tool" | "conversational"
    tool_name: str | None = None
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


_CLASSIFIER_SYSTEM_PROMPT = (
    "You route a user's message to the right handler. Respond with ONLY a "
    "JSON object and nothing else -- no markdown fences, no commentary. Shape:\n"
    '{{"kind": "rag" | "tool" | "conversational", '
    '"tool_name": string or null, "action": string or null, '
    '"params": object}}\n\n'
    '- "rag": the user is asking a question likely answerable from the '
    "indexed documents.\n"
    '- "tool": the user wants an action performed by one of the tools '
    "listed below. Set tool_name to the tool's name and action to one of "
    "its listed actions, with whatever params that action needs.\n"
    '- "conversational": greetings, small talk, or anything not covered '
    "by the above -- when in doubt between this and \"rag\", prefer \"rag\".\n\n"
    "Available tools:\n{tools_block}"
)


class IntentClassifier(ABC):
    @abstractmethod
    async def classify(self, query: str, tools_block: str) -> Intent:
        """Classifies one user message. ``tools_block`` is
        ToolRegistry.describe_for_classifier()'s output for this tenant,
        so routing decisions are grounded in what's actually usable right
        now rather than every tool that exists in the codebase."""


class LLMIntentClassifier(IntentClassifier):
    """Zero-shot classification through the existing LLM gateway -- no
    separate model or fine-tuning, just a constrained prompt."""

    async def classify(self, query: str, tools_block: str) -> Intent:
        system_prompt = _CLASSIFIER_SYSTEM_PROMPT.format(tools_block=tools_block)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query + _NO_THINK_SUFFIX},
        ]
        raw = await complete_chat(messages, max_tokens=_CLASSIFIER_MAX_TOKENS)
        return _parse_intent(raw)


def _parse_intent(raw: str) -> Intent:
    """Parses the classifier's JSON response, falling back to "rag" on
    any parse failure. A malformed classification must never crash the
    chat turn or silently do nothing -- it degrades to the safest
    default (answer from documents) rather than guessing at a tool call
    from output that didn't parse.
    """
    cleaned = _THINK_BLOCK_RE.sub("", raw)
    # Unmatched closing tag: generation was cut off mid-reasoning and the
    # opening <think> was in a prior chunk already consumed -- everything
    # before </think> is reasoning residue, not part of the JSON answer.
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = _THINK_TAG_RE.sub("", cleaned).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        kind = data.get("kind")
        if kind not in ("rag", "tool", "conversational"):
            raise ValueError(f"unexpected kind {kind!r}")
        return Intent(
            kind=kind,
            tool_name=data.get("tool_name"),
            action=data.get("action"),
            params=data.get("params") or {},
        )
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        log.warning("intent_classification_parse_failed", error=str(exc), raw=raw[:200])
        return Intent(kind="rag", reasoning="fallback: classifier output could not be parsed")


# Register additional implementations here (e.g. "jev": JevClassifier) and
# set INTENT_CLASSIFIER to switch -- see the module docstring.
_CLASSIFIERS: dict[str, type[IntentClassifier]] = {
    "llm": LLMIntentClassifier,
}

_classifier_instance: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        classifier_cls = _CLASSIFIERS.get(INTENT_CLASSIFIER)
        if classifier_cls is None:
            raise RuntimeError(
                f"Unknown INTENT_CLASSIFIER {INTENT_CLASSIFIER!r}; "
                f"registered: {list(_CLASSIFIERS)}"
            )
        _classifier_instance = classifier_cls()
    return _classifier_instance


def reset_classifier() -> None:
    """Clears the cached classifier instance (tests, or a config change)."""
    global _classifier_instance
    _classifier_instance = None
