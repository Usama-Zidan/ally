"""
Builds the chat-format message list sent to the LLM gateway: a system
prompt that constrains the model to the retrieved context and instructs
it to cite sources by number, prior conversation turns, and the current
question with its numbered context block attached.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the "
    "numbered sources provided with each question. Follow these rules "
    "strictly:\n"
    "1. Answer only using information present in the sources below.\n"
    "2. Every claim you make must be followed by a citation marker like "
    "[1] or [2] referencing the source number it came from.\n"
    "3. If the sources do not contain enough information to answer, say "
    "so plainly instead of guessing or using outside knowledge.\n"
    "4. Keep answers concise and directly responsive to the question."
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the indexed documents relevant to that "
    "question. Could you rephrase it, or ask about something covered in "
    "the uploaded documents?"
)


def build_context_block(sources: list[dict]) -> str:
    """Formats retrieved chunks into a numbered block the model is
    instructed to cite by number, e.g.:

        [1] (hr_policy.pdf, page 3): Employees receive 21 days...
        [2] (leave_policy.pdf, page 2): Leave requests must be submitted...

    The numbering here must match the "index" field callers attach to the
    same sources when sending the {"type": "sources"} WebSocket message,
    so a citation marker like [2] in the streamed answer and the second
    entry in that sources list refer to the same passage.
    """
    lines = []
    for index, source in enumerate(sources, start=1):
        lines.append(
            f"[{index}] ({source['filename']}, page {source['page_number']}): "
            f"{source['text']}"
        )
    return "\n\n".join(lines)


def build_messages(
    history: list[dict], sources: list[dict], user_query: str
) -> list[dict]:
    """Assembles the full chat-format message list for one turn.

    Args:
        history: Prior turns as returned by services.chat.history.get_history,
            oldest first.
        sources: Retrieved chunks for the current question, in the same
            order used to build the sources WebSocket message (so citation
            numbers line up).
        user_query: The current user message.
    """
    context_block = build_context_block(sources)
    user_content = (
        f"Sources:\n{context_block}\n\nQuestion: {user_query}"
        if sources
        else f"Question: {user_query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]
