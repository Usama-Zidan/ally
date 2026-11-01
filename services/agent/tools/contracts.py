"""Strict cloud-tool arguments, checked before credential lookup or effects."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from services.agent.tools.base import ToolError

Title = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Query = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
Description = Annotated[str, StringConstraints(max_length=8192)]
Content = Annotated[str, StringConstraints(max_length=16384)]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
]
TicketKey = Annotated[
    str,
    StringConstraints(max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}-[A-Za-z0-9]{1,32}$"),
]


def _instant(value: str) -> str:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if "T" not in value or instant.utcoffset() is None:
        raise ValueError("An ISO timestamp with a timezone is required")
    return value


Instant = Annotated[str, StringConstraints(min_length=1, max_length=64), AfterValidator(_instant)]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*")
    @classmethod
    def nonempty(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str) and info.field_name not in {"description", "content"}:
            if not value.strip():
                raise ValueError("Required text must not be empty")
        return value

    @model_validator(mode="after")
    def ordered_range(self) -> Self:
        data = self.model_dump()
        for start, end in (("start_time", "end_time"), ("start", "end")):
            if data.get(start) is not None and data.get(end) is not None:
                if datetime.fromisoformat(
                    data[end].replace("Z", "+00:00")
                ) <= datetime.fromisoformat(data[start].replace("Z", "+00:00")):
                    raise ValueError("Range end must be after its start")
        return self


class CreateTicket(Arguments):
    summary: Title
    description: Description = ""


class Search(Arguments):
    query: Query


class GetTicket(Arguments):
    ticket_key: TicketKey


class AddComment(GetTicket):
    comment: Annotated[str, StringConstraints(min_length=1, max_length=8192)]


class GetFile(Arguments):
    file_id: Identifier


class CreateFile(Arguments):
    title: Title
    content: Content = ""


class CreateEvent(Arguments):
    title: Title
    start_time: Instant
    end_time: Instant | None = None


class ListEvents(Arguments):
    start: Instant | None = None
    end: Instant | None = None


class DeleteEvent(Arguments):
    event_id: Identifier


_CONTRACTS: dict[tuple[str, str], type[Arguments]] = {
    ("jira", "create_ticket"): CreateTicket,
    ("jira", "search_tickets"): Search,
    ("jira", "get_ticket"): GetTicket,
    ("jira", "add_comment"): AddComment,
    ("drive", "search_files"): Search,
    ("drive", "get_file"): GetFile,
    ("drive", "create_file"): CreateFile,
    ("calendar", "create_event"): CreateEvent,
    ("calendar", "list_events"): ListEvents,
    ("calendar", "delete_event"): DeleteEvent,
}


def validate_tool_arguments(tool: str, action: str, params: dict[str, Any]) -> dict[str, Any]:
    contract = _CONTRACTS.get((tool, action))
    if contract is None:
        raise ToolError("Unknown cloud tool action")
    try:
        validated = contract.model_validate(params)
    except ValidationError as exc:
        # Pydantic's raw error contains input values; keep it out of user errors/logs.
        raise ToolError("Invalid arguments for the selected tool action") from exc
    return validated.model_dump(exclude_unset=True, exclude_none=True)
