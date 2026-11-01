"""Issue short-lived socket tickets through an authenticated HTTP boundary."""

from __future__ import annotations

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from services.auth.dependencies import get_current_principal
from services.auth.principal import AuthorizationDenied, Principal
from services.auth.tickets import AUDIENCE_SCOPE, Audience, TicketCapacityExceeded

router = APIRouter(prefix="/auth", tags=["authentication"])


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audience: Audience


@router.post("/ws-tickets")
async def issue_websocket_ticket(
    payload: TicketRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
) -> JSONResponse:
    security = request.app.state.websocket_security
    if security.settings.require_tls and request.url.scheme != "https":
        raise HTTPException(status_code=400, detail="HTTPS is required")
    try:
        principal.require(AUDIENCE_SCOPE[payload.audience])
        token = await run_in_threadpool(security.tickets.issue, principal, payload.audience)
    except AuthorizationDenied as exc:
        raise HTTPException(
            status_code=403, detail="Credential cannot connect to this socket"
        ) from exc
    except TicketCapacityExceeded as exc:
        raise HTTPException(
            status_code=429, detail="Too many outstanding WebSocket tickets"
        ) from exc
    except (psycopg2.Error, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Ticket service is temporarily unavailable"
        ) from exc
    return JSONResponse(
        {
            "ticket": token,
            "expires_in": security.settings.ticket_ttl_seconds,
            "audience": payload.audience,
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
