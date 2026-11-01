"""FastAPI dependencies for resolving the caller's tenant from an API key."""

from __future__ import annotations

import psycopg2
from fastapi import Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from services.auth.api_keys import resolve_principal
from services.auth.principal import AuthorizationDenied, Principal


async def authenticate_key(raw_key: str | None) -> Principal | None:
    """Distinguish an authentication-store outage from an invalid credential."""
    try:
        return await run_in_threadpool(resolve_principal, raw_key)
    except (psycopg2.Error, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Authentication is temporarily unavailable"
        ) from exc


async def get_current_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    principal = await authenticate_key(x_api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return principal


async def get_ingestion_tenant(x_api_key: str | None = Header(default=None)) -> str:
    principal = await get_current_principal(x_api_key)
    try:
        principal.require("documents:ingest")
    except AuthorizationDenied as exc:
        raise HTTPException(status_code=403, detail="Document ingestion is not permitted") from exc
    return principal.tenant_id


async def get_current_tenant(x_api_key: str | None = Header(default=None)) -> str:
    """REST dependency: requires a valid ``X-API-Key`` header.

    resolve_tenant() makes a blocking psycopg2 call, so it's run in a
    threadpool rather than directly in this async function — otherwise
    every request would block the event loop for the duration of that
    database round-trip (see run_in_threadpool's use elsewhere for the
    same reason, e.g. warm_reranker in services.api.main's lifespan).

    Raises:
        HTTPException: 401 if the header is missing or the key is
            unknown/revoked.
    """
    return (await get_current_principal(x_api_key)).tenant_id


async def get_current_tenant_ws(api_key: str | None) -> str | None:
    """Resolve a WebSocket key supplied by the transport.

    Native clients use X-API-Key; query parameters remain a legacy browser
    path only in development; browser clients should use scoped tickets. Returns None so
    the caller can close the socket with a proper WebSocket close code
    rather than surfacing an HTTP error page.
    """
    principal = await authenticate_key(api_key)
    return principal.tenant_id if principal else None
