"""FastAPI dependencies for resolving the caller's tenant from an API key."""
from __future__ import annotations

from fastapi import Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from services.auth.api_keys import resolve_tenant


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
    tenant_id = await run_in_threadpool(resolve_tenant, x_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return tenant_id


async def get_current_tenant_ws(api_key: str | None) -> str | None:
    """WebSocket variant: browsers can't set custom headers during the
    WebSocket handshake, so the key arrives as a query parameter instead
    (see services.api.main.chat_ws). Returns None instead of raising so
    the caller can close the socket with a proper WebSocket close code
    rather than surfacing an HTTP error page.
    """
    return await run_in_threadpool(resolve_tenant, api_key)
