"""Native companion redemption of operator-issued enrollment secrets."""

from __future__ import annotations

import asyncio
import psycopg2
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from services.auth.admission import AdmissionDenied
from services.auth.principal import AuthorizationDenied

router = APIRouter(prefix="/auth/devices", tags=["device enrollment"])


class PairingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    token: str = Field(min_length=32, max_length=128)


@router.post("/redeem")
async def redeem_device(request: Request) -> JSONResponse:
    state = request.app.state
    if state.websocket_security.settings.require_tls and request.url.scheme != "https":
        raise HTTPException(400, "HTTPS is required")
    if request.headers.get("origin") is not None:
        raise HTTPException(403, "Pair from the native companion")
    host = request.client.host if request.client else "unknown"
    try:
        with state.pairing_admission.admit(host):
            data = bytearray()
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        if len(data) + len(chunk) > 1024:
                            raise HTTPException(413, "Pairing request exceeds size limit")
                        data.extend(chunk)
            except TimeoutError as exc:
                raise HTTPException(408, "Pairing request body timed out") from exc
            try:
                payload = PairingRequest.model_validate_json(bytes(data))
            except ValueError as exc:
                raise HTTPException(422, "Invalid pairing request") from exc
            enrollment = await run_in_threadpool(state.device_registry.redeem, payload.token)
    except AdmissionDenied as exc:
        raise HTTPException(429, "Pairing rate limit reached") from exc
    except AuthorizationDenied as exc:
        raise HTTPException(403, "Invalid or expired pairing token") from exc
    except psycopg2.Error as exc:
        raise HTTPException(503, "Device enrollment is temporarily unavailable") from exc
    return JSONResponse(
        {"device_id": enrollment.device_id, "api_key": enrollment.api_key},
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
