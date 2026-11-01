"""Atomic PostgreSQL-backed, credential- and audience-bound WebSocket tickets."""

from __future__ import annotations

import hashlib
import secrets
from typing import Literal, Protocol

from services.auth.database import connect_auth
from services.auth.principal import AuthorizationDenied, Principal
from services.auth.settings import SecuritySettings

Audience = Literal["chat", "pc-agent"]
AUDIENCE_SCOPE: dict[Audience, str] = {"chat": "chat:use", "pc-agent": "pc:connect"}


class TicketCapacityExceeded(Exception):
    """Outstanding ticket quota is exhausted; never evict an active ticket."""


class TicketStore(Protocol):
    def issue(self, principal: Principal, audience: Audience) -> str: ...
    def consume(self, token: str, audience: Audience) -> Principal | None: ...


def ticket_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PostgresTicketStore:
    def __init__(self, settings: SecuritySettings) -> None:
        self.settings = settings

    def issue(self, principal: Principal, audience: Audience) -> str:
        principal.require(AUDIENCE_SCOPE[audience])
        token = secrets.token_urlsafe(32)
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                # Serialize bounded issuance across processes. Consumption remains concurrent.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('ally:ws-ticket-issue', 0));"
                )
                cur.execute(
                    "SELECT id, tenant_id, scopes FROM api_keys WHERE id = %s "
                    "AND tenant_id = %s AND revoked_at IS NULL FOR SHARE;",
                    (principal.credential_id, principal.tenant_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise AuthorizationDenied("Credential is no longer active")
                current = Principal(str(row[1]), str(row[0]), frozenset(row[2]))
                current.require(AUDIENCE_SCOPE[audience])
                cur.execute("DELETE FROM websocket_tickets WHERE expires_at <= NOW();")
                cur.execute(
                    "SELECT COUNT(*), COUNT(*) FILTER (WHERE credential_id = %s) FROM websocket_tickets;",
                    (principal.credential_id,),
                )
                counts = cur.fetchone()
                if counts is None:
                    raise ValueError("Ticket quota query returned no row")
                total, owned = counts
                if (
                    total >= self.settings.max_tickets
                    or owned >= self.settings.max_tickets_per_credential
                ):
                    raise TicketCapacityExceeded("Too many outstanding WebSocket tickets")
                cur.execute(
                    "INSERT INTO websocket_tickets (token_hash, credential_id, tenant_id, audience, expires_at) "
                    "VALUES (%s, %s, %s, %s, NOW() + %s * INTERVAL '1 second');",
                    (
                        ticket_hash(token),
                        principal.credential_id,
                        principal.tenant_id,
                        audience,
                        self.settings.ticket_ttl_seconds,
                    ),
                )
            conn.commit()
            return token
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def consume(self, token: str, audience: Audience) -> Principal | None:
        if not 32 <= len(token) <= 128:
            return None
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM websocket_tickets AS t USING api_keys AS k "
                    "WHERE t.token_hash = %s AND t.audience = %s AND t.expires_at > NOW() "
                    "AND k.id = t.credential_id AND k.tenant_id = t.tenant_id AND k.revoked_at IS NULL "
                    "RETURNING k.id, k.tenant_id, k.scopes;",
                    (ticket_hash(token), audience),
                )
                row = cur.fetchone()
                principal = Principal(str(row[1]), str(row[0]), frozenset(row[2])) if row else None
            conn.commit()
            if principal is not None:
                principal.require(AUDIENCE_SCOPE[audience])
            return principal
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
