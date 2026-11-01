"""PostgreSQL companion enrollment. Administrative issuance is CLI-only.

The one-time token is an enrollment secret transferred to the intended device.
It proves possession, not an individual user's identity or hardware attestation.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from psycopg2.extensions import cursor

from services.auth.api_keys import generate_api_key, hash_key
from services.auth.database import connect_auth
from services.auth.principal import AuthorizationDenied, Principal

PAIRING_TTL_SECONDS = 120
MAX_PENDING_PAIRINGS = 4096
DEVICE_SCOPES = frozenset({"pc:connect"})


class PairingCapacityExceeded(Exception):
    """Enrollment admission is temporarily full."""


@dataclass(frozen=True)
class PairedDevice:
    tenant_id: str
    credential_id: str
    device_id: str


@dataclass(frozen=True)
class DeviceEnrollment:
    device_id: str
    api_key: str = field(repr=False)


class DeviceRegistry:
    def issue_pairing(self, tenant_id: str) -> str:
        """Operator authority comes from database access, never a tenant key.

        Only the latest token for a tenant remains usable. Issuance does not
        disconnect its current device; redemption explicitly replaces it.
        """
        tenant_id = str(UUID(tenant_id))
        token = secrets.token_urlsafe(32)
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('ally:pairing-issue', 0))"
                )
                cur.execute("SELECT id FROM tenants WHERE id=%s FOR NO KEY UPDATE", (tenant_id,))
                if cur.fetchone() is None:
                    raise AuthorizationDenied("Tenant does not exist")
                cur.execute("DELETE FROM companion_pairings WHERE expires_at <= NOW()")
                cur.execute("SELECT COUNT(*) FROM companion_pairings")
                row = cur.fetchone()
                if row is None or row[0] >= MAX_PENDING_PAIRINGS:
                    raise PairingCapacityExceeded("Too many pending enrollments")
                cur.execute(
                    "INSERT INTO companion_pairings(tenant_id,token_hash,expires_at) "
                    "VALUES (%s,%s,NOW()+%s*INTERVAL '1 second') "
                    "ON CONFLICT (tenant_id) DO UPDATE SET token_hash=EXCLUDED.token_hash, "
                    "expires_at=EXCLUDED.expires_at, created_at=NOW()",
                    (tenant_id, hash_key(token), PAIRING_TTL_SECONDS),
                )
            conn.commit()
            return token
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def redeem(self, token: str) -> DeviceEnrollment:
        if not 32 <= len(token) <= 128:
            raise AuthorizationDenied("Invalid or expired pairing token")
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                # Lock tenant before the pairing row, matching issue/revoke order.
                cur.execute(
                    "SELECT t.id FROM tenants t JOIN companion_pairings p ON p.tenant_id=t.id "
                    "WHERE p.token_hash=%s FOR NO KEY UPDATE OF t",
                    (hash_key(token),),
                )
                row = cur.fetchone()
                if row is None:
                    raise AuthorizationDenied("Invalid or expired pairing token")
                tenant_id = str(row[0])
                cur.execute(
                    "DELETE FROM companion_pairings WHERE tenant_id=%s AND token_hash=%s "
                    "AND expires_at>clock_timestamp() RETURNING tenant_id",
                    (tenant_id, hash_key(token)),
                )
                if cur.fetchone() is None:
                    raise AuthorizationDenied("Invalid or expired pairing token")
                self._revoke(cur, tenant_id)
                device_id, credential_id = str(uuid4()), str(uuid4())
                raw_key, key_hash = generate_api_key()
                cur.execute(
                    "INSERT INTO api_keys(id,tenant_id,key_hash,scopes) VALUES (%s,%s,%s,%s)",
                    (credential_id, tenant_id, key_hash, sorted(DEVICE_SCOPES)),
                )
                cur.execute(
                    "INSERT INTO companion_devices(id,tenant_id,credential_id) VALUES (%s,%s,%s)",
                    (device_id, tenant_id, credential_id),
                )
            conn.commit()
            return DeviceEnrollment(device_id, raw_key)
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _revoke(cur: cursor, tenant_id: str) -> None:
        cur.execute(
            "UPDATE api_keys k SET revoked_at=NOW() FROM companion_devices d "
            "WHERE d.tenant_id=%s AND d.credential_id=k.id AND d.revoked_at IS NULL "
            "AND k.revoked_at IS NULL",
            (tenant_id,),
        )
        cur.execute(
            "UPDATE companion_devices SET revoked_at=NOW() WHERE tenant_id=%s AND revoked_at IS NULL",
            (tenant_id,),
        )

    def revoke(self, tenant_id: str) -> None:
        tenant_id = str(UUID(tenant_id))
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM tenants WHERE id=%s FOR NO KEY UPDATE", (tenant_id,))
                if cur.fetchone() is None:
                    raise AuthorizationDenied("Tenant does not exist")
                self._revoke(cur, tenant_id)
                cur.execute("DELETE FROM companion_pairings WHERE tenant_id=%s", (tenant_id,))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def bind(self, principal: Principal, device_id: str) -> PairedDevice:
        if principal.scopes != DEVICE_SCOPES:
            raise AuthorizationDenied("A dedicated paired device credential is required")
        try:
            device_id = str(UUID(device_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise AuthorizationDenied("Invalid paired device identity") from exc
        device = PairedDevice(principal.tenant_id, principal.credential_id, device_id)
        self.require_active(device)
        return device

    def require_active(self, device: PairedDevice) -> None:
        conn = connect_auth()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM companion_devices d JOIN api_keys k "
                    "ON k.id=d.credential_id AND k.tenant_id=d.tenant_id "
                    "WHERE d.id=%s AND d.tenant_id=%s AND d.credential_id=%s "
                    "AND d.revoked_at IS NULL AND k.revoked_at IS NULL AND k.scopes=%s::TEXT[]",
                    (
                        device.device_id,
                        device.tenant_id,
                        device.credential_id,
                        sorted(DEVICE_SCOPES),
                    ),
                )
                if cur.fetchone() is None:
                    raise AuthorizationDenied("Paired device is no longer active")
        finally:
            conn.close()
