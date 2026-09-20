"""
API-key authentication, resolving a raw key to its owning tenant_id.

Keys are stored as SHA-256 hashes (never plaintext) in the api_keys
table — see infra/postgres/002_auth_and_tenancy.sql for the schema and
the rationale for using a fast hash here instead of bcrypt/argon2.
"""
from __future__ import annotations

import hashlib
import secrets

import psycopg2

from config import POSTGRES_DSN

# Raw API keys carry this prefix so they're recognizable at a glance
# (e.g. in a support ticket or an accidentally-committed .env file)
# without needing to look them up, the same convention as Stripe/GitHub
# tokens.
API_KEY_PREFIX = "ally_"


def hash_key(raw_key: str) -> str:
    """Returns the SHA-256 hex digest of ``raw_key``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generates a new random API key.

    Returns:
        A (raw_key, key_hash) pair. The raw key is only ever returned
        here — callers must show it to the tenant immediately and persist
        only the hash, since it cannot be recovered afterward.
    """
    raw_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    return raw_key, hash_key(raw_key)


def resolve_tenant(raw_key: str | None) -> str | None:
    """Looks up the tenant_id owning ``raw_key``.

    Returns:
        The tenant_id as a string, or None if the key is missing, unknown,
        or has been revoked.
    """
    if not raw_key:
        return None

    key_hash = hash_key(raw_key)
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id FROM api_keys "
                "WHERE key_hash = %s AND revoked_at IS NULL;",
                (key_hash,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None
    finally:
        conn.close()


def create_tenant_with_key(tenant_name: str) -> tuple[str, str]:
    """Creates (or reuses) a tenant by name and issues it a fresh API key.

    Intended for an admin/CLI provisioning flow — deliberately not
    exposed as an HTTP endpoint, since anyone who could call it could
    mint themselves a valid tenant + key.

    Returns:
        A (tenant_id, raw_api_key) pair. The raw key is shown once; only
        its hash is persisted.
    """
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (name) VALUES (%s) "
                "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                "RETURNING id;",
                (tenant_name,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to create or retrieve tenant")
            tenant_id = str(row[0])

            raw_key, key_hash = generate_api_key()
            cur.execute(
                "INSERT INTO api_keys (tenant_id, key_hash) VALUES (%s, %s);",
                (tenant_id, key_hash),
            )
        conn.commit()
        return tenant_id, raw_key
    finally:
        conn.close()


def revoke_api_key(raw_key: str) -> bool:
    """Revokes ``raw_key`` so resolve_tenant() no longer accepts it.

    Returns:
        True if a matching, not-already-revoked key was found and revoked.
    """
    key_hash = hash_key(raw_key)
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET revoked_at = NOW() "
                "WHERE key_hash = %s AND revoked_at IS NULL;",
                (key_hash,),
            )
            revoked = cur.rowcount > 0
        conn.commit()
        return revoked
    finally:
        conn.close()
