"""
Per-tenant, encrypted storage for connected-account OAuth tokens (Jira,
Google Drive, Google Calendar, ...).

Tokens are encrypted with Fernet (AES-128-CBC + HMAC) before they ever
reach Postgres -- the database only ever stores ciphertext, so a DB
dump or a stray backup doesn't leak usable credentials. Fernet (not
SHA-256, unlike the API-key hashing in services.auth.api_keys) is used
here deliberately: API keys are hashed because the server only ever
needs to *verify* a key was presented, never recover it, whereas OAuth
tokens must be *decrypted* back to their original value to actually call
Jira/Google's APIs on the tenant's behalf -- that requires reversible
encryption, not a one-way hash.

This module only handles storage. The OAuth dance itself (redirecting a
tenant to Jira/Google, exchanging a code for tokens) isn't built yet --
that requires real client IDs/secrets from each provider. Until then,
save_connected_account() is called with tokens obtained however a tool
integration currently gets them (manually, for development), and every
concrete tool (services/agent/tools/*) already reads credentials through
get_connected_account() -- so wiring in a real OAuth flow later only
means calling save_connected_account() from a new callback endpoint,
with no changes needed in the tools themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg2
from cryptography.fernet import Fernet, InvalidToken

from config import CONNECTED_ACCOUNTS_ENCRYPTION_KEY, POSTGRES_DSN

SUPPORTED_PROVIDERS = {"jira", "google_drive", "google_calendar"}


class VaultNotConfigured(RuntimeError):
    """Raised when CONNECTED_ACCOUNTS_ENCRYPTION_KEY isn't set. Tokens
    are never stored or decrypted without a real key -- there is no
    plaintext fallback."""


@dataclass
class ConnectedAccount:
    tenant_id: str
    provider: str
    access_token: str
    refresh_token: str | None
    expires_at: datetime | None
    scopes: list[str]


def _get_fernet() -> Fernet:
    if not CONNECTED_ACCOUNTS_ENCRYPTION_KEY:
        raise VaultNotConfigured(
            "CONNECTED_ACCOUNTS_ENCRYPTION_KEY is not set -- generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and set it before connecting any account."
        )
    return Fernet(CONNECTED_ACCOUNTS_ENCRYPTION_KEY.encode("utf-8"))


def save_connected_account(
    tenant_id: str,
    provider: str,
    access_token: str,
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
    scopes: list[str] | None = None,
) -> None:
    """Encrypts and upserts a tenant's credentials for ``provider``.

    Raises:
        ValueError: If ``provider`` isn't one of SUPPORTED_PROVIDERS.
        VaultNotConfigured: If no encryption key is configured.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}")

    fernet = _get_fernet()
    encrypted_access = fernet.encrypt(access_token.encode("utf-8")).decode("utf-8")
    encrypted_refresh = (
        fernet.encrypt(refresh_token.encode("utf-8")).decode("utf-8") if refresh_token else None
    )

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO connected_accounts
                    (tenant_id, provider, encrypted_access_token, encrypted_refresh_token,
                     token_expires_at, scopes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (tenant_id, provider) DO UPDATE SET
                    encrypted_access_token = EXCLUDED.encrypted_access_token,
                    encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                    token_expires_at = EXCLUDED.token_expires_at,
                    scopes = EXCLUDED.scopes,
                    updated_at = NOW();
                """,
                (tenant_id, provider, encrypted_access, encrypted_refresh, expires_at, scopes or []),
            )
        conn.commit()
    finally:
        conn.close()


def get_connected_account(tenant_id: str, provider: str) -> ConnectedAccount | None:
    """Returns the tenant's decrypted credentials for ``provider``, or
    None if no account is connected.

    Raises:
        VaultNotConfigured: If no encryption key is configured.
        InvalidToken: If the stored ciphertext can't be decrypted with
            the current key (e.g. the key was rotated without
            re-encrypting existing rows).
    """
    fernet = _get_fernet()

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT encrypted_access_token, encrypted_refresh_token, token_expires_at, scopes
                FROM connected_accounts
                WHERE tenant_id = %s AND provider = %s;
                """,
                (tenant_id, provider),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    encrypted_access, encrypted_refresh, expires_at, scopes = row
    try:
        access_token = fernet.decrypt(encrypted_access.encode("utf-8")).decode("utf-8")
        refresh_token = (
            fernet.decrypt(encrypted_refresh.encode("utf-8")).decode("utf-8")
            if encrypted_refresh
            else None
        )
    except InvalidToken:
        raise

    return ConnectedAccount(
        tenant_id=tenant_id,
        provider=provider,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        scopes=list(scopes or []),
    )


def disconnect_account(tenant_id: str, provider: str) -> bool:
    """Deletes a tenant's stored credentials for ``provider``.

    Returns:
        True if an account was actually deleted.
    """
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM connected_accounts WHERE tenant_id = %s AND provider = %s;",
                (tenant_id, provider),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def list_connected_providers(tenant_id: str) -> list[str]:
    """Returns the providers a tenant currently has connected, without
    decrypting anything -- used to tell the classifier/tool registry
    which tools are actually usable for this tenant right now."""
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT provider FROM connected_accounts WHERE tenant_id = %s;", (tenant_id,)
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
