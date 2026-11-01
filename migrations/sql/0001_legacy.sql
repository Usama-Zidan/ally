-- Frozen legacy baseline: 001_document_chunks.sql
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (filename, page_number, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_filename
    ON document_chunks (filename);


-- Frozen legacy baseline: 002_auth_and_tenancy.sql
-- Adds multi-tenancy and API-key authentication.
--
-- Every chunk in document_chunks must belong to a tenant so that
-- retrieval, ingestion, and deletion can all be scoped per-tenant — see
-- services/auth (API-key resolution) and services/retrieval/qdrant_store.py
-- (the equivalent tenant_id payload field/filter on the Qdrant side).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys (tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active_lookup
    ON api_keys (key_hash) WHERE revoked_at IS NULL;

-- Seed a default tenant so any document_chunks rows that predate
-- multi-tenancy (from 001_document_chunks.sql alone) have somewhere to
-- attach to when this migration's ALTER TABLE runs below.
INSERT INTO tenants (id, name)
VALUES ('00000000-0000-0000-0000-000000000000', 'default')
ON CONFLICT (name) DO NOTHING;

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS tenant_id UUID NOT NULL
        DEFAULT '00000000-0000-0000-0000-000000000000'
        REFERENCES tenants(id);

-- The default above only exists to backfill pre-existing rows; new rows
-- must always specify tenant_id explicitly rather than silently falling
-- back to "default".
ALTER TABLE document_chunks ALTER COLUMN tenant_id DROP DEFAULT;

-- A filename is only unique within a tenant now — two tenants can each
-- upload a file called "policy.pdf" without colliding.
ALTER TABLE document_chunks
    DROP CONSTRAINT IF EXISTS document_chunks_filename_page_number_chunk_index_key;

ALTER TABLE document_chunks
    ADD CONSTRAINT document_chunks_tenant_filename_page_chunk_key
    UNIQUE (tenant_id, filename, page_number, chunk_index);

CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant ON document_chunks (tenant_id);


-- Frozen legacy baseline: 003_agent_integrations.sql
-- Adds per-tenant storage for Phase 4's agent tool integrations:
--   - connected_accounts: encrypted OAuth tokens for external tools
--     (Jira, Google Drive, Google Calendar, ...). Tokens are encrypted
--     at the application layer (services/agent/connected_accounts.py,
--     Fernet/AES-128-CBC via CONNECTED_ACCOUNTS_ENCRYPTION_KEY) before
--     ever reaching this table -- Postgres only ever sees ciphertext.
--   - pc_agent_capabilities: per-tenant toggles for what the local
--     Windows companion agent is allowed to do. These are the
--     server-declared defaults a tenant has opted into; the local agent
--     process is the actual enforcement point (see local_agent/), since
--     the server cannot force a local machine to execute anything it
--     hasn't independently agreed to run.

CREATE TABLE IF NOT EXISTS connected_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    provider TEXT NOT NULL CHECK (provider IN ('jira', 'google_drive', 'google_calendar')),
    -- Fernet ciphertext (base64 text), never plaintext.
    encrypted_access_token TEXT NOT NULL,
    encrypted_refresh_token TEXT,
    token_expires_at TIMESTAMPTZ,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, provider)
);
CREATE INDEX IF NOT EXISTS idx_connected_accounts_tenant ON connected_accounts (tenant_id);

CREATE TABLE IF NOT EXISTS pc_agent_capabilities (
    tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    -- Safer-by-default capabilities, on by default.
    open_apps BOOLEAN NOT NULL DEFAULT TRUE,
    file_rw BOOLEAN NOT NULL DEFAULT TRUE,
    allowed_folders TEXT[] NOT NULL DEFAULT '{}',
    -- Higher-risk capabilities: opt-in only, off by default.
    shell_exec BOOLEAN NOT NULL DEFAULT FALSE,
    mouse_keyboard_control BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

