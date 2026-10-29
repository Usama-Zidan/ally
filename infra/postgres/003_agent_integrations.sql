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
