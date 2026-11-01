"""Add scoped credentials and a shared single-use WebSocket ticket store.

Revision ID: 0002_auth
"""

from alembic import op

revision = "0002_auth"
down_revision = "0001_legacy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill explicitly preserves old key behavior. New keys get a narrow default.
    op.execute("""
        ALTER TABLE api_keys ADD COLUMN scopes TEXT[] NOT NULL DEFAULT
        ARRAY['documents:ingest','documents:read','chat:use','chat:history',
              'tools:read','tools:write','pc:connect','pc:execute']::TEXT[];
        ALTER TABLE api_keys ALTER COLUMN scopes SET DEFAULT
        ARRAY['documents:ingest','documents:read','chat:use','chat:history']::TEXT[];
        ALTER TABLE api_keys ADD CONSTRAINT api_keys_known_scopes CHECK (
            scopes <@ ARRAY['documents:ingest','documents:read','chat:use','chat:history',
                           'tools:read','tools:write','pc:connect','pc:execute','accounts:manage']::TEXT[]
            AND array_position(scopes, NULL) IS NULL
        );
        CREATE TABLE websocket_tickets (
            token_hash TEXT PRIMARY KEY,
            credential_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
            tenant_id UUID NOT NULL REFERENCES tenants(id),
            audience TEXT NOT NULL CHECK (audience IN ('chat', 'pc-agent')),
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX websocket_tickets_expiry ON websocket_tickets(expires_at);
        CREATE INDEX websocket_tickets_credential ON websocket_tickets(credential_id);
    """)


def downgrade() -> None:
    # Operator-only rollback: discards ephemeral tickets and scope grants.
    # A legacy application then has broad tenant-key behavior again.
    op.drop_table("websocket_tickets")
    op.drop_column("api_keys", "scopes")
