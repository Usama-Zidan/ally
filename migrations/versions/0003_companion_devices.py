"""Credential-bound companion enrollment; existing tenant keys are not devices."""

from alembic import op

revision = "0003_devices"
down_revision = "0002_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE api_keys ADD CONSTRAINT api_keys_identity_tenant UNIQUE (id, tenant_id);
        CREATE TABLE companion_devices (
            id UUID PRIMARY KEY,
            tenant_id UUID NOT NULL REFERENCES tenants(id),
            credential_id UUID NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            revoked_at TIMESTAMPTZ,
            FOREIGN KEY (credential_id, tenant_id) REFERENCES api_keys(id, tenant_id)
        );
        CREATE UNIQUE INDEX companion_one_active_device ON companion_devices(tenant_id)
            WHERE revoked_at IS NULL;
        CREATE TABLE companion_pairings (
            tenant_id UUID PRIMARY KEY REFERENCES tenants(id),
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX companion_pairings_expiry ON companion_pairings(expires_at);
    """)


def downgrade() -> None:
    # A legacy binary would accept broad tenant keys as devices again.
    # A forward repair keeps the security boundary and enrollment audit intact.
    raise RuntimeError("Device enrollment downgrade is unsafe; use a reviewed forward migration")
