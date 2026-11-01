"""Adopt the complete legacy schema, or bootstrap an empty database.

Revision ID: 0001_legacy
"""

from pathlib import Path

from alembic import op
from sqlalchemy import inspect

revision = "0001_legacy"
down_revision = None
branch_labels = None
depends_on = None

_REQUIRED = {
    "tenants": {"id", "name"},
    "api_keys": {"id", "tenant_id", "key_hash", "revoked_at"},
    "document_chunks": {"tenant_id", "filename", "page_number", "chunk_index", "text"},
    "connected_accounts": {"tenant_id", "provider", "encrypted_access_token"},
    "pc_agent_capabilities": {"tenant_id", "shell_exec"},
}


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    existing = set(inspector.get_table_names()) & _REQUIRED.keys()
    if existing:
        if existing != _REQUIRED.keys() or any(
            not columns <= {column["name"] for column in inspector.get_columns(table)}
            for table, columns in _REQUIRED.items()
        ):
            raise RuntimeError(
                "Legacy schema is incomplete; reconcile a database copy before adoption"
            )
        return
    # Keep historical upgrades reproducible when Docker init SQL evolves later.
    snapshot = Path(__file__).resolve().parents[1] / "sql" / "0001_legacy.sql"
    op.execute(snapshot.read_text(encoding="utf-8"))


def downgrade() -> None:
    raise RuntimeError("Baseline adoption cannot be reversed by dropping existing application data")
