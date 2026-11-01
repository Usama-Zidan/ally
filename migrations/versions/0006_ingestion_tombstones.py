"""Record soft-deleted ingestion documents for index reconciliation.

Revision ID: 0006_ingestion_tombstones
"""

from alembic import op
from sqlalchemy import text

revision = "0006_ingestion_tombstones"
down_revision = "0005_ingestion_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ingestion_tombstones (
            document_id UUID PRIMARY KEY,
            tenant_id UUID NOT NULL,
            deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reason TEXT CHECK (reason IS NULL OR reason ~ '^[a-z0-9_]{1,64}$'),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES ingestion_documents (id, tenant_id) ON DELETE RESTRICT
        );
        CREATE INDEX ingestion_tombstones_deleted_at
            ON ingestion_tombstones (deleted_at);
        """
    )


def downgrade() -> None:
    has_tombstones = (
        op.get_bind()
        .execute(text("SELECT EXISTS (SELECT 1 FROM ingestion_tombstones)"))
        .scalar_one()
    )
    if has_tombstones:
        raise RuntimeError("Ingestion tombstone downgrade is unsafe while tombstones exist")
    op.execute("DROP INDEX ingestion_tombstones_deleted_at; DROP TABLE ingestion_tombstones;")
