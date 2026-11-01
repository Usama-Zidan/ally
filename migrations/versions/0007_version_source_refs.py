"""Preserve immutable source refs on each ingestion version.

Revision ID: 0007_version_source_refs
"""

from alembic import op
from sqlalchemy import text

revision = "0007_version_source_refs"
down_revision = "0006_ingestion_tombstones"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE document_versions ADD COLUMN storage_ref TEXT;")


def downgrade() -> None:
    has_refs = (
        op.get_bind()
        .execute(
            text("SELECT EXISTS (SELECT 1 FROM document_versions WHERE storage_ref IS NOT NULL)")
        )
        .scalar_one()
    )
    if has_refs:
        raise RuntimeError("Version source-ref downgrade is unsafe while version refs exist")
    op.execute("ALTER TABLE document_versions DROP COLUMN storage_ref;")
