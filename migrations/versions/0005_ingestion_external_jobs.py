"""Persist external ingestion jobs for retry correlation.

Revision ID: 0005_ingestion_jobs
"""

from alembic import op
from sqlalchemy import text

revision = "0005_ingestion_jobs"
down_revision = "0004_ingestion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ingestion_external_jobs (
            version_id UUID PRIMARY KEY,
            provider TEXT NOT NULL CHECK (provider IN ('textract')),
            request_token TEXT NOT NULL UNIQUE,
            external_job_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('started', 'in_progress', 'succeeded', 'failed')
            ),
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            FOREIGN KEY (version_id)
                REFERENCES document_versions (id) ON DELETE CASCADE
        );
        CREATE INDEX ingestion_external_jobs_status_updated
            ON ingestion_external_jobs (provider, status, updated_at);
        """
    )


def downgrade() -> None:
    has_jobs = (
        op.get_bind()
        .execute(text("SELECT EXISTS (SELECT 1 FROM ingestion_external_jobs)"))
        .scalar_one()
    )
    if has_jobs:
        raise RuntimeError("External ingestion job downgrade is unsafe while job records exist")
    op.execute(
        """
        DROP INDEX ingestion_external_jobs_status_updated;
        DROP TABLE ingestion_external_jobs;
        """
    )
