"""Add stable document, attempt and version identities.

Revision ID: 0004_ingestion
"""

from alembic import op
from sqlalchemy import text

revision = "0004_ingestion"
down_revision = "0003_devices"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ingestion_documents (
            id UUID PRIMARY KEY,
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            storage_ref TEXT,
            content_sha256 TEXT CHECK (
                content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
            ),
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'processing', 'indexed', 'failed', 'superseded', 'deleted')
            ),
            active_version_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (id, tenant_id),
            CHECK (status = 'deleted' OR storage_ref IS NOT NULL)
        );
        CREATE INDEX ingestion_documents_tenant_filename
            ON ingestion_documents (tenant_id, filename);
        CREATE INDEX ingestion_documents_tenant_status
            ON ingestion_documents (tenant_id, status);

        CREATE TABLE ingestion_attempts (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            workflow_id TEXT NOT NULL UNIQUE,
            attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'processing', 'indexed', 'failed', 'cancelled')
            ),
            current_stage TEXT,
            error_code TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (document_id, attempt_number),
            UNIQUE (id, document_id, tenant_id),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES ingestion_documents (id, tenant_id) ON DELETE CASCADE
        );
        CREATE INDEX ingestion_attempts_document_status
            ON ingestion_attempts (document_id, status, attempt_number DESC);

        CREATE TABLE document_versions (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            attempt_id UUID NOT NULL,
            version_number INTEGER NOT NULL CHECK (version_number > 0),
            content_sha256 TEXT CHECK (
                content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
            ),
            status TEXT NOT NULL CHECK (
                status IN ('staged', 'active', 'superseded', 'failed')
            ),
            chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            activated_at TIMESTAMPTZ,
            UNIQUE (document_id, version_number),
            UNIQUE (id, document_id, tenant_id),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES ingestion_documents (id, tenant_id) ON DELETE CASCADE,
            FOREIGN KEY (attempt_id, document_id, tenant_id)
                REFERENCES ingestion_attempts (id, document_id, tenant_id) ON DELETE CASCADE
        );
        CREATE INDEX document_versions_document_status
            ON document_versions (document_id, status, version_number DESC);

        ALTER TABLE ingestion_documents
            ADD CONSTRAINT ingestion_documents_active_version_fk
            FOREIGN KEY (active_version_id, id, tenant_id)
            REFERENCES document_versions (id, document_id, tenant_id)
            DEFERRABLE INITIALLY DEFERRED;

        ALTER TABLE document_chunks
            ADD COLUMN document_id UUID,
            ADD COLUMN version_id UUID,
            ADD CONSTRAINT document_chunks_document_version_pair
                CHECK ((document_id IS NULL) = (version_id IS NULL)),
            ADD CONSTRAINT document_chunks_document_fk
                FOREIGN KEY (document_id, tenant_id)
                REFERENCES ingestion_documents (id, tenant_id),
            ADD CONSTRAINT document_chunks_version_fk
                FOREIGN KEY (version_id, document_id, tenant_id)
                REFERENCES document_versions (id, document_id, tenant_id);

        CREATE INDEX document_chunks_version_lookup
            ON document_chunks (tenant_id, document_id, version_id);

        CREATE TABLE document_version_chunks (
            tenant_id UUID NOT NULL,
            document_id UUID NOT NULL,
            version_id UUID NOT NULL,
            filename TEXT NOT NULL,
            page_number INTEGER NOT NULL CHECK (page_number > 0),
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            text TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, version_id, page_number, chunk_index),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES ingestion_documents (id, tenant_id) ON DELETE CASCADE,
            FOREIGN KEY (version_id, document_id, tenant_id)
                REFERENCES document_versions (id, document_id, tenant_id) ON DELETE CASCADE
        );
        CREATE INDEX document_version_chunks_document
            ON document_version_chunks (tenant_id, document_id, version_id);
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    has_versioned_data = bind.execute(
        text(
            """
            SELECT EXISTS (SELECT 1 FROM ingestion_documents)
                OR EXISTS (SELECT 1 FROM document_chunks WHERE version_id IS NOT NULL)
                OR EXISTS (SELECT 1 FROM document_version_chunks)
            """
        )
    ).scalar_one()
    if has_versioned_data:
        raise RuntimeError(
            "Ingestion identity downgrade is unsafe after documents or versioned chunks exist"
        )

    op.execute(
        """
        DROP INDEX document_chunks_version_lookup;
        DROP INDEX document_version_chunks_document;
        DROP TABLE document_version_chunks;
        ALTER TABLE document_chunks
            DROP CONSTRAINT document_chunks_version_fk,
            DROP CONSTRAINT document_chunks_document_fk,
            DROP CONSTRAINT document_chunks_document_version_pair,
            DROP COLUMN version_id,
            DROP COLUMN document_id;
        ALTER TABLE ingestion_documents
            DROP CONSTRAINT ingestion_documents_active_version_fk;
        DROP TABLE document_versions;
        DROP TABLE ingestion_attempts;
        DROP INDEX ingestion_documents_tenant_status;
        DROP INDEX ingestion_documents_tenant_filename;
        DROP TABLE ingestion_documents;
        """
    )
