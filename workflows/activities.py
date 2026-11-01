"""
Temporal activities for ingestion, indexing and maintenance workflows.

Current workflows persist attempt/version state and stage data idempotently;
legacy workflow histories retain their earlier activity sequence through
Temporal patching.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import structlog
from temporalio import activity

from config import (
    AWS_REGION,
    INGESTION_ARTIFACT_MAX_BYTES,
    INGESTION_S3_BUCKET,
    INGESTION_STORAGE_BACKEND,
    POSTGRES_DSN,
    TEXTRACT_S3_BUCKET,
    UPLOAD_DIR,
)
from services.ingestion.artifacts import IngestionArtifactStore
from services.retrieval.qdrant_store import QdrantStore
from services.runtime.task_drain import DrainableTasks

log = structlog.get_logger()
_qdrant_store: QdrantStore | None = None
_worker_work: DrainableTasks | None = None


def _cleanup_temporary_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("ingestion_temp_cleanup_failed", error_type=type(exc).__name__)


@asynccontextmanager
async def _materialized_source(storage_ref: str) -> AsyncIterator[str]:
    """Yield a worker-local file for a legacy path or durable S3 object ref."""
    parsed = urlsplit(storage_ref)
    if parsed.scheme != "s3":
        yield url2pathname(parsed.path) if parsed.scheme == "file" else storage_ref
        return

    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    if not bucket or not key or ".." in key.split("/"):
        raise ValueError("Invalid ingestion object reference")
    import boto3

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ally-ingestion-", delete=False) as temp_file:
            temp_path = temp_file.name
        client = boto3.client("s3", region_name=AWS_REGION)
        cleanup_path = temp_file.name
        try:
            await _run_sync_with_heartbeat(
                client.download_file,
                bucket,
                key,
                temp_path,
                on_cancel=partial(_cleanup_temporary_file, cleanup_path),
            )
        except asyncio.CancelledError:
            # The tracked download may still be writing; transfer temp-file
            # cleanup to its completion callback instead of unlinking early.
            temp_path = None
            raise
        yield temp_path
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def set_qdrant_store(store: QdrantStore | None) -> None:
    """Set the worker-owned Qdrant connection used by indexing activities."""
    global _qdrant_store
    _qdrant_store = store


def set_worker_work(tracker: DrainableTasks | None) -> None:
    """Set the worker-owned tracker that drains blocking indexing work."""
    global _worker_work
    _worker_work = tracker


async def _run_sync_with_heartbeat(
    function: Any,
    *args: Any,
    heartbeat_interval: float = 10,
    on_cancel: Callable[[], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Offload blocking I/O, heartbeat while it runs and retain it through cancellation."""
    operation = asyncio.to_thread(function, *args, **kwargs)
    tracker = _worker_work
    task = tracker.start(operation) if tracker is not None else asyncio.create_task(operation)
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat_interval)
            except TimeoutError:
                activity.heartbeat()
        return await task
    except asyncio.CancelledError:
        if on_cancel is not None:
            task.add_done_callback(lambda _: on_cancel())
        raise


def _artifact_store() -> IngestionArtifactStore:
    return IngestionArtifactStore(
        backend=INGESTION_STORAGE_BACKEND,
        bucket=INGESTION_S3_BUCKET,
        region=AWS_REGION,
        upload_dir=Path(UPLOAD_DIR),
        max_bytes=INGESTION_ARTIFACT_MAX_BYTES,
    )


async def _store_artifact(
    identity: dict[str, str], stage: str, payload: list[dict[str, Any]]
) -> dict[str, Any]:
    reference = await _run_sync_with_heartbeat(_artifact_store().put_json, identity, stage, payload)
    return {"storage_ref": reference, "stage": stage, "item_count": len(payload)}


async def _load_artifact(
    descriptor: dict[str, Any], identity: dict[str, str], expected_stage: str
) -> list[dict[str, Any]]:
    if descriptor.get("stage") != expected_stage:
        raise ValueError("Unexpected ingestion artifact stage")
    store = _artifact_store()
    reference = descriptor.get("storage_ref")
    if not isinstance(reference, str) or reference != store.reference(identity, expected_stage):
        raise ValueError("Ingestion artifact identity mismatch")
    payload = await _run_sync_with_heartbeat(store.get_json, reference)
    if not isinstance(payload, list) or len(payload) != descriptor.get("item_count"):
        raise ValueError("Ingestion artifact shape mismatch")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("Invalid ingestion artifact record")
    return payload


@activity.defn
async def cleanup_ingestion_artifacts(identity: dict[str, str]) -> None:
    """Remove intermediate artifacts after successful indexing."""
    store = _artifact_store()
    for stage in ("pages", "chunks"):
        await _run_sync_with_heartbeat(store.delete, identity, stage)


@activity.defn
async def janitor_ingestion_artifacts(finished_before: str, active_workflow_ids: list[str]) -> int:
    """Delete expired failed/cancelled artifacts, excluding live Temporal runs."""
    identities = await _run_sync_with_heartbeat(
        _expired_artifact_identities_sync, finished_before, active_workflow_ids
    )
    store = _artifact_store()
    deleted = 0
    for identity in identities:
        for stage in ("pages", "chunks"):
            await _run_sync_with_heartbeat(store.delete, identity, stage)
            deleted += 1
    return deleted


def _expired_artifact_identities_sync(
    finished_before: str, active_workflow_ids: list[str]
) -> list[dict[str, str]]:
    """Select only terminal, aged attempts whose workflow is not currently running."""
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT documents.tenant_id, documents.id, versions.id
                FROM ingestion_attempts AS attempts
                JOIN ingestion_documents AS documents
                  ON documents.id = attempts.document_id
                 AND documents.tenant_id = attempts.tenant_id
                JOIN document_versions AS versions
                  ON versions.attempt_id = attempts.id
                 AND versions.document_id = attempts.document_id
                 AND versions.tenant_id = attempts.tenant_id
                WHERE attempts.status IN ('failed', 'cancelled')
                  AND attempts.finished_at < %s::timestamptz
                  AND attempts.workflow_id <> ALL(%s)
                  AND versions.status = 'failed'
                  AND documents.active_version_id IS DISTINCT FROM versions.id
                """,
                (finished_before, active_workflow_ids),
            )
            return [
                {"tenant_id": str(row[0]), "document_id": str(row[1]), "version_id": str(row[2])}
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


@activity.defn
async def register_ingestion(
    document_id: str,
    attempt_id: str,
    version_id: str,
    storage_ref: str,
    content_sha256: str,
    filename: str,
    tenant_id: str,
    workflow_id: str,
) -> bool:
    """Idempotently persist the identities before extraction starts."""
    return await _run_sync_with_heartbeat(
        _register_ingestion_sync,
        document_id,
        attempt_id,
        version_id,
        storage_ref,
        content_sha256,
        filename,
        tenant_id,
        workflow_id,
    )


def _register_ingestion_sync(
    document_id: str,
    attempt_id: str,
    version_id: str,
    storage_ref: str,
    content_sha256: str,
    filename: str,
    tenant_id: str,
    workflow_id: str,
) -> bool:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_documents
                    (id, tenant_id, filename, storage_ref, content_sha256, status)
                VALUES (%s, %s, %s, %s, %s, 'queued')
                ON CONFLICT (id) DO UPDATE SET
                    storage_ref = EXCLUDED.storage_ref,
                    content_sha256 = EXCLUDED.content_sha256,
                    status = CASE
                        WHEN ingestion_documents.status IN ('indexed', 'failed') THEN 'queued'
                        ELSE ingestion_documents.status
                    END,
                    updated_at = NOW()
                WHERE ingestion_documents.tenant_id = EXCLUDED.tenant_id
                  AND ingestion_documents.filename = EXCLUDED.filename
                  AND ingestion_documents.status <> 'deleted'
                RETURNING id
                """,
                (document_id, tenant_id, filename, storage_ref, content_sha256),
            )
            if cur.fetchone() is None:
                raise ValueError("Ingestion document identity conflict")

            cur.execute(
                """
                SELECT id, attempt_number, document_id, tenant_id
                FROM ingestion_attempts WHERE workflow_id = %s
                """,
                (workflow_id,),
            )
            existing_attempt = cur.fetchone()
            if existing_attempt is not None:
                if (
                    str(existing_attempt[0]) != attempt_id
                    or str(existing_attempt[2]) != document_id
                    or str(existing_attempt[3]) != tenant_id
                ):
                    raise ValueError("Ingestion attempt identity conflict")
                attempt_number = existing_attempt[1]
            else:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) + 1
                    FROM ingestion_attempts WHERE document_id = %s AND tenant_id = %s
                    """,
                    (document_id, tenant_id),
                )
                next_attempt_number = cur.fetchone()
                if next_attempt_number is None:
                    raise RuntimeError("Failed to allocate ingestion attempt number")
                attempt_number = next_attempt_number[0]
                cur.execute(
                    """
                    INSERT INTO ingestion_attempts
                        (id, document_id, tenant_id, workflow_id, attempt_number, status)
                    VALUES (%s, %s, %s, %s, %s, 'queued')
                    """,
                    (attempt_id, document_id, tenant_id, workflow_id, attempt_number),
                )

            cur.execute(
                """
                SELECT document_id, tenant_id, attempt_id, content_sha256, storage_ref
                FROM document_versions WHERE id = %s
                """,
                (version_id,),
            )
            existing_version = cur.fetchone()
            if existing_version is not None:
                if tuple(str(value) for value in existing_version) != (
                    document_id,
                    tenant_id,
                    attempt_id,
                    content_sha256,
                    storage_ref,
                ):
                    raise ValueError("Ingestion version identity conflict")
            else:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(version_number), 0) + 1
                    FROM document_versions WHERE document_id = %s AND tenant_id = %s
                    """,
                    (document_id, tenant_id),
                )
                next_version_number = cur.fetchone()
                if next_version_number is None:
                    raise RuntimeError("Failed to allocate ingestion version number")
                version_number = next_version_number[0]
                cur.execute(
                    """
                    INSERT INTO document_versions
                        (id, document_id, tenant_id, attempt_id, version_number,
                         content_sha256, storage_ref, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'staged')
                    """,
                    (
                        version_id,
                        document_id,
                        tenant_id,
                        attempt_id,
                        version_number,
                        content_sha256,
                        storage_ref,
                    ),
                )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@activity.defn
async def update_ingestion_stage(
    document_id: str, attempt_id: str, tenant_id: str, stage: str
) -> bool:
    return await _run_sync_with_heartbeat(
        _update_ingestion_stage_sync, document_id, attempt_id, tenant_id, stage
    )


def _update_ingestion_stage_sync(
    document_id: str, attempt_id: str, tenant_id: str, stage: str
) -> bool:
    allowed_stages = {"detecting", "extracting", "chunking", "persisting", "indexing"}
    if stage not in allowed_stages:
        raise ValueError("Unknown ingestion stage")
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_attempts
                SET status = 'processing', current_stage = %s,
                    started_at = COALESCE(started_at, NOW()), updated_at = NOW()
                WHERE id = %s AND document_id = %s AND tenant_id = %s
                  AND status IN ('queued', 'processing')
                """,
                (stage, attempt_id, document_id, tenant_id),
            )
            if cur.rowcount != 1:
                raise ValueError("Ingestion attempt cannot enter requested stage")
            cur.execute(
                """
                UPDATE ingestion_documents
                SET status = 'processing', updated_at = NOW()
                WHERE id = %s AND tenant_id = %s AND status IN ('queued', 'processing')
                """,
                (document_id, tenant_id),
            )
            if cur.rowcount != 1:
                raise ValueError("Ingestion document cannot enter processing state")
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@activity.defn
async def finish_ingestion_attempt(
    document_id: str,
    attempt_id: str,
    version_id: str,
    tenant_id: str,
    status: str,
    error_code: str | None = None,
) -> bool:
    return await _run_sync_with_heartbeat(
        _finish_ingestion_attempt_sync,
        document_id,
        attempt_id,
        version_id,
        tenant_id,
        status,
        error_code,
    )


@activity.defn
async def activate_ingestion_version(document_id: str, version_id: str, tenant_id: str) -> bool:
    return await _run_sync_with_heartbeat(
        _activate_ingestion_version_sync, document_id, version_id, tenant_id
    )


@activity.defn
async def tombstone_ingestion_document(
    document_id: str, tenant_id: str, reason: str = "user_deleted"
) -> bool:
    safe_reason = reason if re.fullmatch(r"[a-z0-9_]{1,64}", reason) else "user_deleted"
    return await _run_sync_with_heartbeat(
        _tombstone_ingestion_document_sync, document_id, tenant_id, safe_reason
    )


def _tombstone_ingestion_document_sync(document_id: str, tenant_id: str, reason: str) -> bool:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, status
                FROM ingestion_documents
                WHERE id = %s AND tenant_id = %s
                FOR UPDATE
                """,
                (document_id, tenant_id),
            )
            document = cur.fetchone()
            if document is None:
                raise ValueError("Ingestion document does not exist")
            filename, status = document
            if status != "deleted":
                cur.execute(
                    """
                    UPDATE document_versions SET status = 'superseded'
                    WHERE document_id = %s AND tenant_id = %s AND status = 'active'
                    """,
                    (document_id, tenant_id),
                )
                cur.execute(
                    """
                    UPDATE ingestion_documents
                    SET status = 'deleted', active_version_id = NULL, updated_at = NOW()
                    WHERE id = %s AND tenant_id = %s AND status <> 'deleted'
                    """,
                    (document_id, tenant_id),
                )
            cur.execute(
                """
                INSERT INTO ingestion_tombstones (document_id, tenant_id, reason)
                VALUES (%s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id, reason = EXCLUDED.reason
                WHERE ingestion_tombstones.tenant_id = EXCLUDED.tenant_id
                """,
                (document_id, tenant_id, reason),
            )
            if cur.rowcount != 1:
                raise ValueError("Ingestion tombstone identity conflict")
            # Legacy rows are filename-scoped and cannot be reliably mapped to
            # a document ID; this schema already treats tenant+filename as the
            # legacy document identity.
            cur.execute(
                "DELETE FROM document_chunks WHERE tenant_id = %s AND filename = %s "
                "AND version_id IS NULL",
                (tenant_id, filename),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    from services.retrieval.qdrant_store import reconcile_document_version

    if _qdrant_store is None:
        reconcile_document_version(tenant_id, document_id, None, filename, deleted=True)
    else:
        reconcile_document_version(
            tenant_id, document_id, None, filename, deleted=True, store=_qdrant_store
        )
    from services.retrieval.bm25_index import rebuild_index_from_postgres

    rebuild_index_from_postgres(force=True)
    return True


@activity.defn
async def reconcile_ingestion_indexes() -> int:
    return await _run_sync_with_heartbeat(_reconcile_ingestion_indexes_sync)


def _reconcile_ingestion_indexes_sync() -> int:
    import psycopg2
    from services.retrieval.qdrant_store import reconcile_document_version

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, tenant_id, filename, active_version_id, status
                FROM ingestion_documents
                ORDER BY tenant_id, id
                """
            )
            documents = cur.fetchall()
    finally:
        conn.close()

    reconciled = 0
    failures = 0
    for document_id, tenant_id, filename, active_version_id, status in documents:
        try:
            if _qdrant_store is None:
                reconcile_document_version(
                    str(tenant_id),
                    str(document_id),
                    str(active_version_id) if active_version_id else None,
                    filename,
                    deleted=status == "deleted",
                    delete_legacy=active_version_id is not None,
                )
            else:
                reconcile_document_version(
                    str(tenant_id),
                    str(document_id),
                    str(active_version_id) if active_version_id else None,
                    filename,
                    deleted=status == "deleted",
                    delete_legacy=active_version_id is not None,
                    store=_qdrant_store,
                )
            reconciled += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log.warning(
                "ingestion_index_reconciliation_failed",
                error_type=type(exc).__name__,
            )
    from services.retrieval.bm25_index import rebuild_index_from_postgres

    rebuild_index_from_postgres(force=True)
    if failures:
        raise RuntimeError(f"Ingestion index reconciliation failed for {failures} documents")
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_attempts AS attempts
                SET status = 'indexed', current_stage = 'indexed', error_code = NULL,
                    finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                FROM document_versions AS versions
                JOIN ingestion_documents AS documents
                  ON documents.id = versions.document_id
                 AND documents.tenant_id = versions.tenant_id
                 AND documents.active_version_id = versions.id
                WHERE attempts.id = versions.attempt_id
                  AND versions.status = 'active'
                  AND attempts.status IN ('processing', 'failed')
                  AND (attempts.error_code IS NULL
                       OR attempts.error_code = 'workflow_activity_failed')
                """
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return reconciled


def _activate_ingestion_version_sync(document_id: str, version_id: str, tenant_id: str) -> bool:
    """Make a fully indexed staged version authoritative, then reconcile Qdrant flags."""
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, active_version_id, status
                FROM ingestion_documents
                WHERE id = %s AND tenant_id = %s
                FOR UPDATE
                """,
                (document_id, tenant_id),
            )
            document = cur.fetchone()
            if document is None or document[2] == "deleted":
                raise ValueError("Ingestion document cannot activate a version")
            filename, previous_version_id, _document_status = document
            cur.execute(
                """
                SELECT status, chunk_count, attempt_id
                FROM document_versions
                WHERE id = %s AND document_id = %s AND tenant_id = %s
                FOR UPDATE
                """,
                (version_id, document_id, tenant_id),
            )
            version = cur.fetchone()
            if version is None or version[0] not in {"staged", "active"}:
                raise ValueError("Ingestion version is not ready for activation")
            if previous_version_id != version_id:
                if previous_version_id is not None:
                    cur.execute(
                        """
                        UPDATE document_versions SET status = 'superseded'
                        WHERE id = %s AND document_id = %s AND tenant_id = %s
                          AND status = 'active'
                        """,
                        (previous_version_id, document_id, tenant_id),
                    )
                cur.execute(
                    """
                    UPDATE document_versions
                    SET status = 'active', activated_at = COALESCE(activated_at, NOW())
                    WHERE id = %s AND document_id = %s AND tenant_id = %s
                      AND status IN ('staged', 'active')
                    """,
                    (version_id, document_id, tenant_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("Ingestion version could not be activated")
                cur.execute(
                    """
                    UPDATE ingestion_documents
                    SET active_version_id = %s, status = 'indexed', updated_at = NOW()
                    WHERE id = %s AND tenant_id = %s AND status <> 'deleted'
                    """,
                    (version_id, document_id, tenant_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("Ingestion document could not activate the version")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    from services.retrieval.qdrant_store import activate_document_version

    try:
        if _qdrant_store is None:
            activate_document_version(tenant_id, document_id, version_id, filename)
        else:
            activate_document_version(
                tenant_id, document_id, version_id, filename, store=_qdrant_store
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("ingestion_vector_activation_pending", error_type=type(exc).__name__)
        raise
    try:
        from services.retrieval.bm25_index import rebuild_index_from_postgres

        rebuild_index_from_postgres()
    except Exception as exc:  # noqa: BLE001
        log.warning("ingestion_bm25_activation_pending", error_type=type(exc).__name__)
        raise
    return True


def _finish_ingestion_attempt_sync(
    document_id: str,
    attempt_id: str,
    version_id: str,
    tenant_id: str,
    status: str,
    error_code: str | None,
) -> bool:
    if status not in {"indexed", "failed", "cancelled"}:
        raise ValueError("Invalid terminal ingestion status")
    safe_error_code = (
        error_code if error_code and re.fullmatch(r"[a-z0-9_]{1,64}", error_code) else None
    )
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_attempts
                SET status = %s, error_code = %s,
                    current_stage = %s, finished_at = NOW(), updated_at = NOW()
                WHERE id = %s AND document_id = %s AND tenant_id = %s
                  AND status IN ('queued', 'processing')
                """,
                (status, safe_error_code, status, attempt_id, document_id, tenant_id),
            )
            if cur.rowcount != 1:
                cur.execute(
                    """
                    SELECT 1 FROM ingestion_attempts
                    WHERE id = %s AND document_id = %s AND tenant_id = %s
                      AND status = %s AND error_code IS NOT DISTINCT FROM %s
                    """,
                    (attempt_id, document_id, tenant_id, status, safe_error_code),
                )
                if cur.fetchone() is None:
                    raise ValueError("Ingestion attempt cannot enter terminal state")
                conn.commit()
                return True
            if status in {"failed", "cancelled"}:
                cur.execute(
                    """
                    UPDATE document_versions SET status = 'failed'
                    WHERE id = %s AND document_id = %s AND tenant_id = %s
                      AND status = 'staged'
                    """,
                    (version_id, document_id, tenant_id),
                )
                if cur.rowcount != 1:
                    cur.execute(
                        """
                        SELECT 1 FROM document_versions AS versions
                        JOIN ingestion_documents AS documents
                          ON documents.id = versions.document_id
                         AND documents.tenant_id = versions.tenant_id
                         AND documents.active_version_id = versions.id
                        WHERE versions.id = %s AND versions.document_id = %s
                          AND versions.tenant_id = %s AND versions.status = 'active'
                        """,
                        (version_id, document_id, tenant_id),
                    )
                    if cur.fetchone() is None:
                        raise ValueError("Ingestion version cannot be failed")
                else:
                    document_status = "failed" if status == "failed" else "queued"
                    cur.execute(
                        """
                        UPDATE ingestion_documents SET status = %s, updated_at = NOW()
                        WHERE id = %s AND tenant_id = %s AND status = 'processing'
                        """,
                        (document_status, document_id, tenant_id),
                    )
                    if cur.rowcount != 1:
                        raise ValueError("Ingestion document cannot enter terminal state")
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@activity.defn
async def detect_document_type(file_path: str, filename: str) -> str:
    """Cheap heuristic router: PDFs get a quick check for an embedded text
    layer (native PDFs go to Unstructured.io; image-only/scanned PDFs and
    raw images go to Textract). DOCX always goes to Unstructured.io."""
    async with _materialized_source(file_path) as local_path:
        if os.path.splitext(filename)[1].lower() == ".pdf":
            has_text_layer = await _run_sync_with_heartbeat(_pdf_has_text_layer, local_path)
            return "native" if has_text_layer else "scanned"
        return _detect_document_type(local_path, filename)


def _detect_document_type(file_path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".tiff"):
        return "scanned"

    if ext == ".pdf":
        has_text_layer = _pdf_has_text_layer(file_path)
        return "native" if has_text_layer else "scanned"

    return "native"  # .docx, .txt, etc.


def _pdf_has_text_layer(file_path: str) -> bool:
    """Uses pypdf to sample the first few pages for extractable text.
    Falls back to treating the doc as scanned if extraction fails."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        sample_pages = reader.pages[:3]
        text = "".join(page.extract_text() or "" for page in sample_pages)
        return len(text.strip()) > 20
    except Exception as exc:  # noqa: BLE001
        log.warning("pdf_text_layer_check_failed", error=str(exc))
        return False


@activity.defn
async def extract_with_unstructured(
    file_path: str, artifact_identity: dict[str, str] | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    """Extracts elements and groups them into one text blob per page.

    Unstructured returns one element per title/paragraph/table, not per
    page. Returning those elements individually (the previous behavior)
    fed chunk_pages a stream where a lone section heading became its own
    tiny chunk — polluting both the dense index and BM25 with near-empty,
    low-signal entries. Joining elements per page here gives chunk_pages
    real paragraph-length text to work with, same shape as
    extract_with_textract's output.
    """
    async with _materialized_source(file_path) as local_path:
        pages = await _extract_with_unstructured(local_path)
    if artifact_identity is not None:
        return await _store_artifact(artifact_identity, "pages", pages)
    return pages


async def _extract_with_unstructured(file_path: str) -> list[dict[str, Any]]:
    try:
        from unstructured.partition.auto import partition

        elements = await _run_sync_with_heartbeat(
            partition, filename=file_path, include_page_breaks=True
        )
    except ImportError as exc:
        if os.path.splitext(file_path)[1].lower() != ".pdf":
            raise

        log.warning(
            "unstructured_pdf_import_failed_using_pypdf",
            file_path=file_path,
            error=str(exc),
        )
        from pypdf import PdfReader

        def extract_pdf() -> list[dict[str, Any]]:
            reader = PdfReader(file_path)
            return [
                {"page_number": page_number, "text": text, "category": "Text"}
                for page_number, page in enumerate(reader.pages, start=1)
                if (text := (page.extract_text() or "").strip())
            ]

        return await _run_sync_with_heartbeat(extract_pdf)

    pages: dict[int, list[str]] = {}
    last_known_page = 1
    for el in elements:
        # Elements without page metadata (common for footers/page-number
        # artifacts, and generally more common in .docx than PDF since
        # Word documents don't have fixed pagination the way PDFs do)
        # previously fell back to a hardcoded page 1 regardless of where
        # in the document they actually occurred — silently mislabeling,
        # e.g., a page-3 footer as page 1 and breaking its citation.
        # Carrying forward the last element's real page number is a much
        # safer default: the element is almost certainly still on (or
        # very near) that page, not back at the start of the document.
        page_number = getattr(el.metadata, "page_number", None)
        if page_number is None:
            page_number = last_known_page
        else:
            last_known_page = page_number

        text = str(el).strip()
        if text:
            pages.setdefault(page_number, []).append(text)

    return [
        {"page_number": page_number, "text": "\n\n".join(texts), "category": "Text"}
        for page_number, texts in sorted(pages.items())
    ]


@activity.defn
async def extract_with_textract(
    file_path: str,
    client_request_token: str | None = None,
    version_id: str | None = None,
    artifact_identity: dict[str, str] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Extracts text via AWS Textract, correctly attributing every line to
    its source page.

    Single-page images (png/jpg/jpeg) use the synchronous analyze_document
    API. Everything else (scanned PDFs, which are usually multi-page) uses
    the asynchronous start_document_analysis API against an S3 object,
    since AnalyzeDocument's inline-Bytes path only supports single-page
    input and silently drops the rest of a multi-page PDF. Page numbers
    come from Textract's own "Page" attribute on each block rather than
    being inferred from block order, which previously produced an
    off-by-one (page 1's lines attributed to page 2, etc.).
    """
    async with _materialized_source(file_path) as local_path:
        pages = await _extract_with_textract(local_path, client_request_token, version_id)
    if artifact_identity is not None:
        return await _store_artifact(artifact_identity, "pages", pages)
    return pages


async def _extract_with_textract(
    file_path: str,
    client_request_token: str | None = None,
    version_id: str | None = None,
) -> list[dict[str, Any]]:
    import boto3

    ext = os.path.splitext(file_path)[1].lower()
    client = boto3.client("textract", region_name=AWS_REGION)

    if ext in (".png", ".jpg", ".jpeg"):
        blocks = await _run_sync_with_heartbeat(_run_sync_textract, client, file_path)
    else:
        blocks = await _run_async_textract(client, file_path, client_request_token, version_id)

    return _blocks_to_pages(blocks)


def _run_sync_textract(client: Any, file_path: str) -> list[dict[str, Any]]:
    """Synchronous path for single-page images only. AnalyzeDocument's
    Bytes input processes exactly one page — never call this for
    multi-page PDFs or TIFFs, it will silently return only page 1."""
    with open(file_path, "rb") as f:
        document_bytes = f.read()

    response = client.analyze_document(
        Document={"Bytes": document_bytes},
        FeatureTypes=["TABLES", "FORMS"],
    )
    return response.get("Blocks", [])


async def _run_async_textract(
    client: Any,
    file_path: str,
    client_request_token: str | None = None,
    version_id: str | None = None,
) -> list[dict[str, Any]]:
    """Async path for (potentially multi-page) PDFs. Uploads to S3, starts
    an analysis job, polls for completion, then pages through every result
    page via NextToken — a single get_document_analysis call truncates
    results for documents with many blocks."""
    import uuid

    import boto3

    if not TEXTRACT_S3_BUCKET:
        raise RuntimeError(
            "TEXTRACT_S3_BUCKET must be set to OCR multi-page documents "
            "(Textract's async API requires an S3 source, not inline bytes)"
        )

    s3 = boto3.client("s3", region_name=AWS_REGION)
    job_key = client_request_token or str(uuid.uuid4())
    s3_key = f"textract-input/{job_key}{os.path.splitext(file_path)[1]}"
    await _run_sync_with_heartbeat(s3.upload_file, file_path, TEXTRACT_S3_BUCKET, s3_key)

    job_id = None
    if version_id and client_request_token:
        job_id = await _run_sync_with_heartbeat(_get_textract_job_sync, version_id)
    if job_id is None:
        start_kwargs: dict[str, Any] = {
            "DocumentLocation": {"S3Object": {"Bucket": TEXTRACT_S3_BUCKET, "Name": s3_key}},
            "FeatureTypes": ["TABLES", "FORMS"],
        }
        if client_request_token is not None:
            start_kwargs["ClientRequestToken"] = client_request_token
        start_response = await _run_sync_with_heartbeat(
            client.start_document_analysis, **start_kwargs
        )
        job_id = start_response["JobId"]
        if version_id and client_request_token:
            await _run_sync_with_heartbeat(
                _save_textract_job_sync,
                version_id,
                client_request_token,
                job_id,
                "started",
            )

    # Poll for completion. Temporal's start_to_close_timeout on the calling
    # workflow bounds the total wait; heartbeat keeps the activity alive
    # across a long OCR job without tripping Temporal's own liveness check.
    while True:
        status_response = await _run_sync_with_heartbeat(client.get_document_analysis, JobId=job_id)
        status = status_response["JobStatus"]
        if status == "SUCCEEDED":
            if version_id:
                await _run_sync_with_heartbeat(
                    _set_textract_job_status_sync, version_id, job_id, "succeeded"
                )
            break
        if status == "FAILED":
            if version_id:
                await _run_sync_with_heartbeat(
                    _set_textract_job_status_sync, version_id, job_id, "failed"
                )
            raise RuntimeError(
                f"Textract job {job_id} failed: {status_response.get('StatusMessage')}"
            )
        if version_id:
            await _run_sync_with_heartbeat(
                _set_textract_job_status_sync, version_id, job_id, "in_progress"
            )
        activity.heartbeat()
        await asyncio.sleep(5)

    blocks: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token
        page_response = await _run_sync_with_heartbeat(client.get_document_analysis, **kwargs)
        blocks.extend(page_response.get("Blocks", []))
        next_token = page_response.get("NextToken")
        if not next_token:
            break

    return blocks


def _get_textract_job_sync(version_id: str) -> str | None:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT external_job_id
                FROM ingestion_external_jobs
                WHERE version_id = %s AND provider = 'textract'
                """,
                (version_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def _save_textract_job_sync(version_id: str, request_token: str, job_id: str, status: str) -> bool:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_external_jobs
                    (version_id, provider, request_token, external_job_id, status)
                VALUES (%s, 'textract', %s, %s, %s)
                ON CONFLICT (version_id) DO UPDATE SET updated_at = NOW()
                WHERE ingestion_external_jobs.provider = EXCLUDED.provider
                  AND ingestion_external_jobs.request_token = EXCLUDED.request_token
                  AND ingestion_external_jobs.external_job_id = EXCLUDED.external_job_id
                RETURNING version_id
                """,
                (version_id, request_token, job_id, status),
            )
            if cur.fetchone() is None:
                raise ValueError("Textract job identity conflict")
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _set_textract_job_status_sync(version_id: str, job_id: str, status: str) -> bool:
    if status not in {"in_progress", "succeeded", "failed"}:
        raise ValueError("Invalid Textract job status")
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_external_jobs
                SET status = %s,
                    completed_at = CASE WHEN %s IN ('succeeded', 'failed') THEN NOW()
                                        ELSE completed_at END,
                    updated_at = NOW()
                WHERE version_id = %s AND provider = 'textract'
                  AND external_job_id = %s AND status <> %s
                """,
                (status, status, version_id, job_id, status),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    SELECT 1 FROM ingestion_external_jobs
                    WHERE version_id = %s AND provider = 'textract'
                      AND external_job_id = %s AND status = %s
                    """,
                    (version_id, job_id, status),
                )
                if cur.fetchone() is None:
                    raise ValueError("Textract job status identity conflict")
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _blocks_to_pages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups LINE blocks by Textract's own "Page" field rather than by
    the order PAGE marker blocks appear in the stream — PAGE blocks are
    emitted *before* the lines that belong to them, which is what caused
    every page's lines to be attributed to the next page number."""
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.get("BlockType") == "LINE":
            page_number = block.get("Page", 1)
            pages.setdefault(page_number, []).append(block.get("Text", ""))

    return [
        {"page_number": page_number, "text": "\n".join(lines), "category": "Text"}
        for page_number, lines in sorted(pages.items())
    ]


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Splits page text into sentences, first on paragraph breaks (as
    joined by extract_with_unstructured/_blocks_to_pages) then on sentence
    punctuation. Falls back to the whole text as a single unit if no
    boundaries are found, so short/unpunctuated OCR text still chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    sentences: list[str] = []
    for paragraph in paragraphs:
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())

    return sentences


def _pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily packs whole sentences into chunks up to max_chars. Each new
    chunk starts with the trailing ~overlap_chars of the previous one, so
    a fact stated right at a chunk boundary is still retrievable from at
    least one chunk — the previous character-slicing approach chunked with
    zero overlap and no regard for word or sentence boundaries, which put
    a hard ceiling on retrieval quality independent of anything in the
    retrieval pipeline itself.

    A single sentence longer than max_chars (rare, but possible with OCR
    output that lacks punctuation) is hard-wrapped on word boundaries
    rather than word-splitting, since cutting inside a word is never
    useful for retrieval.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def build_overlap(budget: int) -> list[str]:
        """Returns the trailing sentences of the just-flushed chunk that fit
        within ``budget`` characters, to prefix the next chunk.

        The budget is the smaller of overlap_chars and whatever space is
        left in the next chunk after the sentence that triggered the flush.
        Without that second constraint, a large overlap_chars could carry
        over more text than the next chunk has room for, producing a chunk
        longer than max_chars.
        """
        overlap: list[str] = []
        overlap_len = 0
        for sentence in reversed(current):
            if overlap_len + len(sentence) + 1 > budget:
                break
            overlap.insert(0, sentence)
            overlap_len += len(sentence) + 1
        return overlap

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the joining space

        if sentence_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words = sentence.split(" ")
            piece: list[str] = []
            piece_len = 0
            for word in words:
                if piece and piece_len + len(word) + 1 > max_chars:
                    chunks.append(" ".join(piece))
                    piece, piece_len = [], 0
                piece.append(word)
                piece_len += len(word) + 1
            if piece:
                chunks.append(" ".join(piece))
            continue

        if current and current_len + sentence_len > max_chars:
            chunks.append(" ".join(current))
            current = build_overlap(min(overlap_chars, max_chars - sentence_len))
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


@activity.defn
async def chunk_pages(
    pages: list[dict[str, Any]] | dict[str, Any],
    max_chars: int = 1500,
    overlap_chars: int = 200,
    artifact_identity: dict[str, str] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Sentence-aware, page-scoped chunking with overlap.

    Never merges text across page boundaries, so every chunk keeps a
    single correct page_number for citations. Within a page, chunks are
    built from whole sentences and consecutive chunks share a trailing
    ~overlap_chars of context, rather than slicing raw character ranges
    with no overlap.
    """
    if artifact_identity is not None:
        if not isinstance(pages, dict):
            raise ValueError("Versioned ingestion requires a page artifact reference")
        pages = await _load_artifact(pages, artifact_identity, "pages")
    elif isinstance(pages, dict):
        raise ValueError("Page artifact identity is required")
    chunks = await _run_sync_with_heartbeat(_chunk_pages_sync, pages, max_chars, overlap_chars)
    if artifact_identity is not None:
        return await _store_artifact(artifact_identity, "chunks", chunks)
    return chunks


def _chunk_pages_sync(
    pages: list[dict[str, Any]], max_chars: int = 1500, overlap_chars: int = 200
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for page in pages:
        sentences = _split_into_sentences(page["text"])
        for chunk_text in _pack_sentences(sentences, max_chars, overlap_chars):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page_number": page["page_number"]})

    return chunks


@activity.defn
async def persist_chunks(
    chunks: list[dict[str, Any]] | dict[str, Any],
    filename: str,
    tenant_id: str,
    artifact_identity: dict[str, str] | None = None,
) -> bool:
    """Replace all PostgreSQL chunk rows for ``filename`` within
    ``tenant_id`` in one transaction.

    Scoped by tenant_id AND filename so two tenants can each have a
    "policy.pdf" without one's re-ingestion touching the other's rows —
    see infra/postgres/002_auth_and_tenancy.sql for the corresponding
    (tenant_id, filename, page_number, chunk_index) unique constraint.

    Each chunk's input position is stored as its ``chunk_index``. Returns
    ``True`` after the replacement is committed.
    """
    if artifact_identity is not None:
        if not isinstance(chunks, dict):
            raise ValueError("Versioned ingestion requires a chunk artifact reference")
        chunks = await _load_artifact(chunks, artifact_identity, "chunks")
    elif isinstance(chunks, dict):
        raise ValueError("Chunk artifact identity is required")
    return await _run_sync_with_heartbeat(
        _persist_chunks_sync, chunks, filename, tenant_id, artifact_identity
    )


def _persist_chunks_sync(
    chunks: list[dict[str, Any]],
    filename: str,
    tenant_id: str,
    artifact_identity: dict[str, str] | None = None,
) -> bool:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            if artifact_identity is not None:
                document_id = artifact_identity["document_id"]
                version_id = artifact_identity["version_id"]
                cur.execute(
                    """
                    DELETE FROM document_version_chunks
                    WHERE tenant_id = %s AND document_id = %s AND version_id = %s
                    """,
                    (tenant_id, document_id, version_id),
                )
                for idx, chunk in enumerate(chunks):
                    cur.execute(
                        """
                        INSERT INTO document_version_chunks
                            (tenant_id, document_id, version_id, filename,
                             page_number, chunk_index, text)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, version_id, page_number, chunk_index)
                        DO UPDATE SET text = EXCLUDED.text, filename = EXCLUDED.filename
                        """,
                        (
                            tenant_id,
                            document_id,
                            version_id,
                            filename,
                            chunk["page_number"],
                            idx,
                            chunk["text"],
                        ),
                    )
                cur.execute(
                    """
                    UPDATE document_versions SET chunk_count = %s
                    WHERE id = %s AND document_id = %s AND tenant_id = %s AND status = 'staged'
                    """,
                    (len(chunks), version_id, document_id, tenant_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("Ingestion version cannot accept staged chunks")
                conn.commit()
                return True

            cur.execute(
                "DELETE FROM document_chunks WHERE tenant_id = %s AND filename = %s;",
                (tenant_id, filename),
            )
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO document_chunks (tenant_id, filename, page_number, chunk_index, text)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, filename, page_number, chunk_index) DO UPDATE
                        SET text = EXCLUDED.text;
                    """,
                    (tenant_id, filename, chunk["page_number"], idx, chunk["text"]),
                )
        conn.commit()
        return True
    finally:
        conn.close()


@activity.defn
async def embed_and_index(
    chunks: list[dict[str, Any]] | dict[str, Any],
    filename: str,
    tenant_id: str,
    artifact_identity: dict[str, str] | None = None,
) -> bool:
    """Upsert ``chunks`` into Qdrant, then rebuild BM25 from PostgreSQL.

    Deletes this tenant's existing Qdrant points for ``filename`` first.
    Without that, re-ingesting a document that now produces fewer chunks
    than before leaves the old, higher-index points behind — they stay
    retrievable even though persist_chunks has already deleted the
    corresponding rows from Postgres, so the two stores silently disagree.
    """
    from services.retrieval.bm25_index import rebuild_index_from_postgres
    from services.retrieval.qdrant_store import (
        Chunk,
        delete_by_filename,
        index_chunks,
        make_chunk_id,
    )

    if artifact_identity is not None:
        if not isinstance(chunks, dict):
            raise ValueError("Versioned ingestion requires a chunk artifact reference")
        chunks = await _load_artifact(chunks, artifact_identity, "chunks")
    elif isinstance(chunks, dict):
        raise ValueError("Chunk artifact identity is required")

    qdrant_store = _qdrant_store

    def index() -> bool:
        if artifact_identity is None:
            if qdrant_store is None:
                delete_by_filename(tenant_id, filename)
            else:
                delete_by_filename(tenant_id, filename, store=qdrant_store)

        retrieval_chunks = [
            Chunk(
                id=(
                    make_chunk_id(
                        tenant_id,
                        filename,
                        chunk["page_number"],
                        index,
                        artifact_identity["version_id"],
                    )
                    if artifact_identity
                    else make_chunk_id(tenant_id, filename, chunk["page_number"], index)
                ),
                text=chunk["text"],
                filename=filename,
                page_number=chunk["page_number"],
                chunk_index=index,
                tenant_id=tenant_id,
                document_id=artifact_identity["document_id"] if artifact_identity else None,
                version_id=artifact_identity["version_id"] if artifact_identity else None,
            )
            for index, chunk in enumerate(chunks)
        ]
        if qdrant_store is None:
            index_chunks(retrieval_chunks)
        else:
            index_chunks(retrieval_chunks, store=qdrant_store)
        rebuild_index_from_postgres()
        return True

    return await _run_sync_with_heartbeat(index)
