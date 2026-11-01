"""
Temporal workflow: IngestDocumentWorkflow

Orchestrates: detect doc type -> extract (Unstructured.io or Textract) ->
chunk (page-aware) -> persist chunks/metadata to Postgres -> embed & index
(Phase 2 hands this off to the retrieval service).

Using Temporal instead of raw RabbitMQ+Celery gives us durable execution
(automatic retries, no lost jobs on worker crash) and built-in visibility
into every ingestion job via the Temporal UI.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporalio.exceptions import CancelledError as TemporalCancelledError

    from workflows import activities


def _is_cancellation_failure(error: BaseException) -> bool:
    """Recognize cancellation wrapped by Temporal's activity failure chain."""
    pending: BaseException | None = error
    visited: set[int] = set()
    while pending is not None and id(pending) not in visited:
        if isinstance(pending, (asyncio.CancelledError, TemporalCancelledError)):
            return True
        visited.add(id(pending))
        pending = getattr(pending, "cause", None) or pending.__cause__
    return False


@workflow.defn
class IngestDocumentWorkflow:
    @workflow.run
    async def run(
        self,
        file_path: str,
        filename: str,
        tenant_id: str,
        document_id: str | None = None,
        content_sha256: str | None = None,
        attempt_id: str | None = None,
        version_id: str | None = None,
    ) -> dict:
        """Extract, chunk, persist, and index a document through retried activities.

        ``tenant_id`` is resolved from the caller's API key at the upload
        endpoint (see services.api.main.upload_document) and carried
        through to persist_chunks and embed_and_index so every row and
        vector this document produces is scoped to the owning tenant.

        Returns a summary containing the filename, chunk count, and persistence
        and indexing results.
        """
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=5,
        )

        if document_id and content_sha256 and attempt_id and version_id:
            await workflow.execute_activity(
                activities.register_ingestion,
                args=[
                    document_id,
                    attempt_id,
                    version_id,
                    file_path,
                    content_sha256,
                    filename,
                    tenant_id,
                    workflow.info().workflow_id,
                ],
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=retry_policy,
            )

        # Preserve already-running workflow histories before adding stage
        # activities or changing the Textract activity argument list.
        lifecycle_tracking = workflow.patched("ingestion-lifecycle-v1")
        artifact_identity = (
            {"tenant_id": tenant_id, "document_id": document_id, "version_id": version_id}
            if lifecycle_tracking and document_id and version_id
            else None
        )

        async def record_stage(stage: str) -> None:
            if lifecycle_tracking and document_id and attempt_id:
                await workflow.execute_activity(
                    activities.update_ingestion_stage,
                    args=[document_id, attempt_id, tenant_id, stage],
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=retry_policy,
                )

        async def finish(status: str, error_code: str | None = None) -> None:
            if lifecycle_tracking and document_id and attempt_id and version_id:
                await workflow.execute_activity(
                    activities.finish_ingestion_attempt,
                    args=[document_id, attempt_id, version_id, tenant_id, status, error_code],
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=retry_policy,
                )

        textract_request_token = (
            hashlib.sha256(version_id.encode("utf-8")).hexdigest() if version_id else None
        )

        try:
            await record_stage("detecting")
            doc_type = await workflow.execute_activity(
                activities.detect_document_type,
                args=[file_path, filename],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=retry_policy,
            )

            await record_stage("extracting")
            # Route: scanned/handwritten -> Textract, everything else -> Unstructured.io
            if doc_type == "scanned":
                # Multi-page scanned PDFs use Textract's async job API. OCR
                # is correlated to the version for idempotent activity retries.
                textract_args: list[Any] = [file_path]
                if version_id is not None and textract_request_token is not None:
                    textract_args.append(textract_request_token)
                    if lifecycle_tracking:
                        textract_args.append(version_id)
                        if artifact_identity is not None:
                            textract_args.append(artifact_identity)
                extracted = await workflow.execute_activity(
                    activities.extract_with_textract,
                    args=textract_args,
                    start_to_close_timeout=timedelta(minutes=20),
                    heartbeat_timeout=timedelta(seconds=30),
                    retry_policy=retry_policy,
                )
            else:
                extracted = await workflow.execute_activity(
                    activities.extract_with_unstructured,
                    args=(
                        [file_path, artifact_identity]
                        if artifact_identity is not None
                        else [file_path]
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=retry_policy,
                )

            await record_stage("chunking")
            chunks = await workflow.execute_activity(
                activities.chunk_pages,
                args=(
                    [extracted, 1500, 200, artifact_identity]
                    if artifact_identity is not None
                    else [extracted]
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry_policy,
            )

            await record_stage("persisting")
            stored = await workflow.execute_activity(
                activities.persist_chunks,
                args=(
                    [chunks, filename, tenant_id, artifact_identity]
                    if artifact_identity is not None
                    else [chunks, filename, tenant_id]
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry_policy,
            )

            await record_stage("indexing")
            indexed = await workflow.execute_activity(
                activities.embed_and_index,
                args=(
                    [chunks, filename, tenant_id, artifact_identity]
                    if artifact_identity is not None
                    else [chunks, filename, tenant_id]
                ),
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy,
            )

            if artifact_identity is not None:
                await workflow.execute_activity(
                    activities.activate_ingestion_version,
                    args=[document_id, version_id, tenant_id],
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=retry_policy,
                )

            await finish("indexed")
            if artifact_identity is not None:
                try:
                    await workflow.execute_activity(
                        activities.cleanup_ingestion_artifacts,
                        args=[artifact_identity],
                        start_to_close_timeout=timedelta(minutes=1),
                        retry_policy=retry_policy,
                    )
                except Exception as cleanup_error:  # noqa: BLE001
                    workflow.logger.warning(
                        "ingestion_artifact_cleanup_failed",
                        error_type=type(cleanup_error).__name__,
                    )

            result = {
                "filename": filename,
                "tenant_id": tenant_id,
                "num_chunks": (chunks["item_count"] if isinstance(chunks, dict) else len(chunks)),
                "stored": stored,
                "indexed": indexed,
            }
            if document_id is not None:
                result["document_id"] = document_id
            if content_sha256 is not None:
                result["content_sha256"] = content_sha256
            return result
        except (asyncio.CancelledError, TemporalCancelledError):
            try:
                await finish("cancelled", "workflow_cancelled")
            except Exception as status_error:  # noqa: BLE001
                workflow.logger.error(
                    "failed_to_record_ingestion_cancellation",
                    error_type=type(status_error).__name__,
                )
            raise
        except Exception as workflow_error:
            cancelled = _is_cancellation_failure(workflow_error)
            try:
                await finish(
                    "cancelled" if cancelled else "failed",
                    "workflow_cancelled" if cancelled else "workflow_activity_failed",
                )
            except Exception as status_error:  # noqa: BLE001
                workflow.logger.error(
                    "failed_to_record_ingestion_cancellation"
                    if cancelled
                    else "failed_to_record_ingestion_failure",
                    error_type=type(status_error).__name__,
                )
            raise


@workflow.defn
class ReconcileIngestionWorkflow:
    """Repair Qdrant visibility from PostgreSQL's active pointers/tombstones."""

    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            activities.reconcile_ingestion_indexes,
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=5),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=1),
                maximum_attempts=3,
            ),
        )


@workflow.defn
class TombstoneIngestionWorkflow:
    """Soft-delete a tenant-owned document and retire its search points."""

    @workflow.run
    async def run(self, document_id: str, tenant_id: str, reason: str = "user_deleted") -> bool:
        return await workflow.execute_activity(
            activities.tombstone_ingestion_document,
            args=[document_id, tenant_id, reason],
            start_to_close_timeout=timedelta(minutes=2),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=5,
            ),
        )


@workflow.defn
class IngestionArtifactJanitorWorkflow:
    """Remove retained artifacts for expired failed attempts."""

    @workflow.run
    async def run(self, finished_before: str, active_workflow_ids: list[str]) -> int:
        return await workflow.execute_activity(
            activities.janitor_ingestion_artifacts,
            args=[finished_before, active_workflow_ids],
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=5),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=1),
                maximum_attempts=3,
            ),
        )
