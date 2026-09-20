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

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows import activities


@workflow.defn
class IngestDocumentWorkflow:
    @workflow.run
    async def run(self, file_path: str, filename: str, tenant_id: str) -> dict:
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

        doc_type = await workflow.execute_activity(
            activities.detect_document_type,
            args=[file_path, filename],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry_policy,
        )

        # Route: scanned/handwritten -> Textract, everything else -> Unstructured.io
        if doc_type == "scanned":
            # Multi-page scanned PDFs go through Textract's async job API
            # (upload -> poll -> paginate), which can take several minutes
            # for large documents — longer timeout + heartbeat than the
            # other activities, which are all synchronous and fast.
            extracted = await workflow.execute_activity(
                activities.extract_with_textract,
                args=[file_path],
                start_to_close_timeout=timedelta(minutes=20),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=retry_policy,
            )
        else:
            extracted = await workflow.execute_activity(
                activities.extract_with_unstructured,
                args=[file_path],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy,
            )

        chunks = await workflow.execute_activity(
            activities.chunk_pages,
            args=[extracted],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry_policy,
        )

        stored = await workflow.execute_activity(
            activities.persist_chunks,
            args=[chunks, filename, tenant_id],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry_policy,
        )

        indexed = await workflow.execute_activity(
            activities.embed_and_index,
            args=[chunks, filename, tenant_id],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry_policy,
        )

        return {
            "filename": filename,
            "tenant_id": tenant_id,
            "num_chunks": len(chunks),
            "stored": stored,
            "indexed": indexed,
        }
