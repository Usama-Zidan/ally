"""
Temporal worker process. Run this alongside the FastAPI app:

    python workflows/worker.py

It polls the "ingestion-task-queue" and executes IngestDocumentWorkflow
plus its activities. Scale horizontally by running multiple copies of
this process — Temporal handles work distribution automatically.
"""

import asyncio
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from config import runtime_settings
from workflows import activities
from workflows.ingestion_workflow import (
    IngestionArtifactJanitorWorkflow,
    IngestDocumentWorkflow,
    ReconcileIngestionWorkflow,
    TombstoneIngestionWorkflow,
)
from services.retrieval.qdrant_store import QdrantStore
from services.retrieval.model_loading import configure_model_inference
from services.runtime.task_drain import DrainableTasks


async def main() -> None:
    """Connect to Temporal and run the ingestion worker until it is stopped."""
    settings = runtime_settings.application
    configure_model_inference(settings)
    async with asyncio.timeout(settings.temporal_connect_timeout_seconds):
        client = await Client.connect(settings.temporal_address)

    qdrant_store = QdrantStore(settings)
    worker_work = DrainableTasks()
    activities.set_worker_work(worker_work)
    activities.set_qdrant_store(qdrant_store)
    try:
        worker = Worker(
            client,
            task_queue="ingestion-task-queue",
            workflows=[
                IngestDocumentWorkflow,
                IngestionArtifactJanitorWorkflow,
                ReconcileIngestionWorkflow,
                TombstoneIngestionWorkflow,
            ],
            activities=[
                activities.register_ingestion,
                activities.update_ingestion_stage,
                activities.finish_ingestion_attempt,
                activities.activate_ingestion_version,
                activities.tombstone_ingestion_document,
                activities.reconcile_ingestion_indexes,
                activities.cleanup_ingestion_artifacts,
                activities.janitor_ingestion_artifacts,
                activities.detect_document_type,
                activities.extract_with_unstructured,
                activities.extract_with_textract,
                activities.chunk_pages,
                activities.persist_chunks,
                activities.embed_and_index,
            ],
        )

        print(
            f"{settings.project_name} worker started, polling task queue 'ingestion-task-queue'..."
        )
        await worker.run()
    finally:
        await worker_work.close()
        activities.set_worker_work(None)
        activities.set_qdrant_store(None)
        await asyncio.to_thread(qdrant_store.close)


if __name__ == "__main__":
    asyncio.run(main())
