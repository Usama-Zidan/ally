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

from config import PROJECT_NAME, TEMPORAL_ADDRESS
from workflows import activities
from workflows.ingestion_workflow import IngestDocumentWorkflow


async def main() -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)

    worker = Worker(
        client,
        task_queue="ingestion-task-queue",
        workflows=[IngestDocumentWorkflow],
        activities=[
            activities.detect_document_type,
            activities.extract_with_unstructured,
            activities.extract_with_textract,
            activities.chunk_pages,
            activities.persist_chunks,
        ],
    )

    print(
        f"{PROJECT_NAME} worker started, "
        "polling task queue 'ingestion-task-queue'..."
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
