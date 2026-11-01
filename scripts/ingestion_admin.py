"""Run tenant-safe ingestion maintenance workflows.

Examples::

    python scripts/ingestion_admin.py reconcile
    python scripts/ingestion_admin.py janitor --retention-days 7
    python scripts/ingestion_admin.py tombstone --tenant-id TENANT_UUID \
        --document-id DOCUMENT_UUID --confirm-document-id DOCUMENT_UUID

The ingestion worker must be running. Tombstoning is a soft-delete: document
and version rows/source objects remain for recovery, while retrieval visibility
is removed and recorded in ``ingestion_tombstones``. The janitor deletes only
derived page/chunk artifacts from aged terminal attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def _run(args: argparse.Namespace) -> int | bool:
    from temporalio.client import Client

    from config import runtime_settings
    from workflows.ingestion_workflow import (
        IngestionArtifactJanitorWorkflow,
        ReconcileIngestionWorkflow,
        TombstoneIngestionWorkflow,
    )

    settings = runtime_settings.application
    client = await Client.connect(settings.temporal_address)
    if args.command == "reconcile":
        handle = await client.start_workflow(
            ReconcileIngestionWorkflow.run,
            id=f"reconcile-ingestion-{uuid.uuid4()}",
            task_queue="ingestion-task-queue",
        )
    elif args.command == "janitor":
        active_workflow_ids = []
        async for execution in client.list_workflows(query="ExecutionStatus='Running'"):
            workflow_id = execution.id
            if workflow_id.startswith("ingest-"):
                active_workflow_ids.append(workflow_id)
        finished_before = (
            datetime.now(timezone.utc) - timedelta(days=args.retention_days)
        ).isoformat()
        handle = await client.start_workflow(
            IngestionArtifactJanitorWorkflow.run,
            args=[finished_before, active_workflow_ids],
            id=f"ingestion-artifact-janitor-{uuid.uuid4()}",
            task_queue="ingestion-task-queue",
        )
    else:
        handle = await client.start_workflow(
            TombstoneIngestionWorkflow.run,
            args=[args.document_id, args.tenant_id, args.reason],
            id=f"tombstone-ingestion-{args.document_id}-{uuid.uuid4()}",
            task_queue="ingestion-task-queue",
        )
    return await handle.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reconcile", help="repair Qdrant/BM25 visibility from PostgreSQL")
    janitor = commands.add_parser(
        "janitor", help="delete artifacts for failed runs past the retention window"
    )
    janitor.add_argument(
        "--retention-days", type=int, default=7, help="diagnostic retention window (default: 7)"
    )
    tombstone = commands.add_parser("tombstone", help="soft-delete a tenant-owned document")
    tombstone.add_argument("--tenant-id", required=True)
    tombstone.add_argument("--document-id", required=True)
    tombstone.add_argument("--reason", default="user_deleted")
    tombstone.add_argument(
        "--confirm-document-id",
        required=True,
        help="must exactly match --document-id to authorize the soft-delete",
    )
    args = parser.parse_args()
    if args.command == "janitor" and not 1 <= args.retention_days <= 3650:
        parser.error("--retention-days must be between 1 and 3650")
    if args.command == "tombstone" and args.confirm_document_id != args.document_id:
        parser.error("--confirm-document-id must exactly match --document-id")
    result = asyncio.run(_run(args))
    if args.command == "reconcile":
        print(f"Reconciled {result} ingestion documents.")
    elif args.command == "janitor":
        print(f"Removed {result} expired ingestion artifact objects.")
    else:
        print(f"Tombstone workflow completed: {result}.")


if __name__ == "__main__":
    main()
