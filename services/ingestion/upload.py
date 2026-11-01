"""Document upload orchestration behind storage and workflow ports."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit

import structlog


ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}
log = structlog.get_logger()


def _s3_client(region: str) -> Any:
    import boto3

    return boto3.client("s3", region_name=region)


def _parse_s3_ref(storage_ref: str, *, default_bucket: str | None = None) -> tuple[str, str]:
    parsed = urlsplit(storage_ref)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = unquote(parsed.path.lstrip("/"))
    else:
        bucket = default_bucket or ""
        key = storage_ref
    if not bucket or not key or ".." in key.split("/"):
        raise ValueError("Invalid S3 object reference")
    return bucket, key


class UploadSource(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


class UploadStorage(Protocol):
    def destination(
        self,
        upload_dir: Path,
        tenant_id: str,
        document_id: str,
        version_id: str,
        filename: str,
    ) -> str: ...

    async def save(
        self, destination: str, source: UploadSource, max_bytes: int
    ) -> "StoredUpload": ...

    def delete_if_exists(self, path: str) -> None: ...


class WorkflowStarter(Protocol):
    async def start_workflow(
        self,
        workflow_type: str,
        *,
        args: list[Any],
        id: str,
        task_queue: str,
    ) -> Any: ...


class UploadFailure(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class StoredUpload:
    """Identity and content digest for an upload written to shared storage."""

    storage_ref: str
    size_bytes: int
    content_sha256: str


class LocalUploadStorage:
    """Stream uploads to local disk without buffering the whole request."""

    def destination(
        self, upload_dir: Path, tenant_id: str, document_id: str, version_id: str, filename: str
    ) -> str:
        del tenant_id
        return str(upload_dir / f"{document_id}_{version_id}_{filename}")

    async def save(self, destination: str, source: UploadSource, max_bytes: int) -> StoredUpload:
        bytes_written = 0
        digest = hashlib.sha256()
        with open(destination, "wb") as output:
            while chunk := await source.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise UploadFailure(400, f"File exceeds the {max_bytes} byte upload limit")
                output.write(chunk)
                digest.update(chunk)
        return StoredUpload(destination, bytes_written, digest.hexdigest())

    def delete_if_exists(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)


@dataclass(frozen=True)
class S3UploadStorage:
    """Stream request bodies to S3 through a bounded disk spool."""

    bucket: str
    region: str

    def destination(
        self, upload_dir: Path, tenant_id: str, document_id: str, version_id: str, filename: str
    ) -> str:
        del upload_dir
        return f"tenants/{tenant_id}/documents/{document_id}/versions/{version_id}/{filename}"

    async def save(self, destination: str, source: UploadSource, max_bytes: int) -> StoredUpload:
        bytes_written = 0
        digest = hashlib.sha256()
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="ally-upload-", delete=False) as output:
                temp_path = output.name
                while chunk := await source.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise UploadFailure(400, f"File exceeds the {max_bytes} byte upload limit")
                    output.write(chunk)
                    digest.update(chunk)
            client = _s3_client(self.region)
            upload = asyncio.create_task(
                asyncio.to_thread(client.upload_file, temp_path, self.bucket, destination)
            )
            try:
                await asyncio.shield(upload)
            except asyncio.CancelledError:
                # Keep the spool alive until boto3 stops reading it, then
                # remove the object because Temporal has not been started.
                try:
                    await asyncio.shield(upload)
                    await asyncio.to_thread(
                        client.delete_object, Bucket=self.bucket, Key=destination
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("cancelled_upload_cleanup_failed", error_type=type(exc).__name__)
                raise
            return StoredUpload(
                f"s3://{self.bucket}/{quote(destination, safe='/')}",
                bytes_written,
                digest.hexdigest(),
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def delete_if_exists(self, storage_ref: str) -> None:
        bucket, key = _parse_s3_ref(storage_ref, default_bucket=self.bucket)
        _s3_client(self.region).delete_object(Bucket=bucket, Key=key)


@dataclass(frozen=True)
class UploadUseCase:
    upload_dir: Path
    max_upload_bytes: int
    storage: UploadStorage
    workflow: WorkflowStarter | None
    make_id: Callable[[], str] = field(default_factory=lambda: lambda: str(uuid.uuid4()))

    async def execute(self, source: UploadSource, tenant_id: str) -> dict[str, str]:
        if self.workflow is None:
            raise UploadFailure(503, "Temporal service unavailable; workflow cannot be started")

        safe_filename = os.path.basename(source.filename or "")
        extension = os.path.splitext(safe_filename)[1].lower()
        if not safe_filename or extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise UploadFailure(
                400,
                f"Unsupported file type {extension!r}; allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
            )

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ally-document:{tenant_id}:{safe_filename}"))
        attempt_id = self.make_id()
        version_id = self.make_id()
        destination = self.storage.destination(
            self.upload_dir, tenant_id, doc_id, version_id, safe_filename
        )
        try:
            stored = await self.storage.save(destination, source, self.max_upload_bytes)
        except asyncio.CancelledError:
            self.storage.delete_if_exists(destination)
            raise
        except UploadFailure:
            self.storage.delete_if_exists(destination)
            raise
        except Exception as exc:  # noqa: BLE001
            self.storage.delete_if_exists(destination)
            raise UploadFailure(400, "Upload failed; please try again later.") from exc

        workflow_id = f"ingest-{doc_id}-{version_id}"
        try:
            await self.workflow.start_workflow(
                "IngestDocumentWorkflow",
                args=[
                    stored.storage_ref,
                    safe_filename,
                    tenant_id,
                    doc_id,
                    stored.content_sha256,
                    attempt_id,
                    version_id,
                ],
                id=workflow_id,
                task_queue="ingestion-task-queue",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("ingestion_workflow_start_failed", error_type=type(exc).__name__)
            raise UploadFailure(
                503, "Temporal workflow could not start; please try again later."
            ) from exc

        log.info("ingestion_started", doc_id=doc_id, workflow_id=workflow_id, tenant_id=tenant_id)
        return {"doc_id": doc_id, "workflow_id": workflow_id, "status": "queued"}
