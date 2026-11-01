"""Bounded durable JSON artifacts used between ingestion activities."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote, unquote, urlsplit
from urllib.request import url2pathname


class ArtifactSizeExceeded(ValueError):
    """Raised when a workflow artifact exceeds its configured size bound."""


class _BoundedWriter:
    def __init__(self, stream: TextIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._written = 0

    def write(self, value: str) -> int:
        self._written += len(value.encode("utf-8"))
        if self._written > self._limit:
            raise ArtifactSizeExceeded("Ingestion artifact exceeds its uncompressed size limit")
        return self._stream.write(value)


class _BoundedReader:
    def __init__(self, stream: TextIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._read = 0

    def read(self, size: int = -1) -> str:
        remaining = self._limit + 1
        value = self._stream.read(remaining if size < 0 else min(size, remaining))
        self._read += len(value.encode("utf-8"))
        if self._read > self._limit:
            raise ArtifactSizeExceeded("Ingestion artifact exceeds its uncompressed size limit")
        return value


class IngestionArtifactStore:
    """Store bounded compressed JSON by tenant/document/version/stage identity."""

    def __init__(
        self,
        *,
        backend: str,
        bucket: str,
        region: str,
        upload_dir: Path,
        max_bytes: int,
    ) -> None:
        if backend not in {"local", "s3"} or backend == "s3" and not bucket:
            raise ValueError("Invalid ingestion artifact storage configuration")
        self.backend = backend
        self.bucket = bucket
        self.region = region
        self.upload_dir = upload_dir
        self.max_bytes = max_bytes

    @staticmethod
    def _key(identity: dict[str, str], stage: str) -> str:
        if stage not in {"pages", "chunks"}:
            raise ValueError("Unknown ingestion artifact stage")
        parts = (identity.get("tenant_id"), identity.get("document_id"), identity.get("version_id"))
        if any(
            not value or "/" in value or "\\" in value or value in {".", ".."} for value in parts
        ):
            raise ValueError("Invalid ingestion artifact identity")
        tenant_id, document_id, version_id = parts
        return (
            f"tenants/{tenant_id}/documents/{document_id}/versions/{version_id}/"
            f"artifacts/{stage}.json.gz"
        )

    def reference(self, identity: dict[str, str], stage: str) -> str:
        key = self._key(identity, stage)
        if self.backend == "s3":
            return f"s3://{self.bucket}/{quote(key, safe='/')}"
        return (self.upload_dir / "artifacts" / Path(*key.split("/"))).resolve().as_uri()

    def _client(self) -> Any:
        import boto3

        return boto3.client("s3", region_name=self.region)

    def put_json(self, identity: dict[str, str], stage: str, payload: Any) -> str:
        key = self._key(identity, stage)
        base = self.upload_dir / "artifacts"
        target = base.joinpath(*key.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".ally-artifact-", suffix=".tmp", dir=target.parent, delete=False
            ) as temp_file:
                temp_path = temp_file.name
            with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as output:
                json.dump(payload, _BoundedWriter(output, self.max_bytes), ensure_ascii=False)
            if Path(temp_path).stat().st_size > self.max_bytes:
                raise ArtifactSizeExceeded("Ingestion artifact exceeds its compressed size limit")

            if self.backend == "s3":
                self._client().upload_file(
                    temp_path,
                    self.bucket,
                    key,
                    ExtraArgs={"ContentType": "application/gzip"},
                )
                return self.reference(identity, stage)

            os.replace(temp_path, target)
            temp_path = None
            return self.reference(identity, stage)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def get_json(self, storage_ref: str) -> Any:
        parsed = urlsplit(storage_ref)
        temp_path: str | None = None
        if parsed.scheme == "s3":
            if parsed.netloc != self.bucket:
                raise ValueError("Ingestion artifact bucket does not match configuration")
            key = unquote(parsed.path.lstrip("/"))
            if not key.startswith("tenants/") or ".." in key.split("/"):
                raise ValueError("Invalid ingestion artifact reference")
            temp_path = self._download_s3(key)
            source_path = Path(temp_path)
        elif parsed.scheme == "file":
            source_path = Path(url2pathname(parsed.path))
            root = (self.upload_dir / "artifacts").resolve()
            try:
                source_path.resolve().relative_to(root)
            except ValueError:
                raise ValueError("Ingestion artifact path is outside the artifact store") from None
        else:
            raise ValueError("Unsupported ingestion artifact reference")

        try:
            if source_path.stat().st_size > self.max_bytes:
                raise ArtifactSizeExceeded("Ingestion artifact exceeds its compressed size limit")
            with gzip.open(source_path, "rt", encoding="utf-8") as source:
                return json.load(_BoundedReader(source, self.max_bytes))
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def delete(self, identity: dict[str, str], stage: str) -> None:
        key = self._key(identity, stage)
        if self.backend == "s3":
            self._client().delete_object(Bucket=self.bucket, Key=key)
            return
        (self.upload_dir / "artifacts" / Path(*key.split("/"))).unlink(missing_ok=True)

    def _download_s3(self, key: str) -> str:
        client = self._client()
        metadata = client.head_object(Bucket=self.bucket, Key=key)
        if metadata.get("ContentLength", 0) > self.max_bytes:
            raise ArtifactSizeExceeded("Ingestion artifact exceeds its compressed size limit")
        with tempfile.NamedTemporaryFile(
            prefix="ally-artifact-", suffix=".gz", delete=False
        ) as output:
            temp_path = output.name
        try:
            client.download_file(self.bucket, key, temp_path)
            return temp_path
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise
