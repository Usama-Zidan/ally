"""Opt-in real S3-compatible checks for durable ingestion artifacts."""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from services.ingestion.artifacts import IngestionArtifactStore


@unittest.skipUnless(
    os.environ.get("RUN_S3_COMPAT_INTEGRATION_TESTS") == "1",
    "set RUN_S3_COMPAT_INTEGRATION_TESTS=1 to use disposable S3-compatible storage",
)
class S3CompatibleArtifactTest(unittest.TestCase):
    def test_put_get_and_delete_round_trip(self) -> None:
        endpoint = os.environ.get("S3_COMPAT_ENDPOINT", "http://127.0.0.1:59000")
        access_key = os.environ.get("S3_COMPAT_ACCESS_KEY", "ally_test")
        secret_key = os.environ.get("S3_COMPAT_SECRET_KEY", "ally-test-only-password")
        bucket = f"ally-ingestion-{uuid.uuid4().hex[:16]}"
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(s3={"addressing_style": "path"}),
        )
        client.create_bucket(Bucket=bucket)
        identity = {
            "tenant_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
        }
        value = [{"page_number": 1, "text": "durable test artifact"}]

        try:
            with tempfile.TemporaryDirectory() as directory:
                store = IngestionArtifactStore(
                    backend="s3",
                    bucket=bucket,
                    region="us-east-1",
                    upload_dir=Path(directory),
                    max_bytes=10_000,
                )
                with patch.object(store, "_client", return_value=client):
                    reference = store.put_json(identity, "pages", value)
                    self.assertEqual(store.get_json(reference), value)
                    store.delete(identity, "pages")
                    with self.assertRaises(ClientError) as missing:
                        store.get_json(reference)
                    self.assertEqual(
                        missing.exception.response["ResponseMetadata"]["HTTPStatusCode"], 404
                    )
        finally:
            for item in client.list_objects_v2(Bucket=bucket).get("Contents", []):
                client.delete_object(Bucket=bucket, Key=item["Key"])
            client.delete_bucket(Bucket=bucket)


if __name__ == "__main__":
    unittest.main()
