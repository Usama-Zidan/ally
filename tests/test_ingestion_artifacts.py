import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

from services.ingestion.artifacts import ArtifactSizeExceeded, IngestionArtifactStore


class IngestionArtifactStoreTest(unittest.TestCase):
    def test_local_artifact_is_compressed_scoped_and_round_trips(self):
        identity = {
            "tenant_id": "tenant-1",
            "document_id": "document-1",
            "version_id": "version-1",
        }
        value = [{"page_number": 1, "text": "private content" * 100}]
        with tempfile.TemporaryDirectory() as directory:
            store = IngestionArtifactStore(
                backend="local",
                bucket="",
                region="us-east-1",
                upload_dir=Path(directory),
                max_bytes=10_000,
            )
            reference = store.put_json(identity, "pages", value)

            self.assertEqual(store.get_json(reference), value)
            self.assertTrue(reference.startswith("file:"))
            local_path = Path(url2pathname(urlsplit(reference).path))
            self.assertLess(local_path.stat().st_size, 200)
            self.assertEqual(store.reference(identity, "pages"), reference)
            store.delete(identity, "pages")
            store.delete(identity, "pages")
            self.assertFalse(local_path.exists())

    def test_artifact_limits_apply_to_expanded_json_and_paths_are_scoped(self):
        identity = {
            "tenant_id": "tenant-1",
            "document_id": "document-1",
            "version_id": "version-1",
        }
        with tempfile.TemporaryDirectory() as directory:
            store = IngestionArtifactStore(
                backend="local",
                bucket="",
                region="us-east-1",
                upload_dir=Path(directory),
                max_bytes=64,
            )
            with self.assertRaises(ArtifactSizeExceeded):
                store.put_json(identity, "pages", [{"text": "x" * 100}])
            with self.assertRaisesRegex(ValueError, "identity"):
                store.reference({**identity, "tenant_id": "../other"}, "pages")
            with self.assertRaisesRegex(ValueError, "outside the artifact store"):
                store.get_json(Path(directory, "private.json.gz").as_uri())


class IngestionArtifactActivitiesTest(unittest.IsolatedAsyncioTestCase):
    async def test_janitor_deletes_only_database_selected_failed_run_stages(self):
        from workflows import activities

        identity = {
            "tenant_id": "tenant-1",
            "document_id": "document-1",
            "version_id": "version-2",
        }
        store = Mock()

        async def run_sync(function, *args):
            return function(*args)

        with (
            patch.object(activities, "_artifact_store", return_value=store),
            patch.object(activities, "_run_sync_with_heartbeat", side_effect=run_sync),
            patch.object(
                activities,
                "_expired_artifact_identities_sync",
                return_value=[identity],
            ),
        ):
            removed = await activities.janitor_ingestion_artifacts(
                "2026-10-01T00:00:00+00:00", ["ingest-live"]
            )

        self.assertEqual(removed, 2)
        store.delete.assert_has_calls([call(identity, "pages"), call(identity, "chunks")])

    async def test_extraction_and_chunking_return_references_not_payloads(self):
        from workflows import activities

        identity = {
            "tenant_id": "tenant-1",
            "document_id": "document-1",
            "version_id": "version-1",
        }
        pages = [{"page_number": 1, "text": "First sentence. Second sentence."}]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(activities, "INGESTION_STORAGE_BACKEND", "local"),
            patch.object(activities, "INGESTION_S3_BUCKET", ""),
            patch.object(activities, "INGESTION_ARTIFACT_MAX_BYTES", 10_000),
            patch.object(activities, "UPLOAD_DIR", directory),
            patch.object(
                activities, "_extract_with_unstructured", new=AsyncMock(return_value=pages)
            ),
        ):
            page_artifact = await activities.extract_with_unstructured("input.txt", identity)
            chunk_artifact = await activities.chunk_pages(
                page_artifact, max_chars=20, overlap_chars=0, artifact_identity=identity
            )
            chunks = await activities._load_artifact(chunk_artifact, identity, "chunks")

        self.assertIsInstance(page_artifact, dict)
        self.assertIsInstance(chunk_artifact, dict)
        self.assertEqual(page_artifact["item_count"], 1)
        self.assertEqual(chunk_artifact["item_count"], len(chunks))
        self.assertEqual([chunk["page_number"] for chunk in chunks], [1, 1])
