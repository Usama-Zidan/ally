"""Startup validation, confidential values and process snapshot contracts."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.requests import Request
from pydantic import ValidationError

import config
from services.api import main
from services.runtime.settings import ConfigurationError, load_runtime_settings


class RuntimeSettingsTest(unittest.TestCase):
    def test_defaults_and_legacy_exports_preserve_models_and_types(self):
        snapshot = load_runtime_settings({})
        self.assertEqual(snapshot.application.qwen_model_name, "Qwen/Qwen3-7B")
        self.assertEqual(snapshot.application.embedding_model, "BAAI/bge-base-en-v1.5")
        self.assertEqual(snapshot.application.reranker_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(snapshot.application.llm_fallback_model, "gpt-4o-mini")
        self.assertEqual(snapshot.application.chat_retrieval_top_k, 5)
        for name in type(snapshot.application).model_fields:
            value = getattr(config.runtime_settings.application, name)
            if hasattr(value, "get_secret_value"):
                value = value.get_secret_value()
            if isinstance(value, Path):
                value = str(value)
            self.assertEqual(getattr(config, name.upper()), value, name)

    def test_one_snapshot_is_immutable_and_independent_of_environment_changes(self):
        values = {"PROJECT_NAME": "One", "MONGO_DB_NAME": "one", "WS_MAX_CONNECTIONS": "9"}
        first = load_runtime_settings(values)
        values.update({"PROJECT_NAME": "Two", "MONGO_DB_NAME": "two"})
        second = load_runtime_settings(values)
        self.assertEqual(first.application.project_name, "One")
        self.assertEqual(first.history.database, "one")
        self.assertEqual(first.security.max_connections, 9)
        self.assertEqual(second.application.project_name, "Two")
        for owner, field, value in (
            (first.application, "project_name", "mutated"),
            (first.history, "database", "mutated"),
            (first.security, "max_connections", 10),
        ):
            with self.assertRaises(ValidationError):
                setattr(owner, field, value)

    def test_invalid_numbers_are_rejected_with_no_values_in_error_or_traceback(self):
        cases = {
            "MAX_UPLOAD_BYTES": ["0", "-1", str(1024**3 + 1), "not-a-number"],
            "BM25_REBUILD_MIN_INTERVAL_SECONDS": ["NaN", "inf", "-1"],
            "LLM_MAX_RETRIES": ["-1", "11", "1.5"],
            "LLM_REQUEST_TIMEOUT_SECONDS": ["0", "nan", "inf", "601"],
            "CIRCUIT_BREAKER_FAIL_MAX": ["0", "1001"],
            "CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS": ["0", "3601"],
            "CHAT_RETRIEVAL_TOP_K": ["0", "21"],
            "MODEL_INFERENCE_CONCURRENCY": ["0", "65"],
            "MODEL_INFERENCE_MAX_WAITERS": ["-1", "1025"],
            "MODEL_INFERENCE_WAIT_TIMEOUT_SECONDS": ["0", "601", "inf"],
            "QDRANT_OPERATION_TIMEOUT_SECONDS": ["0", "301", "1.5"],
            "PC_AGENT_COMMAND_TIMEOUT_SECONDS": ["NaN", "0", "301"],
            "TEMPORAL_CONNECT_TIMEOUT_SECONDS": ["0", "121", "inf"],
            "TEMPORAL_PROBE_TIMEOUT_SECONDS": ["0", "61"],
            "WS_SEND_TIMEOUT_SECONDS": ["nan", "inf"],
        }
        for field, values in cases.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ConfigurationError) as raised,
                ):
                    load_runtime_settings({field: value, "OPENAI_API_KEY": "secret-sentinel"})
                self.assertNotIn("secret-sentinel", str(raised.exception))
                self.assertTrue(raised.exception.__suppress_context__)

    def test_secret_representations_and_json_are_redacted(self):
        secret = "private-config-sentinel"
        snapshot = load_runtime_settings(
            {
                "POSTGRES_DSN": f"postgresql://user:{secret}@localhost/db",
                "REDIS_URL": f"redis://:{secret}@localhost:6379",
                "MONGO_URI": f"mongodb://user:{secret}@localhost",
                "OPENAI_API_KEY": secret,
                "QDRANT_API_KEY": secret,
                "LANGFUSE_SECRET_KEY": secret,
                "JIRA_API_TOKEN": secret,
                "GOOGLE_CALENDAR_CREDENTIALS_JSON": secret,
                "AWS_SECRET_ACCESS_KEY": secret,
            }
        )
        self.assertNotIn(secret, repr(snapshot))
        self.assertNotIn(secret, snapshot.model_dump_json())
        self.assertNotIn(secret, repr(config._legacy_settings(snapshot)))
        self.assertEqual(snapshot.application.openai_api_key.get_secret_value(), secret)
        self.assertEqual(config._legacy_settings(snapshot).qdrant_api_key, secret)

    def test_invalid_urls_targets_paths_and_optional_vault_keys_are_sanitized(self):
        for field, value in (
            ("QDRANT_URL", "https://user:secret-sentinel@host"),
            ("QWEN_VLLM_BASE_URL", "http:///missing-host"),
            ("LANGFUSE_HOST", "https://host?token=secret-sentinel"),
            ("JIRA_BASE_URL", "file:///private"),
            ("POSTGRES_DSN", "mysql://user:secret-sentinel@host/db"),
            ("POSTGRES_DSN", "postgresql://host:99999/db"),
            ("REDIS_URL", "https://redis"),
            ("TEMPORAL_ADDRESS", "http://secret-sentinel@host:7233"),
            ("TEMPORAL_ADDRESS", "host"),
            ("TEMPORAL_ADDRESS", "host:99999"),
            ("UPLOAD_DIR", ""),
            ("INGESTION_STORAGE_BACKEND", "ftp"),
            ("BM25_INDEX_PATH", "bad\x00path"),
            ("CONNECTED_ACCOUNTS_ENCRYPTION_KEY", "secret-sentinel"),
            ("INTENT_CLASSIFIER", "bad name"),
        ):
            with self.subTest(field=field), self.assertRaises(ConfigurationError) as raised:
                load_runtime_settings({field: value})
            self.assertNotIn("secret-sentinel", str(raised.exception))
        valid = load_runtime_settings(
            {"TEMPORAL_ADDRESS": "[::1]:7233", "BM25_REBUILD_MIN_INTERVAL_SECONDS": "0"}
        )
        self.assertEqual(valid.application.temporal_address, "[::1]:7233")
        self.assertEqual(valid.application.bm25_rebuild_min_interval_seconds, 0)
        legacy = load_runtime_settings({"TEMPORAL_ADDRESS": "https://temporal.test:7233"})
        self.assertEqual(legacy.application.temporal_address, "https://temporal.test:7233")
        s3_storage = load_runtime_settings(
            {"INGESTION_STORAGE_BACKEND": "s3", "INGESTION_S3_BUCKET": "ally-ingestion"}
        )
        self.assertEqual(s3_storage.application.ingestion_storage_backend, "s3")
        with self.assertRaises(ConfigurationError):
            load_runtime_settings({"INGESTION_STORAGE_BACKEND": "s3"})

    def test_security_and_history_validation_are_part_of_the_same_startup_gate(self):
        for values in (
            {"CHAT_HISTORY_MAX_MESSAGES": "5"},
            {"MONGO_MAX_POOL_SIZE": "0"},
            {"ALLY_ENV": "production", "WS_ALLOW_QUERY_API_KEYS": "true"},
            {"WS_ALLOWED_ORIGINS": "https://user:secret-sentinel@host"},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                load_runtime_settings(values)
        with self.assertRaises(ConfigurationError):
            load_runtime_settings({"ALLY_ENV": "production"})
        production = load_runtime_settings(
            {
                "ALLY_ENV": "production",
                "INGESTION_STORAGE_BACKEND": "s3",
                "INGESTION_S3_BUCKET": "ally-ingestion",
            }
        )
        self.assertTrue(production.security.require_tls)
        self.assertFalse(production.security.allow_query_api_keys)

    def test_clean_import_and_loading_have_no_sdk_or_filesystem_side_effects(self):
        # Fresh process avoids an already-imported API hiding an import dependency.
        script = """
import json, os, pathlib, socket, sys
from unittest.mock import patch
def forbidden(*args, **kwargs):
    raise AssertionError("configuration attempted I/O")
with patch.object(socket, "create_connection", forbidden), patch.object(pathlib.Path, "mkdir", forbidden), patch.object(os, "makedirs", forbidden), patch.object(os, "open", forbidden):
    import config
    from services.runtime.settings import load_runtime_settings
    settings = load_runtime_settings({})
assert not any(name in sys.modules for name in ("pymongo", "temporalio", "litellm", "qdrant_client", "sentence_transformers"))
print(json.dumps({"model": settings.application.qwen_model_name}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"model": "Qwen/Qwen3-7B"})

    def test_blank_bm25_path_keeps_legacy_working_directory_semantics(self):
        snapshot = load_runtime_settings({"BM25_INDEX_PATH": ""})
        self.assertEqual(snapshot.application.bm25_index_path, Path("."))

    def test_temp_paths_use_the_supplied_environment_without_writability_probes(self):
        with patch.object(os, "open", side_effect=AssertionError("must not probe")):
            snapshot = load_runtime_settings({"TMPDIR": "configured-temp", "TEMP": "ignored-temp"})
        self.assertEqual(snapshot.application.upload_dir, Path("configured-temp") / "ally_uploads")
        self.assertEqual(
            snapshot.application.bm25_index_path, Path("configured-temp") / "bm25_index.pkl"
        )

    def test_lifespan_uses_the_provided_security_and_history_snapshot(self):
        app = FastAPI()

        async def run():
            with patch.object(main, "_start_optional_services", new=AsyncMock()):
                async with main.lifespan(app):
                    self.assertEqual(app.state.history_store.settings.database, "isolated")
                    self.assertEqual(app.state.websocket_security.settings.max_connections, 3)

        with TemporaryDirectory() as directory:
            app.state.runtime_settings = load_runtime_settings(
                {"MONGO_DB_NAME": "isolated", "WS_MAX_CONNECTIONS": "3", "UPLOAD_DIR": directory}
            )
            asyncio.run(run())


class TemporalBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_probe_timeout_degrades_and_cancels_its_iterator(self):
        app = FastAPI()
        app.state.runtime_settings = load_runtime_settings(
            {"TEMPORAL_PROBE_TIMEOUT_SECONDS": "0.02"}
        )
        cancelled = asyncio.Event()

        class Probe:
            async def list_workflows(self):
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    cancelled.set()

        app.state.temporal_client = Probe()
        response = await asyncio.wait_for(
            main.health_dependencies(Request({"type": "http", "app": app})), timeout=2
        )
        self.assertEqual(json.loads(response.body), {"temporal": "unavailable"})
        self.assertTrue(cancelled.is_set())

    async def test_api_connect_timeout_degrades_without_leaking_target(self):
        app = FastAPI()
        app.state.runtime_settings = load_runtime_settings(
            {"TEMPORAL_CONNECT_TIMEOUT_SECONDS": "0.02"}
        )
        cancelled = asyncio.Event()

        async def connect(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            patch.object(main.TemporalClient, "connect", side_effect=connect),
            patch.object(main, "warm_reranker"),
        ):
            await asyncio.wait_for(main._start_optional_services(app), timeout=2)
        self.assertTrue(cancelled.is_set())
        self.assertIsNone(app.state.temporal_client)
        self.assertFalse(app.state.temporal_connected)

    async def test_worker_connect_timeout_stops_before_constructing_worker(self):
        from workflows import worker

        async def connect(*args):
            await asyncio.Event().wait()

        snapshot = load_runtime_settings({"TEMPORAL_CONNECT_TIMEOUT_SECONDS": "0.02"})
        with (
            patch.object(worker, "runtime_settings", snapshot),
            patch.object(worker.Client, "connect", side_effect=connect),
            patch.object(worker, "Worker") as build,
        ):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(worker.main(), timeout=2)
        build.assert_not_called()
