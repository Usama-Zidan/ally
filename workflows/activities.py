"""
Temporal activities used by IngestDocumentWorkflow.

Each activity is intentionally small and idempotent so Temporal's retry
policy can safely re-run any single step without side effects (e.g.
persist_chunks upserts by document id rather than inserting blindly).
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from temporalio import activity

from config import AWS_REGION, POSTGRES_DSN

log = structlog.get_logger()


@activity.defn
async def detect_document_type(file_path: str, filename: str) -> str:
    """Cheap heuristic router: PDFs get a quick check for an embedded text
    layer (native PDFs go to Unstructured.io; image-only/scanned PDFs and
    raw images go to Textract). DOCX always goes to Unstructured.io."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".tiff"):
        return "scanned"

    if ext == ".pdf":
        has_text_layer = _pdf_has_text_layer(file_path)
        return "native" if has_text_layer else "scanned"

    return "native"  # .docx, .txt, etc.


def _pdf_has_text_layer(file_path: str) -> bool:
    """Uses pypdf to sample the first few pages for extractable text.
    Falls back to treating the doc as scanned if extraction fails."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        sample_pages = reader.pages[:3]
        text = "".join(page.extract_text() or "" for page in sample_pages)
        return len(text.strip()) > 20
    except Exception as exc:  # noqa: BLE001
        log.warning("pdf_text_layer_check_failed", error=str(exc))
        return False


@activity.defn
async def extract_with_unstructured(file_path: str) -> list[dict[str, Any]]:
    """Extracts structured elements (title, narrative text, table, etc.)
    with page numbers preserved for later citation."""
    from unstructured.partition.auto import partition

    elements = partition(filename=file_path, include_page_breaks=True)

    pages: list[dict[str, Any]] = []
    for el in elements:
        page_number = getattr(el.metadata, "page_number", None) or 1
        pages.append(
            {
                "page_number": page_number,
                "text": str(el),
                "category": el.category,
            }
        )
    return pages


@activity.defn
async def extract_with_textract(file_path: str) -> list[dict[str, Any]]:
    """Sends the document to AWS Textract for OCR + table/form extraction.
    Used for scanned PDFs and raw images where Unstructured.io's local
    parsers can't reliably pull text."""
    import boto3

    client = boto3.client("textract", region_name=AWS_REGION)

    with open(file_path, "rb") as f:
        document_bytes = f.read()

    response = client.analyze_document(
        Document={"Bytes": document_bytes},
        FeatureTypes=["TABLES", "FORMS"],
    )

    pages: list[dict[str, Any]] = []
    current_page = 1
    buffer: list[str] = []

    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            buffer.append(block.get("Text", ""))
        if block["BlockType"] == "PAGE":
            if buffer:
                pages.append({"page_number": current_page, "text": "\n".join(buffer), "category": "Text"})
                buffer = []
            current_page += 1

    if buffer:
        pages.append({"page_number": current_page, "text": "\n".join(buffer), "category": "Text"})

    return pages


@activity.defn
async def chunk_pages(pages: list[dict[str, Any]], max_chars: int = 1500) -> list[dict[str, Any]]:
    """Page-aware chunking: never merges text across page boundaries so
    every chunk keeps a single, correct page_number for citations."""
    chunks: list[dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        for start in range(0, len(text), max_chars):
            chunk_text = text[start:start + max_chars].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page_number": page_number})

    return chunks


@activity.defn
async def persist_chunks(chunks: list[dict[str, Any]], filename: str) -> bool:
    """Upserts chunk rows into Postgres, keyed by (filename, page_number,
    chunk_index) so retries don't create duplicates."""
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_number INT NOT NULL,
                    chunk_index INT NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE (filename, page_number, chunk_index)
                );
                """
            )
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO document_chunks (filename, page_number, chunk_index, text)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (filename, page_number, chunk_index) DO UPDATE
                        SET text = EXCLUDED.text;
                    """,
                    (filename, chunk["page_number"], idx, chunk["text"]),
                )
        conn.commit()
        return True
    finally:
        conn.close()
