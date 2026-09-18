"""
Temporal activities used by IngestDocumentWorkflow.

Each activity is intentionally small and idempotent so Temporal's retry
policy can safely re-run any single step without side effects (e.g.
persist_chunks upserts by document id rather than inserting blindly).
"""
from __future__ import annotations

import os
import re
from typing import Any

import structlog
from temporalio import activity

from config import AWS_REGION, POSTGRES_DSN, TEXTRACT_S3_BUCKET

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
    """Extracts elements and groups them into one text blob per page.

    Unstructured returns one element per title/paragraph/table, not per
    page. Returning those elements individually (the previous behavior)
    fed chunk_pages a stream where a lone section heading became its own
    tiny chunk — polluting both the dense index and BM25 with near-empty,
    low-signal entries. Joining elements per page here gives chunk_pages
    real paragraph-length text to work with, same shape as
    extract_with_textract's output.
    """
    from unstructured.partition.auto import partition

    elements = partition(filename=file_path, include_page_breaks=True)

    pages: dict[int, list[str]] = {}
    last_known_page = 1
    for el in elements:
        # Elements without page metadata (common for footers/page-number
        # artifacts, and generally more common in .docx than PDF since
        # Word documents don't have fixed pagination the way PDFs do)
        # previously fell back to a hardcoded page 1 regardless of where
        # in the document they actually occurred — silently mislabeling,
        # e.g., a page-3 footer as page 1 and breaking its citation.
        # Carrying forward the last element's real page number is a much
        # safer default: the element is almost certainly still on (or
        # very near) that page, not back at the start of the document.
        page_number = getattr(el.metadata, "page_number", None)
        if page_number is None:
            page_number = last_known_page
        else:
            last_known_page = page_number

        text = str(el).strip()
        if text:
            pages.setdefault(page_number, []).append(text)

    return [
        {"page_number": page_number, "text": "\n\n".join(texts), "category": "Text"}
        for page_number, texts in sorted(pages.items())
    ]


@activity.defn
async def extract_with_textract(file_path: str) -> list[dict[str, Any]]:
    """Extracts text via AWS Textract, correctly attributing every line to
    its source page.

    Single-page images (png/jpg/jpeg) use the synchronous analyze_document
    API. Everything else (scanned PDFs, which are usually multi-page) uses
    the asynchronous start_document_analysis API against an S3 object,
    since AnalyzeDocument's inline-Bytes path only supports single-page
    input and silently drops the rest of a multi-page PDF. Page numbers
    come from Textract's own "Page" attribute on each block rather than
    being inferred from block order, which previously produced an
    off-by-one (page 1's lines attributed to page 2, etc.).
    """
    import boto3

    ext = os.path.splitext(file_path)[1].lower()
    client = boto3.client("textract", region_name=AWS_REGION)

    if ext in (".png", ".jpg", ".jpeg"):
        blocks = _run_sync_textract(client, file_path)
    else:
        blocks = await _run_async_textract(client, file_path)

    return _blocks_to_pages(blocks)


def _run_sync_textract(client: Any, file_path: str) -> list[dict[str, Any]]:
    """Synchronous path for single-page images only. AnalyzeDocument's
    Bytes input processes exactly one page — never call this for
    multi-page PDFs or TIFFs, it will silently return only page 1."""
    with open(file_path, "rb") as f:
        document_bytes = f.read()

    response = client.analyze_document(
        Document={"Bytes": document_bytes},
        FeatureTypes=["TABLES", "FORMS"],
    )
    return response.get("Blocks", [])


async def _run_async_textract(client: Any, file_path: str) -> list[dict[str, Any]]:
    """Async path for (potentially multi-page) PDFs. Uploads to S3, starts
    an analysis job, polls for completion, then pages through every result
    page via NextToken — a single get_document_analysis call truncates
    results for documents with many blocks."""
    import asyncio
    import uuid

    import boto3

    if not TEXTRACT_S3_BUCKET:
        raise RuntimeError(
            "TEXTRACT_S3_BUCKET must be set to OCR multi-page documents "
            "(Textract's async API requires an S3 source, not inline bytes)"
        )

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_key = f"textract-input/{uuid.uuid4()}{os.path.splitext(file_path)[1]}"
    s3.upload_file(file_path, TEXTRACT_S3_BUCKET, s3_key)

    start_response = client.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": TEXTRACT_S3_BUCKET, "Name": s3_key}},
        FeatureTypes=["TABLES", "FORMS"],
    )
    job_id = start_response["JobId"]

    # Poll for completion. Temporal's start_to_close_timeout on the calling
    # workflow bounds the total wait; heartbeat keeps the activity alive
    # across a long OCR job without tripping Temporal's own liveness check.
    while True:
        status_response = client.get_document_analysis(JobId=job_id)
        status = status_response["JobStatus"]
        if status == "SUCCEEDED":
            break
        if status == "FAILED":
            raise RuntimeError(
                f"Textract job {job_id} failed: {status_response.get('StatusMessage')}"
            )
        activity.heartbeat()
        await asyncio.sleep(5)

    blocks: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token
        page_response = client.get_document_analysis(**kwargs)
        blocks.extend(page_response.get("Blocks", []))
        next_token = page_response.get("NextToken")
        if not next_token:
            break

    return blocks


def _blocks_to_pages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups LINE blocks by Textract's own "Page" field rather than by
    the order PAGE marker blocks appear in the stream — PAGE blocks are
    emitted *before* the lines that belong to them, which is what caused
    every page's lines to be attributed to the next page number."""
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.get("BlockType") == "LINE":
            page_number = block.get("Page", 1)
            pages.setdefault(page_number, []).append(block.get("Text", ""))

    return [
        {"page_number": page_number, "text": "\n".join(lines), "category": "Text"}
        for page_number, lines in sorted(pages.items())
    ]


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Splits page text into sentences, first on paragraph breaks (as
    joined by extract_with_unstructured/_blocks_to_pages) then on sentence
    punctuation. Falls back to the whole text as a single unit if no
    boundaries are found, so short/unpunctuated OCR text still chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    sentences: list[str] = []
    for paragraph in paragraphs:
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())

    return sentences


def _pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily packs whole sentences into chunks up to max_chars. Each new
    chunk starts with the trailing ~overlap_chars of the previous one, so
    a fact stated right at a chunk boundary is still retrievable from at
    least one chunk — the previous character-slicing approach chunked with
    zero overlap and no regard for word or sentence boundaries, which put
    a hard ceiling on retrieval quality independent of anything in the
    retrieval pipeline itself.

    A single sentence longer than max_chars (rare, but possible with OCR
    output that lacks punctuation) is hard-wrapped on word boundaries
    rather than word-splitting, since cutting inside a word is never
    useful for retrieval.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> list[str]:
        """Returns the sentences to carry into the next chunk as overlap."""
        chunks.append(" ".join(current))
        overlap: list[str] = []
        overlap_len = 0
        for sentence in reversed(current):
            if overlap_len + len(sentence) > overlap_chars:
                break
            overlap.insert(0, sentence)
            overlap_len += len(sentence) + 1
        return overlap

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the joining space

        if sentence_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words = sentence.split(" ")
            piece: list[str] = []
            piece_len = 0
            for word in words:
                if piece and piece_len + len(word) + 1 > max_chars:
                    chunks.append(" ".join(piece))
                    piece, piece_len = [], 0
                piece.append(word)
                piece_len += len(word) + 1
            if piece:
                chunks.append(" ".join(piece))
            continue

        if current and current_len + sentence_len > max_chars:
            current = flush_current()
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


@activity.defn
async def chunk_pages(
    pages: list[dict[str, Any]], max_chars: int = 1500, overlap_chars: int = 200
) -> list[dict[str, Any]]:
    """Sentence-aware, page-scoped chunking with overlap.

    Never merges text across page boundaries, so every chunk keeps a
    single correct page_number for citations. Within a page, chunks are
    built from whole sentences and consecutive chunks share a trailing
    ~overlap_chars of context, rather than slicing raw character ranges
    with no overlap.
    """
    chunks: list[dict[str, Any]] = []

    for page in pages:
        sentences = _split_into_sentences(page["text"])
        for chunk_text in _pack_sentences(sentences, max_chars, overlap_chars):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page_number": page["page_number"]})

    return chunks


@activity.defn
async def persist_chunks(chunks: list[dict[str, Any]], filename: str) -> bool:
    """Replace all PostgreSQL chunk rows for ``filename`` in one transaction.

    Each chunk's input position is stored as its ``chunk_index``. Returns
    ``True`` after the replacement is committed.
    """
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE filename = %s;", (filename,))
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


@activity.defn
async def embed_and_index(chunks: list[dict[str, Any]], filename: str) -> bool:
    """Upsert ``chunks`` into Qdrant, then rebuild BM25 from PostgreSQL.

    Deletes this filename's existing Qdrant points first. Without that,
    re-ingesting a document that now produces fewer chunks than before
    leaves the old, higher-index points behind — they stay retrievable
    even though persist_chunks has already deleted the corresponding rows
    from Postgres, so the two stores silently disagree.
    """
    from services.retrieval.bm25_index import rebuild_index_from_postgres
    from services.retrieval.qdrant_store import (
        Chunk,
        delete_by_filename,
        index_chunks,
        make_chunk_id,
    )

    delete_by_filename(filename)

    retrieval_chunks = [
        Chunk(
            id=make_chunk_id(filename, chunk["page_number"], index),
            text=chunk["text"],
            filename=filename,
            page_number=chunk["page_number"],
            chunk_index=index,
        )
        for index, chunk in enumerate(chunks)
    ]
    index_chunks(retrieval_chunks)
    rebuild_index_from_postgres()
    return True
