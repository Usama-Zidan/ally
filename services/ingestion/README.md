# Ingestion service

`upload.py` validates and stages API uploads, then starts the Temporal workflow
through injected storage and workflow ports. `artifacts.py` provides the local
and S3-compatible storage adapters used for source files and bounded intermediate
workflow artifacts. OCR, extraction, indexing and maintenance activities remain
in `workflows/activities.py`.
