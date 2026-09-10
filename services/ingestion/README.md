# Ingestion service

Business logic lives in workflows/activities.py (Temporal activities).
This folder is reserved for standalone helper modules (e.g. file-type
sniffing, S3 upload utilities) as the ingestion pipeline grows beyond
what fits cleanly in a single activities.py file.
