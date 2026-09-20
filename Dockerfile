FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by unstructured[all-docs] at runtime:
#   - poppler-utils: pdfinfo/pdftoppm, used for PDF parsing and rasterization
#   - tesseract-ocr: local OCR fallback used by unstructured's PDF/image partitioners
#   - libmagic1: file-type sniffing (python-magic, a transitive dependency)
# Without these, the first PDF or image upload fails at runtime rather
# than at build time, since pip installing unstructured[all-docs] does
# NOT pull in these system binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Unstructured uses spaCy for sentence detection and otherwise tries to
# install this model lazily at runtime. Install it while the image build is
# still running as root so the non-root API and worker users can load it.
RUN python -m spacy download en_core_web_sm

# Pre-download the embedding and reranker models into the image so a
# fresh container doesn't pull ~1GB of weights from the Hub on its first
# request (and so it still starts in an offline/air-gapped environment).
# Uses the same env-configurable model name as services/retrieval/embeddings.py.
ARG EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
ARG RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')"

# Run as a non-root user. python:3.11-slim doesn't ship one by default.
RUN useradd --create-home --uid 1000 appuser
RUN mkdir -p /home/appuser/.cache/huggingface /tmp/ally_uploads && chown -R appuser:appuser /home/appuser/.cache /tmp/ally_uploads

# Set environment variables for HuggingFace cache directories to avoid permission issues when running as a non-root user.
ENV HOME=/home/appuser
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers
ENV UPLOAD_DIR=/tmp/ally_uploads

COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
