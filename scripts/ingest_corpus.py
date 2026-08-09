r"""Explicit, idempotent corpus ingestion script.

Ingests all clinical PDFs from ``dataset/textos/`` into the RAG vector store
via the ``DocumentService`` API.  Must be run explicitly — never at startup.

Usage::

    python scripts\ingest_corpus.py

The script:
- Scans ``dataset/textos/`` for PDF files.
- Reads each file and computes its SHA-256 content hash.
- Uses ``DocumentService.upload()``, which applies the content-hash duplicate
  policy: re-running the script is safe (idempotent) — identical files are
  recognised by hash and skipped without creating new registry records.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so backend imports work.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv()

from backend.documents.service import DocumentService
from backend.rag.config import RagConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("ingest_corpus")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_TEXTOS = _project_root / "dataset" / "textos"
"""Directory containing subdirectories of clinical PDFs organised by procedure."""

UPLOAD_DIR = Path(os.getenv("DOCUMENTS_UPLOAD_DIR", "uploads"))
DB_PATH = Path(os.getenv("DOCUMENTS_DB_PATH", "data/documents.db"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Ingest all PDFs from the corpus directory."""
    if not DATASET_TEXTOS.is_dir():
        logger.error("Corpus directory not found: %s", DATASET_TEXTOS)
        sys.exit(1)

    pdf_files = sorted(DATASET_TEXTOS.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found under %s", DATASET_TEXTOS)
        return

    config = RagConfig()
    service = DocumentService(upload_dir=UPLOAD_DIR, db_path=DB_PATH)

    logger.info(
        "Starting corpus ingestion: %d PDF files in %s",
        len(pdf_files),
        DATASET_TEXTOS,
    )

    seen_hashes: set[str] = set()
    new_count = 0
    duplicate_count = 0
    error_count = 0

    for pdf_path in pdf_files:
        try:
            content = pdf_path.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()

            if content_hash in seen_hashes:
                logger.info(
                    "  %s — DUPLICATE (same content as earlier file in this run)",
                    pdf_path.relative_to(DATASET_TEXTOS),
                )
                duplicate_count += 1
                continue

            seen_hashes.add(content_hash)

            doc = service.upload(content, pdf_path.name, config)
            if doc.status.value == "ready":
                logger.info(
                    "  %s — %s (ready)",
                    pdf_path.relative_to(DATASET_TEXTOS),
                    doc.document_id,
                )
                new_count += 1
            elif doc.status.value == "failed":
                logger.error(
                    "  %s — FAILED: %s",
                    pdf_path.relative_to(DATASET_TEXTOS),
                    doc.error_message or "unknown error",
                )
                error_count += 1
        except Exception as exc:
            logger.error(
                "  %s — ERROR: %s",
                pdf_path.relative_to(DATASET_TEXTOS),
                exc,
            )
            error_count += 1

    logger.info(
        "Corpus ingestion complete: %d ingested, %d duplicates (skipped), "
        "%d errors. Rerunning is safe — identical content is idempotent.",
        new_count,
        duplicate_count,
        error_count,
    )


if __name__ == "__main__":
    main()
