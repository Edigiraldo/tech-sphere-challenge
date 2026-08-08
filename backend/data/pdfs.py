"""Read-only PDF resolver for the clinical reference documents.

Maps procedure keys (``modulo_synthea`` values) to PDF file paths under
``dataset/textos/``.  Does **not** read or parse PDF content — downstream RAG
ingestion uses the returned paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import PDFReference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Procedure-key → directory-name mapping (derived from live filesystem)
# ---------------------------------------------------------------------------

# The XLSX ``modulo_synthea`` values use underscores while some directories
# were created with spaces or capitalisation.  This table records the actual
# directory names found on disk.
_MODULO_TO_DIR: Dict[str, str] = {
    "appendicitis": "Appendicitis",
    "breast_cancer": "breast_cancer",
    "cholecystitis": "cholecystitis",
    "colorectal_cancer": "colorectal cancer",
    "total_joint_replacement": "total joint replacement",
}

# Reverse mapping for normalisation
_DIR_TO_MODULO: Dict[str, str] = {v: k for k, v in _MODULO_TO_DIR.items()}


def _textos_root() -> Path:
    """Path to ``dataset/textos/``."""
    # Walk up from this file: pdfs.py -> data/ -> backend/ -> project root
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "dataset" / "textos"


def _available_directories() -> Dict[str, Path]:
    """Return a dict of {modulo_key: absolute_textos_dir} for directories
    that actually exist on disk.

    Logs a warning for directories that cannot be accessed (e.g. Unicode-path
    failures on the host filesystem) instead of silently skipping them.
    """
    root = _textos_root()
    result: Dict[str, Path] = {}
    for modulo, dirname in _MODULO_TO_DIR.items():
        candidate = root / dirname
        try:
            if candidate.is_dir():
                result[modulo] = candidate
            else:
                logger.warning(
                    "Expected PDF directory not found: %s (modulo=%s)",
                    candidate, modulo,
                )
        except OSError as exc:
            logger.error(
                "Cannot access PDF directory %s (modulo=%s): %s",
                candidate, modulo, exc,
            )
    return result


def list_pdfs() -> List[PDFReference]:
    """List every PDF in the ``dataset/textos/`` tree.

    Returns a flat list of ``PDFReference`` instances, one per file.
    Logs a warning for any file whose path cannot be resolved (e.g.
    Unicode-encoding failures on the host filesystem).
    """
    refs: List[PDFReference] = []
    dirs = _available_directories()
    for modulo, dirpath in sorted(dirs.items()):
        try:
            children = sorted(dirpath.iterdir())
        except OSError as exc:
            logger.error(
                "Cannot iterate PDF directory %s (modulo=%s): %s",
                dirpath, modulo, exc,
            )
            continue
        for child in children:
            try:
                is_file = child.is_file()
            except OSError:
                logger.warning(
                    "Cannot stat %s (modulo=%s), skipping", child, modulo,
                )
                continue
            if is_file and child.suffix.lower() == ".pdf":
                refs.append(
                    PDFReference(
                        procedure=modulo,
                        filename=child.name,
                        path=child.resolve(),
                    )
                )
    return refs


def get_pdfs_by_procedure() -> Dict[str, List[PDFReference]]:
    """All PDFs grouped by procedure key (``modulo_synthea`` value)."""
    result: Dict[str, List[PDFReference]] = {}
    for ref in list_pdfs():
        result.setdefault(ref.procedure, []).append(ref)
    return result


def resolve_pdf_path(procedure: str, filename: str) -> Optional[Path]:
    """Resolve a single PDF path given a procedure key and filename.

    Returns ``None`` when the file does not exist.
    """
    norm_procedure = procedure.strip().lower().replace(" ", "_")
    # Try exact modulo key first, then try mapped directory name
    dirs = _available_directories()
    dirpath = dirs.get(procedure) or dirs.get(norm_procedure)
    if dirpath is None:
        # Fallback: try all directories
        for modulo, dp in _available_directories().items():
            if modulo.replace("_", " ") == procedure.replace("_", " "):
                dirpath = dp
                break
    if dirpath is None:
        return None
    candidate = dirpath / filename
    return candidate.resolve() if candidate.is_file() else None


def count_pdfs() -> int:
    """Return the total number of PDF files available."""
    return len(list_pdfs())


def get_procedure_names() -> List[str]:
    """Return the list of procedure keys for which PDFs exist on disk."""
    return sorted(_available_directories().keys())
