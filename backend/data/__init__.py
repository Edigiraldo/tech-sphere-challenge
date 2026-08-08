"""Normalized read-only dataset access (``backend.data``).

This package provides typed, immutable access to the four synthetic-dataset
XLSX files and the clinical PDF corpus.  The source files are never modified.

Public API
----------
* ``Patient``, ``Trajectory``, ``Conversation``, ``ConversationTurn``,
  ``PDFReference`` — dataclass models.
* ``load_patients()`` — merged clinical + demographic profiles.
* ``load_trajectories()`` — post-op trajectories grouped by patient.
* ``load_conversations()`` — runtime-safe conversation loader
  (``label_ground_truth`` always ``None``).
* ``get_dataset_path()`` — absolute path to ``dataset/``.
* ``list_pdfs()``, ``get_pdfs_by_procedure()``, ``resolve_pdf_path()``,
  ``count_pdfs()``, ``get_procedure_names()`` — PDF corpus resolver.

**``label_ground_truth`` isolation**

The evaluation-only ``load_conversations_for_evaluation()`` function is
deliberately **not** re-exported here.  Runtime modules (``conversation/``,
``llm/``, ``decision/``, etc.) must never depend on ground-truth labels.
Import it explicitly from ``backend.data.loader`` when needed for offline
evaluation or testing.
"""

from .loader import (
    get_dataset_path,
    load_conversations,
    load_patients,
    load_trajectories,
)
from .models import (
    Conversation,
    ConversationTurn,
    PDFReference,
    Patient,
    Trajectory,
)
from .pdfs import (
    count_pdfs,
    get_pdfs_by_procedure,
    get_procedure_names,
    list_pdfs,
    resolve_pdf_path,
)

__all__ = [
    # Models
    "Patient",
    "Trajectory",
    "Conversation",
    "ConversationTurn",
    "PDFReference",
    # Loaders (runtime-safe)
    "load_patients",
    "load_trajectories",
    "load_conversations",
    "get_dataset_path",
    # PDF resolver
    "list_pdfs",
    "get_pdfs_by_procedure",
    "resolve_pdf_path",
    "count_pdfs",
    "get_procedure_names",
]
