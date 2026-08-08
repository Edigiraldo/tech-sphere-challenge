"""Structured call summary generation.

This package provides a pure, deterministic summary generator that
produces typed ``SummaryResult`` objects from completed call data.

The generator is **text-only, stdlib-only, and deterministic**: it does
not call any LLM, RAG, voice service, or network. It structures the
raw data it receives — patient context, conversation turns, escalation
results, and source references — into a complete Spanish-language
postoperative call summary.

Public API
----------
``generate_summary(call_id, patient_context, turns, escalation_results, sources, **kw)``
    Produce a typed ``SummaryResult`` for a completed call.
``SummaryResult``
    Frozen dataclass with ``summary_id``, ``call_id``, ``patient_summary``,
    ``procedure``, ``symptoms``, ``decision``, ``sources``, ``next_steps``,
    and ``created_at``.
``SummarySection``
    A named section with a ``heading`` and ``content``.
``SourceReference``
    A traceable document citation (``document_id``, ``source_filename``,
    ``page_number``).
"""

from __future__ import annotations

from backend.summaries.generator import generate_summary
from backend.summaries.models import SourceReference, SummaryResult, SummarySection

__all__ = [
    "generate_summary",
    "SourceReference",
    "SummaryResult",
    "SummarySection",
]
