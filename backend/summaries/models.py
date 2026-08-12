"""Typed summary domain models.

These dataclasses represent the structured output of a call summary.
They are pure data containers with no persistence, RAG, LLM, or
network dependencies.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SummarySection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummarySection:
    """A named section within a call summary.

    Attributes
    ----------
    heading : str
        Short section label in Spanish (e.g. ``"Paciente"``,
        ``"Procedimiento"``, ``"Dolor"``, ``"Decisión de escalamiento"``).
    content : str
        Spanish-language content for this section. Non-empty after stripping.
    """

    heading: str
    content: str

    def __post_init__(self) -> None:
        if not self.heading.strip():
            raise ValueError("heading must be non-empty after stripping")
        if not self.content.strip():
            raise ValueError("content must be non-empty after stripping")


# ---------------------------------------------------------------------------
# SourceReference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A traceable document citation used during the call.

    Attributes
    ----------
    document_id : str
        Unique identifier of the source document (UUID hex string).
    source_filename : str
        Human-readable filename of the source.
    page_number : int
        Page number where the cited content appears (1-based).
    """

    document_id: str
    source_filename: str
    page_number: int = 0

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must be non-empty")
        if not self.source_filename.strip():
            raise ValueError("source_filename must be non-empty")
        if self.page_number < 0:
            raise ValueError(
                f"page_number must be >= 0, got {self.page_number}"
            )


# ---------------------------------------------------------------------------
# SummaryResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummaryResult:
    """A complete structured call summary.

    Contains all required sections: patient demographic summary,
    procedure context, per-domain symptom responses, escalation
    decision, traceable sources, and recommended next steps.

    Attributes
    ----------
    summary_id : str
        Unique identifier for this summary.
    call_id : str
        The call this summary corresponds to.
    patient_summary : SummarySection
        Patient demographics (name, age, city, EPS).
    procedure : SummarySection
        Procedure name, post-operative day.
    symptoms : list[SummarySection]
        One section per symptom domain assessed during the call.
    decision : SummarySection
        Escalation decision, severity, and clinical rationale.
    sources : list[SourceReference]
        All traceable document citations referenced during the call.
    next_steps : SummarySection
        Recommended next steps in Spanish.
    created_at : datetime.datetime
        UTC timestamp when the summary was generated.
    """

    summary_id: str
    call_id: str
    patient_summary: SummarySection
    procedure: SummarySection
    symptoms: list[SummarySection] = field(default_factory=list)
    decision: SummarySection = field(default_factory=lambda: SummarySection(
        heading="Decisión de escalamiento",
        content="No se requirió escalamiento durante esta llamada.",
    ))
    sources: list[SourceReference] = field(default_factory=list)
    next_steps: SummarySection = field(default_factory=lambda: SummarySection(
        heading="Próximos pasos",
        content="Continuar con el seguimiento postoperatorio según lo programado.",
    ))
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def __post_init__(self) -> None:
        if not self.summary_id.strip():
            raise ValueError("summary_id must be non-empty")
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        # Validate each symptom section
        for i, section in enumerate(self.symptoms):
            if not isinstance(section, SummarySection):
                raise TypeError(
                    f"symptoms[{i}] must be SummarySection, "
                    f"got {type(section).__name__}"
                )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def all_section_headings(self) -> tuple[str, ...]:
        """Return all section headings in order (patient, procedure,
        symptoms..., decision, next_steps)."""
        symptom_headings = tuple(s.heading for s in self.symptoms)
        return (
            self.patient_summary.heading,
            self.procedure.heading,
            *symptom_headings,
            self.decision.heading,
            self.next_steps.heading,
        )

    @property
    def total_sources(self) -> int:
        """Number of unique source references in the summary."""
        return len(self.sources)

    @property
    def has_escalation(self) -> bool:
        """``True`` when the decision section indicates a **conclusive**
        escalation (RED, accumulated YELLOW, or LLM-upgraded RED).

        Non-conclusive YELLOW observations (first YELLOW per domain) are
        recorded for the audit trail but do not constitute an escalation
        at the call level.
        """
        content_lower = self.decision.content.lower()
        # Explicitly exclude phrasing that states escalation was NOT required
        if "no se requiri" in content_lower:
            return False
        if "no fue necesario" in content_lower:
            return False
        # Conclusive escalation indicators (excludes bare "amarillo" which
        # may appear in non-conclusive per-domain observations)
        escalation_indicators = [
            "escalamiento inmediato",
            "escalamiento por acumulacion",
            "rojo",
            "alerta roja",
        ]
        return any(indicator in content_lower for indicator in escalation_indicators)
