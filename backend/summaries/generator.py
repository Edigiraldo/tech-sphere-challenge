"""Pure typed summary generator for completed postoperative calls.

The ``generate_summary()`` function is the sole public entry point.
It is **text-only, stdlib-only, deterministic**: no LLM, RAG, voice
service, network, or external dependencies. It structures the raw
data it receives into a complete ``SummaryResult``.

The generator receives:
* ``PatientContext`` — patient demographics and procedure.
* ``ConversationTurnRecord`` list — all turns recorded during the call.
* ``EscalationResult`` list — escalation classifications per patient turn.
* ``SourceReference`` list — all document citations used during the call.

And produces a ``SummaryResult`` with these sections:
1. Patient demographics (name, age, city, EPS).
2. Procedure and post-operative day.
3. Per-domain symptom responses.
4. Escalation decision and rationale.
5. Traceable sources.
6. Recommended next steps.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

from backend.conversation.context import PatientContext
from backend.decision.models import EscalationResult
from backend.persistence.sqlite import ConversationTurnRecord

from .models import SourceReference, SummaryResult, SummarySection

# ---------------------------------------------------------------------------
# Domain-to-heading mapping
# ---------------------------------------------------------------------------

_DOMAIN_HEADINGS: dict[str, str] = {
    "dolor": "Dolor",
    "fiebre": "Fiebre",
    "herida": "Herida quirurgica",
    "apetito": "Apetito",
    "sueno": "Sueno",
    "movilidad": "Movilidad",
}

# Ordered domain names matching the follow-up question sequence
_DOMAIN_ORDER: list[str] = [
    "dolor",
    "fiebre",
    "herida",
    "apetito",
    "sueno",
    "movilidad",
]


# ===================================================================
# Public API
# ===================================================================


def generate_summary(
    call_id: str,
    patient_context: PatientContext,
    turns: list[ConversationTurnRecord],
    escalation_results: list[EscalationResult],
    sources: list[SourceReference],
    *,
    created_at: Optional[datetime.datetime] = None,
) -> SummaryResult:
    """Generate a structured ``SummaryResult`` from completed call data.

    Parameters
    ----------
    call_id : str
        Unique identifier for the call being summarised.
    patient_context : PatientContext
        Patient demographics and procedure context.
    turns : list[ConversationTurnRecord]
        All turns recorded during the call, in order.
    escalation_results : list[EscalationResult]
        Escalation classifications, one per patient turn (may be empty).
    sources : list[SourceReference]
        All document citations referenced during the call.
    created_at : datetime.datetime or None
        UTC timestamp for the summary. Uses ``datetime.now(UTC)`` if ``None``.

    Returns
    -------
    SummaryResult
        A complete typed call summary with all required sections.

    Raises
    ------
    ValueError
        If *call_id* is empty or any required input is invalid.
    """
    if not call_id.strip():
        raise ValueError("call_id must be non-empty")

    now = created_at or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")

    summary_id = uuid.uuid4().hex

    # --- 1. Patient summary ---
    patient_section = _build_patient_section(patient_context)

    # --- 2. Procedure summary ---
    procedure_section = _build_procedure_section(patient_context)

    # --- 3. Per-domain symptom sections ---
    symptom_sections = _build_symptom_sections(turns, escalation_results)

    # --- 4. Decision section ---
    decision_section = _build_decision_section(escalation_results)

    # --- 5. Next steps ---
    next_steps_section = _build_next_steps(escalation_results)

    return SummaryResult(
        summary_id=summary_id,
        call_id=call_id,
        patient_summary=patient_section,
        procedure=procedure_section,
        symptoms=symptom_sections,
        decision=decision_section,
        sources=list(sources),
        next_steps=next_steps_section,
        created_at=now,
    )


# ===================================================================
# Section builders
# ===================================================================


def _build_patient_section(
    pc: PatientContext,
) -> SummarySection:
    """Build the patient demographics section."""
    p = pc.patient
    content = (
        f"Paciente: {p.nombre_completo}. "
        f"Documento: {p.documento_cc}. "
        f"Edad: {p.edad} anos. "
        f"Ciudad: {p.ciudad}, {p.departamento}. "
        f"EPS: {p.eps}."
    )
    return SummarySection(heading="Paciente", content=content)


def _build_procedure_section(
    pc: PatientContext,
) -> SummarySection:
    """Build the procedure and post-operative day section."""
    content = (
        f"Procedimiento: {pc.procedimiento}. "
        f"Dia postoperatorio: {pc.dia_postop}. "
        f"Fecha de cirugia: {pc.patient.fecha_cirugia}."
    )
    return SummarySection(heading="Procedimiento", content=content)


def _build_symptom_sections(
    turns: list[ConversationTurnRecord],
    escalation_results: list[EscalationResult],
) -> list[SummarySection]:
    """Build one section per assessed symptom domain.

    Extracts patient responses from the turns and pairs them with their
    escalation classification (if available).
    """
    # Collect patient responses grouped by approximate question index.
    # The conversation flow asks 6 questions in sequence during the
    # QUESTIONS phase.  Patient turns in that phase correspond to the
    # current question.  We also look at the escalation results to annotate
    # each domain with its severity.
    patient_texts_by_domain: dict[str, list[str]] = {
        d: [] for d in _DOMAIN_ORDER
    }

    # Count patient turns to estimate which question they answered.
    # This is a heuristic: the first patient turn during QUESTIONS
    # answers question 0, the second answers question 1, etc.
    # Agent turns are interspersed.
    patient_turn_count = 0
    for turn in turns:
        if turn.role == "PATIENT":
            idx = patient_turn_count
            if idx < len(_DOMAIN_ORDER):
                domain = _DOMAIN_ORDER[idx]
                patient_texts_by_domain[domain].append(turn.text)
            patient_turn_count += 1

    # Build severity lookup from escalation results
    severity_by_domain: dict[str, str] = {}
    for result in escalation_results:
        if result.domain is not None and result.domain in _DOMAIN_ORDER:
            # Keep the highest severity seen for this domain
            existing = severity_by_domain.get(result.domain)
            if existing is None or _severity_gt(result.severity.name, existing):
                severity_by_domain[result.domain] = result.severity.name

    sections: list[SummarySection] = []
    for domain in _DOMAIN_ORDER:
        heading = _DOMAIN_HEADINGS.get(domain, domain.capitalize())
        responses = patient_texts_by_domain.get(domain, [])

        if not responses:
            content = f"No se registro respuesta del paciente para {heading.lower()}."
        else:
            joined = " ".join(responses)
            if len(joined) > 300:
                joined = joined[:297] + "..."
            content = f"El paciente reporto: {joined}"

        # Annotate with severity if available
        sev = severity_by_domain.get(domain)
        if sev:
            if sev == "RED":
                content += " [ALERTA ROJA: se recomendo escalamiento inmediato]"
            elif sev == "YELLOW":
                content += " [ALERTA AMARILLA: requiere seguimiento adicional]"

        sections.append(SummarySection(heading=heading, content=content))

    return sections


def _build_decision_section(
    escalation_results: list[EscalationResult],
) -> SummarySection:
    """Build the escalation decision section."""
    if not escalation_results:
        return SummarySection(
            heading="Decision de escalamiento",
            content=(
                "No se realizaron clasificaciones de escalamiento "
                "durante esta llamada. No se requirio escalamiento."
            ),
        )

    # Find the highest severity
    highest = "GREEN"
    red_reason = None
    yellow_reasons: list[str] = []

    for result in escalation_results:
        sev_name = result.severity.name
        if _severity_gt(sev_name, highest):
            highest = sev_name
            if sev_name == "RED":
                red_reason = result.reason
        if sev_name == "YELLOW":
            yellow_reasons.append(f"{result.domain or 'general'}: {result.reason}")

    if highest == "RED":
        content = (
            f"ESCALAMIENTO INMEDIATO (ROJO). "
            f"Razon: {red_reason}. "
            f"Se recomienda transferir al medico tratante de inmediato."
        )
    elif highest == "YELLOW":
        if len(escalation_results) >= 2:
            content = (
                f"ESCALAMIENTO POR ACUMULACION (AMARILLO x{len(escalation_results)}). "
                f"Se detectaron multiples indicadores de precaucion. "
                f"Detalles: {'; '.join(yellow_reasons[:3])}. "
                f"Se recomienda seguimiento prioritario."
            )
        else:
            content = (
                f"PRECAUCION (AMARILLO). "
                f"{yellow_reasons[0] if yellow_reasons else 'Indicador de precaucion detectado.'} "
                f"Se recomienda seguimiento cercano."
            )
    else:
        content = (
            "No se detectaron senales de alarma en esta llamada. "
            "Todos los indicadores evaluados estan en rango normal (VERDE)."
        )

    return SummarySection(
        heading="Decision de escalamiento",
        content=content,
    )


def _build_next_steps(
    escalation_results: list[EscalationResult],
) -> SummarySection:
    """Build the next-steps section based on escalation results."""
    if not escalation_results:
        return SummarySection(
            heading="Proximos pasos",
            content=(
                "Continuar con el seguimiento postoperatorio segun lo "
                "programado. Proxima llamada de seguimiento en la fecha "
                "indicada por el protocolo."
            ),
        )

    highest = "GREEN"
    for result in escalation_results:
        if _severity_gt(result.severity.name, highest):
            highest = result.severity.name

    if highest == "RED":
        content = (
            "1. Transferir inmediatamente al medico tratante. "
            "2. Notificar a la EPS. "
            "3. Registrar la alerta en el sistema de seguimiento. "
            "4. Hacer seguimiento en 24 horas para verificar atencion recibida."
        )
    elif highest == "YELLOW":
        content = (
            "1. Programar llamada de seguimiento adicional en 48 horas. "
            "2. Solicitar al paciente monitorear los sintomas reportados. "
            "3. Indicar al paciente acudir a urgencias si los sintomas empeoran. "
            "4. Informar al medico tratante sobre los indicadores de precaucion."
        )
    else:
        content = (
            "Continuar con el seguimiento postoperatorio segun lo "
            "programado. Proxima llamada de seguimiento en la fecha "
            "indicada por el protocolo."
        )

    return SummarySection(heading="Proximos pasos", content=content)


# ===================================================================
# Helpers
# ===================================================================


def _severity_gt(a: str, b: str) -> bool:
    """Return ``True`` when severity *a* is higher than *b*.
    
    Order: RED > YELLOW > GREEN.
    """
    _order = {"RED": 3, "YELLOW": 2, "GREEN": 1}
    return _order.get(a, 0) > _order.get(b, 0)
