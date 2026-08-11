"""LLM second-approval for deterministic escalation classification.

After the deterministic classifier (``backend/decision/rules.py``) produces a
non-RED verdict on a patient answer during QUESTIONS, this module asks the LLM
to act as a conservative safety reviewer.  The LLM may:

* **Confirm** the deterministic classification (any severity).
* **Upgrade** severity (GREEN → YELLOW/RED, or YELLOW → RED).
* **Request clarification** when the answer is truly ambiguous (stay on same
  question, ask one follow-up).
* **Request RAG for a doubt** when the model wants clinical context before
  deciding (run RAG retrieval in QUESTIONS, then continue/finish).

The LLM **must never downgrade** a deterministic YELLOW or RED.  GREEN may be
upgraded but never downgraded (it is already the lowest severity).  RED answers
bypass this module entirely — the orchestrator short-circuits to ENDED before
any LLM call.

On failure, timeout, invalid output, or low-confidence response, this module
returns the original deterministic classification unchanged (safe fallback).

Prompt-injection controls from ``backend/llm/adapter.py`` are applied at the
input boundary before any LLM call.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.decision.models import EscalationResult, Severity
from backend.llm.config import LlmConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approval action enum
# ---------------------------------------------------------------------------

_APPROVAL_ACTION_CONFIRM = "confirm"
_APPROVAL_ACTION_ESCALATE = "escalate"
_APPROVAL_ACTION_CLARIFY = "request_clarification"
_APPROVAL_ACTION_RAG = "request_rag"

_VALID_ACTIONS: frozenset[str] = frozenset({
    _APPROVAL_ACTION_CONFIRM,
    _APPROVAL_ACTION_ESCALATE,
    _APPROVAL_ACTION_CLARIFY,
    _APPROVAL_ACTION_RAG,
})

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LlmApprovalResult:
    """Result of LLM second-approval on a deterministic classification.

    Attributes
    ----------
    severity : Severity
        Final severity **after** approval.  Guaranteed to be >= the
        deterministic classifier's severity (conservative escalation).
    should_escalate : bool
        Whether the call should be escalated.
    reason : str
        Spanish-language clinical rationale.
    next_action : str
        Spanish-language instruction for the agent.
    action : str
        What the orchestrator should do next: ``"confirm"`` (proceed
        normally), ``"escalate"`` (apply upgraded severity), 
        ``"request_clarification"`` (stay on same question, ask one
        follow-up), or ``"request_rag"`` (run RAG retrieval in 
        QUESTIONS before continuing).
    clarification_question : str
        The question to ask the patient (non-empty only when 
        ``action == "request_clarification"``).
    rag_query : str
        The RAG query to use (non-empty only when 
        ``action == "request_rag"``).
    llm_duration_ms : float or None
        LLM inference duration in milliseconds.
    prompt_tokens : int or None
        Input tokens consumed.
    completion_tokens : int or None
        Output tokens consumed.
    llm_used : bool
        ``True`` when the LLM was actually called (``False`` on fallback).
    """

    severity: Severity
    should_escalate: bool
    reason: str
    next_action: str
    action: str = _APPROVAL_ACTION_CONFIRM
    clarification_question: str = ""
    rag_query: str = ""
    llm_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    llm_used: bool = False

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}, "
                f"got {self.action!r}"
            )
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.next_action.strip():
            raise ValueError("next_action must be non-empty")
        if self.severity is Severity.RED and not self.should_escalate:
            raise ValueError("RED severity must have should_escalate=True")
        if self.severity is Severity.GREEN and self.should_escalate:
            raise ValueError("GREEN severity must have should_escalate=False")


# ---------------------------------------------------------------------------
# Prompt-injection detection (reused from adapter.py patterns)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:ignor[aeá]\s+(?:todas\s+)?(?:las\s+)?instrucciones|"
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:eres\s+un\s+asistente|you\s+are\s+(?:a\s+)?(?:helpful\s+)?"
        r"(?:AI\s+)?assistant|system\s*prompt|"
        r"repit[eé]\s+(?:el\s+)?prompt|repeat\s+(?:the\s+)?prompt)",
        re.IGNORECASE,
    ),
    re.compile(
        r'"role"\s*:\s*"system"|'
        r'"role"\s*:\s*"assistant"|'
        r"<\|im_start\|>|<\|im_end\|>|"
        r"\[INST\]|\[/INST\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"-{3,}\s*(?:begin|start|system|instrucciones?|prompt)",
        re.IGNORECASE,
    ),
]

_MAX_QUERY_LENGTH: int = 2000


def _detect_injection(text: str) -> list[str]:
    """Detect prompt injection patterns in *text*."""
    reasons: list[str] = []
    if len(text) > _MAX_QUERY_LENGTH:
        reasons.append(
            f"Consulta demasiado larga ({len(text)} caracteres, "
            f"máximo {_MAX_QUERY_LENGTH})."
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            reasons.append(
                f"Posible intento de inyección detectado: "
                f"patrón coincidente con {pattern.pattern!r}."
            )
    return reasons


# ---------------------------------------------------------------------------
# LLM prompt for second-approval
# ---------------------------------------------------------------------------

_APPROVAL_SYSTEM_PROMPT = """\
Eres un revisor de seguridad clínica para un agente de seguimiento \
postoperatorio en Colombia. Tu ÚNICA tarea es revisar la clasificación \
determinista de la respuesta de un paciente y decidir si es correcta o \
necesita ajuste.

REGLAS ESTRICTAS (no las violes bajo ninguna circunstancia):
1. NUNCA bajes una clasificación YELLOW a GREEN. Yellow es yellow y \
no puede degradarse.
2. NUNCA bajes una clasificación RED. Pero las respuestas RED no \
llegarán a ti — si por alguna razón ves una clasificación RED, \
mantenla como RED.
3. Puedes CONFIRMAR la clasificación si estás de acuerdo.
4. Puedes SUBIR la severidad si detectas señales de alerta que el \
clasificador determinista podría haber pasado por alto:
   - GREEN puede subir a YELLOW o RED si hay indicios de peligro.
   - YELLOW puede subir a RED si el síntoma es grave.
5. Puedes solicitar ACLARACIÓN si la respuesta del paciente es \
realmente ambigua y no puedes determinar la severidad. En ese caso, \
proporciona una pregunta clara para hacerle al paciente (en español \
colombiano). Solo puedes solicitar aclaración UNA vez por pregunta.
6. Puedes solicitar RAG (consulta a fuentes clínicas) si tienes dudas \
sobre si un síntoma descrito amerita escalamiento y necesitas \
información clínica adicional. Proporciona una consulta clara en \
español para buscar en los documentos clínicos.
7. Sé conservador: ante la duda, prefiere escalar. Un falso negativo \
(no escalar cuando debías) es catastrófico.
8. Tu respuesta DEBE ser SOLO el JSON. No incluyas texto fuera del JSON.
9. Responde en español."""

_APPROVAL_USER_PROMPT_TEMPLATE = """\
DOMINIO CLÍNICO: {domain}
DÍA POSTOPERATORIO: {dia_postop}
PROCEDIMIENTO: {procedimiento}

RESPUESTA DEL PACIENTE:
{patient_text}

CLASIFICACIÓN DETERMINISTA:
- Severidad: {severity}
- Razón: {reason}
- Acción recomendada: {next_action}

Tu tarea: revisa esta clasificación y decide qué acción tomar.

Responde EXCLUSIVAMENTE en formato JSON:
{{
  "final_severity": "GREEN", "YELLOW", o "RED",
  "should_escalate": true o false,
  "reason": "tu razonamiento clínico en español",
  "next_action": "instrucción para el agente en español",
  "action": "confirm", "escalate", "request_clarification", o "request_rag",
  "clarification_question": "pregunta para el paciente (solo cuando action=request_clarification, si no string vacío)",
  "rag_query": "consulta para buscar en documentos clínicos (solo cuando action=request_rag, si no string vacío)"
}}

RECUERDA:
- NUNCA bajes YELLOW a GREEN.
- Si action="confirm", final_severity debe ser IGUAL a la clasificación determinista (GREEN o YELLOW).
- Si action="escalate", final_severity debe ser MAYOR (GREEN→YELLOW, GREEN→RED, o YELLOW→RED).
- Si action="request_clarification", final_severity debe ser YELLOW y clarification_question no vacío.
- Si action="request_rag", final_severity debe ser YELLOW y rag_query no vacío.
- should_escalate solo es true si final_severity es RED o si estás escalando un segundo YELLOW.
"""


def _build_approval_prompt(
    patient_text: str,
    domain: str,
    classification: EscalationResult,
    dia_postop: int,
    procedimiento: str,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the approval LLM call."""
    user_prompt = _APPROVAL_USER_PROMPT_TEMPLATE.format(
        domain=domain,
        dia_postop=dia_postop,
        procedimiento=procedimiento or "No especificado",
        patient_text=patient_text,
        severity=classification.severity.value,
        reason=classification.reason,
        next_action=classification.next_action,
    )
    return _APPROVAL_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_severity(raw: str) -> Severity | None:
    """Parse a severity string returned by the LLM."""
    upper = raw.strip().upper()
    if upper == "GREEN":
        return Severity.GREEN
    elif upper == "YELLOW":
        return Severity.YELLOW
    elif upper == "RED":
        return Severity.RED
    return None


def _validate_non_downgrade(
    llm_severity: Severity,
    deterministic_severity: Severity,
) -> bool:
    """Return ``True`` when the LLM severity is NOT a downgrade.

    Allowed transitions:
        GREEN → GREEN, YELLOW, RED  (upgrade only)
        YELLOW → YELLOW, RED        (no downgrade to GREEN)
        RED → RED                   (no downgrade)
    """
    if deterministic_severity is Severity.RED:
        return llm_severity is Severity.RED
    if deterministic_severity is Severity.YELLOW:
        return llm_severity in (Severity.YELLOW, Severity.RED)
    if deterministic_severity is Severity.GREEN:
        return True  # GREEN is the lowest — any severity >= GREEN is valid
    return False


def _parse_and_validate_llm_output(
    parsed: dict[str, Any],
    deterministic_classification: EscalationResult,
) -> tuple[LlmApprovalResult | None, str]:
    """Parse LLM JSON output and validate safety constraints.

    Returns ``(result, error_msg)``.  When ``result`` is ``None``,
    *error_msg* describes why validation failed.
    """
    # Extract fields
    final_severity_str = str(parsed.get("final_severity", "")).strip()
    should_escalate_flag = bool(parsed.get("should_escalate", False))
    reason = str(parsed.get("reason", "")).strip()
    next_action = str(parsed.get("next_action", "")).strip()
    action = str(parsed.get("action", "")).strip().lower()
    clarification_question = str(parsed.get("clarification_question", "")).strip()
    rag_query = str(parsed.get("rag_query", "")).strip()

    # Validate severity parseable
    llm_severity = _parse_severity(final_severity_str)
    if llm_severity is None:
        return None, (
            f"Severidad no reconocida: {final_severity_str!r}. "
            f"Se esperaba GREEN, YELLOW, o RED."
        )

    # Non-downgrade check
    if not _validate_non_downgrade(llm_severity, deterministic_classification.severity):
        return None, (
            f"El LLM intentó bajar severidad de "
            f"{deterministic_classification.severity.value} a "
            f"{llm_severity.value}. Rechazado."
        )

    # Validate action
    if action not in _VALID_ACTIONS:
        return None, f"Acción no reconocida: {action!r}."

    # Non-empty reason
    if not reason:
        return None, "El campo 'reason' está vacío."

    # Non-empty next_action
    if not next_action:
        return None, "El campo 'next_action' está vacío."

    # Action-specific validations
    if action == _APPROVAL_ACTION_CLARIFY:
        if not clarification_question:
            return None, (
                "Acción 'request_clarification' requiere "
                "'clarification_question' no vacío."
            )
        # Clarification forces YELLOW (not escalated yet)
        llm_severity = Severity.YELLOW

    if action == _APPROVAL_ACTION_RAG:
        if not rag_query:
            return None, (
                "Acción 'request_rag' requiere 'rag_query' no vacío."
            )
        # RAG doubt forces YELLOW — we don't know enough to confirm GREEN
        llm_severity = Severity.YELLOW

    # Consistency: confirm must keep same severity
    if action == _APPROVAL_ACTION_CONFIRM:
        if llm_severity != deterministic_classification.severity:
            return None, (
                f"Acción 'confirm' requiere final_severity igual a la "
                f"clasificación determinista "
                f"({deterministic_classification.severity.value}), "
                f"pero el LLM devolvió {llm_severity.value}."
            )

    # Consistency: escalate must increase severity
    if action == _APPROVAL_ACTION_ESCALATE:
        det_sev = deterministic_classification.severity
        if det_sev is Severity.RED:
            return None, (
                "No se puede escalar una clasificación RED (ya es máxima)."
            )
        if det_sev is Severity.YELLOW and llm_severity is Severity.YELLOW:
            return None, (
                "Acción 'escalate' requiere aumento de severidad, pero el "
                "LLM devolvió YELLOW (mismo nivel)."
            )
        if det_sev is Severity.GREEN and llm_severity is Severity.GREEN:
            return None, (
                "Acción 'escalate' requiere aumento de severidad, pero el "
                "LLM devolvió GREEN (mismo nivel)."
            )

    # RED always escalates
    if llm_severity is Severity.RED and not should_escalate_flag:
        return None, "RED severity debe tener should_escalate=true."

    # GREEN never escalates
    if llm_severity is Severity.GREEN and should_escalate_flag:
        return None, "GREEN severity debe tener should_escalate=false."

    return LlmApprovalResult(
        severity=llm_severity,
        should_escalate=should_escalate_flag,
        reason=reason,
        next_action=next_action,
        action=action,
        clarification_question=clarification_question,
        rag_query=rag_query,
        llm_used=False,  # will be set by caller
    ), ""


# ---------------------------------------------------------------------------
# Groq transport (minimal, mirrors adapter.py pattern)
# ---------------------------------------------------------------------------


def _call_groq_approval(
    system_prompt: str,
    user_prompt: str,
    config: LlmConfig,
) -> dict[str, Any]:
    """Invoke Llama via Groq for structured approval JSON."""
    if not config.api_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada. "
            "Defina la variable de entorno GROQ_API_KEY."
        )

    try:
        import groq
    except ImportError as exc:
        raise RuntimeError(
            "El paquete groq no está instalado."
        ) from exc

    client = groq.Groq(api_key=config.api_key)

    logger.info(
        "Calling approval LLM %s (temp=%.2f, max_tokens=%d) …",
        config.model_name,
        config.temperature,
        config.max_output_tokens,
    )

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            response_format={"type": "json_object"},
        )
        llm_duration_ms = (time.time() - start) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            f"Error al llamar a {config.model_name}: {exc}"
        ) from exc

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

    raw_text = (response.choices[0].message.content or "").strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse approval JSON: %s", raw_text[:200])
        raise ValueError(
            f"El modelo no devolvió JSON válido: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("El modelo devolvió JSON pero no es un objeto.")

    parsed["_llm_duration_ms"] = llm_duration_ms
    parsed["_prompt_tokens"] = prompt_tokens
    parsed["_completion_tokens"] = completion_tokens

    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_second_approval(
    patient_text: str,
    domain: str,
    deterministic_classification: EscalationResult,
    dia_postop: int,
    procedimiento: str,
    config: LlmConfig,
) -> LlmApprovalResult:
    """Run LLM second-approval on a deterministic classification.

    Parameters
    ----------
    patient_text : str
        The patient's raw answer text.
    domain : str
        Symptom domain being assessed (e.g. ``"dolor"``).
    deterministic_classification : EscalationResult
        The deterministic classifier's verdict (must be non-RED).
    dia_postop : int
        Post-operative day number.
    procedimiento : str
        Surgical procedure name.
    config : LlmConfig
        LLM configuration.

    Returns
    -------
    LlmApprovalResult
        The final verdict, guaranteed to be >= the deterministic severity.
        On any failure, returns the deterministic classification unchanged
        (``action="confirm"``, ``llm_used=False``).

    Raises
    ------
    ValueError
        If *deterministic_classification* has RED severity (RED bypasses
        this module — the caller must not invoke approval for RED).
    """
    if deterministic_classification.severity is Severity.RED:
        raise ValueError(
            "RED classifications must not pass through LLM second-approval. "
            "The orchestrator should short-circuit RED answers to ENDED "
            "before calling this function."
        )

    # Detect prompt injection in patient input
    injection_reasons = _detect_injection(patient_text)
    if injection_reasons:
        logger.warning(
            "Prompt injection detected during approval for query %r: %s",
            patient_text[:120],
            "; ".join(injection_reasons),
        )
        # Fall back to deterministic classification
        return LlmApprovalResult(
            severity=deterministic_classification.severity,
            should_escalate=deterministic_classification.should_escalate,
            reason=deterministic_classification.reason,
            next_action=deterministic_classification.next_action,
            action=_APPROVAL_ACTION_CONFIRM,
            llm_used=False,
        )

    # Build prompt
    system_prompt, user_prompt = _build_approval_prompt(
        patient_text=patient_text,
        domain=domain,
        classification=deterministic_classification,
        dia_postop=dia_postop,
        procedimiento=procedimiento,
    )

    # Call LLM with timeout safety
    try:
        parsed = _call_groq_approval(system_prompt, user_prompt, config)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "LLM approval call failed, falling back to deterministic: %s", exc
        )
        return LlmApprovalResult(
            severity=deterministic_classification.severity,
            should_escalate=deterministic_classification.should_escalate,
            reason=deterministic_classification.reason,
            next_action=deterministic_classification.next_action,
            action=_APPROVAL_ACTION_CONFIRM,
            llm_used=False,
        )

    llm_duration_ms = parsed.pop("_llm_duration_ms", None)
    prompt_tokens = parsed.pop("_prompt_tokens", None)
    completion_tokens = parsed.pop("_completion_tokens", None)

    # Parse and validate
    result, error_msg = _parse_and_validate_llm_output(
        parsed, deterministic_classification
    )

    if result is None:
        logger.warning(
            "LLM approval output rejected: %s. Falling back to deterministic.",
            error_msg,
        )
        return LlmApprovalResult(
            severity=deterministic_classification.severity,
            should_escalate=deterministic_classification.should_escalate,
            reason=deterministic_classification.reason,
            next_action=deterministic_classification.next_action,
            action=_APPROVAL_ACTION_CONFIRM,
            llm_used=False,
        )

    # Attach metrics and mark LLM as used
    # LlmApprovalResult is frozen — must use object.__setattr__
    object.__setattr__(result, "llm_duration_ms", llm_duration_ms)
    object.__setattr__(result, "prompt_tokens", prompt_tokens)
    object.__setattr__(result, "completion_tokens", completion_tokens)
    object.__setattr__(result, "llm_used", True)

    logger.info(
        "LLM approval: deterministic=%s → llm=%s action=%s",
        deterministic_classification.severity.value,
        result.severity.value,
        result.action,
    )

    return result
