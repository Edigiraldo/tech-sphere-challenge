"""Deterministic escalation rule engine.

The ``classify()`` function is the sole public entry point.  It processes a
patient's response text against the Spanish symptom lexicon and numeric
thresholds, returning a typed ``EscalationResult``.

Architecture:
    The engine is **text-only, stdlib-only, deterministic**.  It does not
    call any language model, load any dataset, or touch persistence.  It
    has no side effects.  It can be called synchronously on every patient
    turn during the QUESTIONS phase.

Classification logic (priority order):
    1. **Invalid / empty input** → INVALID (severity=YELLOW, source="invalid")
    2. **Cross-cutting red flags** → RED regardless of domain
    3. **Domain-specific red flags** → RED
    4. **Numeric threshold breach** → RED or YELLOW (pain, temperature)
    5. **Ambiguous / uncertain** → YELLOW (source="ambig")
    6. **Yellow triggers** → YELLOW (source="rule")
    7. **Green indicators** → GREEN (source="rule")
    8. **Fallback** → YELLOW (source="incomplete" — unable to classify)
"""

from __future__ import annotations

import re

from backend.decision.lexicon import (
    ALL_DOMAINS,
    AMBIGUITY_PHRASES,
    CROSS_CUTTING_RED_FLAGS,
    DOMAIN_DISPATCH,
    GREEN_INDICATORS,
    NEGATION_MARKERS,
    PAIN_RED_THRESHOLD,
    PAIN_YELLOW_THRESHOLD,
    RED_FLAGS,
    TEMP_RED_THRESHOLD,
    TEMP_YELLOW_THRESHOLD,
    YELLOW_TRIGGERS,
)
from backend.decision.models import EscalationResult, Severity

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

# Extract the first numeric value from text (handles integer and decimal).
# This is used for pain scores (``"7"``, ``"7.5"``) and temperatures
# (``"37.8"``, ``"38"``).
_NUMBER_RE = re.compile(r"(\d+)(?:[.,](\d+))?")


def _extract_number(text: str) -> float | None:
    """Extract the first integer or decimal number from *text*.

    Returns ``None`` if no numeric sequence is found.
    """
    m = _NUMBER_RE.search(text)
    if m is None:
        return None
    whole = m.group(1)
    frac = m.group(2)
    if frac:
        return float(f"{whole}.{frac}")
    return float(whole)


# ---------------------------------------------------------------------------
# Negation check
# ---------------------------------------------------------------------------


def _is_negated(lowered: str, keyword: str, idx: int) -> bool:
    """Check whether *keyword* at position *idx* in *lowered* is negated.

    A keyword is considered negated when a negation marker appears within a
    window of ~5 tokens (roughly 60 characters) before the match.
    """
    prefix = lowered[max(0, idx - 60):idx].rstrip()
    # Simple token-based check: split the prefix into "words"
    tokens = prefix.split()
    # Check the last 5 tokens for negation markers
    recent = tokens[-5:] if len(tokens) >= 5 else tokens
    for marker in NEGATION_MARKERS:
        if marker in recent:
            return True
        # Also check multi-word markers as exact substring of the prefix tail
        if " " in marker and marker in " ".join(recent):
            return True
    return False


# ---------------------------------------------------------------------------
# Keyword match helpers
# ---------------------------------------------------------------------------


def _find_matches(
    lowered: str, keywords: list[str], respect_negation: bool = True
) -> list[str]:
    """Return all *keywords* found in *lowered* (negation-aware).

    When *respect_negation* is ``True``, a keyword preceded by a negation
    marker is excluded from the result.
    """
    matched: list[str] = []
    for kw in keywords:
        idx = lowered.find(kw)
        if idx == -1:
            continue
        if respect_negation and _is_negated(lowered, kw, idx):
            continue
        matched.append(kw)
    return matched


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify(
    patient_text: str,
    domain: str,
    dia_postop: int = 0,
    procedimiento: str = "",
) -> EscalationResult:
    """Classify a patient response into a safety level.

    Parameters
    ----------
    patient_text : str
        The patient's spoken response (may be empty or whitespace-only).
    domain : str
        The symptom domain being assessed.  Must be one of the recognised
        domain keys (``"dolor"``, ``"fiebre"``, ``"herida"``,
        ``"apetito"``, ``"sueño"`` / ``"sueno"``, ``"movilidad"``).
    dia_postop : int
        Post-operative day number.  Must be >= 0.
    procedimiento : str
        Name of the surgical procedure (may be empty when not available).

    Returns
    -------
    EscalationResult
        Frozen verdict with ``severity``, ``should_escalate``, ``reason``,
        and ``next_action``.

    Raises
    ------
    ValueError
        If *dia_postop* or *domain* is invalid.
    """
    # ---------- validate arguments ----------
    if dia_postop < 0:
        raise ValueError(f"dia_postop must be >= 0, got {dia_postop}")

    canon = DOMAIN_DISPATCH.get(domain.lower())
    if canon is None:
        raise ValueError(
            f"Unknown domain {domain!r}.  Must be one of: "
            f"{', '.join(sorted(ALL_DOMAINS))}"
        )

    # ---------- step 1: invalid / empty input ----------
    stripped = patient_text.strip()
    if not stripped:
        return EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="El paciente no proporcionó información sobre este punto.",
            next_action="Solicitar aclaración: preguntar nuevamente si "
            "presenta algún síntoma en este aspecto.",
            domain=canon,
            source="invalid",
        )

    lowered = stripped.lower()

    # ---------- step 2: cross-cutting red flags ----------
    for flag in CROSS_CUTTING_RED_FLAGS:
        idx = lowered.find(flag)
        if idx != -1 and not _is_negated(lowered, flag, idx):
            return EscalationResult(
                severity=Severity.RED,
                should_escalate=True,
                reason=f"Señal de alerta crítica detectada: '{flag}'. "
                f"Requiere atención médica inmediata.",
                next_action="Transferir al médico tratante de inmediato. "
                "Finalizar la llamada con instrucciones claras.",
                domain=canon,
                source="rule",
            )

    # ---------- step 3: domain-specific red flags ----------
    domain_red = RED_FLAGS.get(canon, [])
    red_matches = _find_matches(lowered, domain_red, respect_negation=True)
    if red_matches:
        first = red_matches[0]
        return EscalationResult(
            severity=Severity.RED,
            should_escalate=True,
            reason=f"Síntoma de alerta roja en {canon}: '{first}'. "
            f"Posible complicación postoperatoria.",
            next_action="Transferir al médico tratante de inmediato. "
            "Finalizar la llamada con instrucciones claras.",
            domain=canon,
            source="rule",
        )

    # ---------- step 4: numeric threshold checks ----------
    numeric = _extract_number(lowered)

    if canon == "dolor" and numeric is not None and 0 <= numeric <= 10:
        if numeric >= PAIN_RED_THRESHOLD:
            return EscalationResult(
                severity=Severity.RED,
                should_escalate=True,
                reason=f"Dolor reportado de {numeric}/10 — umbral crítico "
                f"(>= {PAIN_RED_THRESHOLD}).",
                next_action="Transferir al médico tratante de inmediato. "
                "Finalizar la llamada con instrucciones claras.",
                domain=canon,
                source="numeric",
            )
        if numeric >= PAIN_YELLOW_THRESHOLD:
            return EscalationResult(
                severity=Severity.YELLOW,
                should_escalate=False,
                reason=f"Dolor reportado de {numeric}/10 — nivel moderado "
                f"(entre {PAIN_YELLOW_THRESHOLD} y "
                f"{PAIN_RED_THRESHOLD - 1}).",
                next_action="Reevaluar en la siguiente pregunta. Si el dolor "
                "persiste o empeora, escalar.",
                domain=canon,
                source="numeric",
            )
        # numeric < YELLOW threshold → fall through to green indicators

    if canon == "fiebre" and numeric is not None:
        if numeric >= TEMP_RED_THRESHOLD:
            return EscalationResult(
                severity=Severity.RED,
                should_escalate=True,
                reason=f"Temperatura reportada de {numeric}°C — umbral "
                f"crítico (>= {TEMP_RED_THRESHOLD}°C).",
                next_action="Transferir al médico tratante de inmediato. "
                "Finalizar la llamada con instrucciones claras.",
                domain=canon,
                source="numeric",
            )
        if numeric >= TEMP_YELLOW_THRESHOLD:
            return EscalationResult(
                severity=Severity.YELLOW,
                should_escalate=False,
                reason=f"Temperatura reportada de {numeric}°C — febrícula "
                f"(entre {TEMP_YELLOW_THRESHOLD} y "
                f"{TEMP_RED_THRESHOLD - 0.1}°C).",
                next_action="Reevaluar en la siguiente pregunta. Si la "
                "temperatura sube o aparecen escalofríos, escalar.",
                domain=canon,
                source="numeric",
            )

    # ---------- step 5: ambiguity / uncertainty ----------
    if any(phrase in lowered for phrase in AMBIGUITY_PHRASES):
        return EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="El paciente no dio una respuesta clara sobre este "
            "aspecto.",
            next_action="Solicitar aclaración: preguntar nuevamente si "
            "presenta algún síntoma específico en este aspecto.",
            domain=canon,
            source="ambig",
        )

    # ---------- step 6: yellow triggers ----------
    domain_yellow = YELLOW_TRIGGERS.get(canon, [])
    yellow_matches = _find_matches(lowered, domain_yellow, respect_negation=True)
    if yellow_matches:
        first = yellow_matches[0]
        return EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason=f"Síntoma amarillo en {canon}: '{first}'. "
            f"Requiere seguimiento cercano.",
            next_action="Continuar monitoreo. Si se repite un síntoma "
            "amarillo en este dominio o aparece otro síntoma nuevo, "
            "escalar.",
            domain=canon,
            source="rule",
        )

    # ---------- step 7: green indicators ----------
    domain_green = GREEN_INDICATORS.get(canon, [])
    green_matches = _find_matches(lowered, domain_green, respect_negation=False)
    if green_matches:
        return EscalationResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason=f"El paciente reporta evolución favorable en {canon}.",
            next_action="Continuar con el seguimiento normal.",
            domain=canon,
            source="rule",
        )

    # ---------- step 8: fallback — unable to classify ----------
    return EscalationResult(
        severity=Severity.YELLOW,
        should_escalate=False,
        reason=f"No se pudo clasificar la respuesta del paciente en el "
        f"dominio '{canon}'.",
        next_action="Solicitar aclaración al paciente sobre este aspecto "
        "específico de su recuperación.",
        domain=canon,
        source="incomplete",
    )
