"""Centralized prompt-injection and content-safety detection.

Single source of truth for injection patterns, Unicode/zero-width
normalisation, length bounds, output scanning, and document density
scanning.  All modules that currently duplicate ``_detect_injection``
(``adapter.py``, ``approval.py``) import from this module instead.

Modules
-------
* ``detect_input_injection(text)`` — entry-boundary scanner for user-facing
  text (STT transcriptions, RAG queries, patient input).
* ``detect_output_injection(text)`` — conservative output scanner that
  flags model-produced text containing instruction-injection markers.
* ``scan_document_density(text, filename)`` — ingestion-time density
  scanner that warns about documents with high concentration of
  injection-like patterns without rejecting legitimate clinical content.
* ``normalize_unicode(text)`` — strip zero-width characters and normalise
  Unicode to NFC for canonical pattern matching.
* ``safe_log_preview(text, max_chars=120)`` — produce a privacy-safe
  log snippet suitable for use in warnings and audit logs.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum safe length for user-facing queries (character count).
# Extremely long queries can be used for prompt-stuffing attacks.
_MAX_INPUT_LENGTH: int = 2000

# Maximum safe length for output scanning (character count).
_MAX_OUTPUT_LENGTH: int = 5000

# Zero-width and invisible Unicode categories / codepoints that can be
# used to bypass pattern-based filters.
_ZERO_WIDTH_CHARS: re.Pattern[str] = re.compile(
    "[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060\u2061\u2062\u2063\u2064"
    "\u180E\u034F\u061C\u115F\u1160\u17B4\u17B5\u2028\u2029\u202A\u202B"
    "\u202C\u202D\u202E\u2066\u2067\u2068\u2069\uFFF9\uFFFA\uFFFB]"
)

# ---------------------------------------------------------------------------
# Injection detection patterns (expanded)
# ---------------------------------------------------------------------------

# Each pattern category is a separate list for maintainability and
# precise logging of which category triggered.

_PATTERNS_ROLE_SWITCHING: list[re.Pattern[str]] = [
    # Instruction override / role-switching (Spanish + English)
    re.compile(
        r"(?:ignor[aeá]\s+(?:todas\s+)?(?:las\s+)?instrucciones\s+(?:anteriores|previas|de\s+arriba|de\s+seguridad)|"
        r"ignor[aeá]\s+(?:todas\s+)?(?:las\s+)?instrucciones|"
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions|"
        r"olvid[aeá]\s+(?:todas\s+)?(?:las\s+)?instrucciones|"
        r"forget\s+(?:all\s+)?(?:previous\s+)?instructions|"
        r"no\s+sigas\s+(?:las\s+)?instrucciones|"
        r"do\s+not\s+follow\s+(?:the\s+)?instructions|"
        r"desobedece\s+(?:las\s+)?instrucciones|"
        r"disregard\s+(?:all\s+)?(?:previous\s+)?instructions)",
        re.IGNORECASE,
    ),
    # Role impersonation / override
    re.compile(
        r"(?:ahora\s+(?:eres|act[uú]as\s+como)|"
        r"now\s+you\s+are|"
        r"a\s+partir\s+de\s+ahora\s+(?:eres|act[uú]as)|"
        r"from\s+now\s+on\s+you\s+are|"
        r"pretende[s]?\s+que\s+(?:eres|est[aá]s)|"
        r"pretend\s+(?:you\s+are|to\s+be)|"
        r"haz\s+como\s+si\s+(?:fueras|estuvieras)|"
        r"act\s+as\s+if\s+you\s+(?:are|were))",
        re.IGNORECASE,
    ),
]

_PATTERNS_SYSTEM_EXTRACTION: list[re.Pattern[str]] = [
    # System prompt extraction / leakage
    re.compile(
        r"(?:you\s+are\s+(?:a\s+)?(?:helpful\s+)?(?:AI\s+)?assistant|"
        r"cu[aá]l\s+es\s+(?:tu|el)\s+(?:rol|prop[oó]sito|objetivo|funci[oó]n|tarea)|"
        r"what\s+is\s+your\s+(?:role|purpose|objective|function|task)|"
        r"system\s*prompt|"
        r"system\s*message|"
        r"prompt\s+(?:del\s+)?sistema|"
        r"repit[eé]\s+(?:el\s+)?prompt|"
        r"repeat\s+(?:the\s+)?prompt|"
        r"mu[eé]strame\s+(?:tus\s+)?(?:instrucciones|reglas|prompt)|"
        r"show\s+me\s+your\s+(?:instructions|rules|prompt)|"
        r"dime\s+(?:tus\s+)?(?:instrucciones|reglas)|"
        r"tell\s+me\s+your\s+(?:instructions|rules)|"
        r"cu[aá]les\s+son\s+tus\s+(?:instrucciones|reglas|l[ií]mites|restricciones)|"
        r"what\s+are\s+your\s+(?:instructions|rules|limits|restrictions))",
        re.IGNORECASE,
    ),
    # "eres un asistente" — only match when combined with extraction
    # keywords (instrucciones, reglas, prompt, rol, etc.) or followed
    # by an injection continuation (sin, que, dime, etc.).  Bare
    # "eres un asistente" in polite clinical text is not blocked.
    re.compile(
        r"eres\s+un\s+asistente\s+(?:sin|que\s+(?:puede|ignora|no)|"
        r"y\s+(?:dime|cu[eé]ntame|mu[eé]strame|expl[ií]came|rep[ií]teme)|"
        r"dime|cu[eé]ntame|mu[eé]strame|expl[ií]came|rep[ií]teme|"
        r"di\b|cu[aá]l|qu[eé])",
        re.IGNORECASE,
    ),
]

_PATTERNS_TOOL_EXECUTION: list[re.Pattern[str]] = [
    # Request to execute code, commands, or call internal tools
    re.compile(
        r"(?:ejecut[aeá]\s+(?:el\s+)?(?:c[oó]digo|comando|funci[oó]n|script|programa)|"
        r"run\s+(?:the\s+)?(?:code|command|function|script|program)|"
        r"corr[aeá]\s+(?:el\s+)?(?:c[oó]digo|comando|funci[oó]n|script)|"
        r"sudo\b|\bexec\s*\(|"
        r"\\x[0-9a-fA-F]{2}|"  # hex encoding bypass
        r"chr\s*\(\s*\d+|"     # chr() encoding bypass
        r"eval\s*\(|"
        r"exec\s*\(|"
        r"__import__\s*\(|"
        r"os\.system\s*\(|"
        r"subprocess\.)",
        re.IGNORECASE,
    ),
    # Function/tool calling through delimiters
    re.compile(
        r"(?:llama\s+(?:a|la|al)\s+(?:funci[oó]n|herramienta|m[oó]dulo|api|endpoint)|"
        r"call\s+(?:the\s+)?(?:function|tool|module|api|endpoint)|"
        r"invoca\s+(?:la|el)\s+(?:funci[oó]n|herramienta|api)|"
        r"invoke\s+(?:the\s+)?(?:function|tool|api))",
        re.IGNORECASE,
    ),
]

_PATTERNS_ROLE_TAGS: list[re.Pattern[str]] = [
    # Structured injection via role tags (common LLM formats)
    re.compile(
        r'"role"\s*:\s*"(?:system|assistant|user|developer|tool)"|'
        r"<\|im_start\|>|<\|im_end\|>|"
        r"\[INST\]|\[/INST\]|"
        r"<\|system\|>|<\|user\|>|<\|assistant\|>|"
        r"\[system\]|\[/system\]|\[user\]|\[/user\]|\[assistant\]|\[/assistant\]|"
        r"<system>|</system>|<user>|</user>|<assistant>|</assistant>|"
        r"<\|begin_of_text\|>|<\|end_of_text\|>|"
        r"<s>|</s>|"
        r"\[SYS\]|\[/SYS\]|"
        r"<\|start_header_id\|>|<\|end_header_id\|>",
        re.IGNORECASE,
    ),
]

_PATTERNS_DELIMITER_ATTACK: list[re.Pattern[str]] = [
    # Delimiter-based injection ("--- BEGIN SYSTEM ---", etc.)
    re.compile(
        r"-{3,}\s*(?:begin|start|system|instrucciones?|prompt|"
        r"reglas?|contexto?|nuevo|new|override|sobrescribir)",
        re.IGNORECASE,
    ),
    # Triple-backtick injection
    re.compile(
        r"```(?:system|json|python|bash|sql|javascript|html)",
        re.IGNORECASE,
    ),
    # XML/CDATA wrapping
    re.compile(
        r"<!\[CDATA\[|]]>|"
        r"<\?xml|"
        r"<!(?:DOCTYPE|ENTITY)",
        re.IGNORECASE,
    ),
]

_PATTERNS_ENCODING_BYPASS: list[re.Pattern[str]] = [
    # Base64-encoded prompts
    re.compile(
        r"(?:[A-Za-z0-9+/]{40,}={0,2})",
    ),
    # URL-encoded injection payloads
    re.compile(
        r"(?:%[0-9A-Fa-f]{2}){20,}",
    ),
    # Unicode escape obfuscation
    re.compile(
        r"\\u[0-9a-fA-F]{4}",
    ),
    # ROT13 / Caesar hints
    re.compile(
        r"(?:(?:rot|caesar)[- _]?13|"
        r"decodif[ai]c[aeá]r?\s+(?:en\s+)?(?:base64|url|unicode)|"
        r"decode\s+(?:as\s+)?(?:base64|url|unicode))",
        re.IGNORECASE,
    ),
]

_PATTERNS_EXFILTRATION: list[re.Pattern[str]] = [
    # Attempts to exfiltrate internal data
    re.compile(
        r"(?:env[ií][aeá]\s+(?:la\s+)?(?:respuesta|informaci[oó]n|datos|resultado)|"
        r"send\s+(?:the\s+)?(?:response|information|data|result)|"
        r"reenv[ií][aeá]\s+(?:la\s+)?(?:conversaci[oó]n|historial)|"
        r"forward\s+(?:the\s+)?(?:conversation|history|transcript)|"
        r"guarda\s+(?:la|esta)\s+(?:conversaci[oó]n|respuesta|informaci[oó]n)|"
        r"save\s+(?:the|this)\s+(?:conversation|response|information)|"
        r"copia\s+(?:la|toda)\s+(?:conversaci[oó]n|informaci[oó]n)|"
        r"copy\s+(?:the|all)\s+(?:conversation|information)|"
        r"imprime\s+(?:toda\s+)?(?:la\s+)?(?:conversaci[oó]n|informaci[oó]n|memoria)|"
        r"print\s+(?:the|all)\s+(?:conversation|information|memory)|"
        r"exporta\s+(?:la\s+)?(?:conversaci[oó]n|informaci[oó]n)|"
        r"export\s+(?:the\s+)?(?:conversation|information))",
        re.IGNORECASE,
    ),
    # URL-based exfiltration — match URLs that contain data-exfiltration
    # keywords anywhere in the path.  Non-greedy match to avoid consuming
    # the keyword into the wildcard.
    re.compile(
        r"(?:https?://[^\s]*?(?:webhook|callback|collect|steal|exfil|log|dump)[^\s]*)",
        re.IGNORECASE,
    ),
]

_PATTERNS_PROMPT_INJECTION_KEYWORDS: list[re.Pattern[str]] = [
    # Explicit prompt injection / jailbreak terminology
    re.compile(
        r"(?:prompt\s*injection|"
        r"inyecci[oó]n\s+(?:de|del)\s*prompt|"
        r"jailbreak|"
        r"bypass\s+(?:las\s+)?(?:restricciones|filtros|reglas|límites)|"
        r"bypass\s+(?:the\s+)?(?:restrictions|filters|rules|limits)|"
        r"elud[aeá]s?\s+(?:las\s+)?(?:restricciones|reglas|filtros)|"
        r"elude[s]?\s+(?:the\s+)?(?:restrictions|rules|filters)|"
        r"romp[aeá]s?\s+(?:las\s+)?(?:restricciones|reglas)|"
        r"break\s+(?:the\s+)?(?:restrictions|rules)|"
        r"hack|exploit|"
        r"DAN\b|"
        r"developer\s*mode|"
        r"modo\s*desarrollador)",
        re.IGNORECASE,
    ),
]

# Aggregate all injection patterns with category labels
_InjectionCategory = tuple[str, list[re.Pattern[str]]]

_ALL_INJECTION_CATEGORIES: list[_InjectionCategory] = [
    ("role_switching", _PATTERNS_ROLE_SWITCHING),
    ("system_extraction", _PATTERNS_SYSTEM_EXTRACTION),
    ("tool_execution", _PATTERNS_TOOL_EXECUTION),
    ("role_tags", _PATTERNS_ROLE_TAGS),
    ("delimiter_attack", _PATTERNS_DELIMITER_ATTACK),
    ("encoding_bypass", _PATTERNS_ENCODING_BYPASS),
    ("exfiltration", _PATTERNS_EXFILTRATION),
    ("prompt_injection_keywords", _PATTERNS_PROMPT_INJECTION_KEYWORDS),
]

# ---------------------------------------------------------------------------
# Output-specific patterns (conservative — only clear instruction
# injection markers, not clinical Spanish)
# ---------------------------------------------------------------------------

_OUTPUT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Embedded role tags in output
    re.compile(
        r'"role"\s*:\s*"(?:system|assistant|user|developer|tool)"|'
        r"<\|im_start\|>|<\|im_end\|>|"
        r"\[INST\]|\[/INST\]|"
        r"<\|system\|>|<\|user\|>|<\|assistant\|>",
        re.IGNORECASE,
    ),
    # System prompt disclosure in output
    re.compile(
        r"(?:system\s*prompt[:=]|"
        r"prompt\s+(?:del\s+)?sistema[:=]|"
        r"instrucciones\s+del\s+sistema[:=]|"
        r"You are a\s+(?:helpful\s+)?(?:clinical\s+)?(?:AI\s+)?assistant)",
        re.IGNORECASE,
    ),
    # Code execution markers
    re.compile(
        r"(?:sudo\b|\bexec\s*\(|os\.system\s*\(|subprocess\.)",
    ),
    # JSON injection structures
    re.compile(
        r'"insufficient_knowledge"\s*:\s*false.*"cited_chunk_ids"\s*:',
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Document density patterns (warnings only, never reject)
# ---------------------------------------------------------------------------

# Patterns that trigger density warnings but never reject a legitimate
# clinical document.  These match content that overlaps with injection
# patterns when found at unusually high frequencies in ingested text.
_DOCUMENT_DENSITY_PATTERNS: list[re.Pattern[str]] = [
    # System prompt / assistant language in clinical text
    re.compile(
        r"(?:asistente\s+(?:cl[ií]nico|virtual|m[eé]dico)|"
        r"clinical\s+assistant|"
        r"system\s*prompt)",
        re.IGNORECASE,
    ),
    # Delimiter-like sequences that could be chunk artifacts
    re.compile(
        r"-{4,}",
    ),
    # Role-tag-like content in documents
    re.compile(
        r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]",
    ),
    # Base64-like strings in clinical text (unusual)
    re.compile(
        r"[A-Za-z0-9+/]{40,}={0,2}",
    ),
]

# Thresholds for document density scanning
_DOCUMENT_DENSITY_THRESHOLD_RATIO: float = 0.02  # 2% of lines matching
_DOCUMENT_DENSITY_MIN_MATCHES: int = 3  # Minimum matches before warning

# ---------------------------------------------------------------------------
# Spanish safe fallback message
# ---------------------------------------------------------------------------

_INJECTION_FALLBACK_ES: str = (
    "No puedo procesar esta consulta. Por favor, reformule su pregunta "
    "o comuníquese con su médico tratante para recibir orientación."
)

# ---------------------------------------------------------------------------
# Public API — data types
# ---------------------------------------------------------------------------


@dataclass
class InjectionResult:
    """Result of injection detection on a single input.

    Attributes
    ----------
    blocked : bool
        ``True`` when the input should be blocked (contains clear
        injection markers at the defined thresholds).
    reasons : list[str]
        Human-readable reasons for blocking (empty when not blocked).
    categories : list[str]
        Category labels for each triggered pattern (for audit logging).
    normalized_text : str
        The Unicode-normalised, zero-width-cleaned version of the
        input (for downstream processing).
    original_length : int
        Original character count before normalisation.
    """

    blocked: bool
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    normalized_text: str = ""
    original_length: int = 0


@dataclass
class DensityScanResult:
    """Result of document density scanning.

    Attributes
    ----------
    warning : bool
        ``True`` when density thresholds were exceeded — the document
        should be logged and audited but **never rejected**.
    match_count : int
        Number of lines matching density patterns.
    total_lines : int
        Total number of lines scanned.
    ratio : float
        Match ratio (match_count / total_lines).
    matched_categories : list[str]
        Pattern categories that were matched.
    """

    warning: bool
    match_count: int = 0
    total_lines: int = 0
    ratio: float = 0.0
    matched_categories: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API — helpers
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Normalise Unicode and strip zero-width / invisible characters.

    Steps:
    1. Strip zero-width and invisible characters (U+200B, U+FEFF,
       etc.) that can be used to bypass pattern-based filters.
    2. Normalise to NFC for canonical representation (composes
       combining diacritics into precomposed characters).

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Normalised text with invisible characters removed.
    """
    # Remove zero-width and invisible codepoints
    cleaned = _ZERO_WIDTH_CHARS.sub("", text)

    # NFC normalisation for canonical representation
    return unicodedata.normalize("NFC", cleaned)


def safe_log_preview(text: str, max_chars: int = 120) -> str:
    """Produce a privacy-safe log snippet.

    Truncates the input to *max_chars* and appends ``"…"`` when
    truncation occurred.  Strips trailing whitespace.

    Parameters
    ----------
    text : str
        Full text to preview.
    max_chars : int
        Maximum characters in the preview.

    Returns
    -------
    str
        Safe, truncated preview suitable for log messages.
    """
    if len(text) <= max_chars:
        return text
    return (text[:max_chars] + "…")


# ---------------------------------------------------------------------------
# Public API — input injection detection
# ---------------------------------------------------------------------------


def detect_input_injection(
    text: str,
    *,
    max_length: int = _MAX_INPUT_LENGTH,
) -> InjectionResult:
    """Detect prompt injection / jailbreak patterns in user-facing input.

    This is the entry-boundary scanner for all user-originated text:
    STT transcriptions, RAG queries, patient responses in the
    orchestrator.  It normalises Unicode, strips zero-width chars,
    checks length bounds, and scans all pattern categories.

    When the input is blocked, the caller must:
    - NOT pass the text to any LLM call.
    - NOT advance conversation state.
    - Log the event with the reasons and categories.
    - Return a safe Spanish fallback message.

    Parameters
    ----------
    text : str
        Raw user-facing input text.
    max_length : int
        Maximum allowed length in characters (default 2000).

    Returns
    -------
    InjectionResult
        Detection result with ``blocked``, ``reasons``, ``categories``,
        and the ``normalized_text``.
    """
    reasons: list[str] = []
    categories: list[str] = []

    original_length = len(text)

    # 1. Normalise Unicode and strip zero-width chars
    normalized = normalize_unicode(text)

    # 2. Length check (after normalisation — attackers may pad with
    #    zero-width characters to obscure length attacks)
    if len(normalized) > max_length:
        reasons.append(
            f"Input exceeds maximum length ({len(normalized)} > {max_length} "
            f"characters)."
        )
        categories.append("length_exceeded")
        return InjectionResult(
            blocked=True,
            reasons=reasons,
            categories=categories,
            normalized_text=normalized,
            original_length=original_length,
        )

    if not normalized.strip():
        return InjectionResult(
            blocked=False,
            reasons=[],
            categories=[],
            normalized_text=normalized,
            original_length=original_length,
        )

    # 3. Check for zero-width character presence (even if they were
    #    stripped, log that they were present for audit)
    if len(normalized) < original_length and original_length > 0:
        categories.append("zero_width_chars_removed")

    # 4. Scan all pattern categories
    for category_name, patterns in _ALL_INJECTION_CATEGORIES:
        for pattern in patterns:
            if pattern.search(normalized):
                reasons.append(
                    f"Injection pattern matched in category "
                    f"{category_name!r}."
                )
                categories.append(category_name)
                break  # One match per category is enough

    blocked = len(reasons) > 0

    if blocked:
        logger.warning(
            "Input injection detected: categories=%s, preview=%r",
            categories,
            safe_log_preview(normalized),
        )

    return InjectionResult(
        blocked=blocked,
        reasons=reasons,
        categories=categories,
        normalized_text=normalized,
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# Public API — output injection detection (conservative)
# ---------------------------------------------------------------------------


def detect_output_injection(
    text: str,
    *,
    max_length: int = _MAX_OUTPUT_LENGTH,
) -> InjectionResult:
    """Scan LLM-produced output for instruction-injection markers.

    This is a **conservative** scanner focused on clear structural
    injection signals (role tags, system prompt disclosure, code
    execution markers).  It deliberately avoids matching clinical
    Spanish text, medical terminology, or legitimate RAG citations.

    When output is flagged, the caller should replace it with a safe
    fallback or mark ``insufficient_knowledge=True``.

    Parameters
    ----------
    text : str
        LLM-produced output text.
    max_length : int
        Maximum allowed length (default 5000).

    Returns
    -------
    InjectionResult
        Detection result.
    """
    reasons: list[str] = []
    categories: list[str] = []

    original_length = len(text)
    normalized = normalize_unicode(text)

    # Length check on output
    if len(normalized) > max_length:
        reasons.append(
            f"Output exceeds maximum safe length "
            f"({len(normalized)} > {max_length} characters)."
        )
        categories.append("output_length_exceeded")

    # Scan output-specific patterns
    for pattern in _OUTPUT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            reasons.append("Output contains injection-like markers.")
            categories.append("output_injection")
            break  # One match is sufficient

    blocked = len(reasons) > 0

    if blocked:
        logger.warning(
            "Output injection detected: categories=%s, preview=%r",
            categories,
            safe_log_preview(normalized),
        )

    return InjectionResult(
        blocked=blocked,
        reasons=reasons,
        categories=categories,
        normalized_text=normalized,
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# Public API — document density scanning
# ---------------------------------------------------------------------------


def scan_document_density(
    text: str,
    filename: str = "",
) -> DensityScanResult:
    """Scan document text for injection-pattern density during ingestion.

    This scanner is **never used to block or reject** a clinical
    document — it only produces warnings for audit/logging.  Legitimate
    clinical PDFs should never trigger density warnings; high-density
    matches suggest a document may contain injection-like content.

    Parameters
    ----------
    text : str
        Full extracted text of the document (all pages concatenated).
    filename : str
        Source filename for log context.

    Returns
    -------
    DensityScanResult
        Scan result with warning flag, match counts, and categories.
    """
    normalized = normalize_unicode(text)
    lines = normalized.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return DensityScanResult(warning=False)

    matched_categories: list[str] = []
    match_count = 0

    for line in lines:
        if not line.strip():
            continue
        for pattern in _DOCUMENT_DENSITY_PATTERNS:
            if pattern.search(line):
                match_count += 1
                break  # Count each line once

    ratio = match_count / total_lines if total_lines > 0 else 0.0

    warning = (
        match_count >= _DOCUMENT_DENSITY_MIN_MATCHES
        and ratio >= _DOCUMENT_DENSITY_THRESHOLD_RATIO
    )

    if warning:
        # Determine which categories triggered
        for pattern in _DOCUMENT_DENSITY_PATTERNS:
            if pattern.search(normalized):
                matched_categories.append(pattern.pattern[:40])

        logger.warning(
            "Document density warning for %r: %d/%d lines matched (%.2f%%). "
            "Categories: %s. Document NOT rejected.",
            filename or "<unknown>",
            match_count,
            total_lines,
            ratio * 100,
            matched_categories or ["unknown"],
        )

    return DensityScanResult(
        warning=warning,
        match_count=match_count,
        total_lines=total_lines,
        ratio=ratio,
        matched_categories=matched_categories,
    )


# ---------------------------------------------------------------------------
# Public API — convenience: get the injection fallback message
# ---------------------------------------------------------------------------


def get_injection_fallback() -> str:
    """Return the safe Spanish fallback message used when input is blocked.

    Returns
    -------
    str
        Safe fallback in Spanish.
    """
    return _INJECTION_FALLBACK_ES
