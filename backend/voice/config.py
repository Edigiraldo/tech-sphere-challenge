"""Groq Whisper STT configuration.

Frozen, Spanish-first defaults for the Groq Whisper Large V3 transcription
endpoint.  Values are read from environment variables at initialisation time
so no code changes are needed for deployment tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Fixed model identifier
# ---------------------------------------------------------------------------
# Whisper Large V3 is the model exposed through Groq Cloud's free tier and
# recommended in the challenge's ``stack-tecnico.md`` for ultra-low-latency
# Spanish transcription.

_FIXED_WHISPER_MODEL: str = "whisper-large-v3"
_FIXED_LANGUAGE: str = "es"


@dataclass(frozen=True)
class GroqWhisperConfig:
    """Immutable configuration for the Groq Whisper STT adapter.

    The model is fixed to ``whisper-large-v3`` and the language to ``"es"``
    (Spanish-first).  Both are read-only constants — they cannot be changed
    through environment variables or constructor arguments.

    Callers that need a different STT provider must supply a different
    ``SttProvider`` implementation, not reconfigure this one.
    """

    # -- Model selection ---------------------------------------------------
    model: str = _FIXED_WHISPER_MODEL
    """Model identifier — always ``"whisper-large-v3"``."""

    language: str = _FIXED_LANGUAGE
    """Transcription language — always ``"es"`` (Spanish-first)."""

    api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    """Groq Cloud API key (required for Whisper transcriptions)."""

    # -- Generation parameters ---------------------------------------------
    temperature: float = field(
        default_factory=lambda: float(os.getenv("STT_TEMPERATURE", "0.0"))
    )
    """Sampling temperature.  0.0 → deterministic (recommended for clinical
    transcription).  Range: [0.0, 1.0]."""

    response_format: str = field(
        default_factory=lambda: os.getenv("STT_RESPONSE_FORMAT", "verbose_json")
    )
    """Whisper response format.  ``"verbose_json"`` includes segment-level
    timestamps and language detection confidence."""

    def __post_init__(self) -> None:
        if self.model != "whisper-large-v3":
            raise ValueError(
                f"model must be 'whisper-large-v3', got {self.model!r}"
            )
        if self.language != "es":
            raise ValueError(
                f"language must be 'es', got {self.language!r}"
            )
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                f"STT_TEMPERATURE must be in [0.0, 1.0], got {self.temperature}"
            )
        if self.response_format not in ("json", "verbose_json", "text"):
            raise ValueError(
                f"STT_RESPONSE_FORMAT must be one of "
                f"('json', 'verbose_json', 'text'), "
                f"got {self.response_format!r}"
            )
