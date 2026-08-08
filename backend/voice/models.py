"""Shared STT models — normalised transcription result and error hierarchy.

These types are provider-agnostic: every STT adapter returns a
``TranscriptionResult`` and raises only the exceptions defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Normalised transcription result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionResult:
    """Normalised, provider-agnostic transcription result.

    Attributes:
        text: The transcribed text.  Empty string when no speech was detected.
        language: ISO 639-1 language code (default ``"es"`` for Spanish-first).
        duration_seconds: Duration of the processed audio in seconds.
        model: Provider-specific model identifier used for transcription.
        metadata: Provider-specific diagnostics (e.g. segment-level timestamps,
            confidence scores).  Callers must not rely on a particular shape.
    """

    text: str
    language: str = "es"
    duration_seconds: float = 0.0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class SttError(Exception):
    """Base exception for all STT-related errors."""


class SttConfigError(SttError):
    """Raised when the STT provider is misconfigured (missing API key, invalid
    model identifier, etc.)."""


class SttProviderError(SttError):
    """Raised when the STT provider API returns an error or is unreachable
    (network, rate-limit, authentication, server 5xx)."""


class SttAudioError(SttError):
    """Raised when the supplied audio data is empty, unreadable, or in an
    unsupported format."""
