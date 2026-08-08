"""TTS interfaces — Protocol, result type and error class.

The ``TTSProvider`` Protocol defines the structural contract that every TTS
adapter must satisfy.  Callers in ``backend/conversation/`` receive a
provider at construction time and never depend on a concrete adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.voice.tts.config import TTSConfig


@dataclass
class TTSResult:
    """Normalised synthesis result suitable for browser playback.

    ``audio_bytes`` is always a valid WAV container (16-bit PCM mono) that
    a browser ``<audio>`` element or Web Audio API can decode without
    additional serverside processing.
    """

    audio_bytes: bytes
    """Normalised WAV bytes (RIFF container, 16-bit PCM, mono)."""

    sample_rate: int
    """Sample rate in Hz (e.g. 24 000)."""

    duration_ms: float
    """Synthesised audio duration in milliseconds."""

    text: str
    """The input text that was synthesised."""

    voice: str
    """Voice identifier used for synthesis."""

    format: str = "wav"
    """Audio container format — always ``"wav"``."""

    @property
    def duration_seconds(self) -> float:
        """Synthesis duration in seconds (convenience accessor)."""
        return self.duration_ms / 1000.0


class TTSSynthesisError(Exception):
    """Wraps provider failures during TTS synthesis.

    Raised when a TTS adapter cannot produce audio for the given input
    (network error, model loading failure, unsupported configuration, etc.).
    The original cause is chained via ``raise … from exc``.
    """

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message)
        self.provider = provider


@runtime_checkable
class TTSProvider(Protocol):
    """Structural interface for TTS adapters.

    Any object with a ``synthesize`` method matching this signature satisfies
    the protocol — no explicit subclassing or registration is required.
    """

    def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        """Synthesise spoken audio from Spanish text.

        Args:
            text: The text to speak (expected in Spanish).
            config: Voice and synthesis parameters.

        Returns:
            ``TTSResult`` with normalised WAV bytes suitable for the browser.

        Raises:
            TTSSynthesisError: When synthesis fails (provider unreachable,
                unsupported configuration, etc.).
        """
        ...
