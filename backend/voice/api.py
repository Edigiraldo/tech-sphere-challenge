"""Public STT API — dependency-injection entry point.

Callers (conversation orchestrator, WebSocket handler, etc.) use
``transcribe_audio()`` without knowing which STT provider is active.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from backend.voice.models import (
    SttAudioError,
    SttConfigError,
    SttProviderError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SttDependency = Callable[[bytes], Awaitable[TranscriptionResult]]
"""Type alias for an injectable STT function.

Accepts raw audio bytes and returns a ``TranscriptionResult``.
Matches the signature of ``GroqWhisperProvider.transcribe`` — or any
``SttProvider`` implementation — bound to a specific instance.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def transcribe_audio(
    audio_data: bytes,
    stt: SttDependency,
) -> TranscriptionResult:
    """Transcribe Spanish audio using the injected STT provider.

    This is the single public entry point for speech-to-text in the
    application.  Callers — conversation orchestrator, WebSocket handler,
    test harness — inject the concrete provider so they never import a
    specific adapter.

    Args:
        audio_data: Raw audio bytes.
        stt: Injectable STT function bound to a provider instance
            (e.g. ``GroqWhisperProvider(config).transcribe``).

    Returns:
        A normalised ``TranscriptionResult``.

    Raises:
        SttAudioError: If ``audio_data`` is empty or too large.
        SttConfigError: If the provider is misconfigured.
        SttProviderError: If the provider API fails.
    """
    logger.debug(
        "transcribe_audio called with %d bytes of audio data", len(audio_data)
    )

    if not callable(stt):
        raise TypeError(
            "El argumento 'stt' debe ser un callable async "
            f"(recibido: {type(stt).__name__})"
        )

    try:
        result = await stt(audio_data)
    except (SttAudioError, SttConfigError, SttProviderError):
        raise
    except Exception as exc:
        # Catch-all for unexpected errors from the dependency.
        # Convert to SttProviderError so callers never see raw exceptions.
        logger.exception(
            "Unexpected error from injected STT dependency"
        )
        raise SttProviderError(
            f"Error inesperado durante la transcripción: {exc}"
        ) from exc

    if not isinstance(result, TranscriptionResult):
        raise TypeError(
            f"El proveedor STT debe devolver TranscriptionResult, "
            f"recibido: {type(result).__name__}"
        )

    return result
