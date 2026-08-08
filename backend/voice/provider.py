"""STT provider Protocol.

Defines the contract that every STT implementation must satisfy.  Callers
depend on this Protocol, never on a concrete adapter.
"""

from __future__ import annotations

from typing import Protocol

from backend.voice.models import TranscriptionResult


class SttProvider(Protocol):
    """Structural contract for speech-to-text adapters.

    Every concrete STT adapter must implement this interface.  The
    ``transcribe`` method accepts raw audio bytes and returns a normalised
    ``TranscriptionResult``.

    Adapters that require provider-specific initialisation (API key, model
    selection, etc.) should accept those parameters at construction time.
    """

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe raw audio bytes to text.

        Args:
            audio_data: Raw audio bytes in the format expected by the provider.

        Returns:
            A ``TranscriptionResult`` with the transcribed Spanish text.

        Raises:
            SttAudioError: If ``audio_data`` is empty, too short, or unreadable.
            SttProviderError: If the provider API is unreachable or returns
                an error.
            SttConfigError: If the provider is misconfigured (missing API key,
                invalid model).
        """
        ...
