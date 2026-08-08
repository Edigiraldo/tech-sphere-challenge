"""Voice adapters — STT and TTS provider abstractions.

This package exposes the common interfaces (Protocols, configuration, and
result types) shared across voice I/O adapters so that callers such as
``backend/conversation/`` never depend on a specific STT or TTS provider.
"""

from backend.voice.api import SttDependency, transcribe_audio
from backend.voice.config import GroqWhisperConfig
from backend.voice.models import TranscriptionResult
from backend.voice.provider import SttProvider

__all__ = [
    "SttProvider",
    "SttDependency",
    "GroqWhisperConfig",
    "TranscriptionResult",
    "transcribe_audio",
]
