"""Voice adapters — STT and TTS provider interfaces.

Provides a typed STT provider Protocol, a frozen Spanish-first Groq Whisper
configuration, error hierarchy, and the public ``transcribe_audio``
dependency-injection API.

TTS adapters are planned for a later phase.
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
