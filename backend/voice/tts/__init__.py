"""TTS (text-to-speech) foundation.

Provides a ``TTSProvider`` Protocol, typed configuration (``TTSConfig``),
normalised ``TTSResult``, ``TTSSynthesisError``, and a Kokoro-82M adapter
with Spanish defaults.
"""

from backend.voice.tts.config import TTSConfig
from backend.voice.tts.kokoro import KokoroAdapter
from backend.voice.tts.protocol import TTSProvider, TTSResult, TTSSynthesisError

__all__ = [
    "KokoroAdapter",
    "TTSConfig",
    "TTSProvider",
    "TTSResult",
    "TTSSynthesisError",
]
