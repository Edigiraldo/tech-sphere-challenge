"""TTS configuration — voice, language, and synthesis parameters.

Defaults are tuned for Colombian Spanish via the Kokoro-82M adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Kokoro-82M default voices for Spanish
# ---------------------------------------------------------------------------
# ``es_002`` is the primary Spanish female voice shipped with Kokoro-82M.
# Alternative Spanish voices (e.g. ``es_001``) may be available depending on
# the installed voice-pack version.
#
# Lang code ``"e"`` maps to Spanish (español) in Kokoro's lang-to-voice
# dispatch table.  Sample rate 24 000 Hz is Kokoro's native output rate.

_DEFAULT_VOICE: str = "es_002"
_DEFAULT_LANG_CODE: str = "e"
_DEFAULT_SAMPLE_RATE: int = 24_000
_DEFAULT_SPEED: float = 1.0


@dataclass(frozen=True)
class TTSConfig:
    """Immutable configuration for a TTS provider.

    All fields have sensible defaults for Colombian Spanish via Kokoro-82M.
    """

    voice: str = _DEFAULT_VOICE
    """Kokoro voice identifier (e.g. ``"es_002"``)."""

    lang_code: str = _DEFAULT_LANG_CODE
    """Kokoro language code (``"e"`` = Spanish)."""

    speed: float = _DEFAULT_SPEED
    """Speech speed multiplier (1.0 = normal, > 1 faster, < 1 slower)."""

    sample_rate: int = _DEFAULT_SAMPLE_RATE
    """Audio sample rate in Hz (Kokoro native: 24 000)."""

    # --- Validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )
        if self.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.speed}")
        if not self.voice.strip():
            raise ValueError("voice must be a non-empty string")
        if not self.lang_code.strip():
            raise ValueError("lang_code must be a non-empty string")

    # --- Helpers ------------------------------------------------------------

    def with_overrides(
        self,
        *,
        voice: str | None = None,
        lang_code: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> TTSConfig:
        """Return a new ``TTSConfig`` with the given fields replaced."""
        return TTSConfig(
            voice=voice if voice is not None else self.voice,
            lang_code=lang_code if lang_code is not None else self.lang_code,
            speed=speed if speed is not None else self.speed,
            sample_rate=(
                sample_rate if sample_rate is not None else self.sample_rate
            ),
        )
