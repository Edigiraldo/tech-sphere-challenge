"""Provider startup wiring.

Called once during FastAPI application startup to construct concrete STT and TTS
adapters and inject them into the API endpoint layer.  Construction errors are
caught and logged without crashing application startup so that the health
endpoint and other non-voice routes remain available even when a provider is
misconfigured.
"""

from __future__ import annotations

import logging

from backend.voice.api import SttDependency
from backend.voice.config import GroqWhisperConfig
from backend.voice.groq import GroqWhisperProvider
from backend.voice.tts.config import TTSConfig
from backend.voice.tts.kokoro import KokoroAdapter

logger = logging.getLogger(__name__)


def configure_providers() -> None:
    """Construct STT and TTS adapters and wire them into the API layer.

    Construction of each provider is wrapped in its own try/except so that a
    failure in one does not prevent the other from being wired — and neither
    failure crashes application startup.

    Providers that fail to construct are left as ``None`` in the API layer,
    which causes the corresponding endpoints to return a clear 500 / 502 error
    at request time rather than a cryptic internal exception.
    """

    # -----------------------------------------------------------------------
    # STT: Groq Whisper Large V3
    # -----------------------------------------------------------------------
    _wired_stt: SttDependency | None = None

    try:
        from backend.api.calls import set_stt  # noqa: WPS433

        stt_config = GroqWhisperConfig()
        stt_provider = GroqWhisperProvider(stt_config)
        # Inject the bound transcribe method as the STT dependency.
        set_stt(stt_provider.transcribe)
        _wired_stt = stt_provider.transcribe
        logger.info(
            "STT provider wired: GroqWhisperProvider (model=%s, lang=%s)",
            stt_config.model,
            stt_config.language,
        )
    except Exception:
        logger.exception("Failed to initialise STT provider — voice calls will fail.")

    # -----------------------------------------------------------------------
    # TTS: Kokoro-82M
    # -----------------------------------------------------------------------

    _wired_tts: bool = False

    try:
        from backend.api.calls import set_tts  # noqa: WPS433

        tts_config = TTSConfig()
        tts_adapter = KokoroAdapter(tts_config)
        set_tts(tts_adapter, tts_config)
        _wired_tts = True
        logger.info(
            "TTS provider wired: KokoroAdapter (voice=%s, lang_code=%s)",
            tts_config.voice,
            tts_config.lang_code,
        )
    except Exception:
        logger.exception("Failed to initialise TTS provider — voice calls will fail.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _stt_ready = _wired_stt is not None

    if _stt_ready and _wired_tts:
        logger.info("STT and TTS providers ready for calls.")
    elif _stt_ready:
        logger.info("STT provider ready for calls; TTS provider unavailable.")
    elif _wired_tts:
        logger.info("TTS provider ready for calls; STT provider unavailable.")
    else:
        logger.warning(
            "No voice providers were wired — "
            "voice endpoints will return errors until providers are configured."
        )
