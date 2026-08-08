"""Groq Whisper STT adapter.

Transcribes Spanish speech to text via the Groq Cloud Whisper Large V3
endpoint.  Accepts raw bytes or a file-like object and returns a normalised
``TranscriptionResult``.

All provider-specific details — API shape, error codes, chunk handling —
are encapsulated in this module.  Callers never import ``groq`` directly.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from backend.voice.config import GroqWhisperConfig
from backend.voice.models import (
    SttAudioError,
    SttConfigError,
    SttProviderError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Groq Whisper API limits (documented at https://console.groq.com/docs/speech-to-text)
_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
_MIN_FILE_SIZE_BYTES: int = 1  # at least one byte

# Recognised Groq error patterns for mapping provider errors to our hierarchy.
# Groq returns HTTP 4xx/5xx with a JSON body containing an ``error`` object.
# We map the most common ones to SttProviderError with descriptive messages.
_RATE_LIMIT_MESSAGE_PREFIX: str = "Rate limit"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_audio(audio_data: bytes) -> None:
    """Check that audio data is non-empty and within Groq's size limits.

    Raises:
        SttAudioError: If the audio data is empty, too small, or exceeds
            the 25 MB file-size limit.
    """
    if not isinstance(audio_data, bytes):
        raise SttAudioError(
            f"audio_data debe ser bytes, se recibió {type(audio_data).__name__}"
        )
    if len(audio_data) < _MIN_FILE_SIZE_BYTES:
        raise SttAudioError(
            "El audio está vacío (0 bytes). Proporcione datos de audio válidos."
        )
    if len(audio_data) > _MAX_FILE_SIZE_BYTES:
        raise SttAudioError(
            f"El audio excede el límite de {_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB "
            f"permitido por Groq (tamaño real: {len(audio_data) / (1024 * 1024):.1f} MB)."
        )


def _validate_config(config: GroqWhisperConfig) -> None:
    """Check that the Groq configuration is usable.

    Raises:
        SttConfigError: If the API key is missing.
    """
    if not config.api_key or not config.api_key.strip():
        raise SttConfigError(
            "GROQ_API_KEY no está configurada. "
            "Defina la variable de entorno GROQ_API_KEY con su clave "
            "de Groq Cloud (disponible en https://console.groq.com/keys)."
        )


# ---------------------------------------------------------------------------
# Provider error mapping
# ---------------------------------------------------------------------------


def _map_groq_error(exc: Exception) -> SttProviderError:
    """Convert a raw Groq SDK or HTTP exception into an SttProviderError.

    Inspects the exception chain for known Groq error types and produces a
    descriptive Spanish error message so that upstream layers can safely
    log or relay the error without leaking internal provider details.
    """
    msg = str(exc)

    # Groq SDK wraps HTTP errors in its own exception types (e.g. groq.APIError,
    # groq.RateLimitError, groq.APIConnectionError, groq.AuthenticationError).
    # We check the class name hierarchy so we don't need to import groq here.
    exc_class_name = type(exc).__name__
    module_name = type(exc).__module__

    # Walk the MRO to catch groq-specific types without importing groq
    groq_names: set[str] = set()
    for cls in type(exc).__mro__:
        groq_names.add(cls.__name__)

    if (
        "RateLimitError" in groq_names
        or "rate_limit" in msg.lower()
        or "rate limit" in msg.lower()
    ):
        return SttProviderError(
            "Límite de tasa de Groq excedido. "
            "Espere unos segundos antes de reintentar la transcripción."
        )
    if "AuthenticationError" in groq_names or "401" in msg:
        return SttProviderError(
            "Error de autenticación con Groq. "
            "Verifique que su GROQ_API_KEY sea válida."
        )
    if "BadRequestError" in groq_names or "400" in msg:
        return SttProviderError(
            f"Groq rechazó la solicitud de transcripción: {msg}"
        )
    if "APIConnectionError" in groq_names or "ConnectionError" in groq_names:
        return SttProviderError(
            "No se pudo conectar con la API de Groq. "
            "Verifique su conexión de red."
        )
    if "APIError" in groq_names or "InternalServerError" in groq_names:
        return SttProviderError(
            f"Error interno de la API de Groq: {msg}"
        )

    return SttProviderError(
        f"Error inesperado del proveedor de STT (Groq): {msg}"
    )


# ---------------------------------------------------------------------------
# Groq adapter implementation
# ---------------------------------------------------------------------------


class GroqWhisperProvider:
    """Speech-to-text adapter backed by Groq Cloud Whisper Large V3.

    Instantiate with a ``GroqWhisperConfig``, then call ``transcribe()`` with
    raw audio bytes.  The provider normalises the Groq response into a
    ``TranscriptionResult`` and maps all provider errors into the shared
    ``SttError`` hierarchy.

    Usage::

        config = GroqWhisperConfig()
        provider = GroqWhisperProvider(config)
        result = await provider.transcribe(audio_bytes)
    """

    def __init__(self, config: GroqWhisperConfig) -> None:
        """Initialise the Groq Whisper provider.

        Args:
            config: Frozen Groq Whisper configuration.  The API key, model,
                language, temperature, and response format are read from
                this object at transcription time (not at construction time)
                so that callers can rely on the frozen dataclass semantics.
        """
        self._config = config

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe Spanish speech to text via Groq Whisper Large V3.

        Args:
            audio_data: Raw audio bytes (WAV, MP3, FLAC, OGG, M4A, or OPUS).
                Must be between 1 byte and 25 MB.

        Returns:
            A ``TranscriptionResult`` with the transcribed text, detected
            language, audio duration, and provider metadata.

        Raises:
            SttAudioError: If ``audio_data`` is empty or exceeds size limits.
            SttConfigError: If the API key is missing.
            SttProviderError: If the Groq API is unreachable or returns an error.
        """
        # 1. Guard clauses — fail fast
        _validate_audio(audio_data)
        _validate_config(self._config)

        logger.info(
            "Transcribing %d bytes with %s (lang=%s, temp=%.2f) …",
            len(audio_data),
            self._config.model,
            self._config.language,
            self._config.temperature,
        )

        # 2. Call Groq API
        try:
            raw = await self._call_groq(audio_data)
        except SttProviderError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during Groq transcription")
            raise _map_groq_error(exc) from exc

        # 3. Normalise response
        text = str(raw.get("text", ""))
        detected_lang = str(raw.get("language", self._config.language))
        duration = float(raw.get("duration", 0.0))

        # Build metadata dict with segment info (if verbose_json)
        meta: dict[str, Any] = {}
        segments = raw.get("segments")
        if isinstance(segments, list):
            meta["segments"] = segments
        if "words" in raw:
            meta["words"] = raw["words"]

        logger.info(
            "Transcription complete: %d chars, lang=%s, duration=%.1fs",
            len(text),
            detected_lang,
            duration,
        )

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            duration_seconds=duration,
            model=self._config.model,
            metadata=meta,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _call_groq(self, audio_data: bytes) -> dict[str, Any]:
        """Invoke the Groq Whisper API and return the parsed response dict.

        Wraps the raw audio bytes in a file-like object for the Groq SDK.

        Args:
            audio_data: Validated audio bytes.

        Returns:
            Parsed JSON response from the Groq Whisper endpoint.

        Raises:
            SttProviderError: If the Groq API returns an error.
        """
        try:
            import groq  # noqa: WPS433
        except ImportError as exc:
            raise SttProviderError(
                "El paquete 'groq' no está instalado. "
                "Ejecute: pip install groq"
            ) from exc

        client = groq.AsyncGroq(api_key=self._config.api_key)

        # Groq SDK expects a file-like object (not raw bytes)
        audio_file = BytesIO(audio_data)
        audio_file.name = "audio.wav"

        try:
            transcription = await client.audio.transcriptions.create(
                model=self._config.model,
                file=audio_file,
                language=self._config.language,
                temperature=self._config.temperature,
                response_format=self._config.response_format,
            )
        except Exception as exc:
            raise _map_groq_error(exc) from exc

        # Groq SDK returns a pydantic model or dict depending on
        # ``response_format``.  Normalise to a plain dict.
        if hasattr(transcription, "model_dump"):
            return transcription.model_dump()
        if hasattr(transcription, "dict"):
            return transcription.dict()
        if isinstance(transcription, dict):
            return transcription

        # Fallback: extract known attributes
        result: dict[str, Any] = {}
        for attr in ("text", "language", "duration", "segments", "words"):
            if hasattr(transcription, attr):
                result[attr] = getattr(transcription, attr)
        return result
