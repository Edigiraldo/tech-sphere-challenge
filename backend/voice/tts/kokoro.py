"""Kokoro-82M TTS adapter with lazy loading and Spanish defaults.

All Kokoro-specific imports are deferred to ``synthesize()`` so that the
module is importable in test environments where the optional ``kokoro``
package is not installed.
"""

from __future__ import annotations

import logging
import struct
import time
from typing import TYPE_CHECKING

import numpy as np

from backend.voice.tts.config import TTSConfig
from backend.voice.tts.protocol import TTSProvider, TTSResult, TTSSynthesisError

if TYPE_CHECKING:
    pass  # Deferred imports — see _get_pipeline() below.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAV serialisation helpers
# ---------------------------------------------------------------------------


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 audio (-1..1) to WAV container bytes (16-bit PCM).

    The resulting byte string is a valid RIFF/WAV file that browsers can
    decode via ``<audio>`` or the Web Audio API.
    """
    # Clip to safe range and scale to 16-bit signed integer.
    audio_i16: np.ndarray = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    # Ensure mono (1-D).
    if audio_i16.ndim > 1:
        audio_i16 = audio_i16.flatten()

    n_samples: int = len(audio_i16)
    data_size: int = n_samples * 2  # 16-bit = 2 bytes / sample
    byte_rate: int = sample_rate * 2

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,            # PCM chunk size
        1,             # Audio format: PCM
        1,             # Channels: mono
        sample_rate,
        byte_rate,
        2,             # Block align: 2 bytes
        16,            # Bits per sample
        b"data",
        data_size,
    )
    return header + audio_i16.tobytes()


def _silent_wav_bytes(sample_rate: int, duration_ms: int = 100) -> bytes:
    """Return a short silent WAV payload.

    Used when the input text is empty — callers receive a valid, playable
    audio buffer instead of an invalid or truncated payload.
    """
    n_samples: int = max(1, int(sample_rate * duration_ms / 1000))
    silence: np.ndarray = np.zeros(n_samples, dtype=np.float32)
    return _float32_to_wav_bytes(silence, sample_rate)


# ---------------------------------------------------------------------------
# Kokoro adapter
# ---------------------------------------------------------------------------

# Kokoro sample rates known to be supported.  Other values may work with
# specific model versions but are not guaranteed.
_SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({24_000})


class KokoroAdapter:
    """Kokoro-82M TTS adapter with Spanish defaults.

    The optional ``kokoro`` package (``KPipeline``) is imported lazily on
    the first ``synthesize()`` call so that tests can import this module
    without installing Kokoro.

    Args:
        config: Default synthesis configuration.  May be overridden per-call
            via the ``config`` parameter of ``synthesize()``.
    """

    def __init__(self, config: TTSConfig | None = None) -> None:
        self._config: TTSConfig = config or TTSConfig()
        self._pipeline: object | None = None  # type: ignore[assignment]

    # -- Public API ----------------------------------------------------------

    def synthesize(
        self, text: str, config: TTSConfig | None = None
    ) -> TTSResult:
        """Synthesise Spanish text to WAV audio bytes.

        Empty *text* is handled gracefully — a valid short-silence WAV is
        returned rather than an invalid payload.

        Raises:
            TTSSynthesisError: When the ``kokoro`` package is not installed
                or synthesis fails for any other reason.
        """
        cfg: TTSConfig = config if config is not None else self._config

        _validate_config(cfg)

        # ------------------------------------------------------------------
        # Empty text → silent WAV (valid, playable, minimal duration)
        # ------------------------------------------------------------------
        if not text or not text.strip():
            logger.debug("Empty text — returning silent WAV.")
            silent_bytes: bytes = _silent_wav_bytes(cfg.sample_rate)
            return TTSResult(
                audio_bytes=silent_bytes,
                sample_rate=cfg.sample_rate,
                duration_ms=100.0,
                text=text,
                voice=cfg.voice,
            )

        pipeline = self._get_pipeline(cfg.lang_code)

        t_start = time.perf_counter()

        try:
            # Older Kokoro releases returned (audio, phonemes); current releases
            # yield Result objects whose output contains the audio tensor.
            pipeline_result = pipeline(text, voice=cfg.voice, speed=cfg.speed)  # type: ignore[call-arg]
            if isinstance(pipeline_result, tuple):
                audio = pipeline_result[0]
            else:
                segment_audio: list[np.ndarray] = []
                for segment in pipeline_result:
                    output = getattr(segment, "output", None)
                    audio_tensor = getattr(output, "audio", None)
                    if audio_tensor is not None:
                        segment_audio.append(_coerce_to_array(audio_tensor))
                if not segment_audio:
                    raise ValueError("Kokoro returned no audio segments")
                audio = np.concatenate(segment_audio)
        except Exception as exc:
            raise TTSSynthesisError(
                f"Kokoro synthesis failed for voice={cfg.voice!r}: {exc}",
                provider="kokoro",
            ) from exc

        elapsed_ms: float = (time.perf_counter() - t_start) * 1000.0

        # Kokoro may return a list of segments — concatenate if needed.
        audio_arr: np.ndarray = _coerce_to_array(audio)

        wav_bytes: bytes = _float32_to_wav_bytes(audio_arr, cfg.sample_rate)
        duration_ms: float = (len(audio_arr) / cfg.sample_rate) * 1000.0

        logger.debug(
            "Synthesised %d samples (%.0f ms) with voice=%r in %.0f ms",
            len(audio_arr),
            duration_ms,
            cfg.voice,
            elapsed_ms,
        )

        return TTSResult(
            audio_bytes=wav_bytes,
            sample_rate=cfg.sample_rate,
            duration_ms=duration_ms,
            text=text,
            voice=cfg.voice,
        )

    # -- Internals -----------------------------------------------------------

    def _get_pipeline(self, lang_code: str) -> object:
        """Lazily import and construct ``KPipeline`` for *lang_code*.

        The ``kokoro`` package is imported only when this method is first
        called, so the module remains importable even without it.
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            from kokoro import KPipeline  # noqa: WPS433
        except ImportError as exc:
            raise TTSSynthesisError(
                "The optional `kokoro` package is not installed. "
                "Install it with: pip install kokoro",
                provider="kokoro",
            ) from exc

        try:
            self._pipeline = KPipeline(lang_code=lang_code)
        except Exception as exc:
            raise TTSSynthesisError(
                f"Failed to initialise KPipeline(lang_code={lang_code!r}): {exc}",
                provider="kokoro",
            ) from exc

        return self._pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_config(cfg: TTSConfig) -> None:
    """Raise ``TTSSynthesisError`` when *cfg* is unsupported by Kokoro."""
    if cfg.sample_rate not in _SUPPORTED_SAMPLE_RATES:
        raise TTSSynthesisError(
            f"Unsupported sample_rate={cfg.sample_rate}. "
            f"Kokoro supports: {sorted(_SUPPORTED_SAMPLE_RATES)}",
            provider="kokoro",
        )


def _coerce_to_array(audio: object) -> np.ndarray:
    """Normalise Kokoro pipeline output to a 1-D float32 numpy array.

    Kokoro may return:
    - A single numpy array.
    - A list/sequence of numpy arrays (per-segment output).
    - A torch tensor (when the torch backend is active).
    """
    import numpy as np

    # List of segments → concatenate.
    if isinstance(audio, (list, tuple)):
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        segments: list[np.ndarray] = [
            _coerce_to_array(seg) for seg in audio
        ]
        return np.concatenate(segments)

    # Torch tensor → numpy.
    if hasattr(audio, "numpy"):
        audio = getattr(audio, "numpy")()

    arr: np.ndarray = np.asarray(audio, dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.flatten()

    return arr
