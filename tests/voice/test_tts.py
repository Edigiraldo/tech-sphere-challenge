"""Unit tests for the TTS foundation — config, protocol, Kokoro adapter.

All Kokoro ``KPipeline`` calls are mocked so the tests execute without
installing the optional ``kokoro`` package.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.voice.tts.config import TTSConfig
from backend.voice.tts.kokoro import (
    KokoroAdapter,
    _coerce_to_array,
    _float32_to_wav_bytes,
    _silent_wav_bytes,
    _validate_config,
)
from backend.voice.tts.protocol import TTSProvider, TTSResult, TTSSynthesisError


# ============================================================================
# TTSConfig
# ============================================================================


class TestTTSConfig:
    """Configuration dataclass — defaults, validation, overrides."""

    def test_defaults(self) -> None:
        cfg = TTSConfig()
        assert cfg.voice == "es_002"
        assert cfg.lang_code == "e"
        assert cfg.speed == 1.0
        assert cfg.sample_rate == 24_000

    def test_explicit_values(self) -> None:
        cfg = TTSConfig(voice="es_001", speed=1.2, sample_rate=24_000)
        assert cfg.voice == "es_001"
        assert cfg.speed == 1.2

    def test_frozen(self) -> None:
        cfg = TTSConfig()
        with pytest.raises(Exception):
            cfg.speed = 2.0  # type: ignore[misc]

    # --- Validation ---------------------------------------------------------

    def test_rejects_zero_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            TTSConfig(sample_rate=0)

    def test_rejects_negative_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            TTSConfig(sample_rate=-1)

    def test_rejects_zero_speed(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            TTSConfig(speed=0.0)

    def test_rejects_negative_speed(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            TTSConfig(speed=-0.5)

    def test_rejects_empty_voice(self) -> None:
        with pytest.raises(ValueError, match="voice"):
            TTSConfig(voice="")

    def test_rejects_whitespace_voice(self) -> None:
        with pytest.raises(ValueError, match="voice"):
            TTSConfig(voice="   ")

    def test_rejects_empty_lang_code(self) -> None:
        with pytest.raises(ValueError, match="lang_code"):
            TTSConfig(lang_code="")

    # --- with_overrides -----------------------------------------------------

    def test_with_overrides_returns_new_instance(self) -> None:
        cfg = TTSConfig()
        cfg2 = cfg.with_overrides(speed=1.5)
        assert cfg2 is not cfg
        assert cfg.speed == 1.0
        assert cfg2.speed == 1.5

    def test_with_overrides_keeps_unrelated_fields(self) -> None:
        cfg = TTSConfig(voice="es_001", speed=0.9)
        cfg2 = cfg.with_overrides(lang_code="a")
        assert cfg2.voice == "es_001"
        assert cfg2.lang_code == "a"
        assert cfg2.speed == 0.9
        assert cfg2.sample_rate == 24_000

    def test_with_overrides_none_noop(self) -> None:
        cfg = TTSConfig(voice="es_002")
        cfg2 = cfg.with_overrides()
        assert cfg2.voice == cfg.voice
        assert cfg2.speed == cfg.speed


# ============================================================================
# TTSProvider Protocol
# ============================================================================


class TestTTSProviderProtocol:
    """Structural subtyping — any class with ``synthesize`` matches."""

    def test_custom_provider_satisfies_protocol(self) -> None:
        class FakeProvider:
            def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
                return TTSResult(
                    audio_bytes=b"fake",
                    sample_rate=24000,
                    duration_ms=0,
                    text=text,
                    voice=config.voice,
                )

        provider: TTSProvider = FakeProvider()
        result = provider.synthesize("hola", TTSConfig())
        assert isinstance(result, TTSResult)
        assert result.text == "hola"

    def test_kokoro_adapter_satisfies_protocol(self) -> None:
        """KokoroAdapter structurally matches TTSProvider."""
        # Confirm the adapter has the required method signature.
        assert hasattr(KokoroAdapter, "synthesize")
        # Instance check via structural subtyping.
        adapter: TTSProvider = KokoroAdapter()
        assert isinstance(adapter, TTSProvider)


# ============================================================================
# TTSResult
# ============================================================================


class TestTTSResult:
    """Result dataclass — attributes and helpers."""

    @pytest.fixture
    def result(self) -> TTSResult:
        return TTSResult(
            audio_bytes=b"RIFF...",
            sample_rate=24000,
            duration_ms=1500.0,
            text="Hola paciente",
            voice="es_002",
        )

    def test_basic_attributes(self, result: TTSResult) -> None:
        assert result.audio_bytes == b"RIFF..."
        assert result.sample_rate == 24000
        assert result.duration_ms == 1500.0
        assert result.text == "Hola paciente"
        assert result.voice == "es_002"
        assert result.format == "wav"

    def test_duration_seconds(self, result: TTSResult) -> None:
        assert result.duration_seconds == 1.5

    def test_empty_text_result_is_valid(self) -> None:
        """Empty text still produces a valid result with bytes."""
        result = TTSResult(
            audio_bytes=b"\x00" * 44,
            sample_rate=24000,
            duration_ms=100.0,
            text="",
            voice="es_002",
        )
        assert len(result.audio_bytes) > 0
        assert result.duration_ms > 0


# ============================================================================
# TTSSynthesisError
# ============================================================================


class TestTTSSynthesisError:
    """Error wrapping provider failures."""

    def test_basic_message(self) -> None:
        err = TTSSynthesisError("synthesis failed", provider="kokoro")
        assert str(err) == "synthesis failed"
        assert err.provider == "kokoro"

    def test_with_cause(self) -> None:
        cause = RuntimeError("pipeline crashed")
        err = TTSSynthesisError("wrapped", provider="kokoro")
        err.__cause__ = cause
        assert isinstance(err.__cause__, RuntimeError)


# ============================================================================
# WAV serialisation helpers
# ============================================================================


class TestWavSerialisation:
    """Normalised WAV bytes suitable for browser playback."""

    def test_float32_to_wav_basic(self) -> None:
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = _float32_to_wav_bytes(audio, 24000)

        # WAV header: 44 bytes + 5 samples * 2 bytes = 54 bytes total.
        assert len(wav) == 44 + 10

        # Verify RIFF header.
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_float32_to_wav_stereo_flattens(self) -> None:
        """Multi-dimensional audio is flattened to mono."""
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        wav = _float32_to_wav_bytes(audio, 24000)
        # 44 bytes header + 4 samples * 2 bytes.
        assert len(wav) == 44 + 8

    def test_float32_to_wav_clips(self) -> None:
        audio = np.array([-2.0, 2.0, 0.0], dtype=np.float32)
        wav = _float32_to_wav_bytes(audio, 24000)

        # Parse PCM data from the WAV (offset 44, signed 16-bit LE).
        pcm_bytes = wav[44:]
        samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)

        # Clipped to int16 range.  Standard float→int16 mapping:
        # clip [-1, 1] → multiply by 32767.  -1.0 yields -32767 and
        # +1.0 yields +32767 (symmetric range).  The value -32768 is
        # the minimum int16 but float normalisation to [-1, 1] never
        # reaches it in practice; asymmetry of 1 LSB is standard.
        assert samples[0] == -32767
        assert samples[1] == 32767
        assert samples[2] == 0

    def test_float32_to_wav_empty_audio(self) -> None:
        audio = np.array([], dtype=np.float32)
        wav = _float32_to_wav_bytes(audio, 24000)

        # 44 bytes header + 0 samples = 44 bytes.
        assert len(wav) == 44
        assert wav[:4] == b"RIFF"

    # --- silent WAV ---------------------------------------------------------

    def test_silent_wav_produces_valid_riff(self) -> None:
        wav = _silent_wav_bytes(24000, duration_ms=100)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

        # 24000 * 0.1 = 2400 samples → 44 + 4800 bytes.
        assert len(wav) == 44 + 4800

    def test_silent_wav_all_samples_zero(self) -> None:
        wav = _silent_wav_bytes(24000, duration_ms=50)
        pcm_bytes = wav[44:]
        samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
        assert all(s == 0 for s in samples)

    def test_silent_wav_minimum_one_sample(self) -> None:
        """Even with very short duration, at least 1 sample is produced."""
        wav = _silent_wav_bytes(24000, duration_ms=0)
        pcm_bytes = wav[44:]
        assert len(pcm_bytes) >= 2  # at least 1 sample = 2 bytes


# ============================================================================
# _validate_config
# ============================================================================


class TestValidateConfig:
    def test_supported_rate_passes(self) -> None:
        _validate_config(TTSConfig(sample_rate=24_000))

    def test_unsupported_rate_raises(self) -> None:
        with pytest.raises(TTSSynthesisError, match="Unsupported sample_rate"):
            _validate_config(TTSConfig(sample_rate=44100))


# ============================================================================
# _coerce_to_array
# ============================================================================


class TestCoerceToArray:
    def test_single_ndarray(self) -> None:
        arr = np.array([0.1, 0.2], dtype=np.float32)
        result = _coerce_to_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_list_of_ndarrays_concat(self) -> None:
        a1 = np.array([0.1, 0.2], dtype=np.float32)
        a2 = np.array([0.3], dtype=np.float32)
        result = _coerce_to_array([a1, a2])
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_empty_list(self) -> None:
        result = _coerce_to_array([])
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_torch_tensor(self) -> None:
        """Simulate a torch-like tensor with a .numpy() method."""
        class FakeTensor:
            def numpy(self) -> np.ndarray:
                return np.array([0.5, -0.5], dtype=np.float32)

        result = _coerce_to_array(FakeTensor())
        np.testing.assert_array_equal(result, np.array([0.5, -0.5], dtype=np.float32))

    def test_multidimensional_flattens(self) -> None:
        arr = np.array([[0.1], [0.2]], dtype=np.float32)
        result = _coerce_to_array(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, np.array([0.1, 0.2], dtype=np.float32))


# ============================================================================
# KokoroAdapter
# ============================================================================


class TestKokoroAdapter:
    """Kokoro-82M adapter — construction, empty text, mocked synthesis."""

    def test_default_config(self) -> None:
        adapter = KokoroAdapter()
        assert isinstance(adapter, TTSProvider)

    def test_custom_config(self) -> None:
        cfg = TTSConfig(voice="es_001", speed=0.8)
        adapter = KokoroAdapter(config=cfg)
        # Internal config should be our custom one.
        assert adapter._config.voice == "es_001"

    # -- Empty text ----------------------------------------------------------

    def test_empty_string_returns_valid_wav(self) -> None:
        """Empty text must produce a valid, playable WAV — not an error."""
        adapter = KokoroAdapter()
        result = adapter.synthesize("")

        assert isinstance(result, TTSResult)
        assert len(result.audio_bytes) > 44  # has PCM data
        assert result.audio_bytes[:4] == b"RIFF"
        assert result.text == ""
        assert result.duration_ms == 100.0
        assert result.voice == "es_002"

    def test_whitespace_only_returns_valid_wav(self) -> None:
        adapter = KokoroAdapter()
        result = adapter.synthesize("   \t\n  ")

        assert isinstance(result, TTSResult)
        assert len(result.audio_bytes) > 44
        assert result.audio_bytes[:4] == b"RIFF"
        assert result.duration_ms == 100.0

    # -- Unsupported config --------------------------------------------------

    def test_unsupported_sample_rate_raises(self) -> None:
        cfg = TTSConfig(sample_rate=44100)
        adapter = KokoroAdapter(config=cfg)
        with pytest.raises(TTSSynthesisError, match="Unsupported sample_rate"):
            adapter.synthesize("hola")

    def test_per_call_config_overrides_adapter_config(self) -> None:
        adapter = KokoroAdapter()  # default 24_000
        cfg_bad = TTSConfig(sample_rate=44100)
        with pytest.raises(TTSSynthesisError):
            adapter.synthesize("hola", config=cfg_bad)

    # -- Mocked synthesis (happy path) ---------------------------------------

    def test_synthesize_with_mock_pipeline(self) -> None:
        """Full synthesis path with mocked Kokoro KPipeline."""
        adapter = KokoroAdapter()

        # Pre-inject a mocked pipeline.
        mock_pipeline = MagicMock()
        # KPipeline returns (audio_array, phoneme_list).
        mock_audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        mock_pipeline.return_value = (mock_audio, ["HH", "OW", "L", "AA"])
        adapter._pipeline = mock_pipeline

        result = adapter.synthesize("hola")
        assert isinstance(result, TTSResult)
        assert result.text == "hola"
        assert result.voice == "es_002"
        assert result.sample_rate == 24_000
        assert result.format == "wav"
        assert len(result.audio_bytes) > 44  # has PCM

        # Duration: 3 samples / 24000 Hz * 1000 ms = 0.125 ms
        expected_duration = (3 / 24000) * 1000
        assert result.duration_ms == pytest.approx(expected_duration)

        # Verify the pipeline was called with correct args.
        mock_pipeline.assert_called_once_with("hola", voice="es_002", speed=1.0)

    def test_synthesize_with_mock_pipeline_custom_config(self) -> None:
        cfg = TTSConfig(voice="es_001", speed=0.9)
        adapter = KokoroAdapter(config=cfg)

        mock_pipeline = MagicMock()
        mock_audio = np.zeros(100, dtype=np.float32)
        mock_pipeline.return_value = (mock_audio, [])
        adapter._pipeline = mock_pipeline

        result = adapter.synthesize("adiós")
        assert result.voice == "es_001"
        mock_pipeline.assert_called_once_with("adiós", voice="es_001", speed=0.9)

    # -- Mocked synthesis (error wrapping) -----------------------------------

    def test_pipeline_failure_wraps_in_synthesis_error(self) -> None:
        adapter = KokoroAdapter()

        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("Kokoro backend crash")
        adapter._pipeline = mock_pipeline

        with pytest.raises(TTSSynthesisError, match="Kokoro synthesis failed"):
            adapter.synthesize("hola")

    def test_pipeline_import_error_wraps_in_synthesis_error(self) -> None:
        """When kokoro is not installed, a clear TTSSynthesisError is raised."""
        adapter = KokoroAdapter()

        # Remove kokoro from sys.modules so the lazy import attempt fails.
        with patch.dict("sys.modules", {"kokoro": None}):
            with pytest.raises(TTSSynthesisError, match="kokoro"):
                # Reset pipeline so synthesize() triggers a fresh import.
                adapter._pipeline = None
                adapter.synthesize("hola")

    def test_multi_segment_output_concatenated(self) -> None:
        """Kokoro may return a list of arrays (per-segment output)."""
        adapter = KokoroAdapter()

        mock_pipeline = MagicMock()
        seg1 = np.array([0.1, 0.2], dtype=np.float32)
        seg2 = np.array([0.3, 0.4], dtype=np.float32)
        mock_pipeline.return_value = ([seg1, seg2], [[], []])
        adapter._pipeline = mock_pipeline

        result = adapter.synthesize("hola mundo")
        assert len(result.audio_bytes) == 44 + 4 * 2  # 4 samples
        assert result.duration_ms == pytest.approx((4 / 24000) * 1000)

    def test_single_text_chunk_no_leading_trailing_spaces_stripped(self) -> None:
        """Input text is passed verbatim — no silent stripping."""
        adapter = KokoroAdapter()

        mock_pipeline = MagicMock()
        mock_audio = np.array([0.0], dtype=np.float32)
        mock_pipeline.return_value = (mock_audio, [])
        adapter._pipeline = mock_pipeline

        result = adapter.synthesize("  hola  ")
        # Non-empty → pipeline called (strip only checked for empty path).
        assert result.text == "  hola  "

    def test_lazy_pipeline_creation(self) -> None:
        """Pipeline is not created during __init__ — only on first call."""
        adapter = KokoroAdapter()
        assert adapter._pipeline is None

        # Inject a mock without actually importing kokoro.
        mock_pipeline = MagicMock()
        mock_audio = np.array([0.0], dtype=np.float32)
        mock_pipeline.return_value = (mock_audio, [])
        adapter._pipeline = mock_pipeline

        result = adapter.synthesize("test")
        assert result is not None
        mock_pipeline.assert_called_once()
