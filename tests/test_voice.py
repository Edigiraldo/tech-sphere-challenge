"""Unit tests for the STT adapter foundation.

All Groq API calls are mocked so the tests execute quickly without network
access or an API key.
"""

from __future__ import annotations

import sys
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.voice.api import transcribe_audio
from backend.voice.config import GroqWhisperConfig
from backend.voice.groq import (
    GroqWhisperProvider,
    _map_groq_error,
    _validate_audio,
    _validate_config,
)
from backend.voice.models import (
    SttAudioError,
    SttConfigError,
    SttError,
    SttProviderError,
    TranscriptionResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_wav_bytes() -> bytes:
    """A minimal but non-empty byte sequence for audio input testing."""
    # 44-byte minimal WAV header + 1 byte of silence
    return b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"


@pytest.fixture
def groq_config_with_key() -> GroqWhisperConfig:
    return GroqWhisperConfig(api_key="gsk_test_fake_key")


@pytest.fixture
def mock_groq_transcription() -> dict[str, Any]:
    """Simulate a verbose_json Whisper response."""
    return {
        "text": "Me duele la herida desde ayer.",
        "language": "es",
        "duration": 2.8,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.8,
                "text": "Me duele la herida desde ayer.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.25,
                "compression_ratio": 1.5,
                "no_speech_prob": 0.05,
            }
        ],
    }


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_defaults(self) -> None:
        r = TranscriptionResult(text="Hola")
        assert r.text == "Hola"
        assert r.language == "es"
        assert r.duration_seconds == 0.0
        assert r.model == ""
        assert r.metadata == {}

    def test_full_construction(self) -> None:
        r = TranscriptionResult(
            text="Sí, doctor.",
            language="es",
            duration_seconds=1.5,
            model="whisper-large-v3",
            metadata={"segments": [{"text": "Sí, doctor."}]},
        )
        assert r.text == "Sí, doctor."
        assert r.language == "es"
        assert r.duration_seconds == 1.5
        assert r.model == "whisper-large-v3"
        assert r.metadata["segments"][0]["text"] == "Sí, doctor."

    def test_is_frozen(self) -> None:
        r = TranscriptionResult(text="Hola")
        with pytest.raises(Exception):
            r.text = "Adiós"  # type: ignore[misc]

    def test_empty_text(self) -> None:
        """Empty text is valid — represents no-speech detection."""
        r = TranscriptionResult(text="")
        assert r.text == ""


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_stt_error_is_base(self) -> None:
        assert issubclass(SttConfigError, SttError)
        assert issubclass(SttProviderError, SttError)
        assert issubclass(SttAudioError, SttError)
        assert issubclass(SttError, Exception)

    def test_stt_error_is_not_stt_audio(self) -> None:
        """Ensure base SttError is not a subclass of specific errors."""
        assert not issubclass(SttError, SttAudioError)
        assert not issubclass(SttError, SttConfigError)
        assert not issubclass(SttError, SttProviderError)

    def test_config_error_can_be_raised_and_caught_as_stt_error(self) -> None:
        with pytest.raises(SttError) as exc:
            raise SttConfigError("bad config")
        assert isinstance(exc.value, SttConfigError)
        assert "bad config" in str(exc.value)

    def test_provider_error_can_be_caught_as_stt_error(self) -> None:
        with pytest.raises(SttError) as exc:
            raise SttProviderError("api down")
        assert isinstance(exc.value, SttProviderError)

    def test_audio_error_can_be_caught_as_stt_error(self) -> None:
        with pytest.raises(SttError) as exc:
            raise SttAudioError("empty file")
        assert isinstance(exc.value, SttAudioError)


# ---------------------------------------------------------------------------
# GroqWhisperConfig
# ---------------------------------------------------------------------------


class TestGroqWhisperConfig:
    def test_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        cfg = GroqWhisperConfig()
        assert cfg.model == "whisper-large-v3"
        assert cfg.language == "es"
        assert cfg.temperature == 0.0
        assert cfg.response_format == "verbose_json"
        assert cfg.api_key == ""

    def test_model_is_frozen(self) -> None:
        cfg = GroqWhisperConfig()
        assert cfg.model == "whisper-large-v3"
        # explicit pass of valid model is OK
        cfg2 = GroqWhisperConfig(model="whisper-large-v3")
        assert cfg2.model == "whisper-large-v3"

    def test_model_wrong_raises(self) -> None:
        with pytest.raises(ValueError, match="model must be"):
            GroqWhisperConfig(model="whisper-large-v2")

    def test_language_is_frozen(self) -> None:
        cfg = GroqWhisperConfig()
        assert cfg.language == "es"

    def test_language_wrong_raises(self) -> None:
        with pytest.raises(ValueError, match="language must be"):
            GroqWhisperConfig(language="en")

    def test_temperature_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="STT_TEMPERATURE"):
            GroqWhisperConfig(temperature=1.5)

    def test_temperature_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="STT_TEMPERATURE"):
            GroqWhisperConfig(temperature=-0.1)

    def test_response_format_invalid(self) -> None:
        with pytest.raises(ValueError, match="STT_RESPONSE_FORMAT"):
            GroqWhisperConfig(response_format="srt")

    def test_response_format_valid_alternatives(self) -> None:
        for fmt in ("json", "verbose_json", "text"):
            cfg = GroqWhisperConfig(response_format=fmt)
            assert cfg.response_format == fmt


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


class TestValidateAudio:
    def test_non_bytes_raises(self) -> None:
        with pytest.raises(SttAudioError, match="bytes"):
            _validate_audio("not bytes")  # type: ignore[arg-type]

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(SttAudioError, match="vacío"):
            _validate_audio(b"")

    def test_oversize_raises(self) -> None:
        too_big = b"\x00" * (26 * 1024 * 1024)  # 26 MB
        with pytest.raises(SttAudioError, match="excede"):
            _validate_audio(too_big)

    def test_valid_bytes_pass(self, minimal_wav_bytes) -> None:
        _validate_audio(minimal_wav_bytes)

    def test_exactly_25_mb_passes(self) -> None:
        exact = b"\x00" * (25 * 1024 * 1024)
        _validate_audio(exact)


class TestValidateConfig:
    def test_missing_api_key_raises(self) -> None:
        cfg = GroqWhisperConfig(api_key="")
        with pytest.raises(SttConfigError, match="GROQ_API_KEY"):
            _validate_config(cfg)

    def test_whitespace_only_api_key_raises(self) -> None:
        cfg = GroqWhisperConfig(api_key="   ")
        with pytest.raises(SttConfigError, match="GROQ_API_KEY"):
            _validate_config(cfg)

    def test_valid_config_passes(self, groq_config_with_key) -> None:
        _validate_config(groq_config_with_key)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestMapGroqError:
    """Test _map_groq_error converts known provider exceptions correctly."""

    def test_rate_limit(self) -> None:
        # Simulate a groq.RateLimitError without importing groq
        class FakeRateLimitError(Exception):
            pass

        exc = FakeRateLimitError("Rate limit exceeded")
        # Patch MRO to include RateLimitError name
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "Límite de tasa" in str(mapped)

    def test_authentication_error(self) -> None:
        exc = Exception("401 Unauthorized")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "autenticación" in str(mapped).lower()

    def test_bad_request_error(self) -> None:
        exc = Exception("400 Bad Request: invalid audio format")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "rechazó" in str(mapped)

    def test_connection_error(self) -> None:
        exc = ConnectionError("Connection refused")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "conectar" in str(mapped)

    def test_generic_exception(self) -> None:
        exc = RuntimeError("something broke")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "Error inesperado" in str(mapped)
        assert "something broke" in str(mapped)

    def test_authentication_error_by_class_name(self) -> None:
        """Error with 'AuthenticationError' in MRO should map to auth error."""
        class FakeAuthError(Exception):
            pass

        FakeAuthError.__name__ = "AuthenticationError"
        exc = FakeAuthError("Invalid API key")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "autenticación" in str(mapped).lower()

    def test_rate_limit_by_message_substring(self) -> None:
        exc = Exception("rate_limit exceeded for this minute")
        mapped = _map_groq_error(exc)
        assert isinstance(mapped, SttProviderError)
        assert "Límite de tasa" in str(mapped)


# ---------------------------------------------------------------------------
# GroqWhisperProvider
# ---------------------------------------------------------------------------


class TestGroqWhisperProvider:
    def test_construction(self, groq_config_with_key) -> None:
        provider = GroqWhisperProvider(groq_config_with_key)
        assert provider._config is groq_config_with_key

    async def test_transcribe_success_bytes(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
        mock_groq_transcription,
    ) -> None:
        """Full successful transcription with raw bytes input.

        Proves ``groq.AsyncGroq`` is instantiated and the transcription
        create call is awaited (non-blocking async I/O instead of
        synchronous SDK usage).
        """
        mock_client = MagicMock()
        mock_transcriptions = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = mock_groq_transcription
        mock_transcriptions.create = AsyncMock(return_value=create_response)
        mock_client.audio.transcriptions = mock_transcriptions
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            result = await provider.transcribe(minimal_wav_bytes)

        assert result.text == "Me duele la herida desde ayer."
        assert result.language == "es"
        assert result.duration_seconds == 2.8
        assert result.model == "whisper-large-v3"
        assert "segments" in result.metadata

        # Verify AsyncGroq (not sync Groq) was instantiated
        mock_groq.AsyncGroq.assert_called_once_with(
            api_key="gsk_test_fake_key"
        )
        mock_groq.Groq.assert_not_called()
        # Prove the create call was awaited
        mock_transcriptions.create.assert_awaited_once()
        call_kwargs = mock_transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-large-v3"
        assert call_kwargs["language"] == "es"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"] == "verbose_json"
        # Verify file-like object was passed
        assert call_kwargs["file"] is not None
        assert call_kwargs["file"].name == "audio.wav"

    async def test_transcribe_empty_audio_raises(
        self, groq_config_with_key
    ) -> None:
        provider = GroqWhisperProvider(groq_config_with_key)
        with pytest.raises(SttAudioError, match="vacío"):
            await provider.transcribe(b"")

    async def test_transcribe_oversize_audio_raises(
        self, groq_config_with_key
    ) -> None:
        provider = GroqWhisperProvider(groq_config_with_key)
        too_big = b"\x00" * (26 * 1024 * 1024)
        with pytest.raises(SttAudioError, match="excede"):
            await provider.transcribe(too_big)

    async def test_transcribe_non_bytes_raises(
        self, groq_config_with_key
    ) -> None:
        provider = GroqWhisperProvider(groq_config_with_key)
        with pytest.raises(SttAudioError, match="bytes"):
            await provider.transcribe("not bytes")  # type: ignore[arg-type]

    async def test_transcribe_missing_api_key_raises(
        self, minimal_wav_bytes
    ) -> None:
        cfg = GroqWhisperConfig(api_key="")
        provider = GroqWhisperProvider(cfg)
        with pytest.raises(SttConfigError, match="GROQ_API_KEY"):
            await provider.transcribe(minimal_wav_bytes)

    async def test_transcribe_rate_limit_maps_to_stt_provider_error(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
    ) -> None:
        """Rate-limit from Groq SDK should surface as SttProviderError."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            with pytest.raises(SttProviderError, match="Límite de tasa"):
                await provider.transcribe(minimal_wav_bytes)

    async def test_transcribe_network_error_maps(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
    ) -> None:
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            with pytest.raises(SttProviderError, match="conectar"):
                await provider.transcribe(minimal_wav_bytes)

    async def test_transcribe_groq_not_installed(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
        monkeypatch,
    ) -> None:
        """If groq package is not importable, a clear SttProviderError is raised."""
        # Remove groq from sys.modules (if present) and prevent import
        monkeypatch.delitem(sys.modules, "groq", raising=False)
        # Patch builtins.__import__ to reject 'groq'
        import builtins

        original_import = builtins.__import__

        def reject_groq(name, *args, **kwargs):
            if name == "groq" or name.startswith("groq."):
                raise ImportError("No module named 'groq'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=reject_groq):
            provider = GroqWhisperProvider(groq_config_with_key)
            with pytest.raises(SttProviderError, match="groq"):
                await provider.transcribe(minimal_wav_bytes)

    async def test_transcribe_empty_text_handled(
        self, groq_config_with_key
    ) -> None:
        """Whisper may return empty text for silence/no-speech."""
        mock_response = {
            "text": "",
            "language": "es",
            "duration": 5.0,
        }
        mock_client = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = mock_response
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=create_response
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            result = await provider.transcribe(b"\x00" * 100)

        assert result.text == ""
        assert result.duration_seconds == 5.0

    async def test_transcribe_with_file_bytesio(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
    ) -> None:
        """Verify the adapter wraps audio bytes in BytesIO correctly."""
        mock_client = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = {
            "text": "OK",
            "language": "es",
            "duration": 0.5,
        }
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=create_response
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            result = await provider.transcribe(minimal_wav_bytes)

        assert result.text == "OK"
        # Verify a BytesIO was passed with the correct name
        call_args = mock_client.audio.transcriptions.create.call_args
        file_arg = call_args.kwargs["file"]
        assert isinstance(file_arg, BytesIO)
        assert file_arg.name == "audio.wav"

    async def test_async_groq_client_instantiated_and_awaited(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
    ) -> None:
        """Prove the adapter uses `groq.AsyncGroq` and awaits the API call.

        This is a focused structural test: it verifies that
        ``GroqWhisperProvider._call_groq`` constructs an ``AsyncGroq``
        client (not the synchronous ``Groq``) and that
        ``transcriptions.create`` is actually awaited — meaning the
        adapter performs non-blocking async I/O instead of blocking the
        event loop.
        """
        mock_client = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = {
            "text": "hola",
            "language": "es",
            "duration": 1.0,
        }
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=create_response
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            await provider.transcribe(minimal_wav_bytes)

        # Core assertions: AsyncGroq, not Groq
        mock_groq.AsyncGroq.assert_called_once()
        mock_groq.Groq.assert_not_called()
        # The transcription create was awaited
        mock_client.audio.transcriptions.create.assert_awaited_once()

    async def test_transcribe_metadata_preserves_segments(
        self,
        groq_config_with_key,
        minimal_wav_bytes,
        mock_groq_transcription,
    ) -> None:
        mock_client = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = mock_groq_transcription
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=create_response
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            provider = GroqWhisperProvider(groq_config_with_key)
            result = await provider.transcribe(minimal_wav_bytes)

        assert len(result.metadata["segments"]) == 1
        assert result.metadata["segments"][0]["text"] == mock_groq_transcription["text"]


# ---------------------------------------------------------------------------
# transcribe_audio (public injection API)
# ---------------------------------------------------------------------------


class TestTranscribeAudio:
    async def test_success_with_mock_dependency(self, minimal_wav_bytes) -> None:
        expected = TranscriptionResult(
            text="Gracias",
            language="es",
            duration_seconds=1.0,
            model="whisper-large-v3",
        )

        async def mock_stt(audio: bytes) -> TranscriptionResult:
            assert audio == minimal_wav_bytes
            return expected

        result = await transcribe_audio(minimal_wav_bytes, mock_stt)
        assert result is expected
        assert result.text == "Gracias"

    async def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            await transcribe_audio(b"data", "not_callable")  # type: ignore[arg-type]

    async def test_none_dependency_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            await transcribe_audio(b"data", None)  # type: ignore[arg-type]

    async def test_dependency_raises_stt_audio_error(self) -> None:
        async def failing_stt(audio: bytes) -> TranscriptionResult:
            raise SttAudioError("audio corrupto")

        with pytest.raises(SttAudioError, match="audio corrupto"):
            await transcribe_audio(b"data", failing_stt)

    async def test_dependency_raises_stt_provider_error(self) -> None:
        async def failing_stt(audio: bytes) -> TranscriptionResult:
            raise SttProviderError("API timeout")

        with pytest.raises(SttProviderError, match="API timeout"):
            await transcribe_audio(b"data", failing_stt)

    async def test_dependency_raises_stt_config_error(self) -> None:
        async def failing_stt(audio: bytes) -> TranscriptionResult:
            raise SttConfigError("no key")

        with pytest.raises(SttConfigError, match="no key"):
            await transcribe_audio(b"data", failing_stt)

    async def test_dependency_raises_unexpected_error_wraps(self) -> None:
        async def broken_stt(audio: bytes) -> TranscriptionResult:
            raise RuntimeError("something unexpected")

        with pytest.raises(SttProviderError, match="Error inesperado"):
            await transcribe_audio(b"data", broken_stt)

    async def test_dependency_returns_wrong_type_raises(self) -> None:
        async def bad_stt(audio: bytes) -> str:
            return "not a TranscriptionResult"

        with pytest.raises(TypeError, match="TranscriptionResult"):
            await transcribe_audio(b"data", bad_stt)  # type: ignore[arg-type]

    async def test_dependency_returns_none_raises(self) -> None:
        async def none_stt(audio: bytes) -> None:
            return None

        with pytest.raises(TypeError, match="TranscriptionResult"):
            await transcribe_audio(b"data", none_stt)  # type: ignore[arg-type]

    async def test_with_groq_provider_bound_method(
        self,
        minimal_wav_bytes,
        mock_groq_transcription,
    ) -> None:
        """Integration: transcribe_audio with GroqWhisperProvider.transcribe."""
        mock_client = MagicMock()
        create_response = MagicMock()
        create_response.model_dump.return_value = mock_groq_transcription
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=create_response
        )
        mock_groq = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}):
            cfg = GroqWhisperConfig(api_key="gsk_test")
            provider = GroqWhisperProvider(cfg)
            # Inject the bound method
            result = await transcribe_audio(minimal_wav_bytes, provider.transcribe)

        assert result.text == mock_groq_transcription["text"]
        assert result.model == "whisper-large-v3"
