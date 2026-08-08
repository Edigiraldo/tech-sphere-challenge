"""Focused tests for provider startup wiring.

Covers:
- Successful wiring of both STT and TTS.
- Missing ``GROQ_API_KEY`` — graceful handling, no crash.
- Construction error in one provider → other still wired.
- Injection override: ``set_stt`` / ``set_tts`` called after
  ``configure_providers()`` correctly replaces the provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.voice.initialization import configure_providers


# ---------------------------------------------------------------------------
# Helpers — use module-level access (not a local ``from ... import`` binding)
# so that every call sees the live ``backend.api.calls._stt`` / ``_tts``.
# ---------------------------------------------------------------------------


def _get_stt():
    """Read the current injected STT via the calls module's namespace."""
    import backend.api.calls as _calls_mod

    return _calls_mod._stt


def _get_tts():
    """Read the current injected TTS via the calls module's namespace."""
    import backend.api.calls as _calls_mod

    return _calls_mod._tts


def _reset_injections():
    """Set both injection slots to None for test isolation."""
    from backend.api.calls import set_stt, set_tts

    set_stt(None)  # type: ignore[arg-type]
    set_tts(None)  # type: ignore[arg-type]


def _fake_transcribe(audio_data: bytes):
    """No-op STT stub used by overrides."""
    from backend.voice.models import TranscriptionResult

    return TranscriptionResult(
        text="fake transcription",
        language="es",
        model="stub",
    )


def _fake_tts():
    """Return a MagicMock that satisfies the TTSProvider protocol."""
    from backend.voice.tts.protocol import TTSResult

    tts = MagicMock()
    tts.synthesize.return_value = TTSResult(
        audio_bytes=b"RIFF....WAVE....",
        sample_rate=24000,
        duration_ms=100.0,
        text="mock",
        voice="stub",
    )
    return tts


# ---------------------------------------------------------------------------
# Autouse fixture — ensure clean state before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_before_test():
    """Nuke injected STT/TTS before each test so no test sees stale state."""
    _reset_injections()
    yield
    _reset_injections()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigureProvidersSuccess:
    """Happy path: both providers wire successfully."""

    def test_both_providers_wired(self, monkeypatch) -> None:
        """When GROQ_API_KEY is set, both STT and TTS are injected into
        ``backend.api.calls``."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_123")

        # Prevent numpy from crashing (KokoroAdapter imports numpy at module
        # level; mock it so tests pass without optional voice dependencies).
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        configure_providers()

        stt = _get_stt()
        tts = _get_tts()

        assert stt is not None, "STT should be wired when GROQ_API_KEY is set"
        assert callable(stt), "STT injection should be callable"
        assert tts is not None, "TTS should be wired"
        assert hasattr(tts, "synthesize"), (
            "TTS injection should have synthesize method"
        )


class TestMissingGroqApiKey:
    """When GROQ_API_KEY is absent, ``GroqWhisperConfig`` constructs with
    an empty key (validation is deferred to ``transcribe()`` call time).
    ``configure_providers()`` must not crash — STT *is* wired, but actual
    transcription will fail at request time."""

    def test_no_crash_when_key_missing(self, monkeypatch) -> None:
        """Startup completes without exception even with no API key."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        # Should not raise
        configure_providers()

        # STT gets wired because GroqWhisperProvider.__init__ does not
        # validate the API key — that happens at transcribe() time.
        stt = _get_stt()
        assert stt is not None, (
            "STT should be wired (constructor does not validate API key)"
        )


class TestConstructionErrors:
    """An unrecoverable error in one provider must not prevent wiring of
    the other, and startup must not crash."""

    def test_stt_constructor_raises_does_not_crash(self, monkeypatch) -> None:
        """If ``GroqWhisperProvider.__init__`` throws, configure_providers
        logs and continues.  TTS is still wired."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        from backend.voice import groq as groq_mod

        with patch.object(
            groq_mod.GroqWhisperProvider,
            "__init__",
            side_effect=RuntimeError("GPU on fire"),
        ):
            configure_providers()

        assert _get_stt() is None, "STT should be None after constructor crash"
        assert _get_tts() is not None, "TTS should still be wired"

    def test_tts_constructor_raises_does_not_crash(self, monkeypatch) -> None:
        """If ``KokoroAdapter.__init__`` throws, STT is still wired."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        from backend.voice.tts import kokoro as kokoro_mod

        with patch.object(
            kokoro_mod.KokoroAdapter,
            "__init__",
            side_effect=RuntimeError("no memory"),
        ):
            configure_providers()

        stt = _get_stt()
        assert stt is not None, "STT should still be wired"
        assert callable(stt)
        assert _get_tts() is None, "TTS should be None after constructor crash"

    def test_both_providers_raise_does_not_crash(self, monkeypatch) -> None:
        """Dual failure → startup still completes without exception."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        from backend.voice import groq as groq_mod
        from backend.voice.tts import kokoro as kokoro_mod

        with patch.object(
            groq_mod.GroqWhisperProvider,
            "__init__",
            side_effect=RuntimeError("STT failure"),
        ), patch.object(
            kokoro_mod.KokoroAdapter,
            "__init__",
            side_effect=RuntimeError("TTS failure"),
        ):
            configure_providers()

        assert _get_stt() is None
        assert _get_tts() is None


class TestInjectionOverrideBehavior:
    """After ``configure_providers()`` runs, ``set_stt`` / ``set_tts`` can
    be called again to override the provider (used by tests).  The override
    must take effect and not be reverted by subsequent calls."""

    def test_override_stt_after_configure(self, monkeypatch) -> None:
        """Calling set_stt after configure_providers replaces the STT."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        configure_providers()

        from backend.api.calls import set_stt

        assert _get_stt() is not None
        original_id = id(_get_stt())

        set_stt(_fake_transcribe)
        assert _get_stt() is not None
        assert id(_get_stt()) != original_id, (
            "Override should replace previous STT"
        )
        assert _get_stt() is _fake_transcribe

    def test_override_tts_after_configure(self, monkeypatch) -> None:
        """Calling set_tts after configure_providers replaces the TTS."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        configure_providers()

        from backend.api.calls import set_tts

        assert _get_tts() is not None
        original_id = id(_get_tts())

        new_tts = _fake_tts()
        set_tts(new_tts)
        assert _get_tts() is not None
        assert id(_get_tts()) != original_id, (
            "Override should replace previous TTS"
        )
        assert _get_tts() is new_tts

    def test_override_both_after_configure(self, monkeypatch) -> None:
        """Both can be overridden and the overrides persist."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        configure_providers()

        from backend.api.calls import set_stt, set_tts

        new_tts = _fake_tts()
        set_stt(_fake_transcribe)
        set_tts(new_tts)

        assert _get_stt() is _fake_transcribe
        assert _get_tts() is new_tts


class TestSummaryLogging:
    """The summary log at the end of ``configure_providers()`` must distinguish
    STT and TTS readiness independently, not claim all providers are ready when
    only one succeeded."""

    def test_both_ready_logs_both(self, monkeypatch, caplog) -> None:
        """When both STT and TTS wire, the summary log says both are ready."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_123")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        with caplog.at_level("INFO", logger="backend.voice.initialization"):
            configure_providers()

        assert "STT and TTS providers ready for calls." in caplog.text

    def test_only_stt_ready_logs_stt_ready(self, monkeypatch, caplog) -> None:
        """When only STT wires (TTS fails), summary says TTS is unavailable."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())
        monkeypatch.setattr(
            "backend.voice.initialization.KokoroAdapter",
            MagicMock(side_effect=RuntimeError("TTS unavailable")),
        )

        with caplog.at_level("INFO", logger="backend.voice.initialization"):
            configure_providers()

        assert "STT provider ready for calls; TTS provider unavailable." in caplog.text

    def test_only_tts_ready_logs_tts_ready(self, monkeypatch, caplog) -> None:
        """When only TTS wires (STT fails), summary says STT is unavailable."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        from backend.voice import groq as groq_mod

        with caplog.at_level("INFO", logger="backend.voice.initialization"):
            with patch.object(
                groq_mod.GroqWhisperProvider,
                "__init__",
                side_effect=RuntimeError("STT unavailable"),
            ):
                configure_providers()

        assert "TTS provider ready for calls; STT provider unavailable." in caplog.text

    def test_neither_ready_logs_warning(self, monkeypatch, caplog) -> None:
        """When neither provider wires, summary logs the existing warning."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        from backend.voice import groq as groq_mod
        from backend.voice.tts import kokoro as kokoro_mod

        with caplog.at_level("WARNING", logger="backend.voice.initialization"):
            with patch.object(
                groq_mod.GroqWhisperProvider,
                "__init__",
                side_effect=RuntimeError("STT failure"),
            ), patch.object(
                kokoro_mod.KokoroAdapter,
                "__init__",
                side_effect=RuntimeError("TTS failure"),
            ):
                configure_providers()

        assert "No voice providers were wired" in caplog.text

    def test_both_ready_does_not_log_unavailable(self, monkeypatch, caplog) -> None:
        """When both providers are ready, the summary must NOT claim either
        is unavailable."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_123")
        monkeypatch.setattr("backend.voice.tts.kokoro.np", MagicMock())

        with caplog.at_level("INFO", logger="backend.voice.initialization"):
            configure_providers()

        assert "unavailable" not in caplog.text
