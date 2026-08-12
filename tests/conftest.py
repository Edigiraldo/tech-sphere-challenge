"""Shared test configuration.

Existing adapter tests exercise the Groq transport with mocked calls.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "live_server: tests that require a running server on alternate port 18000"
    )


@pytest.fixture(autouse=True)
def use_mocked_groq_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep provider-specific unit tests deterministic and network-free."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-70b-versatile")
