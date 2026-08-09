"""LLM configuration — provider, model selection, credentials, and parameters.

The application uses the current Groq successor Llama 3.3 70B Versatile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model and provider defaults
# ---------------------------------------------------------------------------
# Llama 3.3 70B Versatile is the current successor to the originally suggested
# Groq model, as authorized by the challenge organizers when a provider retires
# a suggested model.
#
# The ``model_name`` field is kept for adapter compatibility so that callers
# (``backend/llm/adapter.py``, ``backend/api/rag.py``) can read the model
# identifier without change.

_DEFAULT_PROVIDER: str = "groq"
_DEFAULT_MODEL_NAME: str = "llama-3.3-70b-versatile"

@dataclass(frozen=True)
class LlmConfig:
    """Immutable configuration for the LLM adapter.

    The default is the current Groq successor **Llama 3.3 70B Versatile**.
    The provider is fixed to Groq for the delivery configuration.
    """

    provider: str = _DEFAULT_PROVIDER
    """LLM provider: ``groq``."""

    # -- Model selection ---------------------------------------------------
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL", _DEFAULT_MODEL_NAME))
    """Provider model identifier."""

    api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    """Groq Cloud API key, used by the LLM and STT providers."""

    # -- Generation parameters ---------------------------------------------
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    """Sampling temperature. Low values (≤0.3) favour deterministic,
    source-grounded clinical answers."""

    max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )
    """Maximum tokens in the generated response."""

    def __post_init__(self) -> None:
        if self.provider != "groq":
            raise ValueError(f"provider must be 'groq', got {self.provider!r}")
        if self.model_name not in {"llama-3.1-70b-versatile", "llama-3.3-70b-versatile"}:
            raise ValueError(
                f"model {self.model_name!r} is not allowed for provider "
                f"{self.provider!r}"
            )
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(
                f"LLM_TEMPERATURE must be in [0, 2], got {self.temperature}"
            )
        if self.max_output_tokens < 1:
            raise ValueError(
                f"LLM_MAX_TOKENS must be >= 1, got {self.max_output_tokens}"
            )
