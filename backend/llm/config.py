"""LLM configuration — model selection, API credentials, generation params.

The language model is fixed to Llama 3.1 70B Versatile via Groq Cloud
(the only model currently integrated).  API key and generation parameters
are read from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Fixed model identifier (the only model currently integrated)
# ---------------------------------------------------------------------------
# Llama 3.1 70B Versatile was selected because it is a permitted challenge
# model (``.challenge-docs/stack-tecnico.md``) hosted on Groq Cloud with
# fast inference, strong Spanish-language performance, and native structured
# JSON output support.  Other permitted models (Llama 3.2, Phi-3.5) are not
# integrated.
#
# The ``model_name`` field is kept for adapter compatibility so that callers
# (``backend/llm/adapter.py``, ``backend/api/rag.py``) can read the model
# identifier without change.

_FIXED_MODEL_NAME: str = "llama-3.1-70b-versatile"


@dataclass(frozen=True)
class LlmConfig:
    """Immutable configuration for the LLM adapter.

    The model is fixed to **Llama 3.1 70B Versatile** (Groq Cloud) — the
    only model currently integrated.  ``model_name`` is a read-only
    constant exposed for adapter compatibility.
    """

    # -- Model selection ---------------------------------------------------
    model_name: str = _FIXED_MODEL_NAME
    """Model identifier — always ``"llama-3.1-70b-versatile"``."""

    api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    """Groq Cloud API key (required for Llama models on Groq)."""

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
        if self.model_name != "llama-3.1-70b-versatile":
            raise ValueError(
                f"model_name must be 'llama-3.1-70b-versatile', "
                f"got {self.model_name!r}"
            )
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(
                f"LLM_TEMPERATURE must be in [0, 2], got {self.temperature}"
            )
        if self.max_output_tokens < 1:
            raise ValueError(
                f"LLM_MAX_TOKENS must be >= 1, got {self.max_output_tokens}"
            )
