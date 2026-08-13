"""Cost estimation for language-model token consumption.

Defines a frozen ``CostConfig`` with per-million-token rates and a
standalone ``estimate_cost`` function.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class CostConfig:
    """Immutable per-million-token pricing configuration.

    Attributes
    ----------
    input_cost_per_million : float
        Cost in USD per one million input (prompt) tokens.  Must be >= 0.
    output_cost_per_million : float
        Cost in USD per one million output (completion) tokens.  Must be >= 0.
    """

    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    def __post_init__(self) -> None:
        if self.input_cost_per_million < 0:
            raise ValueError(
                "input_cost_per_million must be >= 0, "
                f"got {self.input_cost_per_million}"
            )
        if self.output_cost_per_million < 0:
            raise ValueError(
                "output_cost_per_million must be >= 0, "
                f"got {self.output_cost_per_million}"
            )


def default_groq_llama33_cost_config() -> CostConfig:
    """Return configurable Groq Llama 3.3 pricing per million tokens."""
    return CostConfig(
        input_cost_per_million=float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0.24")),
        output_cost_per_million=float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0.24")),
    )
def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    """Estimate the USD cost of a single LLM call.

    Parameters
    ----------
    input_tokens : int
        Number of input (prompt) tokens.  Must be >= 0.
    output_tokens : int
        Number of output (completion) tokens.  Must be >= 0.
    input_cost_per_million : float
        USD cost per 1 000 000 input tokens.
    output_cost_per_million : float
        USD cost per 1 000 000 output tokens.

    Returns
    -------
    float
        Estimated cost in USD.  Always >= 0.
    """
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be >= 0, got {input_tokens}")
    if output_tokens < 0:
        raise ValueError(f"output_tokens must be >= 0, got {output_tokens}")

    input_cost = (input_tokens / 1_000_000.0) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000.0) * output_cost_per_million
    return input_cost + output_cost
