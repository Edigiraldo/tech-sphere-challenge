"""Tests for ``backend.metrics.cost`` — CostConfig and estimate_cost."""

from __future__ import annotations

import pytest

from backend.metrics.cost import CostConfig, estimate_cost


class TestCostConfig:
    """CostConfig construction and validation."""

    def test_defaults(self):
        cfg = CostConfig()
        assert cfg.input_cost_per_million == 0.0
        assert cfg.output_cost_per_million == 0.0

    def test_custom_rates(self):
        cfg = CostConfig(
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
        )
        assert cfg.input_cost_per_million == 0.50
        assert cfg.output_cost_per_million == 1.50

    def test_zero_rates_allowed(self):
        cfg = CostConfig(
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
        )
        assert cfg.input_cost_per_million == 0.0

    def test_negative_input_rate_raises(self):
        with pytest.raises(ValueError, match="input_cost_per_million"):
            CostConfig(input_cost_per_million=-0.01)

    def test_negative_output_rate_raises(self):
        with pytest.raises(ValueError, match="output_cost_per_million"):
            CostConfig(output_cost_per_million=-0.01)

    def test_frozen(self):
        cfg = CostConfig()
        with pytest.raises(Exception):
            cfg.input_cost_per_million = 1.0  # type: ignore[misc]


class TestEstimateCost:
    """estimate_cost function."""

    def test_zero_tokens_zero_cost(self):
        cost = estimate_cost(
            input_tokens=0,
            output_tokens=0,
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )
        assert cost == 0.0

    def test_exact_million_tokens(self):
        cost = estimate_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
        )
        expected = (1.0 * 0.50) + (0.5 * 1.50)  # 0.50 + 0.75
        assert cost == pytest.approx(1.25)

    def test_fractional_tokens(self):
        cost = estimate_cost(
            input_tokens=100,
            output_tokens=50,
            input_cost_per_million=1.00,
            output_cost_per_million=2.00,
        )
        expected = (100 / 1e6) * 1.0 + (50 / 1e6) * 2.0
        assert cost == pytest.approx(expected)

    def test_only_input_tokens(self):
        cost = estimate_cost(
            input_tokens=1_000,
            output_tokens=0,
            input_cost_per_million=0.10,
            output_cost_per_million=1.00,
        )
        expected = (1000 / 1e6) * 0.10
        assert cost == pytest.approx(expected)

    def test_only_output_tokens(self):
        cost = estimate_cost(
            input_tokens=0,
            output_tokens=500,
            input_cost_per_million=0.10,
            output_cost_per_million=1.00,
        )
        expected = (500 / 1e6) * 1.00
        assert cost == pytest.approx(expected)

    def test_negative_input_tokens_raises(self):
        with pytest.raises(ValueError, match="input_tokens must be >= 0"):
            estimate_cost(
                input_tokens=-1,
                output_tokens=10,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
            )

    def test_negative_output_tokens_raises(self):
        with pytest.raises(ValueError, match="output_tokens must be >= 0"):
            estimate_cost(
                input_tokens=10,
                output_tokens=-1,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
            )
