"""Tests for ``backend.metrics.percentiles`` — linear-interpolation
percentile computation."""

from __future__ import annotations

import pytest

from backend.metrics.percentiles import percentile


class TestPercentile:
    """percentile() edge cases and correctness."""

    # -- edge cases ---------------------------------------------------------

    def test_empty_returns_none(self):
        assert percentile([], 50) is None

    def test_single_element(self):
        assert percentile([42.0], 0) == 42.0
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 100) == 42.0

    def test_two_elements(self):
        assert percentile([10.0, 20.0], 0) == 10.0
        assert percentile([10.0, 20.0], 50) == 15.0
        assert percentile([10.0, 20.0], 100) == 20.0

    # -- P50 / median -------------------------------------------------------

    def test_p50_odd_length(self):
        values = [1.0, 3.0, 2.0, 5.0, 4.0]
        result = percentile(values, 50)
        assert result == 3.0  # sorted: [1,2,3,4,5] → index 2 → 3

    def test_p50_even_length(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = percentile(values, 50)
        # index = 0.5 * 3 = 1.5 → interpolate between idx 1 (2.0) and 2 (3.0)
        assert result == 2.5

    # -- P95 ----------------------------------------------------------------

    def test_p95_interpolation(self):
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = percentile(values, 95)
        # index = 0.95 * 4 = 3.8 → between idx 3 (400) and 4 (500)
        # result = 400 * 0.2 + 500 * 0.8 = 80 + 400 = 480
        assert result == pytest.approx(480.0)

    def test_p95_small_dataset(self):
        values = [10.0, 90.0]
        result = percentile(values, 95)
        # index = 0.95 * 1 = 0.95 → between idx 0 (10) and 1 (90)
        # result = 10 * 0.05 + 90 * 0.95 = 0.5 + 85.5 = 86.0
        assert result == pytest.approx(86.0)

    def test_p0_is_min(self):
        values = [5.0, 1.0, 3.0, 9.0, 7.0]
        assert percentile(values, 0) == 1.0

    def test_p100_is_max(self):
        values = [5.0, 1.0, 3.0, 9.0, 7.0]
        assert percentile(values, 100) == 9.0

    # -- boundary validation -------------------------------------------------

    def test_p_negative_raises(self):
        with pytest.raises(ValueError, match="p must be in"):
            percentile([1.0, 2.0], -0.1)

    def test_p_above_100_raises(self):
        with pytest.raises(ValueError, match="p must be in"):
            percentile([1.0, 2.0], 100.1)

    # -- floats and ints ----------------------------------------------------

    def test_with_ints(self):
        values = [1, 2, 3, 4, 5]
        result = percentile(values, 50)
        assert result == 3.0

    def test_with_mixed_floats(self):
        values = [1.5, 2.5, 3.5]
        result = percentile(values, 50)
        assert result == 2.5

    # -- deterministic ------------------------------------------------------

    def test_same_input_same_output(self):
        values = [42.0, 17.0, 99.0, 3.0]
        a = percentile(values, 90)
        b = percentile(values, 90)
        assert a == b

    def test_unsorted_preserves_output(self):
        sorted_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        unsorted_vals = [30.0, 50.0, 10.0, 40.0, 20.0]
        assert percentile(sorted_vals, 75) == percentile(unsorted_vals, 75)
