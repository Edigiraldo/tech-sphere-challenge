"""Percentile computation via linear interpolation.

Provides a standalone ``percentile()`` function suitable for latency
and duration distributions.
"""

from __future__ import annotations

import math
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float | None:
    """Compute the *p*-th percentile via linear interpolation.

    Uses the same method as ``numpy.percentile`` with
    ``method='linear'`` (the "C = 1" variant from Hyndman & Fan 1996).

    Parameters
    ----------
    values : sequence of float
        Non-empty sequence of numeric observations.
    p : float
        Percentile in [0, 100].  ``p=50`` computes the median;
        ``p=95`` computes the 95th percentile.

    Returns
    -------
    float or None
        The computed percentile, or ``None`` when *values* is empty.

    Raises
    ------
    ValueError
        If *p* is outside [0, 100].
    """
    if not (0 <= p <= 100):
        raise ValueError(f"p must be in [0, 100], got {p}")

    if len(values) == 0:
        return None

    sorted_vals = sorted(values)

    # Fractional index for linear interpolation (C=1 method).
    n = len(sorted_vals)
    index = (p / 100.0) * (n - 1)

    lower = int(math.floor(index))
    upper = int(math.ceil(index))

    if lower == upper:
        return sorted_vals[lower]

    fraction = index - lower
    return sorted_vals[lower] * (1.0 - fraction) + sorted_vals[upper] * fraction
