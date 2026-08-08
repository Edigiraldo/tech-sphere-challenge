"""Metrics instrumentation — latency, token, and cost observability.

Exports the ``MetricsCollector`` Protocol, typed data models
(``TurnMetrics``, ``CallMetrics``, ``MetricsSummary``, ``CostConfig``),
the thread-safe ``InMemoryMetricsCollector``, the ``estimate_cost``
function, and the ``percentile`` helper.
"""

from backend.metrics.collector import InMemoryMetricsCollector, MetricsCollector
from backend.metrics.cost import CostConfig, estimate_cost
from backend.metrics.models import CallMetrics, MetricsSummary, TurnMetrics
from backend.metrics.percentiles import percentile

__all__ = [
    "CallMetrics",
    "CostConfig",
    "InMemoryMetricsCollector",
    "MetricsCollector",
    "MetricsSummary",
    "TurnMetrics",
    "estimate_cost",
    "percentile",
]
