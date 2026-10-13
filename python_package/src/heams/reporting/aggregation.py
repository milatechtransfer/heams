"""Benchmark-agnostic aggregation primitives shared by report sections.

These helpers hold the reusable mechanics that every benchmark report relies
on: labeling a run with its memory backend, iterating runs alongside their
scenario, accumulating weighted pass rates, and grouping per-run summaries by
memory backend. Benchmark-specific modules (for example the transparency and
Evo-Memory report metrics) build on these instead of re-deriving denominator
handling or run iteration.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from heams.memory_systems.factory import MemorySystemVariant
from heams.results import RunResult
from heams.scenarios.base import Scenario

#: A single rendered metric row: a label plus one formatted cell per column.
MetricRow = tuple[str, list[str]]

_SummaryT = TypeVar("_SummaryT")


def memory_label(scenario: Scenario, index: int) -> str:
    """Stable display label for the memory backend at ``index``."""
    memory_entry = scenario.memory_systems[index]
    if isinstance(memory_entry, MemorySystemVariant):
        return memory_entry.id or memory_entry.backend_config.type
    return getattr(memory_entry, "type", f"memory#{index}")


def iter_labeled_runs(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> Iterator[tuple[str, Scenario, RunResult]]:
    """Yield ``(memory_label, scenario, run)`` for every recorded run."""
    by_id = {scenario.id: scenario for scenario in scenarios}
    for scenario_id, runs in results.items():
        scenario = by_id.get(scenario_id)
        if scenario is None:
            continue
        for run in runs:
            yield memory_label(scenario, run.memory_index), scenario, run


def weighted_rate(passed: float, total: float) -> float | None:
    """Return ``passed / total`` or ``None`` when there is nothing to score."""
    if total <= 0:
        return None
    return passed / total


@dataclass
class RateAccumulator:
    """Accumulates passed/total counts for weighted pass-rate reporting."""

    passed: float = 0.0
    total: float = 0.0

    def add(self, passed: float, total: float) -> None:
        self.passed += passed
        self.total += total

    def merge(self, other: "RateAccumulator") -> None:
        self.add(other.passed, other.total)

    @property
    def rate(self) -> float | None:
        return weighted_rate(self.passed, self.total)

    @property
    def failures(self) -> float:
        return self.total - self.passed


def group_summaries_by_memory(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    summarize_fn: Callable[[Scenario, RunResult], _SummaryT | None],
) -> dict[str, list[_SummaryT]]:
    """Group non-``None`` per-run summaries by memory backend label."""
    grouped: dict[str, list[_SummaryT]] = defaultdict(list)
    for label, scenario, run in iter_labeled_runs(scenarios, results):
        summary = summarize_fn(scenario, run)
        if summary is not None:
            grouped[label].append(summary)
    return dict(grouped)


def order_memory_labels(
    requested: Sequence[str],
    available: Sequence[str],
) -> list[str]:
    """Keep the requested order, then append any extra labels alphabetically."""
    available_set = set(available)
    ordered = [label for label in requested if label in available_set]
    seen = set(ordered)
    ordered.extend(sorted(label for label in available_set if label not in seen))
    return ordered


__all__ = [
    "MetricRow",
    "RateAccumulator",
    "group_summaries_by_memory",
    "iter_labeled_runs",
    "memory_label",
    "order_memory_labels",
    "weighted_rate",
]
