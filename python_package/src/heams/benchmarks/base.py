"""Shared contract and errors for config-driven benchmark workloads.

Each typed benchmark config is dispatched through ``create_benchmark`` to a
``BaseBenchmark`` that exposes ordered ``BenchmarkSample`` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from heams.benchmarks.sample import BenchmarkSample


@dataclass(frozen=True)
class SkipReason:
    """Bookkeeping for a scenario that could not be included in a run.

    The CLI surfaces these in both the stdout table and the Markdown report
    so regressions are visible rather than silent.
    """

    path: Path
    reason: str


class BenchmarkDatasetUnavailableError(RuntimeError):
    """A benchmark's default dataset could not be located or materialized.

    Raised (instead of silently skipping) when a required dataset is missing
    and cannot be downloaded, so the CLI can surface an actionable message and
    exit non-zero rather than reporting an empty run.
    """


class BaseBenchmark(ABC):
    """Eagerly materialized workload emitted by a typed benchmark config."""

    skipped: list[SkipReason]

    @property
    @abstractmethod
    def samples(self) -> Sequence[BenchmarkSample]:
        """Return runnable samples in benchmark-defined order."""

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
