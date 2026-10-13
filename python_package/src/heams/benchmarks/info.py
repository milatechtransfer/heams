"""What each benchmark declares about itself for reports and the results browser.

Every benchmark config class sets ``report_info: ClassVar[BenchmarkInfo]``
next to its ``type``. Reports and ``heams browse`` read benchmark metadata
(display name, section, ordering, criteria, run scope) only from here, so a
new benchmark shows up correctly without changes to reporting or UI code.

Hooks receive the saved ``benchmark_config`` mapping (as written to
``experiment_config.*.yaml``) rather than a validated config, so they keep
working on results produced by older versions of a config schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any


class ReportDimension(StrEnum):
    """Report section a benchmark belongs to, in display order."""

    PERFORMANCE = "Performance"
    PRIVACY = "Privacy"
    EFFICIENCY = "Efficiency"
    """System metrics (duration, tokens, memory operations); not a benchmark section."""


class RunScope(StrEnum):
    """How much of a benchmark's dataset a run covers."""

    FULL = "full"
    """The whole dataset."""
    SAMPLE = "sample"
    """A cap on the same dataset (``limit``): comparable, but smaller."""
    PARTIAL = "partial"
    """A different slice (for example some Retention Policy policies only):
    its scores cannot stand for the benchmark."""


def _limit_scope(config: Mapping[str, Any]) -> RunScope:
    return RunScope.SAMPLE if config.get("limit") is not None else RunScope.FULL


@dataclass(frozen=True)
class BenchmarkInfo:
    """Report metadata a benchmark declares on its config class."""

    display_name: str
    dimension: ReportDimension
    criterion_tags: tuple[tuple[str, ...], ...] = ()
    """Judge-record tag groups naming a criterion, tried in order; the first
    group whose tags are all set wins (``("task_type", "phase")`` gives
    "erase / pre"). A record tagged ``criterion`` always uses that value."""
    context_tokens: int | None = None
    """Typical conversation history per sample, in tokens. Performance
    benchmarks are listed by increasing context size; ``None`` sorts last."""
    variant: Callable[[Mapping[str, Any]], str | None] | None = None
    """Name of a config variant reported as its own benchmark row (BEAM
    splits), or ``None``."""
    variant_context_tokens: Callable[[str], int | None] | None = None
    """``context_tokens`` of a variant, when it differs from the benchmark's."""
    scope: Callable[[Mapping[str, Any]], RunScope] = _limit_scope
    """Dataset coverage of a run with this config."""
    criterion_from_key: Callable[[str], str | None] | None = None
    """Criterion of an audit key, for judges that write no report records."""

    def variant_of(self, config: Mapping[str, Any]) -> str | None:
        return self.variant(config) if self.variant else None

    def label(self, variant: str | None) -> str:
        return f"{self.display_name} {variant}" if variant else self.display_name

    def tokens(self, variant: str | None) -> int | None:
        if variant and self.variant_context_tokens:
            return self.variant_context_tokens(variant)
        return self.context_tokens


def token_count(label: str) -> int | None:
    """Tokens in a size label such as "100K" or "10M"."""
    multipliers = {"K": 1_000, "M": 1_000_000}
    suffix = label[-1:].upper()
    if suffix in multipliers and label[:-1].isdigit():
        return int(label[:-1]) * multipliers[suffix]
    return int(label) if label.isdigit() else None


@cache
def benchmark_infos() -> dict[str, BenchmarkInfo]:
    """``{config type: BenchmarkInfo}`` for every registered benchmark."""
    from typing import get_args

    from heams.benchmarks.factory import BenchmarkConfig

    union = get_args(BenchmarkConfig)[0]
    return {
        config_class.model_fields["type"].default: config_class.report_info
        for config_class in get_args(union)
    }


def benchmark_info(benchmark_type: str) -> BenchmarkInfo | None:
    """The declared info of a benchmark type, or ``None`` if it is unknown
    (for example results of a benchmark this version no longer has)."""
    return benchmark_infos().get(benchmark_type)


__all__ = [
    "BenchmarkInfo",
    "ReportDimension",
    "RunScope",
    "benchmark_info",
    "benchmark_infos",
    "token_count",
]
