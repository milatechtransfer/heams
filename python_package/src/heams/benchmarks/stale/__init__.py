"""STALE benchmark package (CLI factory path).

STALE tests whether a memory-augmented agent recognizes that a stored memory has
become *stale* after a user's state changed. Keep this module light: importing
``judge`` must not pull ``benchmarks.base``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.stale.judge import (
    StaleLlmJudgeConfig,
    StaleQA,
    StaleQAGroundTruth,
)

if TYPE_CHECKING:
    from heams.benchmarks.stale.benchmark import StaleBenchmark
    from heams.benchmarks.stale.config import StaleBenchmarkConfig
    from heams.benchmarks.stale.report import (
        render_stale_metrics_summary,
        render_stale_rollup_sections,
        write_stale_rollup_report,
    )

# Suggested agent instructions for YAML / docs; runtime policy lives in the
# agent config files referenced by an experiment. Kept deliberately neutral
# (mirrors STALE's target-model prompt): the agent is NOT told to prefer recent
# state, because detecting staleness on its own is exactly what STALE measures.
STALE_AGENT_INSTRUCTIONS = (
    "You are a helpful assistant. Use the memory recall tool to review the "
    "conversation history with the user, then accurately answer the user's "
    "latest query."
)

__all__ = [
    "STALE_AGENT_INSTRUCTIONS",
    "StaleBenchmark",
    "StaleBenchmarkConfig",
    "StaleLlmJudgeConfig",
    "StaleQA",
    "StaleQAGroundTruth",
    "render_stale_metrics_summary",
    "render_stale_rollup_sections",
    "write_stale_rollup_report",
]

_REPORT_EXPORTS = {
    "render_stale_metrics_summary",
    "render_stale_rollup_sections",
    "write_stale_rollup_report",
}


def __getattr__(name: str) -> Any:
    if name == "StaleBenchmark":
        from heams.benchmarks.stale.benchmark import StaleBenchmark

        return StaleBenchmark
    if name == "StaleBenchmarkConfig":
        from heams.benchmarks.stale.config import StaleBenchmarkConfig

        return StaleBenchmarkConfig
    if name in _REPORT_EXPORTS:
        from heams.benchmarks.stale import report as _report

        return getattr(_report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
