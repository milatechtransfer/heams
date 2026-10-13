"""Transparency benchmark package (CLI factory path).

Keep this module light: importing ``judge`` must not pull ``benchmarks.base``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.transparency.judge import (
    TransparencyLlmJudgeConfig,
    TransparencyQA,
    TransparencyQAGroundTruth,
)

if TYPE_CHECKING:
    from heams.benchmarks.transparency.benchmark import TransparencyBenchmark
    from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig
    from heams.benchmarks.transparency.report import (
        build_transparency_report,
        render_transparency_detail_sections,
        render_transparency_metrics_summary,
        render_transparency_rollup_sections,
    )

# Agent instructions for YAML / docs; runtime policy lives in agent config files.
TRANSPARENCY_AGENT_INSTRUCTIONS = (
    "You are evaluating transparent access to persistent memory for a chat "
    "assistant. Use the memory recall tool to inspect stored memories before "
    "answering audit questions. Disclose only memories that answer the user's "
    "request, include available source or timestamp metadata when asked, and "
    "clearly state when requested audit information is not available rather "
    "than inventing provenance or hidden state."
)

__all__ = [
    "TRANSPARENCY_AGENT_INSTRUCTIONS",
    "TransparencyBenchmark",
    "TransparencyBenchmarkConfig",
    "TransparencyLlmJudgeConfig",
    "TransparencyQA",
    "TransparencyQAGroundTruth",
    "build_transparency_report",
    "render_transparency_detail_sections",
    "render_transparency_metrics_summary",
    "render_transparency_rollup_sections",
]


def __getattr__(name: str) -> Any:
    if name == "TransparencyBenchmark":
        from heams.benchmarks.transparency.benchmark import TransparencyBenchmark

        return TransparencyBenchmark
    if name == "TransparencyBenchmarkConfig":
        from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig

        return TransparencyBenchmarkConfig
    if name in {
        "build_transparency_report",
        "render_transparency_detail_sections",
        "render_transparency_metrics_summary",
        "render_transparency_rollup_sections",
    }:
        from heams.benchmarks.transparency import report as _report

        return getattr(_report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
