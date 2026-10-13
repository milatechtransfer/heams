"""BEAM long-context memory-ability benchmark (CLI factory path).

Keep this module light: importing ``judge`` must not pull ``benchmarks.base``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.beam.judge import (
    BeamLlmJudgeConfig,
    BeamLlmJudgeEvaluator,
    BeamQA,
    BeamQAGroundTruth,
    question_type_from_audit_key,
)

if TYPE_CHECKING:
    from heams.benchmarks.beam.benchmark import BeamBenchmark
    from heams.benchmarks.beam.config import BeamBenchmarkConfig
    from heams.benchmarks.beam.report import (
        render_beam_metrics_summary,
        write_beam_rollup_report,
    )

__all__ = [
    "BeamBenchmark",
    "BeamBenchmarkConfig",
    "BeamLlmJudgeConfig",
    "BeamLlmJudgeEvaluator",
    "BeamQA",
    "BeamQAGroundTruth",
    "question_type_from_audit_key",
    "render_beam_metrics_summary",
    "write_beam_rollup_report",
]


def __getattr__(name: str) -> Any:
    if name == "BeamBenchmark":
        from heams.benchmarks.beam.benchmark import BeamBenchmark

        return BeamBenchmark
    if name == "BeamBenchmarkConfig":
        from heams.benchmarks.beam.config import BeamBenchmarkConfig

        return BeamBenchmarkConfig
    if name == "render_beam_metrics_summary":
        from heams.benchmarks.beam.report import render_beam_metrics_summary

        return render_beam_metrics_summary
    if name == "write_beam_rollup_report":
        from heams.benchmarks.beam.report import write_beam_rollup_report

        return write_beam_rollup_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
