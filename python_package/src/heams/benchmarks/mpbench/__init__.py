"""MPBench memory-poisoning benchmark package (CLI factory path).

Keep this module light: importing ``judge`` must not pull ``benchmarks.base``
(evaluation.factory registers the judge and would otherwise cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.mpbench.judge import (
    MpbenchGroundTruth,
    MpbenchLlmJudgeConfig,
    MpbenchLlmJudgeEvaluator,
    attack_class_from_audit_key,
)

if TYPE_CHECKING:
    from heams.benchmarks.mpbench.benchmark import MpbenchBenchmark
    from heams.benchmarks.mpbench.config import MpbenchBenchmarkConfig
    from heams.benchmarks.mpbench.report import (
        render_mpbench_metrics_summary,
        write_mpbench_rollup_report,
    )

__all__ = [
    "MpbenchBenchmark",
    "MpbenchBenchmarkConfig",
    "MpbenchGroundTruth",
    "MpbenchLlmJudgeConfig",
    "MpbenchLlmJudgeEvaluator",
    "attack_class_from_audit_key",
    "render_mpbench_metrics_summary",
    "write_mpbench_rollup_report",
]


def __getattr__(name: str) -> Any:
    if name == "MpbenchBenchmark":
        from heams.benchmarks.mpbench.benchmark import MpbenchBenchmark

        return MpbenchBenchmark
    if name == "MpbenchBenchmarkConfig":
        from heams.benchmarks.mpbench.config import MpbenchBenchmarkConfig

        return MpbenchBenchmarkConfig
    if name == "render_mpbench_metrics_summary":
        from heams.benchmarks.mpbench.report import render_mpbench_metrics_summary

        return render_mpbench_metrics_summary
    if name == "write_mpbench_rollup_report":
        from heams.benchmarks.mpbench.report import write_mpbench_rollup_report

        return write_mpbench_rollup_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
