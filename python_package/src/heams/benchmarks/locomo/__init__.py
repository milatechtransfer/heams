"""LOCOMO benchmark package (CLI factory path).

Keep this module light: importing ``judge`` must not pull ``benchmarks.base``
(evaluation.factory registers the judge and would otherwise cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.locomo.judge import (
    LocomoLlmJudgeConfig,
    LocomoLlmJudgeEvaluator,
    LocomoQA,
    LocomoQAGroundTruth,
    is_scored_locomo_category,
    locomo_category_label_from_key,
)

if TYPE_CHECKING:
    from heams.benchmarks.locomo.benchmark import LocomoBenchmark
    from heams.benchmarks.locomo.config import LocomoBenchmarkConfig
    from heams.benchmarks.locomo.report import (
        render_locomo_metrics_summary,
        write_locomo_rollup_report,
    )

__all__ = [
    "LocomoBenchmark",
    "LocomoBenchmarkConfig",
    "LocomoLlmJudgeConfig",
    "LocomoLlmJudgeEvaluator",
    "LocomoQA",
    "LocomoQAGroundTruth",
    "is_scored_locomo_category",
    "locomo_category_label_from_key",
    "render_locomo_metrics_summary",
    "write_locomo_rollup_report",
]


def __getattr__(name: str) -> Any:
    if name == "LocomoBenchmark":
        from heams.benchmarks.locomo.benchmark import LocomoBenchmark

        return LocomoBenchmark
    if name == "LocomoBenchmarkConfig":
        from heams.benchmarks.locomo.config import LocomoBenchmarkConfig

        return LocomoBenchmarkConfig
    if name == "render_locomo_metrics_summary":
        from heams.benchmarks.locomo.report import render_locomo_metrics_summary

        return render_locomo_metrics_summary
    if name == "write_locomo_rollup_report":
        from heams.benchmarks.locomo.report import write_locomo_rollup_report

        return write_locomo_rollup_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
