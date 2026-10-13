"""Contextual Integrity (CI) benchmark package (CLI factory path).

Keep this module light: importing ``judge`` must not pull ``benchmarks.base``
(evaluation.factory registers the judge and would otherwise cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.contextual_integrity.judge import (
    CILlmJudgeConfig,
    CILlmJudgeEvaluator,
)

if TYPE_CHECKING:
    from heams.benchmarks.contextual_integrity.benchmark import CIBenchmark
    from heams.benchmarks.contextual_integrity.config import (
        CIBenchmarkConfig,
        CIDatasetSourceConfig,
    )
    from heams.benchmarks.contextual_integrity.report import (
        render_ci_metrics_summary,
        write_ci_rollup_report,
    )

__all__ = [
    "CIBenchmark",
    "CIBenchmarkConfig",
    "CIDatasetSourceConfig",
    "CILlmJudgeConfig",
    "CILlmJudgeEvaluator",
    "render_ci_metrics_summary",
    "write_ci_rollup_report",
]

_LAZY = {
    "CIBenchmark": ("heams.benchmarks.contextual_integrity.benchmark", "CIBenchmark"),
    "CIBenchmarkConfig": (
        "heams.benchmarks.contextual_integrity.config",
        "CIBenchmarkConfig",
    ),
    "CIDatasetSourceConfig": (
        "heams.benchmarks.contextual_integrity.config",
        "CIDatasetSourceConfig",
    ),
    "render_ci_metrics_summary": (
        "heams.benchmarks.contextual_integrity.report",
        "render_ci_metrics_summary",
    ),
    "write_ci_rollup_report": (
        "heams.benchmarks.contextual_integrity.report",
        "write_ci_rollup_report",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    return getattr(import_module(module_name), attr)
