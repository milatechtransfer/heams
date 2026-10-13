"""LongMemEval benchmark package for the config-driven CLI path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heams.benchmarks.long_mem_eval.benchmark import LongMemEvalBenchmark
    from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig
    from heams.benchmarks.long_mem_eval.report import (
        render_long_mem_eval_metrics_summary,
    )

__all__ = [
    "LongMemEvalBenchmark",
    "LongMemEvalBenchmarkConfig",
    "render_long_mem_eval_metrics_summary",
]


def __getattr__(name: str) -> Any:
    if name == "LongMemEvalBenchmark":
        from heams.benchmarks.long_mem_eval.benchmark import LongMemEvalBenchmark

        return LongMemEvalBenchmark
    if name == "LongMemEvalBenchmarkConfig":
        from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig

        return LongMemEvalBenchmarkConfig
    if name == "render_long_mem_eval_metrics_summary":
        from heams.benchmarks.long_mem_eval.report import (
            render_long_mem_eval_metrics_summary,
        )

        return render_long_mem_eval_metrics_summary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
