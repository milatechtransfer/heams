"""Erasability & editability (EE) benchmark package (CLI factory path).

EE tests whether a memory-augmented agent honours an explicit deletion request
and cascades a root-entity change to its dependent entities. It is built on the
MEME dataset (https://huggingface.co/datasets/meme-benchmark/MEME), restricted to
the `erase` (MEME `Del`) and `edit` (MEME `Cas`) task
types.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.benchmarks.erasability_editability.judge import (
    EELlmJudgeConfig,
    EEQa,
    EEQaGroundTruth,
)

if TYPE_CHECKING:
    from heams.benchmarks.erasability_editability.benchmark import EEBenchmark
    from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig
    from heams.benchmarks.erasability_editability.report import (
        render_ee_metrics_summary,
        render_ee_rollup_sections,
        write_ee_rollup_report,
    )

ERASABILITY_EDITABILITY_AGENT_INSTRUCTIONS = (
    "You are a helpful assistant with a long-term memory. Use the memory tools to "
    "record what the user tells you, to recall earlier context before answering, "
    "and to remove information when the user asks you to forget it. Answer the "
    "user's latest message accurately and concisely."
)

__all__ = [
    "ERASABILITY_EDITABILITY_AGENT_INSTRUCTIONS",
    "EEBenchmark",
    "EEBenchmarkConfig",
    "EELlmJudgeConfig",
    "EEQa",
    "EEQaGroundTruth",
    "render_ee_metrics_summary",
    "render_ee_rollup_sections",
    "write_ee_rollup_report",
]

_REPORT_EXPORTS = {
    "render_ee_metrics_summary",
    "render_ee_rollup_sections",
    "write_ee_rollup_report",
}


def __getattr__(name: str) -> Any:
    if name == "EEBenchmark":
        from heams.benchmarks.erasability_editability.benchmark import EEBenchmark

        return EEBenchmark
    if name == "EEBenchmarkConfig":
        from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig

        return EEBenchmarkConfig
    if name in _REPORT_EXPORTS:
        from heams.benchmarks.erasability_editability import report as _report

        return getattr(_report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
