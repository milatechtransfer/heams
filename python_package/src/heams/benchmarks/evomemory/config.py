"""Typed workload configuration for EvoMemory tasks."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.evomemory.data import DEFAULT_SINGLE_TURN_DIR


class EvoMemorySingleTurnBenchmarkConfig(BaseModel):
    """Declarative workload settings for ordered EvoMemory single-turn tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["evomemory_single_turn"] = "evomemory_single_turn"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="EvoMemory single-turn",
        dimension=ReportDimension.PERFORMANCE,
        criterion_tags=(("dataset",),),
    )
    examples_dir: Path = Field(default=DEFAULT_SINGLE_TURN_DIR)
    limit: int | None = Field(default=None, ge=1)
    limit_stimuli: int | None = Field(default=None, ge=1)


__all__ = [
    "EvoMemorySingleTurnBenchmarkConfig",
]
