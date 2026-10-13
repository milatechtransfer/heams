"""Typed workload config for the LOCOMO benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.locomo.judge import locomo_category_label_from_key
from heams.benchmarks.locomo.judge import LocomoLlmJudgeConfig
from heams.utils.settings import settings

_DEFAULT_DATASET_DIR = settings.datasets_dir / "locomo"


class LocomoBenchmarkConfig(BaseModel):
    """Typed workload config for the LOCOMO benchmark.

    Prompt strings are required and authored in configs/benchmarks/*.yaml —
    there are no Python fallback prompt defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["locomo"] = "locomo"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="LoCoMo",
        dimension=ReportDimension.PERFORMANCE,
        criterion_tags=(("category",),),
        # Median conversation history, at ~4 characters per token.
        context_tokens=26_000,
        criterion_from_key=locomo_category_label_from_key,
    )
    examples_dir: Path = Field(
        default=_DEFAULT_DATASET_DIR,
        description="LOCOMO dataset JSON file or directory containing JSON files.",
    )
    limit: int | None = Field(default=None, ge=1)
    limit_stimuli: int | None = Field(default=None, ge=1)
    agent_conversation: Literal["stateful", "independent"] = Field(
        default="independent",
        description=(
            "independent: reset agent messages before each QA; primed memory stays. "
            "stateful: later QA items see prior agent prompts and answers."
        ),
    )
    evaluator: LocomoLlmJudgeConfig

    answer_prompt: str = Field(
        description="UserStimulus template; must include {question}.",
    )
    answer_prompt_cat5: str = Field(
        description="UserStimulus template for category-5 QA; must include {question}.",
    )


__all__ = ["LocomoBenchmarkConfig", "_DEFAULT_DATASET_DIR"]
