"""Typed workload config for the first-class BEAM benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension, token_count
from heams.benchmarks.beam.judge import BeamLlmJudgeConfig

_BEAM_SPLITS = ("100K", "500K", "1M", "10M")


class BeamBenchmarkConfig(BaseModel):
    """Typed workload config for BEAM long-context memory-ability QA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["beam"] = "beam"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="BEAM",
        dimension=ReportDimension.PERFORMANCE,
        criterion_tags=(("question_type",),),
        # Each split is its own row, sized by its configured history length.
        variant=lambda config: str(config.get("split") or "100K"),
        variant_context_tokens=token_count,
    )
    split: Literal["100K", "500K", "1M", "10M"] = "100K"
    local_path: Path | None = Field(
        default=None,
        description="Optional JSON list of conversations; skips HuggingFace download.",
    )
    limit: int | None = Field(default=None, ge=1)
    limit_stimuli: int | None = Field(default=None, ge=1)
    evaluator: BeamLlmJudgeConfig
    answer_prompt: str = Field(
        description="UserStimulus template; must include {question}.",
    )

    @field_validator("answer_prompt")
    @classmethod
    def _require_question_placeholder(cls, value: str) -> str:
        if "{question}" not in value:
            raise ValueError("answer_prompt must include the {question} placeholder.")
        return value


__all__ = ["BeamBenchmarkConfig", "_BEAM_SPLITS"]
