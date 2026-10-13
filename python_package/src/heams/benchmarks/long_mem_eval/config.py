"""Typed workload config for the LongMemEval benchmark."""

from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.evaluation.factory import LongMemEvalLlmJudgeConfig
from heams.utils.settings import settings

_DEFAULT_DATASET_DIR = settings.datasets_dir / "long_mem_eval"
_REQUIRED_ANSWER_PROMPT_FIELDS = {"question", "question_date"}


def _criterion_from_key(key: str) -> str:
    from heams.evaluation.scenarios.long_mem_eval_llm_judge import (
        long_mem_eval_tags_from_key,
    )

    tags = long_mem_eval_tags_from_key(key)
    return f"{tags['question_type']} / {tags['abstention']}"


class LongMemEvalBenchmarkConfig(BaseModel):
    """Typed workload config for LongMemEval memory-recall QA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["long_mem_eval"] = "long_mem_eval"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="LongMemEval",
        dimension=ReportDimension.PERFORMANCE,
        criterion_tags=(("question_type", "abstention"), ("question_type",)),
        # Median history of the default `s` split, at ~4 characters per token.
        context_tokens=123_000,
        criterion_from_key=_criterion_from_key,
    )
    examples_dir: Path = Field(
        default=_DEFAULT_DATASET_DIR,
        description="LongMemEval dataset JSON file or directory containing JSON files.",
    )
    limit: int | None = Field(default=None, ge=1)
    evaluator: LongMemEvalLlmJudgeConfig
    answer_prompt: str = Field(
        description="UserStimulus template; must include {question} and {question_date}.",
    )

    @field_validator("answer_prompt")
    @classmethod
    def validate_answer_prompt(cls, value: str) -> str:
        field_names = {
            field_name
            for _, field_name, _, _ in Formatter().parse(value)
            if field_name is not None
        }
        if field_names != _REQUIRED_ANSWER_PROMPT_FIELDS:
            raise ValueError(
                "answer_prompt must contain exactly {question} and {question_date}"
            )
        return value


__all__ = ["LongMemEvalBenchmarkConfig", "_DEFAULT_DATASET_DIR"]
