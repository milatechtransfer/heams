"""Typed workload config for the MPBench memory-poisoning benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.mpbench.judge import attack_class_from_audit_key
from heams.benchmarks.mpbench.judge import MpbenchLlmJudgeConfig
from heams.utils.settings import settings

_DEFAULT_DATASET_DIR = settings.datasets_dir / "mpbench"


class MpbenchBenchmarkConfig(BaseModel):
    """Typed workload config for MPBench RSR evaluation.

    Prompt strings are required and authored in configs/benchmarks/*.yaml.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["mpbench"] = "mpbench"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="MPBench",
        dimension=ReportDimension.PRIVACY,
        criterion_tags=(("attack_class",),),
        criterion_from_key=attack_class_from_audit_key,
    )
    examples_dir: Path = Field(
        default=_DEFAULT_DATASET_DIR,
        description="Directory for MPBench JSONL files (downloaded on first use).",
    )
    limit: int | None = Field(default=None, ge=1)
    limit_per_class: int | None = Field(
        default=10,
        ge=1,
        description="Max adversarial samples per canonical attack class.",
    )
    auto_download: bool = Field(
        default=True,
        description="Download upstream MPBench JSONL when missing.",
    )
    evaluator: MpbenchLlmJudgeConfig
    answer_prompt: str = Field(
        description="UserStimulus template; must include {question}.",
    )

    @field_validator("answer_prompt")
    @classmethod
    def _require_question_placeholder(cls, value: str) -> str:
        if "{question}" not in value:
            raise ValueError("answer_prompt must include the {question} placeholder.")
        return value


__all__ = ["MpbenchBenchmarkConfig", "_DEFAULT_DATASET_DIR"]
