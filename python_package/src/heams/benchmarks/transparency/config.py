"""Typed workload config for the Transparency benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.transparency.beam import BeamTransparencyAdapterConfig
from heams.benchmarks.transparency.judge import TransparencyLlmJudgeConfig


class JsonTransparencySourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["json"] = "json"
    path: Path


class BeamTransparencySourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["beam"] = "beam"
    split: Literal["100K", "500K", "1M"] = "100K"
    materialization_limit: int | None = Field(default=None, ge=1)
    adapter: BeamTransparencyAdapterConfig = Field(
        default_factory=BeamTransparencyAdapterConfig
    )


TransparencySourceConfig = Annotated[
    JsonTransparencySourceConfig | BeamTransparencySourceConfig,
    Field(discriminator="type"),
]


class TransparencyBenchmarkConfig(BaseModel):
    """Typed workload config for the Transparency benchmark.

    Prompt strings are required and authored in configs/benchmarks/*.yaml —
    there are no Python fallback prompt defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["transparency"] = "transparency"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="Transparency",
        dimension=ReportDimension.PRIVACY,
        criterion_tags=(("dimension",), ("privacy_domain",), ("scenario_type",)),
    )
    source: TransparencySourceConfig
    limit: int | None = Field(default=None, ge=1)
    audit_prompt: str = Field(
        description="UserStimulus template; must include {question}.",
    )
    evaluator: TransparencyLlmJudgeConfig

    @field_validator("audit_prompt")
    @classmethod
    def _require_question_placeholder(cls, value: str) -> str:
        if "{question}" not in value:
            raise ValueError("audit_prompt must include the {question} placeholder.")
        return value


__all__ = [
    "BeamTransparencySourceConfig",
    "JsonTransparencySourceConfig",
    "TransparencyBenchmarkConfig",
    "TransparencySourceConfig",
]
