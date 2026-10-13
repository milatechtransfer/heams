"""Typed workload config for the STALE benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.stale.judge import StaleLlmJudgeConfig


class StaleSourceConfig(BaseModel):
    """Where the STALE corpus comes from.

    The published dataset is a single JSON file on HuggingFace
    (`STALEproj/STALE` / `T1_T2_400_FULL.json`, ~306 MB). By default the first run
    downloads it (cached under `~/.cache/huggingface`). Set ``local_path`` to read
    a local *_MAIN.json instead, or ``auto_download: false`` to make a missing
    dataset a hard error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str = Field(
        default="STALEproj/STALE", description="HuggingFace dataset repo id."
    )
    filename: str = Field(
        default="T1_T2_400_FULL.json",
        description="Dataset file name inside the HuggingFace repo.",
    )
    revision: str | None = Field(
        default=None, description="Optional HuggingFace revision (branch/tag/commit)."
    )
    auto_download: bool = Field(
        default=True,
        description=(
            "Download from HuggingFace when `local_path` is unset. When false, a "
            "missing dataset is an error."
        ),
    )
    local_path: Path | None = Field(
        default=None,
        description=(
            "Optional local STALE *_MAIN.json; used instead of downloading when set."
        ),
    )


class StaleBenchmarkConfig(BaseModel):
    """Typed workload config for the STALE benchmark.

    The probe prompt and judge prompts are authored in
    ``configs/benchmarks/*.yaml``; there are no Python prompt fallbacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["stale"] = "stale"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="STALE",
        dimension=ReportDimension.PERFORMANCE,
        criterion_tags=(("dimension_label",), ("dimension",), ("conflict_type",)),
    )
    source: StaleSourceConfig = Field(default_factory=StaleSourceConfig)
    limit: int | None = Field(
        default=None, ge=1, description="Cap on source records, applied after loading."
    )
    default_conflict_type: str = Field(
        default="unknown",
        description="Conflict-type label for records that omit `type`/`conflict_type`.",
    )
    probe_prompt: str = Field(
        description="UserStimulus template for each probe; must include {query}.",
    )
    agent_conversation: Literal["stateful", "independent"] = Field(
        default="stateful",
        description=(
            "stateful: the three probes share one conversation. independent: "
            "reset agent messages before each probe, as upstream STALE runs them."
        ),
    )
    evaluator: StaleLlmJudgeConfig

    @field_validator("probe_prompt")
    @classmethod
    def _require_query_placeholder(cls, value: str) -> str:
        if "{query}" not in value:
            raise ValueError("probe_prompt must include the {query} placeholder.")
        return value


__all__ = ["StaleBenchmarkConfig", "StaleSourceConfig"]
