"""Typed workload config for the Contextual Integrity (CI) benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.contextual_integrity.judge import CILlmJudgeConfig
from heams.utils.settings import settings

_DEFAULT_DATASET_DIR = settings.datasets_dir / "contextual_integrity"


class CIDatasetSourceConfig(BaseModel):
    """Where the CI corpus comes from when no local JSON is present.

    The upstream dataset ships as a single parquet on HuggingFace. The first
    run downloads it, parses the two JSON-string columns, and caches the result
    as a local fixture so later runs are offline. Set ``auto_download: false``
    to make a missing dataset a hard error instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    auto_download: bool = Field(
        default=True,
        description=(
            "Download and cache the corpus from HuggingFace when the local "
            "dataset file is absent. When false, a missing file is an error."
        ),
    )
    hf_repo_id: str = Field(
        default="huseyinatahaninan/ContextualIntegritySyntheticDataset",
        description="HuggingFace dataset repo id.",
    )
    split: str = Field(
        default="train",
        description="Dataset split to download; upstream currently ships train only.",
    )
    parquet_path_template: str = Field(
        default="data/{split}-00000-of-00001.parquet",
        description="Path of the parquet inside the HF repo; formatted with {split}.",
    )
    fixture_name_template: str = Field(
        default="ci_{split}_{limit}.json",
        description=(
            "File name for the cached fixture; formatted with {split} and "
            "{limit}. Keep {limit} so runs with different download_limit values "
            "(e.g. smoke vs full) cache to separate files instead of reusing "
            "each other's download."
        ),
    )
    download_limit: int | None = Field(
        default=16,
        ge=1,
        description=(
            "Rows to keep when building the cached fixture. Null caches the "
            "whole split."
        ),
    )


class CIBenchmarkConfig(BaseModel):
    """Typed workload config for the Contextual Integrity benchmark.

    Prompt strings are required and authored in configs/benchmarks/*.yaml;
    there are no Python fallback prompt defaults.

    This config deliberately holds no agent or memory policy. The legacy CI
    benchmark forced `instructions`, `model_args.timeout`, and memory
    `top_k` onto whatever components it was handed; those are component
    settings and now belong in configs/agents/*.yaml and
    configs/memory_systems/*.yaml respectively.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["ci"] = "ci"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="Contextual Integrity",
        dimension=ReportDimension.PRIVACY,
        criterion_tags=(("record_type",),),
    )
    examples_dir: Path = Field(
        default=_DEFAULT_DATASET_DIR,
        description="CI dataset JSON file or directory containing JSON files.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Cap on materialized samples, applied after loading.",
    )
    dataset_source: CIDatasetSourceConfig = Field(
        default_factory=CIDatasetSourceConfig,
        description="Corpus bootstrap settings used when no local JSON exists.",
    )
    user_task_prompt: str = Field(
        description=(
            "UserStimulus template; must include {user_task}. Keep it neutral: "
            "wrapper text that hints disclosure is being tested biases the "
            "measurement."
        ),
    )
    evaluator: CILlmJudgeConfig

    @field_validator("user_task_prompt")
    @classmethod
    def _require_user_task_placeholder(cls, value: str) -> str:
        if "{user_task}" not in value:
            raise ValueError(
                "user_task_prompt must include the {user_task} placeholder."
            )
        return value

    def fixture_name(self) -> str:
        source = self.dataset_source
        limit = "full" if source.download_limit is None else str(source.download_limit)
        return source.fixture_name_template.format(split=source.split, limit=limit)

    def parquet_path(self) -> str:
        source = self.dataset_source
        return source.parquet_path_template.format(split=source.split)


__all__ = ["CIBenchmarkConfig", "CIDatasetSourceConfig", "_DEFAULT_DATASET_DIR"]
