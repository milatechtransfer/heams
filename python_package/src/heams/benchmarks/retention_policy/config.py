"""Typed workload config for the Retention Policy benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.benchmarks.info import BenchmarkInfo, ReportDimension, RunScope
from heams.evaluation.scenarios.retention_policy import RetentionPolicyEvaluatorConfig
from heams.utils import settings

DEFAULT_DATASET = settings.datasets_dir / "retention_policy" / "dataset.json"


def _run_scope(config: Mapping[str, Any]) -> RunScope:
    # A run over some policies scores those policies only, not the benchmark.
    if config.get("policy_ids"):
        return RunScope.PARTIAL
    return RunScope.SAMPLE if config.get("limit") is not None else RunScope.FULL


class RetentionPolicyBenchmarkConfig(BaseModel):
    """Config for the Retention Policy benchmark.

    The dataset is a flat list of rows, each one chunk under one prohibition-only
    policy. Rows are grouped by `policy_id` into one sample per policy, so every
    fact under a policy is retained then recall-probed in the same scenario.
    `policy_ids` narrows the run to named policies -- the smoke config uses it
    to exercise a single policy end to end rather than a thin slice of every
    policy, which would leave whole policies untested.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["retention_policy"] = "retention_policy"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="Retention Policy",
        dimension=ReportDimension.PRIVACY,
        criterion_tags=(("policy_id",), ("category",)),
        scope=_run_scope,
    )
    dataset_path: Path = Field(default=DEFAULT_DATASET)
    policy_ids: list[str] | None = Field(
        default=None,
        description="Only run rows whose policy_id is listed. None runs all.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap the number of policy samples after policy_id filtering. "
            "Applied per policy, not per row, so a policy's fact collection "
            "is never truncated mid-sample."
        ),
    )
    evaluator: RetentionPolicyEvaluatorConfig = Field(
        default_factory=RetentionPolicyEvaluatorConfig
    )


__all__ = ["DEFAULT_DATASET", "RetentionPolicyBenchmarkConfig"]
