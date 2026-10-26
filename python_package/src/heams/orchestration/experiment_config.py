from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heams.agents.factory import AgentConfig
from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.retention_policy.samples import load_rows
from heams.memory_systems.factory import MemorySystemVariant


class ResolvedExperimentExecutionConfig(BaseModel):
    """Per-experiment execution contract for the refactored run orchestrator.

    Agent, memory-system, and benchmark configs are fully typed here so the
    execution boundary receives one resolved experiment tuple.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_schema_version: int = Field(default=1, ge=1)
    run_id: str = Field(..., min_length=1)
    run_name: str = Field(..., min_length=1)
    experiment_name: str = Field(..., min_length=1)
    experiment_dir: Path
    agent_config_file: Path
    memory_system_config_file: Path
    benchmark_config_file: Path
    agent_config: AgentConfig
    memory_system_config: MemorySystemVariant
    benchmark_config: BenchmarkConfig

    @field_validator("experiment_name")
    @classmethod
    def _name_is_safe_directory_component(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or path.name != value or value in {".", ".."}:
            raise ValueError("Experiment names must be a single directory name.")
        return value

    @model_validator(mode="after")
    def _memory_system_accepts_the_retention_policies(self) -> Self:
        """Fail before any scenario if the memory system can't run these policies.

        Each config in the memory-system variant (the backend's and any
        wrapper's) checks the benchmark's policies through the
        `check_retention_policies` contract hook. A system without governance
        passes and runs ungoverned. Checking here, once per experiment, fails one
        bad experiment loudly without risking an in-progress multi-experiment run.
        """
        if self.benchmark_config.type != "retention_policy":
            return self
        policies = sorted(_selected_policy_statements(self.benchmark_config))
        configs = [
            self.memory_system_config.backend_config,
            *self.memory_system_config.wrapper_configs.values(),
        ]
        for config in configs:
            check = getattr(config, "check_retention_policies", None)
            if not callable(check):
                continue
            try:
                check(policies)
            except ValueError as exc:
                raise ValueError(
                    f"Experiment '{self.experiment_name}' "
                    f"({self.memory_system_config_file}): {exc}"
                ) from exc
        return self


def _selected_policy_statements(
    benchmark_config: RetentionPolicyBenchmarkConfig,
) -> set[str]:
    rows = load_rows(benchmark_config.dataset_path)
    wanted = benchmark_config.policy_ids
    return {
        row["policy"]
        for row in rows
        if row.get("policy") and (wanted is None or row.get("policy_id") in wanted)
    }
