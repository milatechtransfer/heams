from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heams.agents.factory import AgentConfig
from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.retention_policy.samples import load_rows
from heams.memory_systems.factory import MemorySystemVariant
from heams.memory_systems.privacy.config import RetentionMode
from heams.memory_systems.privacy.recall import NoOpRecallPolicy, build_recall_policy

#: Retention modes that enforce a policy via its `retention.policy_rules` entry.
_RULE_COMPILING_MODES = frozenset({RetentionMode.DETERMINISTIC, RetentionMode.HYBRID})


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
    def _retention_policy_requires_recall_disabled(self) -> Self:
        """Catch this misconfiguration at resolve time, not mid-run.

        A recall policy filtering results on top of retention makes a
        missing recall ambiguous between "retention blocked it at ingest"
        and "a recall policy withheld it separately" -- exactly what the
        retention_policy benchmark's retain-then-recall design exists to
        remove. Checking here (once per experiment, before any scenario
        runs) fails one bad experiment loudly without risking an
        in-progress multi-experiment run.
        """
        if self.benchmark_config.type != "retention_policy":
            return self
        privacy_config = self.memory_system_config.privacy_config
        if privacy_config is None:
            return self
        # Probe with every compiled policy's rules included: in rule-based
        # modes the benchmark appends them at runtime via
        # `set_retention_policy`, which would silently activate a recall layer
        # that looks rule-less in the YAML. llm mode never appends them.
        retention = privacy_config.retention
        runtime_rules = (
            [rule for rules in retention.policy_rules.values() for rule in rules]
            if retention.mode in _RULE_COMPILING_MODES
            else []
        )
        probe_config = privacy_config.model_copy(
            update={"rules": [*privacy_config.rules, *runtime_rules]}
        )
        # `build_recall_policy` alone, not the full `PrivacyPolicyEngine`: the
        # engine also builds a retention policy this check never reads, which
        # can raise for retention modes unrelated to recall (e.g. the
        # unimplemented `transform` mode) and does real judge-prompt-building
        # work for hybrid/llm retention that would be wasted here.
        recall_policy_is_active = not isinstance(
            build_recall_policy(probe_config), NoOpRecallPolicy
        )
        if recall_policy_is_active:
            raise ValueError(
                f"Experiment '{self.experiment_name}': retention_policy requires "
                "privacy_config.recall.enabled: false on the memory-system config "
                f"({self.memory_system_config_file}). With a recall policy active, "
                "a fact that isn't recalled is ambiguous between 'retention "
                "blocked it at ingest' and 'a recall policy withheld it "
                "separately' -- exactly the ambiguity this benchmark exists to "
                "eliminate."
            )
        return self

    @model_validator(mode="after")
    def _retention_policy_requires_runtime_policy_support(self) -> Self:
        """Fail before any scenario if a sample's `set_retention_policy` would.

        Every retention_policy sample opens by setting its policy on the
        memory system, which needs a `PrivacyMemoryWrapper` (built only for
        an active `privacy_config`) and, for a rule-based layer with a
        `policy_rules` table, an entry for each policy the run will set.
        """
        if self.benchmark_config.type != "retention_policy":
            return self
        privacy_config = self.memory_system_config.privacy_config
        if privacy_config is None or not privacy_config.is_active:
            raise ValueError(
                f"Experiment '{self.experiment_name}': retention_policy sets each "
                "sample's policy via set_retention_policy, which needs an active "
                "privacy_config on the memory-system config "
                f"({self.memory_system_config_file}). For an ungoverned baseline, "
                "use deterministic retention with no rules and no policy_rules."
            )
        retention = privacy_config.retention
        if not retention.policy_rules or retention.mode not in _RULE_COMPILING_MODES:
            return self
        missing = sorted(
            _selected_policy_statements(self.benchmark_config)
            - retention.policy_rules.keys()
        )
        if missing:
            raise ValueError(
                f"Experiment '{self.experiment_name}': retention.policy_rules in "
                f"{self.memory_system_config_file} has no rules for these "
                f"benchmark policies: {missing}."
            )
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
