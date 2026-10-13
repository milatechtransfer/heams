from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heams.agents.factory import AgentConfig
from heams.memory_systems.factory import MemorySystemVariant
from heams.result_ledger import ResultKey
from heams.scenarios.base import Scenario


class ScenarioExecutionConfig(BaseModel):
    """Scenario execution contract consumed by the Orchestrator."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scenarios: list[Scenario] = Field(
        ..., min_length=1, description="Scenarios to execute in order."
    )
    workspace_dir: Path = Field(
        ...,
        description="Directory where per-run memory files are written and wiped.",
    )
    agent_config: AgentConfig | None = Field(
        default=None,
        description=(
            "Resolved experiment agent configuration. When supplied, it takes "
            "precedence over legacy scenario-local agent lists."
        ),
    )
    memory_system_config: MemorySystemVariant | None = Field(
        default=None,
        description=(
            "Resolved experiment memory configuration. When supplied, it takes "
            "precedence over legacy scenario-local memory lists."
        ),
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="Independent repetitions per scenario/component combo.",
    )
    pending_keys: set[ResultKey] | None = Field(
        default=None,
        description=(
            "Scenario/sample keys that still need execution. None means execute all."
        ),
    )

    @model_validator(mode="after")
    def resolved_components_are_supplied_together(self) -> "ScenarioExecutionConfig":
        if (self.agent_config is None) != (self.memory_system_config is None):
            raise ValueError(
                "agent_config and memory_system_config must be supplied together."
            )
        return self
