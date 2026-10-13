from pydantic import BaseModel, ConfigDict, Field

from typing import Any, Literal

from heams.agents.factory import AgentConfig
from heams.environments.factory import EnvironmentConfig
from heams.evaluation.factory import EvaluatorConfig
from heams.memory_systems.shared_types import MemoryEntry
from heams.memory_systems.factory import MemorySystemVariant
from heams.scenarios.actions import MetaAction

AgentConversation = Literal["stateful", "independent"]


class EnvironmentPriming(BaseModel):
    """Data used to populate the environment before the scenario."""

    model_config = ConfigDict(extra="forbid")

    config: EnvironmentConfig | None = Field(
        default=None,
        description="Typed environment config forwarded to create_environment().",
    )


class MemoryPriming(BaseModel):
    """Data used to populate the memory before the scenario."""

    model_config = ConfigDict(extra="forbid")

    entries: list[MemoryEntry] = Field(
        default_factory=list,
        description="Memory entries to retain in the memory system before execution.",
    )


class ScenarioPriming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: EnvironmentPriming = Field(
        default_factory=EnvironmentPriming,
        description="Environment priming for the scenario",
    )
    memory: MemoryPriming = Field(
        default_factory=MemoryPriming, description="Memory priming for the scenario"
    )


class Scenario(BaseModel):
    """The data model for a benchmark test case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier, e.g., 'MOZ-PRIV-001'")
    name: str = Field(..., description="Name of the scenario")

    tags: list[str] = Field(default_factory=list, description="Tags for the scenario")

    scenario_priming: ScenarioPriming = Field(
        default_factory=ScenarioPriming, description="Priming data for the scenario"
    )

    agents: list[AgentConfig] = Field(
        default_factory=list, description="Agents for the scenario"
    )
    memory_systems: list[MemorySystemVariant] = Field(
        default_factory=list, description="Memory systems for the scenario"
    )
    evaluators: list[EvaluatorConfig] = Field(
        default_factory=list, description="Evaluators for the scenario"
    )
    # Typed as Any to avoid import cycles with evaluation schemas; runtime value
    # is BaseGroundTruth | None authored by the benchmark sample.
    ground_truth: Any | None = Field(
        default=None,
        description="Benchmark-authored ground truth consumed by evaluators at audit time.",
    )

    execution_sequence: list[MetaAction] = Field(
        ..., min_length=1, description="Sequence of actions to execute in the scenario"
    )
    agent_conversation: AgentConversation = Field(
        default="stateful",
        description=(
            "stateful: later UserStimulus turns keep prior agent messages. "
            "independent: reset_state() before each UserStimulus; primed memory stays."
        ),
    )
