from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class BaseMetaAction(BaseModel):
    """Shared metadata for every step in a `Scenario.execution_sequence`.

    Subclasses must declare `action_type: Literal["..."] = "..."` to join the
    `MetaAction` discriminated union.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    step_evaluation_markers: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional assertions or thresholds checked immediately after this step executes.",
    )


class CommandAction(BaseMetaAction):
    """Shared shape for meta-actions that dispatch a named command with parameters."""

    command: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class UserStimulus(BaseMetaAction):
    action_type: Literal["user_stimulus"] = "user_stimulus"
    content: str


class EnvironmentManipulation(CommandAction):
    action_type: Literal["env_manipulation"] = "env_manipulation"


class MemoryManipulation(CommandAction):
    action_type: Literal["memory_manipulation"] = "memory_manipulation"


class AgentManipulation(CommandAction):
    action_type: Literal["agent_manipulation"] = "agent_manipulation"


MetaAction = Annotated[
    Union[UserStimulus, EnvironmentManipulation, MemoryManipulation, AgentManipulation],
    Field(discriminator="action_type"),
]
