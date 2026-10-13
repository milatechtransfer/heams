from enum import Enum
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


class MemoryStrategy(str, Enum):
    NO_MEMORY = "no_memory"
    TOOL_USE = "tool_use"  # Memory interactions are entirely mediated through agent tool calls.
    DETERMINISTIC = "deterministic"  # Memory entries are deterministically recalled and injected into the agent prompt; the full turn is retained after each run.
    DETERMINISTIC_WITH_DELETE = "deterministic_with_delete"
    RECALL_ONLY_TOOL_USE = "recall_only_tool_use"
    RECALL_ONLY_DETERMINISTIC = "recall_only_deterministic"

    @property
    def uses_memory_tools(self) -> bool:
        return self in {
            MemoryStrategy.TOOL_USE,
            MemoryStrategy.RECALL_ONLY_TOOL_USE,
        }

    @property
    def uses_deterministic_recall(self) -> bool:
        return self in {
            MemoryStrategy.DETERMINISTIC,
            MemoryStrategy.DETERMINISTIC_WITH_DELETE,
            MemoryStrategy.RECALL_ONLY_DETERMINISTIC,
        }

    @property
    def allows_memory_writes(self) -> bool:
        return self in {
            MemoryStrategy.TOOL_USE,
            MemoryStrategy.DETERMINISTIC,
            MemoryStrategy.DETERMINISTIC_WITH_DELETE,
        }

    @property
    def uses_deterministic_delete(self) -> bool:
        return self is MemoryStrategy.DETERMINISTIC_WITH_DELETE


class AgentParameters(BaseModel):
    type: Literal["agent"] = "agent"
    memory_strategy: MemoryStrategy = MemoryStrategy.NO_MEMORY
    model_id: str
    instructions: str
    model_args: Dict[str, Any] = Field(default_factory=dict)
    max_tool_executions: int | None = Field(default=10, ge=1)
    max_input_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum estimated input tokens sent to each model call.",
    )
    approx_chars_per_token: int = Field(
        default=4,
        gt=0,
        description="Character-to-token ratio used by the input budget guard.",
    )
    # Keep benchmark stdout clean by default; enable when debugging a
    # single scenario interactively.
    enable_console_trace: bool = False


def default_benchmark_agent_config(model_id: str) -> AgentParameters:
    """Defaults for the benchmark CLI harness (deterministic temperature)."""
    return AgentParameters(
        model_id=model_id,
        instructions="",
        model_args={"temperature": 0.0},
    )
