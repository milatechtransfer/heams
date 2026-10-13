import pytest
from pathlib import Path
from typing import Any

from heams.agents.config import (
    AgentParameters,
)
from heams.agents.factory import create_agent
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams.memory_systems.shared_types import MemoryEntry

# Model IDs to be used in test matrix
# Note: The requires_ollama marker is applied to the ollama model parameter
# so tests using this fixture will automatically skip when ollama is not available
MODEL_IDS = [
    pytest.param("ollama:qwen2.5:7b", marks=pytest.mark.requires_ollama),
]


@pytest.fixture(
    params=MODEL_IDS,
    ids=[m.values[0] if hasattr(m, "values") else m for m in MODEL_IDS],
)
def model_id(request):
    return request.param


# Agent configs to be used in test matrix
AGENT_TYPES = {
    "agent": AgentParameters,
}


@pytest.fixture(params=AGENT_TYPES.values(), ids=AGENT_TYPES.keys())
def agent_config(request, model_id):
    overrides = request.param if isinstance(request.param, dict) else {}
    config_type = overrides.get("config_type", request.param)
    if config_type == AgentParameters:
        return AgentParameters(
            model_id=overrides.get("model_id", model_id),
            instructions=overrides.get("instructions", "Reply briefly and clearly."),
            model_args=overrides.get("model_args", {"temperature": 0.0}),
        )
    raise ValueError(f"Unknown agent config type: {config_type}")


# Tool sets to be used in test matrix
@pytest.fixture
def tools():
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together and return the result."""
        return a + b

    return [add_numbers]


# Memory systems to be used in test matrix
@pytest.fixture
def memory_system():
    class DummyMemorySystem(BaseMemorySystem):
        def __init__(self, name: str = "dummy_test_memory", top_k: int = 5):
            super().__init__(
                BaseMemoryConfig(type="dummy_memory", name=name, top_k=top_k),
                storage_dir=Path(),
            )
            self._entries: list[MemoryEntry] = []

        def _retain_impl(
            self, content: str, metadata: dict[str, Any]
        ) -> MemoryOperationResult:
            self._entries.append(MemoryEntry(content=content, metadata=metadata or {}))
            return MemoryOperationResult(value="Successfully saved to memory.")

        def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
            if not top_k:
                return MemoryOperationResult(
                    value=[entry.content for entry in self._entries]
                )
            return MemoryOperationResult(
                value=[entry.content for entry in self._entries[-top_k:]]
            )

    return DummyMemorySystem(name="dummy_test_memory")


# Agents to be used in test matrix
@pytest.fixture
def agent(agent_config, tools, memory_system):
    return create_agent(config=agent_config, tools=tools, memory_system=memory_system)
