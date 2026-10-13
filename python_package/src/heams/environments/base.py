from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from heams.tools import BenchmarkToolSpec


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, environment_config: Any = None) -> None:
        """Initializes the environment for a new run."""

    @abstractmethod
    def get_available_tools(self) -> list[Callable[..., Any]]:
        """Returns the tools an agent can call to interact with the environment.

        Every state change an agent drives flows through these tools; their
        arguments and outputs are captured in the agent trace and become the
        authoritative record consumed by evaluators.
        """

    def get_tool_specs(self) -> list[BenchmarkToolSpec]:
        """Returns typed tool specs for environments migrated to ToolSpec."""
        return []

    @abstractmethod
    def close(self) -> None:
        """Releases any environment resources."""

    def dispatch_command(self, command: str, parameters: dict[str, Any]) -> None:
        """Execute an `EnvironmentManipulation` step emitted by a scenario.

        These are scenario-driven out-of-band mutations (e.g. switching
        workspace, seeding device state) that the orchestrator performs
        between agent turns. They are deliberately separate from
        `get_available_tools`, which exposes the agent-visible surface.
        Subclasses override to expose a narrow allowlist.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support environment manipulation "
            f"commands (got '{command}')."
        )
