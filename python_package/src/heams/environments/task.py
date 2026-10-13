from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from heams.environments.base import BaseEnvironment
from heams.tools import BenchmarkToolSpec, compile_tool, evo_task_tool_specs


class EvoTaskEnvironmentConfig(BaseModel):
    """Shared config fields for Evo-Memory interactive task streams."""

    task_id: str = "task_stream"
    dataset: str
    instructions: str
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    max_steps_per_task: int = Field(default=10, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvoTaskEnvironment(BaseEnvironment, ABC):
    """Shared stream lifecycle for Evo-Memory interactive environments."""

    def __init__(self, config: EvoTaskEnvironmentConfig):
        self.config = config
        self._closed = False
        self._current_task_index = 0
        self._task_step_count = 0
        self._actions_by_task: dict[str, list[str]] = {}
        self._state_by_task: dict[str, dict[str, Any]] = {}
        self._results: list[dict[str, Any]] = []
        self._override_observation: str | None = None

    def reset(self, environment_config: EvoTaskEnvironmentConfig | None = None) -> None:
        if environment_config is not None:
            self.config = environment_config
        self._closed = False
        self._current_task_index = 0
        self._task_step_count = 0
        self._actions_by_task = {}
        self._state_by_task = {}
        self._results = []
        self._override_observation = None

    def get_available_tools(self) -> list[Callable[..., Any]]:
        return [compile_tool(spec) for spec in self.get_tool_specs()]

    def get_tool_specs(self) -> list[BenchmarkToolSpec]:
        return evo_task_tool_specs(self)

    def get_task_state(self) -> dict[str, Any]:
        """Return the current task stream state, observation, and progress."""

        current_task = self._current_task()
        env_state = self._state_for_current_task(current_task)
        return {
            "status": "success" if not self._closed else "error",
            "stream_id": self.config.task_id,
            "dataset": self.config.dataset,
            "task_id": current_task.get("task_id") if current_task else None,
            "task_index": self._current_task_index,
            "total_tasks": len(self.config.tasks),
            "instructions": self.config.instructions,
            "observation": self._current_observation(),
            "step_index": self._task_step_count,
            "max_steps": self.config.max_steps_per_task,
            "valid_actions": self._valid_actions_for_current_task(current_task),
            "actions": list(self._actions_for_current_task()),
            "environment_state": self._public_state_for_task(env_state),
            "completed_tasks": list(self._results),
            "done": self._is_done(),
            "metadata": dict(self.config.metadata),
        }

    def get_valid_actions(self) -> dict[str, Any]:
        """Return the valid action space for the current task."""

        current_task = self._current_task()
        return {
            "status": "success" if current_task is not None else "error",
            "task_id": current_task.get("task_id") if current_task else None,
            "valid_actions": self._valid_actions_for_current_task(current_task),
            "done": self._is_done(),
        }

    def step_task(self, action: str) -> dict[str, Any]:
        """Execute one environment action for the current task."""

        if self._closed:
            return self._error("The environment is closed.")
        current_task = self._current_task()
        if current_task is None:
            return self._error("The task stream is complete.")
        if not isinstance(action, str) or action == "":
            return self._error("action must be a non-empty string.")

        self._actions_for_current_task().append(action)
        self._task_step_count += 1
        if self._task_step_count >= self.config.max_steps_per_task:
            transition = {
                "observation": self._max_steps_observation(),
                "reward": 0.0,
                "done": True,
                "success": False,
                "progress": self._max_steps_progress(),
            }
        else:
            transition = self._execute_action(action, current_task)

        if transition["done"]:
            return self._complete_current_task(
                success=bool(transition["success"]),
                terminal_action=action,
                observation=str(transition["observation"]),
                progress=float(transition["progress"]),
                reward=float(transition.get("reward", 0.0)),
            )

        return {
            **self.get_task_state(),
            "message": transition["observation"],
            "success": transition["success"],
            "progress": transition["progress"],
            "reward": transition.get("reward", 0.0),
        }

    def finish_task(self) -> dict[str, Any]:
        """Finish the current task and advance the stream when solved."""

        if self._closed:
            return self._error("The environment is closed.")
        current_task = self._current_task()
        if current_task is None:
            return self._error("The task stream is complete.")

        success = self._history_satisfies_task(current_task)
        return self._complete_current_task(
            success=success,
            terminal_action="finish_task",
            observation="Task completed!" if success else "Task failed.",
            progress=1.0 if success else 0.4,
            reward=1.0 if success else 0.0,
        )

    def set_observation(self, observation: str) -> None:
        if not isinstance(observation, str) or observation == "":
            raise ValueError("observation must be a non-empty string.")
        self._override_observation = observation

    def dispatch_command(self, command: str, parameters: dict[str, Any]) -> None:
        if command != "set_observation":
            raise ValueError(
                "Unknown task environment command "
                f"{command!r}. Allowed: ['set_observation']."
            )
        self.set_observation(**parameters)

    def close(self) -> None:
        self._closed = True

    def _current_observation(self) -> str | None:
        if self._override_observation is not None:
            return self._override_observation
        current_task = self._current_task()
        if current_task is None:
            return None
        return self._observation_for_task(
            current_task,
            self._state_for_current_task(current_task),
        )

    def _is_done(self) -> bool:
        return self._current_task_index >= len(self.config.tasks)

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "stream_id": self.config.task_id,
            "done": self._is_done(),
        }

    def _current_task(self) -> dict[str, Any] | None:
        if self._current_task_index >= len(self.config.tasks):
            return None
        return self.config.tasks[self._current_task_index]

    def _actions_for_current_task(self) -> list[str]:
        current_task = self._current_task()
        task_id = str(current_task.get("task_id")) if current_task else "<done>"
        return self._actions_by_task.setdefault(task_id, [])

    def _state_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if task is None:
            return {}
        task_id = str(task.get("task_id"))
        if task_id not in self._state_by_task:
            self._state_by_task[task_id] = self._initial_state_for_task(task)
        return self._state_by_task[task_id]

    def _complete_current_task(
        self,
        *,
        success: bool,
        terminal_action: str,
        observation: str,
        progress: float,
        reward: float,
    ) -> dict[str, Any]:
        current_task = self._current_task()
        if current_task is None:
            return self._error("The task stream is complete.")

        result = {
            "task_id": current_task.get("task_id"),
            "goal": current_task.get("goal"),
            "success": success,
            "progress": progress,
            "steps": self._task_step_count,
            "terminal_action": terminal_action,
            "observation": observation,
            "reward": reward,
        }
        self._results.append(result)
        self._current_task_index += 1
        self._task_step_count = 0
        self._override_observation = None

        return {
            "status": "success",
            "message": observation,
            "result": result,
            "success": success,
            "progress": result["progress"],
            "reward": result["reward"],
            "completed_tasks": list(self._results),
            "done": self._is_done(),
            "next_task": self._current_task(),
        }

    def _public_state_for_task(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)

    def _history_satisfies_task(self, task: dict[str, Any]) -> bool:
        return False

    def _max_steps_observation(self) -> str:
        return "Max steps reached."

    def _max_steps_progress(self) -> float:
        return 0.3

    @abstractmethod
    def _initial_state_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return initial environment state for a task."""

    @abstractmethod
    def _observation_for_task(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        """Return task-specific observation text."""

    @abstractmethod
    def _valid_actions_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> list[str]:
        """Return task-specific action space for the active task."""

    @abstractmethod
    def _execute_action(
        self,
        action: str,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply an action and return observation/success/progress/done."""
