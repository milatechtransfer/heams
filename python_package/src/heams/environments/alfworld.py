from __future__ import annotations

from typing import Any, Literal

from heams.environments.task import EvoTaskEnvironment, EvoTaskEnvironmentConfig
from heams.utils import register


class AlfWorldEnvironmentConfig(EvoTaskEnvironmentConfig):
    """Environment config for Evo-Memory AlfWorld task streams."""

    type: Literal["alfworld_environment"] = "alfworld_environment"
    dataset: Literal["alfworld"] = "alfworld"
    instructions: str = (
        "Complete each AlfWorld household goal using task environment tools. "
        "Call get_task_state and get_valid_actions, use step_task(action) for "
        "household actions, and call finish_task when the current goal is complete."
    )
    max_steps_per_task: int = 50


@register("alfworld_environment")
class AlfWorldEnvironment(EvoTaskEnvironment):
    """Simplified AlfWorld household instruction environment."""

    def _initial_state_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "current_location": "kitchen",
            "inventory": [],
            "objects": {
                "apple": {"location": "countertop", "state": "normal"},
                "mug": {"location": "cabinet", "state": "normal"},
                "plate": {"location": "sink", "state": "dirty"},
                "knife": {"location": "drawer", "state": "normal"},
            },
        }

    def _observation_for_task(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        return (
            f"You are in the {state['current_location']}. You see a countertop, "
            f"sink, microwave, and fridge. Your task is: {task.get('goal')}"
        )

    def _valid_actions_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> list[str]:
        if task is None:
            return []
        actions = [
            "go to kitchen",
            "go to living room",
            "go to bedroom",
            "look around",
            "check inventory",
            "examine",
        ]
        for obj in self._state_for_current_task(task)["objects"]:
            actions.extend(
                [
                    f"pick up {obj}",
                    f"put {obj} in/on sink",
                    "use microwave",
                    "use fridge",
                ]
            )
        return actions

    def _execute_action(
        self,
        action: str,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state_for_current_task(task)
        action_lower = action.lower().strip()
        observation = "Nothing happens."
        success = False

        if action_lower.startswith("go to"):
            state["current_location"] = action_lower.replace("go to", "").strip()
            observation = f"You moved to the {state['current_location']}."
        elif action_lower in {"look", "look around"}:
            observation = (
                f"You are in the {state['current_location']}. "
                "You see various objects around."
            )
        elif action_lower.startswith("pick up") or action_lower.startswith("take"):
            obj = action_lower.replace("pick up", "").replace("take", "").strip()
            if obj in state["objects"]:
                state["inventory"].append(obj)
                observation = f"You picked up the {obj}."
                success = str(task.get("kind")) == "pick"
            else:
                observation = f"You don't see a {obj} here."
        elif action_lower.startswith("put"):
            if state["inventory"]:
                obj = state["inventory"].pop()
                observation = f"You put the {obj} down."
                success = str(task.get("kind")) == "put"
            else:
                observation = "You're not holding anything."
        elif "use" in action_lower:
            if (
                "microwave" in action_lower
                and str(task.get("kind")) == "heat"
                and state["inventory"]
            ):
                observation = (
                    f"You heated the {state['inventory'][-1]} in the microwave."
                )
                success = True
            elif (
                "fridge" in action_lower
                and str(task.get("kind")) == "cool"
                and state["inventory"]
            ):
                observation = f"You cooled the {state['inventory'][-1]} in the fridge."
                success = True
            elif "sink" in action_lower and str(task.get("kind")) == "clean":
                observation = "You cleaned the object in the sink."
                success = True
            else:
                observation = "You use the appliance but nothing special happens."
        elif action_lower in {"check inventory", "inventory"}:
            inventory = state["inventory"]
            observation = (
                f"You are carrying: {', '.join(inventory)}"
                if inventory
                else "Your inventory is empty."
            )

        progress = min(self._task_step_count / 10, 0.9)
        return {
            "observation": observation,
            "success": success,
            "progress": 1.0 if success else progress,
            "done": success,
            "reward": 1.0
            if success
            else (
                0.1
                if action_lower.startswith(("pick up", "take"))
                and "picked up" in observation
                else 0.0
            ),
        }

    def _max_steps_observation(self) -> str:
        return "Maximum steps reached."

    def _max_steps_progress(self) -> float:
        return 0.5

    def _history_satisfies_task(self, task: dict[str, Any]) -> bool:
        actions = " ".join(self._actions_for_current_task()).lower()
        goal = str(task.get("goal", "")).lower()
        kind = str(task.get("kind", "")).lower()
        if kind == "clean" or "clean" in goal:
            return "clean" in actions or "wash" in actions or "sink" in actions
        if kind == "heat" or "heat" in goal or "hot" in goal:
            return "heat" in actions or "microwave" in actions
        if kind == "cool" or "cool" in goal or "cooled" in goal:
            return "cool" in actions or "fridge" in actions or "refrigerator" in actions
        if kind == "put" or "put" in goal:
            return "put" in actions or "place" in actions
        if kind == "pick" or "pick" in goal:
            return "pick" in actions or "take" in actions
        return "done" in actions or "finish" in actions
