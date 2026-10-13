from __future__ import annotations

from typing import Any, Literal

from heams.environments.task import EvoTaskEnvironment, EvoTaskEnvironmentConfig
from heams.utils import register

_BABYAI_COLORS = {"red", "blue", "green", "yellow", "purple", "grey"}


class BabyAIEnvironmentConfig(EvoTaskEnvironmentConfig):
    """Environment config for Evo-Memory BabyAI task streams."""

    type: Literal["babyai_environment"] = "babyai_environment"
    dataset: Literal["babyai"] = "babyai"
    instructions: str = (
        "Complete each BabyAI grid-world goal using task environment tools. "
        "Call get_task_state and get_valid_actions, use step_task(action) "
        "for navigation/actions, and call finish_task when the current goal "
        "is complete."
    )
    max_steps_per_task: int = 64


@register("babyai_environment")
class BabyAIEnvironment(EvoTaskEnvironment):
    """Simplified BabyAI grid-world environment."""

    def _initial_state_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_pos": (3, 3),
            "agent_dir": 0,
            "carrying": None,
            "objects": {
                "red_ball": {"pos": (1, 1), "type": "ball", "color": "red"},
                "blue_key": {"pos": (5, 2), "type": "key", "color": "blue"},
                "green_box": {"pos": (2, 5), "type": "box", "color": "green"},
                "yellow_door": {
                    "pos": (6, 3),
                    "type": "door",
                    "color": "yellow",
                    "open": False,
                },
            },
        }

    def _observation_for_task(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        nearby = []
        for obj in state["objects"].values():
            distance = abs(obj["pos"][0] - state["agent_pos"][0]) + abs(
                obj["pos"][1] - state["agent_pos"][1]
            )
            if distance <= 2:
                nearby.append(
                    f"{obj['color']} {obj['type']} to the "
                    f"{self._direction(state['agent_pos'], obj['pos'])}"
                )
        parts = [f"Goal: {task.get('goal')}", f"Position: {state['agent_pos']}"]
        parts.append(
            f"You see: {', '.join(nearby)}" if nearby else "You see nothing nearby."
        )
        if state.get("carrying"):
            parts.append(f"Carrying: {state['carrying']}")
        return " | ".join(parts)

    def _valid_actions_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> list[str]:
        if task is None:
            return []
        state = self._state_for_current_task(task)
        object_actions = []
        for name, obj in sorted(state["objects"].items()):
            object_actions.append(f"go to {name}")
            if obj["type"] in {"ball", "key", "box"}:
                object_actions.append(f"pick up {name}")
            if obj["type"] == "door":
                object_actions.append(f"toggle {name}")
        return [
            "turn left",
            "turn right",
            "go forward",
            *object_actions,
            "drop",
            "done",
        ]

    def _execute_action(
        self,
        action: str,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state_for_current_task(task)
        action_lower = action.lower().strip()
        observation = f"Unknown action: {action}"
        done = False
        success = False

        if action_lower in {"turn left", "left"}:
            state["agent_dir"] = (state["agent_dir"] - 1) % 4
            observation = "You turned left."
        elif action_lower in {"turn right", "right"}:
            state["agent_dir"] = (state["agent_dir"] + 1) % 4
            observation = "You turned right."
        elif action_lower in {"go forward", "forward", "move"}:
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][state["agent_dir"]]
            x, y = state["agent_pos"]
            new_pos = (x + dx, y + dy)
            if 0 <= new_pos[0] < 7 and 0 <= new_pos[1] < 7:
                state["agent_pos"] = new_pos
                observation = f"You moved to {new_pos}."
            else:
                observation = "You hit a wall."
        elif action_lower.startswith("go to"):
            target_name = _object_name_from_action(action_lower, "go to")
            target = state["objects"].get(target_name)
            if target is None:
                observation = f"Unknown target: {target_name}."
            else:
                state["agent_pos"] = target["pos"]
                observation = (
                    f"You moved next to the {target['color']} {target['type']}."
                )
        elif action_lower.startswith("pick up") or action_lower == "pickup":
            target_name = _object_name_from_action(action_lower, "pick up")
            observation = "Nothing to pick up here."
            for name, obj in state["objects"].items():
                if target_name and name != target_name:
                    continue
                if obj["pos"] == state["agent_pos"] and obj["type"] in {
                    "ball",
                    "key",
                    "box",
                }:
                    state["carrying"] = name
                    observation = f"You picked up the {obj['color']} {obj['type']}."
                    break
        elif action_lower == "drop":
            if state["carrying"]:
                state["objects"][state["carrying"]]["pos"] = state["agent_pos"]
                observation = f"You dropped the {state['carrying']}."
                state["carrying"] = None
            else:
                observation = "You're not carrying anything."
        elif action_lower.startswith("toggle") or action_lower == "open":
            target_name = _object_name_from_action(action_lower, "toggle")
            observation = "No door nearby."
            for name, obj in state["objects"].items():
                if target_name and name != target_name:
                    continue
                if obj["type"] != "door":
                    continue
                distance = abs(obj["pos"][0] - state["agent_pos"][0]) + abs(
                    obj["pos"][1] - state["agent_pos"][1]
                )
                if distance <= 1:
                    obj["open"] = not obj["open"]
                    observation = (
                        f"You {'opened' if obj['open'] else 'closed'} "
                        f"the {obj['color']} door."
                    )
                    break
        elif action_lower == "done":
            done = True

        if self._goal_is_complete(task, state):
            success = True
            done = True
            observation = f"{observation} Goal achieved!"
        elif action_lower == "done":
            observation = "Task failed."

        progress = min(self._task_step_count / 20, 0.8)
        return {
            "observation": observation,
            "success": success,
            "progress": 1.0 if success else (0.4 if done else progress),
            "done": done,
            "reward": 1.0 if success else (0.2 if "picked up" in observation else 0.0),
        }

    def _history_satisfies_task(self, task: dict[str, Any]) -> bool:
        return self._goal_is_complete(task, self._state_for_current_task(task))

    def _goal_is_complete(self, task: dict[str, Any], state: dict[str, Any]) -> bool:
        goal = str(task.get("goal", "")).lower()
        if "pick up" in goal or "get" in goal:
            carrying = str(state.get("carrying") or "")
            return any(color in goal and color in carrying for color in _BABYAI_COLORS)
        if "go to" in goal:
            for obj in state["objects"].values():
                if obj["color"] not in goal or obj["type"] not in goal:
                    continue
                distance = abs(obj["pos"][0] - state["agent_pos"][0]) + abs(
                    obj["pos"][1] - state["agent_pos"][1]
                )
                return distance <= 1
        return False

    @staticmethod
    def _direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"


def _object_name_from_action(action: str, prefix: str) -> str:
    name = action.replace(prefix, "", 1).strip()
    return name.replace(" ", "_")
