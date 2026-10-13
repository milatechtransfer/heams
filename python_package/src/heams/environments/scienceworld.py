from __future__ import annotations

from typing import Any, Literal

from heams.environments.task import EvoTaskEnvironment, EvoTaskEnvironmentConfig
from heams.utils import register


class ScienceWorldEnvironmentConfig(EvoTaskEnvironmentConfig):
    """Environment config for Evo-Memory ScienceWorld task streams."""

    type: Literal["scienceworld_environment"] = "scienceworld_environment"
    dataset: Literal["scienceworld"] = "scienceworld"
    instructions: str = (
        "Complete each ScienceWorld experiment using task environment tools. "
        "Inspect lab state, manipulate objects with step_task(action), and call "
        "finish_task when the experiment goal is complete."
    )
    max_steps_per_task: int = 50


@register("scienceworld_environment")
class ScienceWorldEnvironment(EvoTaskEnvironment):
    """Simplified ScienceWorld science experiment environment."""

    def _initial_state_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "inventory": [],
            "current_location": "lab",
            "experiment_state": {},
            "objects": {
                "thermometer": {"location": "lab", "type": "tool"},
                "beaker": {"location": "lab", "type": "container", "contains": None},
                "bunsen_burner": {
                    "location": "lab",
                    "type": "heat_source",
                    "state": "off",
                },
                "ice": {
                    "location": "freezer",
                    "type": "substance",
                    "state": "solid",
                    "temp": 0,
                },
                "water": {
                    "location": "sink",
                    "type": "substance",
                    "state": "liquid",
                    "temp": 20,
                },
                "plant": {
                    "location": "greenhouse",
                    "type": "living",
                    "state": "healthy",
                },
                "soil": {"location": "greenhouse", "type": "substance"},
                "seed": {"location": "storage", "type": "living", "state": "dormant"},
            },
        }

    def _observation_for_task(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        parts = [f"Task: {task.get('goal')}", f"Location: {state['current_location']}"]
        visible = [
            name
            for name, obj in state["objects"].items()
            if obj.get("location") == state["current_location"]
        ]
        if visible:
            parts.append(f"You see: {', '.join(visible)}")
        if state["inventory"]:
            parts.append(f"Inventory: {', '.join(state['inventory'])}")
        if state["experiment_state"]:
            parts.append(f"Experiment: {state['experiment_state']}")
        return " | ".join(parts)

    def _valid_actions_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> list[str]:
        if task is None:
            return []
        state = self._state_for_current_task(task)
        objects = sorted(state["objects"])
        visible_objects = sorted(
            name
            for name, obj in state["objects"].items()
            if obj.get("location") == state["current_location"]
        )
        inventory = sorted(state["inventory"])
        pickup_actions = [f"pick up {obj}" for obj in visible_objects]
        placement_actions = [
            f"put {obj} in beaker"
            for obj in inventory
            if state["objects"].get("beaker", {}).get("location")
            in {state["current_location"], "inventory"}
        ]
        return [
            "go to lab",
            "go to greenhouse",
            "go to storage",
            "go to sink",
            "go to freezer",
            *pickup_actions,
            *placement_actions,
            *[f"heat {obj}" for obj in objects],
            *[f"cool {obj}" for obj in objects],
            *[f"measure {obj}" for obj in objects],
            "mix inventory",
            "look",
        ]

    def _execute_action(
        self,
        action: str,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state_for_current_task(task)
        action_lower = action.lower().strip()
        observation = "Nothing happens."
        reward = 0.0

        if action_lower.startswith("go to"):
            location = action_lower.replace("go to", "").strip()
            if location in {"lab", "greenhouse", "storage", "sink", "freezer"}:
                state["current_location"] = location
                observation = f"You moved to the {location}."
            else:
                observation = f"Unknown location: {location}"
        elif action_lower.startswith("pick up") or action_lower.startswith("take"):
            obj_name = _object_name_from_action(action_lower, ("pick up", "take"))
            if obj_name in state["objects"]:
                obj = state["objects"][obj_name]
                if obj.get("location") == state["current_location"]:
                    state["inventory"].append(obj_name)
                    obj["location"] = "inventory"
                    observation = f"You picked up the {obj_name}."
                    reward = 0.05
                else:
                    observation = f"The {obj_name} is not here."
            else:
                observation = f"No {obj_name} found."
        elif action_lower.startswith("put") or action_lower.startswith("place"):
            observation = self._put_object(action_lower, state)
            if observation.startswith("You put"):
                reward = 0.1
        elif action_lower.startswith("heat"):
            observation = self._heat_object(
                _object_name_from_action(action_lower, ("heat",)),
                state,
            )
            if not observation.startswith("No "):
                reward = 0.2
        elif action_lower.startswith("cool"):
            observation = self._cool_object(
                _object_name_from_action(action_lower, ("cool",)),
                state,
            )
            if not observation.startswith("No "):
                reward = 0.2
        elif action_lower.startswith("measure") or action_lower.startswith(
            "use thermometer"
        ):
            obj_name = _object_name_from_action(
                action_lower,
                ("measure", "use thermometer on"),
            )
            if obj_name in state["objects"]:
                temp = state["objects"][obj_name].get("temp", 20)
                observation = f"The {obj_name} is at {temp}°C."
                state["experiment_state"]["measured"] = {
                    "object": obj_name,
                    "temp": temp,
                }
                reward = 0.1
        elif action_lower.startswith("mix"):
            if len(state["inventory"]) >= 2:
                observation = f"You mixed {' and '.join(state['inventory'])}."
                state["experiment_state"]["mixed"] = list(state["inventory"])
                reward = 0.2
        elif action_lower in {"look", "examine", "look around"}:
            observation = self._observation_for_task(task, state)

        success = self._goal_is_complete(task, state)
        if success:
            observation = f"{observation} Task completed!"
            reward = 1.0
        return {
            "observation": observation,
            "success": success,
            "progress": self._calculate_progress(state),
            "done": success,
            "reward": reward,
        }

    def _history_satisfies_task(self, task: dict[str, Any]) -> bool:
        return self._goal_is_complete(task, self._state_for_current_task(task))

    @staticmethod
    def _put_object(action_lower: str, state: dict[str, Any]) -> str:
        parts = action_lower.replace("put", "").replace("place", "").strip()
        for sep in (" in ", " on "):
            if sep not in parts:
                continue
            obj_name, container = [
                _normalize_object_name(part) for part in parts.split(sep, 1)
            ]
            if obj_name in state["inventory"]:
                state["inventory"].remove(obj_name)
                state["objects"][obj_name]["location"] = container
                return f"You put the {obj_name} in/on the {container}."
        return "Could not place the object."

    @staticmethod
    def _heat_object(obj_name: str, state: dict[str, Any]) -> str:
        if obj_name not in state["objects"]:
            return f"No {obj_name} found."
        obj = state["objects"][obj_name]
        obj["temp"] = obj.get("temp", 20) + 50
        state["experiment_state"]["heated"] = obj_name
        if obj.get("state") == "solid" and obj["temp"] > 0:
            obj["state"] = "liquid"
            return f"The {obj_name} melted!"
        if obj.get("state") == "liquid" and obj["temp"] > 100:
            obj["state"] = "gas"
            return f"The {obj_name} evaporated!"
        return f"You heated the {obj_name}. Temperature: {obj['temp']}°C"

    @staticmethod
    def _cool_object(obj_name: str, state: dict[str, Any]) -> str:
        if obj_name not in state["objects"]:
            return f"No {obj_name} found."
        obj = state["objects"][obj_name]
        obj["temp"] = obj.get("temp", 20) - 30
        state["experiment_state"]["cooled"] = obj_name
        if obj.get("state") == "liquid" and obj["temp"] < 0:
            obj["state"] = "solid"
            return f"The {obj_name} froze!"
        return f"You cooled the {obj_name}. Temperature: {obj['temp']}°C"

    @staticmethod
    def _goal_is_complete(task: dict[str, Any], state: dict[str, Any]) -> bool:
        goal = str(task.get("goal", "")).lower()
        if "melt" in goal and "measure" in goal and "temperature" in goal:
            return bool(
                state["experiment_state"].get("heated")
                and state["experiment_state"].get("measured")
            )
        if "melt" in goal:
            return bool(state["experiment_state"].get("heated"))
        if "freeze" in goal or "solidify" in goal:
            return any(
                obj.get("state") == "solid" and obj.get("temp", 20) < 0
                for obj in state["objects"].values()
            )
        if "measure" in goal and "temperature" in goal:
            return bool(state["experiment_state"].get("measured"))
        if "boil" in goal or "evaporate" in goal:
            return any(obj.get("state") == "gas" for obj in state["objects"].values())
        return False

    @staticmethod
    def _calculate_progress(state: dict[str, Any]) -> float:
        progress = 0.0
        if state["experiment_state"]:
            progress += 0.3
        if state["inventory"]:
            progress += 0.1 * len(state["inventory"])
        if any(obj.get("temp", 20) != 20 for obj in state["objects"].values()):
            progress += 0.2
        return min(progress, 0.9)


def _object_name_from_action(action: str, prefixes: tuple[str, ...]) -> str:
    name = action
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name.replace(prefix, "", 1)
            break
    return _normalize_object_name(name)


def _normalize_object_name(name: str) -> str:
    tokens = [token for token in name.strip().split() if token not in {"object", "the"}]
    return "_".join(tokens)
