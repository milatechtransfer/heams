from __future__ import annotations

import re
from typing import Any, Literal

from heams.environments.task import EvoTaskEnvironment, EvoTaskEnvironmentConfig
from heams.utils import register


class PddlEnvironmentConfig(EvoTaskEnvironmentConfig):
    """Environment config for Evo-Memory PDDL task streams."""

    type: Literal["pddl_environment"] = "pddl_environment"
    dataset: Literal["pddl"] = "pddl"
    instructions: str = (
        "Complete each PDDL Blocksworld planning goal using task environment tools. "
        "Inspect current predicates, choose applicable operators with "
        "step_task(action), and call finish_task when the goal is satisfied."
    )
    max_steps_per_task: int = 30


@register("pddl_environment")
class PddlEnvironment(EvoTaskEnvironment):
    """Simplified PDDL Blocksworld planning environment."""

    def _initial_state_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        init_predicates = task.get("metadata", {}).get("init_state") or task.get(
            "init_state"
        )
        if isinstance(init_predicates, list):
            predicates = set(init_predicates)
        else:
            predicates = {
                "on(a,table)",
                "on(b,table)",
                "on(c,a)",
                "clear(b)",
                "clear(c)",
                "clear(table)",
                "arm-empty",
            }
        return {
            "predicates": predicates,
            "goal_predicates": self._parse_goal(str(task.get("goal", ""))),
        }

    def _public_state_for_task(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "predicates": sorted(state["predicates"]),
            "goal_predicates": sorted(state["goal_predicates"]),
        }

    def _observation_for_task(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        return (
            "Current state: "
            + ", ".join(sorted(state["predicates"]))
            + ". Goal: "
            + ", ".join(sorted(state["goal_predicates"]))
        )

    def _valid_actions_for_current_task(
        self,
        task: dict[str, Any] | None,
    ) -> list[str]:
        if task is None:
            return []
        state = self._state_for_current_task(task)
        predicates: set[str] = state["predicates"]
        actions: list[str] = []
        objects = _objects_from_predicates(predicates | state["goal_predicates"])

        if "arm-empty" in predicates:
            for obj in objects:
                if f"clear({obj})" not in predicates:
                    continue
                surface = _surface_for_object(obj, predicates)
                if surface == "table":
                    actions.append(f"pickup({obj})")
                elif surface:
                    actions.append(f"unstack({obj},{surface})")
        else:
            held = _held_object(predicates)
            if held:
                for dest in objects:
                    if dest != held and f"clear({dest})" in predicates:
                        actions.append(f"stack({held},{dest})")
                actions.append(f"putdown({held})")

        goal_directed_actions = _goal_directed_actions(
            actions,
            predicates,
            state["goal_predicates"],
        )
        if goal_directed_actions:
            return goal_directed_actions
        return _order_actions_for_goal(actions, state["goal_predicates"])

    def _execute_action(
        self,
        action: str,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state_for_current_task(task)
        predicates: set[str] = state["predicates"]
        action = action.lower().strip()
        observation = "Action failed."
        reward = 0.0

        if pickup_match := re.match(r"pickup\s*\(?(\w+)\)?", action):
            obj = pickup_match.group(1)
            if f"clear({obj})" in predicates and "arm-empty" in predicates:
                for pred in list(predicates):
                    if pred.startswith(f"on({obj},"):
                        surface = pred.split(",")[1].rstrip(")")
                        predicates.remove(pred)
                        predicates.add(f"clear({surface})")
                        break
                predicates.remove("arm-empty")
                predicates.remove(f"clear({obj})")
                predicates.add(f"holding({obj})")
                observation = f"Picked up {obj}."
                reward = 0.1

        if putdown_match := re.match(r"putdown\s*\(?(\w+)\)?", action):
            obj = putdown_match.group(1)
            if f"holding({obj})" in predicates:
                predicates.remove(f"holding({obj})")
                predicates.add(f"on({obj},table)")
                predicates.add("arm-empty")
                predicates.add(f"clear({obj})")
                observation = f"Put {obj} on table."
                reward = 0.1

        if stack_match := re.match(r"stack\s*\(?(\w+)\s*,\s*(\w+)\)?", action):
            obj, dest = stack_match.groups()
            if f"holding({obj})" in predicates and f"clear({dest})" in predicates:
                predicates.remove(f"holding({obj})")
                predicates.remove(f"clear({dest})")
                predicates.add(f"on({obj},{dest})")
                predicates.add("arm-empty")
                predicates.add(f"clear({obj})")
                observation = f"Stacked {obj} on {dest}."
                reward = 0.2

        if unstack_match := re.match(r"unstack\s*\(?(\w+)\s*,\s*(\w+)\)?", action):
            obj, surface = unstack_match.groups()
            if (
                f"on({obj},{surface})" in predicates
                and f"clear({obj})" in predicates
                and "arm-empty" in predicates
            ):
                predicates.remove(f"on({obj},{surface})")
                predicates.remove(f"clear({obj})")
                predicates.remove("arm-empty")
                predicates.add(f"clear({surface})")
                predicates.add(f"holding({obj})")
                observation = f"Unstacked {obj} from {surface}."
                reward = 0.1

        success = state["goal_predicates"].issubset(predicates)
        if success:
            observation = f"{observation} Goal achieved!"
        progress = len(state["goal_predicates"] & predicates) / max(
            len(state["goal_predicates"]),
            1,
        )
        return {
            "observation": observation,
            "success": success,
            "progress": 1.0 if success else max(progress, reward),
            "done": success,
            "reward": 1.0 if success else reward,
        }

    def _history_satisfies_task(self, task: dict[str, Any]) -> bool:
        state = self._state_for_current_task(task)
        return state["goal_predicates"].issubset(state["predicates"])

    @staticmethod
    def _parse_goal(goal_text: str) -> set[str]:
        goal_text = goal_text.lower()
        predicates: set[str] = set()
        for obj, surface in re.findall(r"(\w+) on (\w+)", goal_text):
            predicates.add(f"on({obj},{surface})")
        for obj in re.findall(r"(\w+) is clear", goal_text):
            predicates.add(f"clear({obj})")
        for obj in re.findall(r"holding (\w+)", goal_text):
            predicates.add(f"holding({obj})")
        return predicates


def _objects_from_predicates(predicates: set[str]) -> list[str]:
    objects: set[str] = set()
    for predicate in predicates:
        for obj in re.findall(r"\b[a-z]\w*\b", predicate):
            if obj not in {"on", "clear", "holding", "table", "arm", "empty"}:
                objects.add(obj)
    return sorted(objects)


def _surface_for_object(obj: str, predicates: set[str]) -> str | None:
    for predicate in predicates:
        if predicate.startswith(f"on({obj},"):
            return predicate.split(",", 1)[1].rstrip(")")
    return None


def _held_object(predicates: set[str]) -> str | None:
    for predicate in predicates:
        if match := re.match(r"holding\((\w+)\)", predicate):
            return match.group(1)
    return None


def _goal_directed_actions(
    actions: list[str],
    predicates: set[str],
    goal_predicates: set[str],
) -> list[str]:
    directed: list[str] = []
    held = _held_object(predicates)

    for goal in sorted(goal_predicates):
        if goal in predicates:
            continue
        if goal.startswith("holding("):
            obj = goal.removeprefix("holding(").removesuffix(")")
            directed.extend(_actions_to_hold_object(obj, predicates, held))
        elif goal.startswith("on("):
            obj, dest = goal.removeprefix("on(").removesuffix(")").split(",", 1)
            directed.extend(_actions_to_put_object_on(obj, dest, predicates, held))

    return [action for action in directed if action in actions]


def _actions_to_hold_object(
    obj: str,
    predicates: set[str],
    held: str | None,
) -> list[str]:
    if held == obj:
        return []
    if held:
        return [f"putdown({held})"]
    if f"clear({obj})" in predicates:
        surface = _surface_for_object(obj, predicates)
        if surface == "table":
            return [f"pickup({obj})"]
        return [f"unstack({obj},{surface})"]
    blocker = _top_object_on(obj, predicates)
    return [f"unstack({blocker},{obj})"] if blocker else []


def _actions_to_put_object_on(
    obj: str,
    dest: str,
    predicates: set[str],
    held: str | None,
) -> list[str]:
    if held == obj:
        if dest == "table":
            return [f"putdown({obj})"]
        return [f"stack({obj},{dest})"]
    if held:
        return [f"putdown({held})"]
    if f"clear({obj})" not in predicates:
        blocker = _top_object_on(obj, predicates)
        return [f"unstack({blocker},{obj})"] if blocker else []
    if dest != "table" and f"clear({dest})" not in predicates:
        blocker = _top_object_on(dest, predicates)
        return [f"unstack({blocker},{dest})"] if blocker else []
    surface = _surface_for_object(obj, predicates)
    if surface == "table":
        return [f"pickup({obj})"]
    return [f"unstack({obj},{surface})"]


def _top_object_on(surface: str, predicates: set[str]) -> str | None:
    for predicate in predicates:
        if match := re.match(r"on\((\w+),\s*(\w+)\)", predicate):
            obj, current_surface = match.groups()
            if current_surface == surface:
                return obj
    return None


def _order_actions_for_goal(actions: list[str], goal_predicates: set[str]) -> list[str]:
    def score(action: str) -> tuple[int, str]:
        for goal in goal_predicates:
            if goal.startswith("holding(") and action.startswith("pickup"):
                obj = goal.removeprefix("holding(").removesuffix(")")
                if action == f"pickup({obj})":
                    return (0, action)
            if goal.startswith("on(") and action.startswith("stack"):
                obj, dest = goal.removeprefix("on(").removesuffix(")").split(",", 1)
                if action == f"stack({obj},{dest})":
                    return (0, action)
        for goal in goal_predicates:
            if goal.startswith("on(") and action.startswith("pickup"):
                obj = goal.removeprefix("on(").split(",", 1)[0]
                if action == f"pickup({obj})":
                    return (1, action)
        return (2, action)

    return sorted(actions, key=score)
