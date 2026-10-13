from __future__ import annotations

from heams.environments import PddlEnvironmentConfig, create_environment
from heams.environments.pddl import PddlEnvironment


def test_pddl_environment_applies_blocksworld_actions() -> None:
    environment = create_environment(
        PddlEnvironmentConfig(
            task_id="PDDL-STREAM",
            tasks=[
                {
                    "task_id": "pddl_0",
                    "goal": "holding a",
                    "init_state": ["on(a,table)", "clear(a)", "arm-empty"],
                }
            ],
        )
    )
    assert isinstance(environment, PddlEnvironment)
    environment.reset()

    tools = {tool.__name__: tool for tool in environment.get_available_tools()}
    initial = tools["get_task_state"]()
    assert initial["dataset"] == "pddl"
    assert "arm-empty" in initial["observation"]
    assert tools["get_valid_actions"]()["valid_actions"] == ["pickup(a)"]

    result = tools["step_task"](action="pickup(a)")
    assert result["success"] is True
    assert result["result"]["task_id"] == "pddl_0"


def test_pddl_environment_prioritizes_goal_directed_actions() -> None:
    environment = create_environment(
        PddlEnvironmentConfig(
            task_id="PDDL-STREAM",
            tasks=[
                {
                    "task_id": "pddl_0",
                    "goal": "a on b, b on table",
                    "init_state": [
                        "on(a,table)",
                        "on(b,table)",
                        "clear(a)",
                        "clear(b)",
                        "arm-empty",
                    ],
                }
            ],
        )
    )
    environment.reset()

    tools = {tool.__name__: tool for tool in environment.get_available_tools()}
    assert tools["get_valid_actions"]()["valid_actions"] == ["pickup(a)"]

    first = tools["step_task"](action="pickup(a)")
    assert first["success"] is False
    assert first["valid_actions"] == ["stack(a,b)"]

    second = tools["step_task"](action="stack(a,b)")
    assert second["success"] is True
