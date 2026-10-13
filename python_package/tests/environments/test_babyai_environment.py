from __future__ import annotations

from heams.environments import BabyAIEnvironmentConfig, create_environment
from heams.environments.babyai import BabyAIEnvironment


def test_babyai_environment_exposes_grid_world_action_space() -> None:
    environment = create_environment(
        BabyAIEnvironmentConfig(
            task_id="BABYAI-STREAM",
            tasks=[{"task_id": "babyai_0", "goal": "go to the red ball"}],
        )
    )
    assert isinstance(environment, BabyAIEnvironment)
    environment.reset()

    tools = {tool.__name__: tool for tool in environment.get_available_tools()}
    initial = tools["get_task_state"]()
    assert initial["dataset"] == "babyai"
    assert "Goal: go to the red ball" in initial["observation"]
    valid_actions = tools["get_valid_actions"]()["valid_actions"]
    assert "go forward" in valid_actions
    assert "go to red_ball" in valid_actions

    result = tools["step_task"](action="go to red ball")
    assert result["success"] is True
    assert result["result"]["task_id"] == "babyai_0"


def test_babyai_environment_dispatch_can_override_observation() -> None:
    environment = create_environment(
        BabyAIEnvironmentConfig(
            task_id="BABYAI-STREAM",
            tasks=[{"task_id": "babyai_0", "goal": "go to the red ball"}],
        )
    )
    environment.reset()

    environment.dispatch_command(
        "set_observation",
        {"observation": "scenario supplied observation"},
    )

    assert (
        environment.get_task_state()["observation"] == "scenario supplied observation"
    )
