from __future__ import annotations

from heams.environments import AlfWorldEnvironmentConfig, create_environment
from heams.environments.alfworld import AlfWorldEnvironment


def test_alfworld_environment_exposes_household_action_space() -> None:
    environment = create_environment(
        AlfWorldEnvironmentConfig(
            task_id="ALF-STREAM",
            tasks=[
                {
                    "task_id": "alfworld_0",
                    "goal": "Heat the plate and put it on the countertop.",
                    "kind": "heat",
                }
            ],
        )
    )
    assert isinstance(environment, AlfWorldEnvironment)
    environment.reset()

    tools = {tool.__name__: tool for tool in environment.get_available_tools()}
    assert set(tools) == {
        "get_task_state",
        "get_valid_actions",
        "step_task",
        "finish_task",
    }

    initial = tools["get_task_state"]()
    assert initial["dataset"] == "alfworld"
    assert "Heat the plate" in initial["observation"]
    assert "use microwave" in tools["get_valid_actions"]()["valid_actions"]

    pickup_result = tools["step_task"](action="pick up plate")
    assert pickup_result["status"] == "success"
    assert pickup_result["success"] is False

    action_result = tools["step_task"](action="use microwave")
    assert action_result["status"] == "success"
    assert action_result["success"] is True
    assert action_result["result"]["task_id"] == "alfworld_0"
    assert action_result["done"] is True
