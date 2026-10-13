from __future__ import annotations

from heams.environments import ScienceWorldEnvironmentConfig, create_environment
from heams.environments.scienceworld import ScienceWorldEnvironment


def test_scienceworld_environment_runs_experiment_actions() -> None:
    environment = create_environment(
        ScienceWorldEnvironmentConfig(
            task_id="SCIENCE-STREAM",
            tasks=[
                {
                    "task_id": "scienceworld_0",
                    "goal": "Melt the ice and measure its temperature.",
                    "kind": "phase_change",
                }
            ],
        )
    )
    assert isinstance(environment, ScienceWorldEnvironment)
    environment.reset()

    tools = {tool.__name__: tool for tool in environment.get_available_tools()}
    initial = tools["get_task_state"]()
    assert initial["dataset"] == "scienceworld"
    assert "Location: lab" in initial["observation"]
    assert "heat ice" in tools["get_valid_actions"]()["valid_actions"]

    result = tools["step_task"](action="heat ice")
    assert result["success"] is False
    result = tools["step_task"](action="measure object ice")
    assert result["success"] is True
    assert result["result"]["task_id"] == "scienceworld_0"
