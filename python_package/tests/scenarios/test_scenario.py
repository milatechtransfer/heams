import pytest
from pydantic import ValidationError

from heams.scenarios import (
    AgentManipulation,
    EnvironmentManipulation,
    MemoryManipulation,
    Scenario,
    ScenarioPriming,
    UserStimulus,
)
from heams.scenarios.actions import BaseMetaAction


def test_memory_manipulation_uses_parameters_not_data() -> None:
    action = MemoryManipulation(command="inject_fact", parameters={"k": "v"})

    assert action.parameters == {"k": "v"}

    with pytest.raises(ValidationError):
        MemoryManipulation.model_validate(
            {"command": "inject_fact", "data": {"k": "v"}}
        )


def test_scenario_dict_round_trip_preserves_all_fields() -> None:
    data = {
        "id": "MOZ-PRIV-001",
        "name": "Round trip",
        "tags": ["privacy"],
        "scenario_priming": {
            "environment": {
                "config": {
                    "type": "alfworld_environment",
                    "task_id": "ROUNDTRIP",
                    "tasks": [{"task_id": "ROUNDTRIP-0", "goal": "g", "kind": "heat"}],
                },
            },
            "memory": {
                "entries": [
                    {"content": "fact", "metadata": {"source": "user"}, "score": None},
                ],
            },
        },
        "agents": [
            {
                "type": "agent",
                "model_id": "ollama:qwen2.5:7b",
                "instructions": "Reply briefly.",
            }
        ],
        "memory_systems": [
            {
                "id": "example.in_memory",
                "backend_config": {"type": "example.in_memory"},
            }
        ],
        "evaluators": [{"type": "task_qa"}],
        "execution_sequence": [
            {"action_type": "user_stimulus", "content": "go"},
            {"action_type": "env_manipulation", "command": "switch_tab"},
            {"action_type": "memory_manipulation", "command": "inject_fact"},
            {"action_type": "agent_manipulation", "command": "inject_system_prompt"},
        ],
    }

    scenario = Scenario.model_validate(data)

    assert (
        scenario.model_dump()
        == Scenario.model_validate(scenario.model_dump()).model_dump()
    )
    assert scenario.scenario_priming.memory.entries[0].content == "fact"
    env_config = scenario.scenario_priming.environment.config
    assert env_config is not None
    env_dump = env_config.model_dump()
    assert env_dump["type"] == "alfworld_environment"
    assert env_dump["task_id"] == "ROUNDTRIP"


def test_scenario_rejects_legacy_memory_system_payload_shape() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "id": "LEGACY-MEM-1",
                "name": "Legacy memory payload",
                "scenario_priming": {},
                "agents": [
                    {
                        "type": "agent",
                        "model_id": "ollama:qwen2.5:7b",
                        "instructions": "Reply briefly.",
                    }
                ],
                "memory_systems": [{"type": "example.in_memory"}],
                "evaluators": [{"type": "task_qa"}],
                "execution_sequence": [
                    {"action_type": "user_stimulus", "content": "go"}
                ],
            }
        )


def test_base_meta_action_is_not_directly_instantiable_as_a_step() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "id": "x",
                "name": "y",
                "scenario_priming": {},
                "execution_sequence": [BaseMetaAction().model_dump()],
            }
        )


def test_scenario_rejects_empty_execution_sequence() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="x",
            name="y",
            scenario_priming=ScenarioPriming(),
            execution_sequence=[],
        )


def test_scenario_rejects_unknown_keys() -> None:
    base_payload = {
        "id": "x",
        "name": "y",
        "scenario_priming": {},
        "execution_sequence": [{"action_type": "user_stimulus", "content": "go"}],
    }

    with pytest.raises(ValidationError):
        Scenario.model_validate({**base_payload, "unexpected_key": True})

    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                **base_payload,
                "scenario_priming": {"memory": {"entries": [], "typo_key": 1}},
            }
        )


def test_meta_action_discriminator_dispatches_to_correct_subclass() -> None:
    scenario = Scenario.model_validate(
        {
            "id": "x",
            "name": "y",
            "scenario_priming": {},
            "execution_sequence": [
                {"action_type": "user_stimulus", "content": "go"},
                {"action_type": "env_manipulation", "command": "switch_tab"},
                {"action_type": "memory_manipulation", "command": "inject_fact"},
                {
                    "action_type": "agent_manipulation",
                    "command": "inject_system_prompt",
                },
            ],
        }
    )

    stimulus, env_action, mem_action, agent_action = scenario.execution_sequence
    assert isinstance(stimulus, UserStimulus)
    assert isinstance(env_action, EnvironmentManipulation)
    assert isinstance(mem_action, MemoryManipulation)
    assert isinstance(agent_action, AgentManipulation)
