"""The privacy wrapper inside the framework: built by the factory, carried by the
scenario runtime, and consolidated by the orchestrator through the wrapper."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from heams.agents.config import AgentParameters
from heams.scenarios.config import ScenarioExecutionConfig as BenchmarkConfig
from heams.evaluation.schemas import AuditRecord
from heams.evaluation.schemas import BaseGroundTruth
from heams.memory_systems.api import (
    BaseConsolidationConfig,
    MemoryEntry,
    MemoryOperationStatus,
    MemoryOperationType,
)
from heams.memory_systems.factory import MemorySystemVariant, create_memory
from heams.memory_systems.in_context import InContextConfig
from heams.orchestration.orchestrator import Orchestrator
from heams.orchestration.scenario_runtime import setup_scenario_runtime
from heams.orchestration.scenario_execution import execute_scenario
from heams.scenarios import MemoryManipulation, Scenario, UserStimulus
from heams_memory_example.plugin import ExampleMemoryConfig, ExampleMemorySystem
from heams_memory_privacy_wrapper import PrivacyConfig, PrivacyMemoryWrapper


def _agent_config() -> AgentParameters:
    return AgentParameters(
        model_id="stub",
        instructions="",
    )


def _successful_audit() -> AuditRecord:
    return AuditRecord(
        is_successful=True,
        accuracy_score=0.75,
        passed_keys=["scenario-result"],
        failed_keys=[],
        audit_notes="scenario audit",
    )


def _post_session_scenario(memory_system) -> Scenario:
    return Scenario(
        id="POST-SESSION-CONSOLIDATION",
        name="post-session-consolidation",
        agents=[_agent_config()],
        memory_systems=[memory_system],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[UserStimulus(content="question")],
    )


def test_create_memory_wraps_backend_when_privacy_is_enabled(tmp_path) -> None:
    memory = create_memory(
        ExampleMemoryConfig(name="wrapped", top_k=3),
        storage_dir=tmp_path,
        wrapper_configs={"privacy_config": PrivacyConfig()},
    )

    assert isinstance(memory, PrivacyMemoryWrapper)
    assert isinstance(memory.inner, ExampleMemorySystem)
    assert memory.config is memory.inner.config
    assert memory.config.name == "wrapped"


def test_create_memory_returns_unwrapped_backend_when_privacy_is_disabled(
    tmp_path,
) -> None:
    memory = create_memory(
        ExampleMemoryConfig(name="plain", top_k=3),
        storage_dir=tmp_path,
        wrapper_configs={"privacy_config": PrivacyConfig(enabled=False)},
    )

    assert isinstance(memory, ExampleMemorySystem)
    assert not isinstance(memory, PrivacyMemoryWrapper)


def test_privacy_wrapped_consolidation_delegates_after_priming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consolidate_calls = 0
    original_consolidate = ExampleMemorySystem.consolidate

    def _track_consolidate(self, *args, **kwargs) -> str:
        nonlocal consolidate_calls
        consolidate_calls += 1
        return original_consolidate(self, *args, **kwargs)

    monkeypatch.setattr(
        ExampleMemorySystem,
        "consolidate",
        _track_consolidate,
    )
    memory_variant = MemorySystemVariant(
        backend_config=ExampleMemoryConfig(
            consolidation=BaseConsolidationConfig(enabled=True),
        ),
        privacy_config=PrivacyConfig(),
    )
    scenario = Scenario(
        id="runtime-private-consolidation",
        name="runtime-private-consolidation",
        execution_sequence=[UserStimulus(content="question")],
    )

    _, memory, _, _, _, ledger = setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=memory_variant,
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert isinstance(memory, PrivacyMemoryWrapper)
    assert consolidate_calls == 1
    assert len(ledger) == 1
    assert ledger[0].operation_type is MemoryOperationType.CONSOLIDATE
    assert ledger[0].status is MemoryOperationStatus.SUCCESS


def test_setup_scenario_runtime_forwards_memory_entry_privacy_config(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    privacy_config = PrivacyConfig()
    memory_entry = MemorySystemVariant(
        backend_config=ExampleMemoryConfig(),
        privacy_config=privacy_config,
    )

    class Memory:
        config = ExampleMemoryConfig()

        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            pass

        def retain(self, content: str, metadata=None) -> str:
            return "ok"

        def recall(self, query: str, top_k=None) -> list[str]:
            return []

        def delete(self, query: str) -> str:
            return "ok"

        def consolidate(self, context=None, metadata=None) -> str:
            return "ok"

    scenario = Scenario(
        id="runtime-privacy-forwarding",
        name="runtime-privacy-forwarding",
        memory_systems=[memory_entry],
        execution_sequence=[UserStimulus(content="question")],
    )

    def create_memory_fn(*args, **kwargs):
        captured["config"] = args[0]
        captured["privacy_config"] = kwargs.get("wrapper_configs", {}).get(
            "privacy_config"
        )
        return Memory()

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=scenario.memory_systems[0],
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=create_memory_fn,
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert captured["config"] == memory_entry.backend_config
    assert captured["privacy_config"] == privacy_config


def test_post_session_consolidation_uses_privacy_wrapper_and_reaches_final_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original_consolidate = PrivacyMemoryWrapper.consolidate

    def _track_consolidate(self, *args, **kwargs) -> str:
        calls.append("consolidate")
        return original_consolidate(self, *args, **kwargs)

    monkeypatch.setattr(PrivacyMemoryWrapper, "consolidate", _track_consolidate)
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: MagicMock(),
    )

    def _execute(**kwargs):
        calls.append("execute")
        return _successful_audit(), [], []

    monkeypatch.setattr("heams.orchestration.orchestrator.execute_scenario", _execute)
    memory_variant = MemorySystemVariant(
        backend_config=ExampleMemoryConfig(
            consolidation=BaseConsolidationConfig(enabled=True),
        ),
        privacy_config=PrivacyConfig(),
    )

    run = Orchestrator(
        BenchmarkConfig(
            scenarios=[_post_session_scenario(memory_variant)],
            workspace_dir=tmp_path / "ws",
        )
    ).run()["POST-SESSION-CONSOLIDATION"][0]

    assert calls == ["consolidate", "execute", "consolidate"]
    assert run.memory_ledger is not None
    consolidation_events = [
        event
        for event in run.memory_ledger
        if event.operation_type.value == "consolidate"
    ]
    assert len(consolidation_events) == 2
    assert consolidation_events[-1].duration_ms is not None


class _AgentStub:
    """Holds a memory system; scenarios of memory manipulations never call the agent."""

    def __init__(self, memory_system):
        self.memory_system = memory_system


def test_scenario_responses_carry_the_privacy_wrappers_statuses(tmp_path) -> None:
    statement = "Do not retain the user's pet information."
    memory = create_memory(
        InContextConfig(),
        storage_dir=tmp_path,
        wrapper_configs={
            "privacy_config": PrivacyConfig(
                retention={
                    "mode": "deterministic",
                    "policy_rules": {
                        statement: [
                            {"rule_id": "drop-pet", "content_substrings": ["dog"]}
                        ]
                    },
                },
                recall={"enabled": False},
            )
        },
    )
    ledger: list = []
    memory.attach_event_sink(ledger)
    agent = _AgentStub(memory)
    scenario = Scenario(
        id="STATUS",
        name="status",
        execution_sequence=[
            MemoryManipulation(
                command="set_retention_policy",
                parameters={"policies": [statement]},
            ),
            MemoryManipulation(
                command="retain", parameters={"content": "My dog is Rex."}
            ),
            MemoryManipulation(
                command="retain", parameters={"content": "I live in Montreal."}
            ),
            MemoryManipulation(command="recall", parameters={"query": "Montreal"}),
        ],
    )

    _, _, responses = execute_scenario(
        scenario=scenario,
        agent=agent,
        environment=None,
        evaluator=MagicMock(),
        eval_cfg=None,
        build_ground_truth=lambda scenario, eval_cfg: BaseGroundTruth(
            task_description=scenario.name
        ),
        memory_ledger=ledger,
    )

    assert [response.memory_status for response in responses] == [
        "success",
        "dropped",
        "success",
        "success",
    ]
    [dropped] = [event for event in ledger if event.status == "dropped"]
    assert dropped.operation_type == "retain"
    assert dropped.result["policy_id"] == "drop-pet"
