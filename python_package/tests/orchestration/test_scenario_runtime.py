from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from heams.agents.config import (
    AgentParameters,
    MemoryStrategy,
)
from heams.environments.alfworld import AlfWorldEnvironmentConfig
from heams.memory_systems import (
    PrivacyConfig,
    PrivacyMemoryWrapper,
)
from heams_memory_example.plugin import ExampleMemoryConfig, ExampleMemorySystem
from heams.memory_systems.api import BaseConsolidationConfig
from heams.memory_systems.base import BaseMemorySystem, MemoryOperationResult
from heams.memory_systems.factory import MemorySystemVariant
from heams.memory_systems.schemas import MemoryOperationStatus, MemoryOperationType
from heams.memory_systems.shared_types import MemoryEntry
from heams.orchestration.scenario_runtime import (
    build_tools,
    consolidate_if_enabled,
    setup_scenario_runtime,
    teardown_scenario_runtime,
)
from heams.scenarios import Scenario, UserStimulus
from heams.scenarios.base import EnvironmentPriming, MemoryPriming, ScenarioPriming


def _agent_config() -> AgentParameters:
    return AgentParameters(
        model_id="stub",
        instructions="",
    )


def test_enabled_consolidation_runs_once_immediately_after_priming(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Memory:
        config = SimpleNamespace(
            consolidation=SimpleNamespace(enabled=True),
        )

        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            calls.append("prime")

        def consolidate(self) -> str:
            calls.append("consolidate")
            return "ok"

    scenario = Scenario(
        id="runtime-consolidation-enabled",
        name="runtime-consolidation-enabled",
        execution_sequence=[UserStimulus(content="question")],
    )

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: Memory(),
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert calls == ["prime", "consolidate"]


def test_consolidate_if_enabled_skips_disabled_consolidation() -> None:
    memory = MagicMock()
    memory.config = SimpleNamespace(consolidation=SimpleNamespace(enabled=False))

    consolidate_if_enabled(memory)

    memory.consolidate.assert_not_called()


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


def test_setup_scenario_runtime_prefers_memory_priming_capability(
    tmp_path: Path,
) -> None:
    primed_entries: list[MemoryEntry] = []
    retained_calls: list[tuple[str, dict]] = []

    class PrimingMemory:
        config = ExampleMemoryConfig()

        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            primed_entries.extend(entries)

        def retain(self, content: str, metadata=None) -> str:
            retained_calls.append((content, metadata or {}))
            return "ok"

        def recall(self, query: str, top_k=None) -> list[str]:
            return []

        def delete(self, query: str) -> str:
            return "ok"

        def consolidate(self, context=None, metadata=None) -> str:
            return "ok"

    scenario = Scenario(
        id="runtime-priming",
        name="runtime-priming",
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(
                entries=[MemoryEntry(content="fact", metadata={"source": "test"})]
            )
        ),
        execution_sequence=[UserStimulus(content="question")],
    )

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: PrimingMemory(),
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert len(primed_entries) == 1
    assert primed_entries[0].content == "fact"
    assert retained_calls == []


def test_setup_scenario_runtime_cleans_partial_setup_on_agent_failure(
    tmp_path: Path,
) -> None:
    environment = MagicMock()
    memory = MagicMock()
    scenario = Scenario(
        id="runtime-agent-failure",
        name="runtime-agent-failure",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=AlfWorldEnvironmentConfig())
        ),
        execution_sequence=[UserStimulus(content="question")],
    )

    def create_agent_fn(*args, **kwargs):
        raise KeyboardInterrupt("agent creation interrupted")

    storage_dir = tmp_path / "runtime-agent-failure-combo-s0-memory"
    with pytest.raises(KeyboardInterrupt, match="agent creation interrupted"):
        setup_scenario_runtime(
            scenario,
            workspace_dir=tmp_path,
            agent_cfg=_agent_config(),
            mem_cfg=ExampleMemoryConfig(),
            eval_cfg=MagicMock(),
            combo_key="combo",
            sample_index=0,
            create_environment_fn=lambda config: environment,
            create_memory_fn=lambda *args, **kwargs: memory,
            create_agent_fn=create_agent_fn,
        )

    environment.close.assert_called_once_with()
    assert not storage_dir.exists()


def test_setup_scenario_runtime_closes_agent_when_evaluator_fails(
    tmp_path: Path,
) -> None:
    agent = MagicMock()
    memory = MagicMock()
    scenario = Scenario(
        id="runtime-evaluator-failure",
        name="runtime-evaluator-failure",
        execution_sequence=[UserStimulus(content="question")],
    )

    def create_evaluator_fn(*args, **kwargs):
        raise RuntimeError("evaluator creation failed")

    storage_dir = tmp_path / "runtime-evaluator-failure-combo-s0-memory"
    with pytest.raises(RuntimeError, match="evaluator creation failed"):
        setup_scenario_runtime(
            scenario,
            workspace_dir=tmp_path,
            agent_cfg=_agent_config(),
            mem_cfg=ExampleMemoryConfig(),
            eval_cfg=MagicMock(),
            combo_key="combo",
            sample_index=0,
            create_memory_fn=lambda *args, **kwargs: memory,
            create_agent_fn=lambda *args, **kwargs: agent,
            create_evaluator_fn=create_evaluator_fn,
        )

    agent.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    assert not storage_dir.exists()


def test_teardown_scenario_runtime_removes_storage_when_environment_close_fails(
    tmp_path: Path,
) -> None:
    environment = MagicMock()
    environment.close.side_effect = RuntimeError("environment close failed")
    storage_dir = tmp_path / "runtime-storage"
    storage_dir.mkdir()

    with pytest.raises(RuntimeError, match="environment close failed"):
        teardown_scenario_runtime(environment, storage_dir)

    assert not storage_dir.exists()


def test_setup_scenario_runtime_uses_base_memory_priming_fallback(
    tmp_path: Path,
) -> None:
    retained_calls: list[tuple[str, dict]] = []

    class FallbackMemory(BaseMemorySystem):
        config = MagicMock(top_k=5)

        def __init__(self, storage_dir: Path) -> None:
            super().__init__(
                ExampleMemoryConfig(),
                storage_dir=storage_dir,
            )

        def _retain_impl(
            self,
            content: str,
            metadata: dict,
        ) -> MemoryOperationResult:
            retained_calls.append((content, metadata))
            return MemoryOperationResult(value="ok")

        def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
            return MemoryOperationResult(value=[])

        def _delete_impl(self, query: str) -> MemoryOperationResult:
            return MemoryOperationResult(value="ok")

        def _consolidate_impl(
            self,
            context: str | None,
            metadata: dict,
        ) -> MemoryOperationResult:
            return MemoryOperationResult(value="ok")

    scenario = Scenario(
        id="runtime-base-priming",
        name="runtime-base-priming",
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(
                entries=[
                    MemoryEntry(content="fact 1", metadata={"order": 1}),
                    MemoryEntry(content="fact 2", metadata={"order": 2}),
                ]
            )
        ),
        execution_sequence=[UserStimulus(content="question")],
    )

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: FallbackMemory(kwargs["storage_dir"]),
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert retained_calls == [
        ("fact 1", {"order": 1}),
        ("fact 2", {"order": 2}),
    ]


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
        captured["privacy_config"] = kwargs.get("privacy_config")
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


def test_persistent_runtime_hands_backend_a_unique_storage_directory(
    tmp_path: Path,
) -> None:
    memory_config = ExampleMemoryConfig()
    original_config = memory_config.model_dump()
    scenario = Scenario(
        id="persistent-runtime",
        name="persistent-runtime",
        execution_sequence=[UserStimulus(content="question")],
    )

    _, memory, cleanup_path, _, _, _ = setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=_agent_config(),
        mem_cfg=memory_config,
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert cleanup_path == tmp_path / "persistent-runtime-combo-s0-memory"
    assert cleanup_path.is_dir()
    assert memory.storage_dir == cleanup_path
    assert memory_config.model_dump() == original_config


def test_no_memory_runtime_exposes_only_final_answer_tool(
    tmp_path: Path,
) -> None:
    captured_tools = None

    class PrimingMemory:
        config = MagicMock(top_k=5)

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

    def create_agent_fn(*args, **kwargs):
        nonlocal captured_tools
        captured_tools = kwargs["tools"]
        return MagicMock()

    scenario = Scenario(
        id="toolbench",
        name="toolbench",
        tags=["evomemory", "single_turn", "toolbench"],
        execution_sequence=[UserStimulus(content="question")],
    )

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=AgentParameters(
            model_id="openai:gpt-4o-mini",
            instructions="",
        ),
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: PrimingMemory(),
        create_agent_fn=create_agent_fn,
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert [tool.__name__ for tool in captured_tools] == ["final_answer"]


def test_setup_scenario_runtime_preserves_resolved_agent_strategy(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    consolidate_calls = 0

    class Memory:
        config = MagicMock(top_k=5)

        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            pass

        def retain(self, content: str, metadata=None) -> str:
            return "ok"

        def recall(self, query: str, top_k=None) -> list[str]:
            return []

        def delete(self, query: str) -> str:
            return "ok"

        def consolidate(self, context=None, metadata=None) -> str:
            nonlocal consolidate_calls
            consolidate_calls += 1
            return "ok"

    def create_agent_fn(*args, **kwargs):
        captured["config"] = args[0]
        captured["tools"] = kwargs["tools"]
        return MagicMock()

    scenario = Scenario(
        id="runtime-strategy",
        name="runtime-strategy",
        execution_sequence=[UserStimulus(content="question")],
    )

    agent_config = _agent_config()
    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=agent_config,
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: Memory(),
        create_agent_fn=create_agent_fn,
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert captured["config"] is agent_config
    assert captured["config"].memory_strategy == MemoryStrategy.NO_MEMORY
    assert captured["tools"] == []
    assert consolidate_calls == 0


def test_tool_use_runtime_exposes_all_memory_tools_without_startup_consolidation(
    tmp_path: Path,
) -> None:
    captured_tools: list = []
    primed_entries: list[MemoryEntry] = []
    consolidate_calls = 0

    class Memory:
        config = MagicMock(top_k=5)

        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            primed_entries.extend(entries)

        def retain(self, content: str, metadata=None) -> str:
            return "ok"

        def recall(self, query: str, top_k=None) -> list[str]:
            return []

        def delete(self, query: str) -> str:
            return "ok"

        def consolidate(self, context=None, metadata=None) -> str:
            nonlocal consolidate_calls
            consolidate_calls += 1
            return "ok"

    def create_agent_fn(*args, **kwargs):
        nonlocal captured_tools
        captured_tools = kwargs["tools"]
        return MagicMock()

    scenario = Scenario(
        id="runtime-tools",
        name="runtime-tools",
        execution_sequence=[UserStimulus(content="question")],
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(entries=[MemoryEntry(content="fact")])
        ),
    )

    setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=AgentParameters(
            memory_strategy=MemoryStrategy.TOOL_USE,
            model_id="stub",
            instructions="",
        ),
        mem_cfg=ExampleMemoryConfig(),
        eval_cfg=MagicMock(),
        combo_key="combo",
        sample_index=0,
        create_memory_fn=lambda *args, **kwargs: Memory(),
        create_agent_fn=create_agent_fn,
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert [entry.content for entry in primed_entries] == ["fact"]
    assert [tool.__name__ for tool in captured_tools] == [
        "retain",
        "recall",
        "delete",
    ]
    assert consolidate_calls == 0


def test_recall_only_tool_use_exposes_only_recall_tool():
    class Memory:
        def retain(self, content: str, metadata=None) -> str:
            return "retained"

        def recall(self, query: str, top_k=None) -> list[str]:
            return []

        def delete(self, query: str) -> str:
            return "deleted"

        def consolidate(self, context=None, metadata=None) -> str:
            return "consolidated"

    tools = build_tools(
        None,
        Memory(),
        agent_cfg=AgentParameters(
            memory_strategy=MemoryStrategy.RECALL_ONLY_TOOL_USE,
            model_id="stub",
            instructions="",
        ),
    )

    assert [tool.__name__ for tool in tools] == ["recall"]


def test_scenario_rejects_removed_memory_access_strategy() -> None:
    with pytest.raises(ValueError, match="memory_access_strategy"):
        Scenario(
            id="legacy-policy",
            name="legacy policy",
            memory_access_strategy="agent_default",
            execution_sequence=[UserStimulus(content="question")],
        )
