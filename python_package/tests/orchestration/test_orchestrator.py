import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from heams.agents.config import (
    AgentParameters,
    MemoryStrategy,
)
from heams.agents.tiny_agent import (
    ToolExecutionLimitReached,
)
from heams.agents.base import AgentResponse
from heams.benchmarks.locomo import LocomoQA, LocomoQAGroundTruth
from heams.environments.alfworld import (
    AlfWorldEnvironment,
    AlfWorldEnvironmentConfig,
)
from heams.evaluation.factory import (
    EvoMemorySingleTurnTaskQAEvaluatorConfig,
)
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth
from heams.memory_systems import (
    MemorySystemVariant,
    create_memory,
)
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.memory_systems.in_context import InContextConfig
from heams.memory_systems.shared_types import MemoryEntry
from heams.orchestration.orchestrator import Orchestrator
from heams.orchestration.scenario_execution import execute_scenario
from heams.orchestration.scenario_runtime import teardown_scenario_runtime
from heams.results import RunResult
from heams.scenarios import AgentManipulation, Scenario, UserStimulus
from heams.scenarios.actions import (
    EnvironmentManipulation,
    MemoryManipulation,
)
from heams.scenarios.base import (
    EnvironmentPriming,
    MemoryPriming,
    ScenarioPriming,
)
from heams.scenarios.config import ScenarioExecutionConfig as BenchmarkConfig
from tests.benchmarks.locomo.fixtures import load_locomo_judge_config
from tests.benchmarks.retention_policy.fixtures import KeywordGovernedMemory


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


def test_enabled_consolidation_runs_once_after_successful_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Memory:
        config = SimpleNamespace(
            consolidation=SimpleNamespace(enabled=True),
        )

        def prime_memory_entries(self, entries) -> None:
            calls.append("prime")

        def consolidate(self) -> str:
            calls.append("consolidate")
            return "ok"

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory",
        lambda *args, **kwargs: Memory(),
    )
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

    Orchestrator(
        BenchmarkConfig(
            scenarios=[_post_session_scenario(ExampleMemoryConfig())],
            workspace_dir=tmp_path / "ws",
        )
    ).run()

    assert calls == ["prime", "consolidate", "execute", "consolidate", "close"]


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(consolidation=SimpleNamespace(enabled=False)),
        SimpleNamespace(top_k=5),
    ],
    ids=["disabled", "unsupported"],
)
def test_post_session_consolidation_skips_disabled_and_unsupported_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config
) -> None:
    memory = MagicMock()
    memory.config = config
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory",
        lambda *args, **kwargs: memory,
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.execute_scenario",
        lambda **kwargs: (_successful_audit(), [], []),
    )

    Orchestrator(
        BenchmarkConfig(
            scenarios=[_post_session_scenario(ExampleMemoryConfig())],
            workspace_dir=tmp_path / "ws",
        )
    ).run()

    memory.consolidate.assert_not_called()


def test_post_session_consolidation_failure_preserves_audit_and_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Memory:
        config = SimpleNamespace(
            consolidation=SimpleNamespace(enabled=True),
        )

        def prime_memory_entries(self, entries) -> None:
            calls.append("prime")

        def consolidate(self) -> str:
            calls.append("consolidate")
            if calls.count("consolidate") == 2:
                raise RuntimeError("post-session failure")
            return "ok"

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory",
        lambda *args, **kwargs: Memory(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.execute_scenario",
        lambda **kwargs: (_successful_audit(), [], []),
    )

    run = Orchestrator(
        BenchmarkConfig(
            scenarios=[_post_session_scenario(ExampleMemoryConfig())],
            workspace_dir=tmp_path / "ws",
        )
    ).run()["POST-SESSION-CONSOLIDATION"][0]

    assert calls == ["prime", "consolidate", "consolidate", "close"]
    assert run.audit.is_successful is False
    assert run.audit.accuracy_score == 0.75
    assert run.audit.passed_keys == ["scenario-result"]
    assert run.audit.failed_keys == ["consolidation_error: RuntimeError"]
    assert "scenario audit" in run.audit.audit_notes
    assert "post-session failure" in run.audit.audit_notes


def test_execution_config_requires_resolved_components_as_a_pair(
    tmp_path: Path,
) -> None:
    scenario = Scenario(
        id="PAIR-VALIDATION",
        name="pair validation",
        execution_sequence=[UserStimulus(content="q")],
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[],
    )

    with pytest.raises(ValidationError, match="must be supplied together"):
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path,
            agent_config=_agent_config(),
        )


def test_pending_key_filter_skips_runtime_setup_for_completed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_env_step("Personal")
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.setup_scenario_runtime",
        lambda *args, **kwargs: pytest.fail("completed work initialized runtime"),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path / "workspace",
            pending_keys=set(),
        )
    )

    assert orchestrator.run() == {scenario.id: []}

    with pytest.raises(ValidationError, match="must be supplied together"):
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path,
            memory_system_config=MemorySystemVariant(
                id="resolved-memory",
                backend_config=ExampleMemoryConfig(),
            ),
        )


def _alfworld_env_config() -> AlfWorldEnvironmentConfig:
    return AlfWorldEnvironmentConfig(
        task_id="ENV-DISPATCH",
        tasks=[
            {
                "task_id": "ENV-DISPATCH-0",
                "goal": "Heat the plate.",
                "kind": "heat",
            }
        ],
    )


def _scenario_with_env_step(expected_value: str) -> Scenario:
    return Scenario(
        id="ENV-DISPATCH-1",
        name="env-dispatch",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=_alfworld_env_config()),
            memory=MemoryPriming(
                entries=[MemoryEntry(content="ignored", metadata={"source": "test"})],
            ),
        ),
        agents=[
            _agent_config().model_copy(
                update={"memory_strategy": MemoryStrategy.TOOL_USE}
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            EnvironmentManipulation(
                command="set_observation",
                parameters={"observation": expected_value},
            ),
            UserStimulus(
                content="confirm workspace",
                step_evaluation_markers={"expected_answer": "a"},
            ),
        ],
    )


def test_orchestrator_dispatches_environment_manipulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator must route `EnvironmentManipulation` steps to the
    active environment's `dispatch_command`, and must do so before any
    subsequent user stimulus hits the agent."""
    expected_workspace = "Personal"
    scenario = _scenario_with_env_step(expected_workspace)

    captured: dict[str, str | list[str] | None] = {
        "workspace_when_agent_ran": "<unset>",
        "tool_names": [],
    }

    orchestrator_ref: dict[str, Orchestrator] = {}

    def fake_create_agent(config, tools, memory_system=None):
        captured["tool_names"] = [getattr(t, "__name__", repr(t)) for t in tools]
        stub = MagicMock()

        def run(prompt: str) -> AgentResponse:
            environment = orchestrator_ref["orch"].active_environment
            assert isinstance(environment, AlfWorldEnvironment)
            captured["workspace_when_agent_ran"] = environment.get_task_state()[
                "observation"
            ]
            return AgentResponse(
                answer="noop",
                raw_trace=MagicMock(spans=[], final_output="noop"),
            )

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )

    class _PassingEvaluator:
        def audit(self, *args, **kwargs):
            return AuditRecord(
                is_successful=True,
                accuracy_score=1.0,
                passed_keys=["workspace_marker"],
                failed_keys=[],
                audit_notes="stub",
            )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: _PassingEvaluator(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    orchestrator_ref["orch"] = orchestrator
    results = orchestrator.run()

    assert captured["workspace_when_agent_ran"] == expected_workspace
    assert "step_task" in captured["tool_names"]
    assert "retain" in captured["tool_names"]
    assert "recall" in captured["tool_names"]
    assert "delete" in captured["tool_names"]
    assert "consolidate" not in captured["tool_names"]
    assert len(results[scenario.id]) == 1
    assert results[scenario.id][0].audit.is_successful is True


def test_orchestrator_runs_agents_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _scenario_with_env_step("Research")
    observed: list[str] = []
    orchestrator_ref: dict[str, Orchestrator] = {}

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()

        def run(prompt: str, *, run_context=None) -> AgentResponse:
            environment = orchestrator_ref["orch"].active_environment
            assert isinstance(environment, AlfWorldEnvironment)
            observed.append(environment.get_task_state()["observation"])
            return AgentResponse(
                answer=f"ok:{prompt}",
                raw_trace=MagicMock(spans=[], final_output=""),
            )

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="stub",
                )
            },
        )(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    orchestrator_ref["orch"] = orchestrator
    results = orchestrator.run()

    assert observed == ["Research"]
    assert results[scenario.id][0].transcript[0].answer == "ok:confirm workspace"


def test_orchestrator_uses_resolved_components_instead_of_scenario_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _scenario_with_env_step("Research").model_copy(
        update={
            "agents": [_agent_config(), _agent_config()],
            "memory_systems": [
                ExampleMemoryConfig(),
                ExampleMemoryConfig(),
            ],
        }
    )
    resolved_agent = _agent_config()
    resolved_memory = MemorySystemVariant(
        id="resolved-memory",
        backend_config=ExampleMemoryConfig(),
    )
    captured: dict[str, object] = {}

    class _Memory:
        config = ExampleMemoryConfig()

        def prime_memory_entries(self, entries) -> None:
            pass

        def retain(self, content, metadata=None):
            return "ok"

        def recall(self, query, top_k=None):
            return []

        def delete(self, query):
            return "ok"

        def consolidate(self, context=None):
            return "ok"

    def fake_create_memory(config, **kwargs):
        captured["memory_config"] = config
        return _Memory()

    def fake_create_agent(config, tools, memory_system=None):
        captured["agent_config"] = config
        return MagicMock(
            run=lambda prompt: AgentResponse(
                answer="ok",
                raw_trace=MagicMock(spans=[], final_output="ok"),
            )
        )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory", fake_create_memory
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: type(
            "Eval",
            (),
            {
                "audit": lambda self, *args, **kwargs: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="stub",
                )
            },
        )(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path / "ws",
            agent_config=resolved_agent,
            memory_system_config=resolved_memory,
        )
    )
    results = orchestrator.run()

    assert orchestrator.total_runs() == 1
    assert len(results[scenario.id]) == 1
    assert captured["agent_config"] is resolved_agent
    assert captured["memory_config"] is resolved_memory.backend_config


def test_teardown_closes_environment_and_removes_storage_when_agent_close_fails(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "runtime-memory"
    memory_path.mkdir()
    for file_name in ("raw.json", "canonical.json", "canonical.embeddings.json"):
        (memory_path / file_name).write_text("{}", encoding="utf-8")
    environment = MagicMock()
    agent = MagicMock()
    agent.close.side_effect = RuntimeError("agent cleanup failed")
    memory = MagicMock()

    with pytest.raises(RuntimeError, match="agent cleanup failed"):
        teardown_scenario_runtime(environment, memory_path, agent, memory)

    memory.close.assert_called_once_with()
    environment.close.assert_called_once_with()
    assert not memory_path.exists()


def test_orchestrator_runs_task_environment_through_sync_meta_action_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from heams.environments.alfworld import AlfWorldEnvironmentConfig
    from heams.evaluation.factory import TaskEnvironmentEvaluatorConfig

    scenario = Scenario(
        id="TASK-ENV-ORCH",
        name="task env orchestrator",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(
                config=AlfWorldEnvironmentConfig(
                    task_id="TASK-ENV-ORCH",
                    tasks=[
                        {
                            "task_id": "TASK-ENV-ORCH-0",
                            "goal": "Heat the plate and put it on the countertop.",
                            "kind": "heat",
                        }
                    ],
                )
            )
        ),
        agents=[
            _agent_config().model_copy(
                update={"memory_strategy": MemoryStrategy.TOOL_USE}
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[
            TaskEnvironmentEvaluatorConfig(
                expected_task_ids=["TASK-ENV-ORCH-0"],
            )
        ],
        execution_sequence=[UserStimulus(content="Use tools and complete the task.")],
    )
    captured_tool_names: list[str] = []

    def execute_tool_span(tool_name: str, output: object) -> SimpleNamespace:
        return SimpleNamespace(
            name=f"execute_tool {tool_name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.args": "{}",
                "gen_ai.output": repr(output),
            },
        )

    def fake_create_agent(config, tools, memory_system=None):
        captured_tool_names[:] = [getattr(tool, "__name__", "") for tool in tools]
        tool_map = {getattr(tool, "__name__", ""): tool for tool in tools}
        stub = MagicMock()

        def run(prompt: str) -> AgentResponse:
            spans = [
                execute_tool_span("get_task_state", tool_map["get_task_state"]()),
                execute_tool_span(
                    "step_task",
                    tool_map["step_task"](action="pick up plate"),
                ),
                execute_tool_span(
                    "step_task",
                    tool_map["step_task"](action="use microwave"),
                ),
            ]
            return AgentResponse(
                answer="submitted",
                raw_trace=SimpleNamespace(spans=spans, final_output="submitted"),
            )

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )

    results = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    ).run()

    assert results[scenario.id][0].audit.is_successful is True
    assert "get_task_state" in captured_tool_names
    assert "get_valid_actions" in captured_tool_names
    assert "step_task" in captured_tool_names
    assert "finish_task" in captured_tool_names
    assert "retain" in captured_tool_names
    assert "recall" in captured_tool_names
    assert "delete" in captured_tool_names
    assert "consolidate" not in captured_tool_names


def test_orchestrator_fires_progress_callbacks_at_run_and_stimulus_granularity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single integrative check over both progress signals that the CLI
    wires to the tqdm bar and the JSONL checkpointer:

    * `on_stimulus_complete` fires exactly `total_stimulus_calls()`
      times, once per `UserStimulus`, with run coordinates in
      (sample, stimulus) order.
    * `on_run_complete` fires exactly `total_runs()` times with the
      corresponding `RunResult`s.

    Split tests would duplicate the same monkeypatch scaffold and
    scenario fixture; one scenario with multiple stimuli × samples
    exercises both callbacks under realistic fan-out."""
    from heams.orchestration.orchestrator import RunResult, StimulusEvent

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent",
        lambda *a, **kw: type(
            "Stub",
            (),
            {
                "run": lambda self, prompt: AgentResponse(
                    answer="ok", raw_trace=MagicMock(spans=[], final_output="ok")
                )
            },
        )(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="stub",
                )
            },
        )(),
    )

    scenario = Scenario(
        id="STIM_CB",
        name="cb",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(content="q0"),
            UserStimulus(content="q1"),
        ],
    )
    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path / "ws",
            num_samples=2,
        )
    )

    events: list[StimulusEvent] = []
    runs: list[RunResult] = []
    orchestrator.run(
        on_stimulus_complete=events.append,
        on_run_complete=runs.append,
    )

    # 2 stimuli × 2 samples = 4 stimulus events (in (sample, stimulus)
    # order), but only 2 runs (one per sample).
    assert orchestrator.total_stimulus_calls() == 4
    assert orchestrator.total_runs() == 2
    assert len(events) == 4
    assert [(e.sample_index, e.stimulus_index) for e in events] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert all(e.scenario_id == "STIM_CB" and e.ok for e in events)
    assert [r.sample_index for r in runs] == [0, 1]
    assert all(r.scenario_id == "STIM_CB" for r in runs)


def test_orchestrator_records_failing_stimulus_as_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stimulus raising produces a failed run and progress backfill."""
    from heams.orchestration.orchestrator import RunResult

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()

        def run(prompt: str, *, run_context=None) -> AgentResponse:
            if prompt == "boom":
                raise RuntimeError("context_length_exceeded")
            return AgentResponse(
                answer=f"ok:{prompt}",
                raw_trace=MagicMock(spans=[], final_output=""),
            )

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=0.5,
                    passed_keys=["q0"],
                    failed_keys=["q1"],
                    audit_notes="mixed",
                )
            },
        )(),
    )

    scenario = Scenario(
        id="ISO-ASYNC",
        name="iso-async",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(content="ok-first"),
            UserStimulus(content="boom"),
            UserStimulus(content="ok-third"),
        ],
    )
    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    observed: list[RunResult] = []
    results = orchestrator.run(on_run_complete=observed.append)

    assert len(observed) == 1
    assert observed[0].transcript == []
    assert results[scenario.id][0].audit.is_successful is False
    assert results[scenario.id][0].audit.failed_keys == ["run_error: RuntimeError"]


def test_orchestrator_isolates_failing_scenario_and_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncaught failure in the sync execution path (or setup) must
    downgrade the current run to a failed `RunResult` instead of
    tearing down the whole benchmark. Subsequent scenarios keep running
    and the progress callback still fires for the failed run so the
    JSONL checkpoint captures the failure on disk."""
    from heams.orchestration.orchestrator import RunResult

    call_count = {"n": 0}

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()

        def run(prompt: str) -> AgentResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return AgentResponse(
                answer="ok",
                raw_trace=MagicMock(spans=[], final_output=""),
            )

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="ok",
                )
            },
        )(),
    )

    def _mk_scenario(id_: str) -> Scenario:
        return Scenario(
            id=id_,
            name=id_,
            scenario_priming=ScenarioPriming(
                environment=EnvironmentPriming(config=None),
                memory=MemoryPriming(entries=[]),
            ),
            agents=[_agent_config()],
            memory_systems=[ExampleMemoryConfig()],
            evaluators=[{"type": "task_qa"}],
            execution_sequence=[UserStimulus(content="go")],
        )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[_mk_scenario("FAIL-FIRST"), _mk_scenario("OK-SECOND")],
            workspace_dir=tmp_path / "ws",
        )
    )
    observed: list[RunResult] = []
    results = orchestrator.run(on_run_complete=observed.append)

    # Both scenarios produced a run: progress fired for the failing
    # scenario (failure path) and the surviving one.
    assert len(observed) == 2
    assert observed[0].scenario_id == "FAIL-FIRST"
    assert observed[0].audit.is_successful is False
    assert observed[0].audit.accuracy_score == 0.0
    assert any(
        key.startswith("run_error: RuntimeError")
        for key in observed[0].audit.failed_keys
    )
    assert "boom" in observed[0].audit.audit_notes
    # Second scenario still completes normally.
    assert observed[1].scenario_id == "OK-SECOND"
    assert observed[1].audit.is_successful is True
    assert results["OK-SECOND"][0].audit.accuracy_score == 1.0


def test_orchestrator_isolates_teardown_failure_and_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents: list[MagicMock] = []

    def fake_create_agent(config, tools, memory_system=None):
        agent = MagicMock()
        agent.run.return_value = AgentResponse(
            answer="ok",
            raw_trace=MagicMock(spans=[], final_output=""),
        )
        if not agents:
            agent.close.side_effect = RuntimeError("agent cleanup failed")
        agents.append(agent)
        return agent

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="ok",
                )
            },
        )(),
    )

    def _mk_scenario(id_: str) -> Scenario:
        return Scenario(
            id=id_,
            name=id_,
            scenario_priming=ScenarioPriming(
                environment=EnvironmentPriming(config=None),
                memory=MemoryPriming(entries=[]),
            ),
            agents=[_agent_config()],
            memory_systems=[ExampleMemoryConfig()],
            evaluators=[{"type": "task_qa"}],
            execution_sequence=[UserStimulus(content="go")],
        )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[_mk_scenario("TEARDOWN-FAIL"), _mk_scenario("OK-SECOND")],
            workspace_dir=tmp_path / "ws",
        )
    )
    observed: list[RunResult] = []
    results = orchestrator.run(on_run_complete=observed.append)

    assert len(agents) == 2
    assert len(observed) == 2
    assert observed[0].scenario_id == "TEARDOWN-FAIL"
    assert observed[0].audit.is_successful is False
    assert observed[0].audit.accuracy_score == 0.0
    assert observed[0].audit.failed_keys == ["teardown_error: RuntimeError"]
    assert "agent cleanup failed" in observed[0].audit.audit_notes
    assert observed[1].scenario_id == "OK-SECOND"
    assert observed[1].audit.is_successful is True
    assert results["OK-SECOND"][0].audit.accuracy_score == 1.0
    assert orchestrator.active_agent is None


def test_orchestrator_backfills_only_missing_stimulus_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heams.orchestration.orchestrator import StimulusEvent

    calls = {"n": 0}

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()

        def run(prompt: str) -> AgentResponse:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return AgentResponse(answer="ok", raw_trace=None)

        stub.run.side_effect = run
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: MagicMock(),
    )

    scenario = Scenario(
        id="PARTIAL-STIMULUS",
        name="partial-stimulus",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(content="first"),
            UserStimulus(content="second"),
        ],
    )
    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    events: list[StimulusEvent] = []
    results = orchestrator.run(on_stimulus_complete=events.append)

    assert [(event.stimulus_index, event.ok) for event in events] == [
        (0, True),
        (1, False),
    ]
    assert results[scenario.id][0].audit.is_successful is False


def test_zero_stimulus_success_gets_successful_progress_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heams.orchestration.orchestrator import StimulusEvent

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", lambda *a, **kw: MagicMock()
    )

    class _PassingEvaluator:
        def audit(self, *args, **kwargs):
            return AuditRecord(
                is_successful=True,
                accuracy_score=1.0,
                passed_keys=["ok"],
                failed_keys=[],
                audit_notes="stub",
            )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: _PassingEvaluator(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )

    scenario = Scenario(
        id="ZERO-STIM",
        name="zero-stim",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=_alfworld_env_config()),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            EnvironmentManipulation(
                command="set_observation",
                parameters={"observation": "Work"},
            )
        ],
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    events: list[StimulusEvent] = []
    results = orchestrator.run(on_stimulus_complete=events.append)

    assert [(event.stimulus_index, event.ok) for event in events] == [(0, True)]
    assert results[scenario.id][0].audit.is_successful is True


def test_orchestrator_captures_agent_transcript_on_run_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every `UserStimulus` -> agent answer pair must land on
    `RunResult.transcript` so the JSONL checkpoint and the Markdown
    report contain the full Q/A record instead of streaming it to the
    console."""
    from heams.execution_traces import ExecutionTraceRecord
    from heams.orchestration.orchestrator import AgentTurn, RunResult

    answers = iter(["answer-one", "answer-two"])

    class _SyncAgent:
        def run(self, prompt: str) -> AgentResponse:
            return AgentResponse(
                answer=next(answers),
                raw_trace=MagicMock(spans=[], final_output=""),
                context_truncated=prompt == "question one",
            )

        async def run_async(self, prompt: str) -> AgentResponse:  # pragma: no cover
            raise AssertionError("regular Scenario should not use run_async")

    def fake_create_agent(config, tools, memory_system=None):
        return _SyncAgent()

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: type(
            "Eval",
            (),
            {
                "audit": lambda self, *a, **kw: AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ok"],
                    failed_keys=[],
                    audit_notes="stub",
                )
            },
        )(),
    )

    scenario = Scenario(
        id="TRANSCRIPT",
        name="transcript",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(
                content="question one",
                step_evaluation_markers={"expected_answer": "target-one"},
            ),
            UserStimulus(content="question two"),
        ],
    )
    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )

    observed: list[RunResult] = []
    traces: list[ExecutionTraceRecord] = []
    orchestrator.run(
        on_run_complete=observed.append,
        on_execution_trace_complete=traces.append,
        execution_trace_run_id="run-1",
    )

    assert len(observed) == 1
    assert len(traces) == 1
    assert observed[0].trace_id == traces[0].trace_id
    assert traces[0].run_id == "run-1"
    assert [event.type for event in traces[0].events] == [
        "memory_priming",
        "user_stimulus",
        "user_stimulus",
    ]
    transcript = observed[0].transcript
    assert transcript == [
        AgentTurn(
            question="question one",
            answer="answer-one",
            target_answer="target-one",
            context_truncated=True,
        ),
        AgentTurn(question="question two", answer="answer-two"),
    ]

    # Round-trip through JSON to confirm the transcript survives the
    # JSONL checkpoint path (`run.model_dump_json()`).
    import json

    payload = json.loads(observed[0].model_dump_json())
    assert payload["transcript"] == [
        {
            "question": "question one",
            "answer": "answer-one",
            "target_answer": "target-one",
            "tool_results": [],
            "context_truncated": True,
        },
        {"question": "question two", "answer": "answer-two", "tool_results": []},
    ]


def test_async_scenario_primes_memory_and_exposes_declared_memory_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heams.memory_systems.shared_types import MemoryEntry

    captured_tools: dict[str, list[str]] = {}

    class _RecordingMemory:
        config = MagicMock(top_k=5)

        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def prime_memory_entries(self, entries) -> None:
            for entry in entries:
                self.retain(entry.content, entry.metadata)

        def retain(self, content, metadata=None):
            self.calls.append(("retain", content))
            return "retained"

        def recall(self, query, top_k=None):
            return []

        def delete(self, query):
            self.calls.append(("delete", query))
            return "deleted"

        def consolidate(self, context=None):
            self.calls.append(("consolidate", context))
            return "consolidated"

    memory = _RecordingMemory()

    class _SyncAgent:
        def run(self, prompt: str, *, run_context=None) -> AgentResponse:
            return AgentResponse(answer="ok", raw_trace=None)

    def fake_create_agent(config, tools, memory_system=None):
        captured_tools["names"] = [getattr(t, "__name__", repr(t)) for t in tools]
        return _SyncAgent()

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory", lambda *a, **kw: memory
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )

    class _PassingEvaluator:
        def audit(self, *args, **kwargs):
            return AuditRecord(
                is_successful=True,
                accuracy_score=1.0,
                passed_keys=["ok"],
                failed_keys=[],
                audit_notes="stub",
            )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: _PassingEvaluator(),
    )

    scenario = Scenario(
        id="NO-ENV",
        name="no-env",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(
                entries=[
                    MemoryEntry(content="turn one", metadata={"dia_id": "D1"}),
                    MemoryEntry(content="turn two", metadata={"dia_id": "D2"}),
                ]
            ),
        ),
        agents=[
            _agent_config().model_copy(
                update={"memory_strategy": MemoryStrategy.TOOL_USE}
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[UserStimulus(content="recall something")],
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    orchestrator.run()

    names = captured_tools.get("names", [])
    assert names == ["retain", "recall", "delete"]
    assert [name for name, _ in memory.calls] == ["retain", "retain"]


def test_evomemory_runs_through_the_common_synchronous_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StreamMemory:
        config = MagicMock(top_k=4)

        def __init__(self) -> None:
            self.entries: list[str] = []
            self.calls: list[tuple] = []
            self.close_calls = 0

        def prime_memory_entries(self, entries) -> None:
            for entry in entries:
                self.retain(entry.content, entry.metadata)

        def retain(self, content, metadata=None):
            self.calls.append(("retain", content, metadata or {}))
            self.entries.append(content)
            return "retained"

        def recall(self, query, top_k=None):
            self.calls.append(("recall", query, top_k, list(self.entries)))
            return list(self.entries)

        def delete(self, query):
            self.calls.append(("delete", query))
            return "deleted"

        def consolidate(self, context=None):
            self.calls.append(("consolidate", context))
            return "consolidated"

        def bind_run_context(self, run_context=None):
            self.calls.append(("bind", dict(run_context or {})))

        def reset_run_context(self, token):
            return None

        def close(self) -> None:
            self.close_calls += 1

    memory = _StreamMemory()
    prompts: list[str] = []
    captured_tools: list[str] = []
    captured_memory_strategy: list[MemoryStrategy] = []
    created_agents = []

    def fake_create_agent(config, tools, memory_system=None):
        captured_memory_strategy.append(config.memory_strategy)
        captured_tools.extend(getattr(tool, "__name__", repr(tool)) for tool in tools)
        retain_tool = next(
            tool for tool in tools if getattr(tool, "__name__", "") == "retain"
        )
        recall_tool = next(
            tool for tool in tools if getattr(tool, "__name__", "") == "recall"
        )

        class _StatefulAgent:
            def __init__(self) -> None:
                self.memory_system = memory_system
                self.local_state: list[str] = []
                self.state_before_turn: list[list[str]] = []
                self.reset_memory_state: list[tuple[int, int]] = []
                self.recalled_on_task_two: list[str] = []

            def run(self, prompt: str, *, run_context=None) -> AgentResponse:
                self.state_before_turn.append(list(self.local_state))
                self.local_state.append(prompt)
                prompts.append(prompt)
                if prompt.startswith("Correct answer"):
                    retain_tool(prompt, metadata={"source": "evomemory_feedback"})
                    return AgentResponse(answer="feedback retained", raw_trace=None)
                if "grass" in prompt:
                    self.recalled_on_task_two = recall_tool(prompt)
                    return AgentResponse(answer="red", raw_trace=None)
                return AgentResponse(answer="blue", raw_trace=None)

            def reset_state(self) -> None:
                self.reset_memory_state.append(
                    (id(self.memory_system), self.memory_system.close_calls)
                )
                self.local_state.clear()

        agent = _StatefulAgent()
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_memory", lambda *a, **kw: memory
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )

    scenario = Scenario(
        id="EVOMEM-STREAM",
        name="evomemory stream",
        tags=["evomemory", "single_turn"],
        agents=[
            _agent_config().model_copy(
                update={"memory_strategy": MemoryStrategy.TOOL_USE}
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[EvoMemorySingleTurnTaskQAEvaluatorConfig()],
        execution_sequence=[
            UserStimulus(
                content="What color is the sky?",
                step_evaluation_markers={
                    "evomemory_turn": "task",
                    "task_id": "task-1",
                    "expected_answer": "blue",
                    "dataset": "fixture",
                    "metadata": {"year": 2024},
                },
            ),
            UserStimulus(
                content=(
                    "Correct answer for the immediately preceding task: blue\n"
                    "Use this feedback to improve future answers. If useful and "
                    "available, use the memory tools to retain this correction. "
                    "Do not answer the completed task again. After any optional "
                    "memory operation, call the final_answer tool with exactly: "
                    "Feedback recorded."
                ),
                step_evaluation_markers={
                    "evomemory_turn": "feedback",
                    "feedback_for_task_id": "task-1",
                },
            ),
            AgentManipulation(command="reset_state"),
            UserStimulus(
                content="What color is grass?",
                step_evaluation_markers={
                    "evomemory_turn": "task",
                    "task_id": "task-2",
                    "expected_answer": "green",
                    "dataset": "fixture",
                },
            ),
            UserStimulus(
                content=(
                    "Correct answer for the immediately preceding task: green\n"
                    "Use this feedback to improve future answers. If useful and "
                    "available, use the memory tools to retain this correction. "
                    "Do not answer the completed task again. After any optional "
                    "memory operation, call the final_answer tool with exactly: "
                    "Feedback recorded."
                ),
                step_evaluation_markers={
                    "evomemory_turn": "feedback",
                    "feedback_for_task_id": "task-2",
                },
            ),
            AgentManipulation(command="reset_state"),
        ],
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    progress = []
    result = orchestrator.run(on_stimulus_complete=progress.append)[scenario.id][0]
    agent = created_agents[0]

    assert [call[0] for call in memory.calls] == ["retain", "recall", "retain"]
    assert memory.entries == [prompts[1], prompts[3]]
    assert agent.state_before_turn == [
        [],
        [prompts[0]],
        [],
        [prompts[2]],
    ]
    assert agent.recalled_on_task_two == [prompts[1]]
    assert agent.reset_memory_state == [(id(memory), 0), (id(memory), 0)]
    assert memory.close_calls == 1
    assert captured_tools == ["retain", "recall", "delete"]
    assert captured_memory_strategy == [MemoryStrategy.TOOL_USE]
    assert orchestrator.total_stimulus_calls() == 4
    assert [event.stimulus_index for event in progress] == [0, 1, 3, 4]
    assert len(prompts) == 4
    assert prompts[0] == "What color is the sky?"
    assert "immediately preceding task: blue\n" in prompts[1]
    assert "final_answer tool with exactly: Feedback recorded." in prompts[1]
    assert prompts[2] == "What color is grass?"
    assert "immediately preceding task: green\n" in prompts[3]
    assert [turn.answer for turn in result.transcript] == [
        "blue",
        "feedback retained",
        "red",
        "feedback retained",
    ]
    assert result.audit.accuracy_score == 0.5
    assert result.audit.passed_keys == ["task-1"]
    assert result.audit.failed_keys == ["task-2"]


def test_orchestrator_raises_when_env_step_has_no_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issuing an EnvironmentManipulation step without an env config is a
    scenario-authoring bug; the orchestrator must surface it clearly."""
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: MagicMock(),
    )

    scenario = Scenario(
        id="NO-ENV-ERR",
        name="no-env-err",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            EnvironmentManipulation(
                command="set_observation", parameters={"observation": "w"}
            )
        ],
    )
    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    with pytest.raises(ValueError, match="EnvironmentManipulation"):
        orchestrator.run()


def test_orchestrator_still_raises_for_unimplemented_agent_manipulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the explicit reset_state agent manipulation is supported."""
    scenario = Scenario(
        id="MEM-MANIP",
        name="mem-manip",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=_alfworld_env_config())
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            AgentManipulation(command="set_persona", parameters={"persona": "x"}),
        ],
    )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: MagicMock(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    with pytest.raises(ValueError, match="unsupported AgentManipulation"):
        orchestrator.run()


class _RecordingAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.reset_calls = 0

    def run(self, prompt: str, *, run_context=None) -> AgentResponse:
        self.prompts.append(prompt)
        return AgentResponse(answer=f"answer: {prompt}", raw_trace=None)

    def reset_state(self) -> None:
        self.reset_calls += 1


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.response_counts: list[int] = []

    def audit(self, agent_responses, ground_truth) -> AuditRecord:
        self.response_counts.append(len(agent_responses))
        return AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["ok"],
            failed_keys=[],
        )


def _execute_with_recorders(
    scenario: Scenario,
    agent: _RecordingAgent,
    evaluator: _RecordingEvaluator,
    callback=None,
):
    return execute_scenario(
        scenario=scenario,
        agent=agent,
        environment=None,
        evaluator=evaluator,
        eval_cfg=None,
        build_ground_truth=lambda scenario, eval_cfg: BaseGroundTruth(
            task_description=scenario.name
        ),
        on_stimulus_complete=callback,
    )


def test_reset_state_dispatch_does_not_create_an_agent_turn() -> None:
    scenario = Scenario(
        id="RESET",
        name="reset",
        execution_sequence=[
            UserStimulus(content="first"),
            AgentManipulation(command="reset_state"),
            UserStimulus(content="second"),
        ],
    )
    agent = _RecordingAgent()
    evaluator = _RecordingEvaluator()
    progress: list[tuple[int, bool]] = []

    audit, transcript, responses = _execute_with_recorders(
        scenario,
        agent,
        evaluator,
        lambda step_index, ok: progress.append((step_index, ok)),
    )

    assert audit.is_successful is True
    assert agent.prompts == ["first", "second"]
    assert agent.reset_calls == 1
    assert [response.answer for response in responses] == [
        "answer: first",
        "answer: second",
    ]
    assert [turn.question for turn in transcript] == ["first", "second"]
    assert evaluator.response_counts == [2]
    assert progress == [(0, True), (2, True)]


def test_memory_manipulation_responses_have_no_status_without_a_ledger(
    tmp_path,
) -> None:
    agent = _RecordingAgent()
    agent.memory_system = create_memory(InContextConfig(), storage_dir=tmp_path)
    scenario = Scenario(
        id="NO-LEDGER",
        name="no ledger",
        execution_sequence=[
            MemoryManipulation(command="retain", parameters={"content": "fact"})
        ],
    )

    _, _, [response] = _execute_with_recorders(scenario, agent, _RecordingEvaluator())

    assert response.memory_status is None
    assert "memory_status" not in response.model_dump()


def test_unknown_agent_manipulation_names_command_and_scenario() -> None:
    scenario = Scenario(
        id="UNKNOWN-RESET",
        name="unknown reset",
        execution_sequence=[AgentManipulation(command="replace_state")],
    )

    with pytest.raises(ValueError) as error:
        _execute_with_recorders(scenario, _RecordingAgent(), _RecordingEvaluator())

    assert "replace_state" in str(error.value)
    assert scenario.id in str(error.value)


def test_user_stimuli_do_not_implicitly_reset_agent_state() -> None:
    scenario = Scenario(
        id="NO-RESET",
        name="no reset",
        execution_sequence=[
            UserStimulus(content="first"),
            UserStimulus(content="second"),
        ],
    )
    agent = _RecordingAgent()

    _execute_with_recorders(scenario, agent, _RecordingEvaluator())

    assert agent.reset_calls == 0


def test_orchestrator_records_capped_agent_runs_as_failed_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _scenario_with_env_step("Personal")

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()
        stub.run.side_effect = ToolExecutionLimitReached("Exceeded 10 tool executions")
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    )
    results = orchestrator.run()

    run = results[scenario.id][0]
    assert run.audit.is_successful is False
    assert run.audit.accuracy_score == 0.0
    assert run.audit.failed_keys == ["run_error: ToolExecutionLimitReached"]
    assert "Exceeded 10 tool executions" in run.audit.audit_notes


def test_orchestrator_continues_after_a_capped_agent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_scenario = _scenario_with_env_step("Personal")
    second_scenario = _scenario_with_env_step("Work").model_copy(
        update={"id": "ENV-DISPATCH-2", "name": "env-dispatch-2"}
    )

    create_agent_calls = {"count": 0}

    def fake_create_agent(config, tools, memory_system=None):
        stub = MagicMock()
        create_agent_calls["count"] += 1
        if create_agent_calls["count"] == 1:
            stub.run.side_effect = ToolExecutionLimitReached(
                "Exceeded 10 tool executions"
            )
        else:
            stub.run.return_value = AgentResponse(
                answer="noop",
                raw_trace=MagicMock(spans=[], final_output="noop"),
            )
        return stub

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent", fake_create_agent
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._build_ground_truth",
        lambda *args, **kwargs: MagicMock(),
    )

    class _PassingEvaluator:
        def audit(self, *args, **kwargs):
            return AuditRecord(
                is_successful=True,
                accuracy_score=1.0,
                passed_keys=["workspace_marker"],
                failed_keys=[],
                audit_notes="stub",
            )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *args, **kwargs: _PassingEvaluator(),
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[first_scenario, second_scenario],
            workspace_dir=tmp_path / "ws",
        )
    )
    results = orchestrator.run()

    first_run = results[first_scenario.id][0]
    second_run = results[second_scenario.id][0]
    assert first_run.audit.is_successful is False
    assert first_run.audit.failed_keys == ["run_error: ToolExecutionLimitReached"]
    assert "Exceeded 10 tool executions" in first_run.audit.audit_notes
    assert second_run.audit.is_successful is True
    assert second_run.audit.accuracy_score == 1.0


def test_locomo_uses_synchronous_evaluator_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SyncAgent:
        def run(self, prompt: str, *, run_context=None) -> AgentResponse:
            return AgentResponse(answer="stub", raw_trace=None)

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_agent",
        lambda *a, **kw: _SyncAgent(),
    )
    captured: dict[str, object] = {}

    class _InterfaceEvaluator:
        def audit(self, agent_responses, ground_truth):
            assert isinstance(ground_truth, LocomoQAGroundTruth)
            captured["responses"] = agent_responses
            captured["ground_truth"] = ground_truth
            return AuditRecord(
                is_successful=True,
                accuracy_score=0.0,
                passed_keys=[],
                failed_keys=[],
                audit_notes="stub",
            )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.create_evaluator",
        lambda *a, **kw: _InterfaceEvaluator(),
    )

    scenario = Scenario(
        id="TOP-LEVEL-ASYNC",
        name="top-level-async",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[_agent_config()],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[load_locomo_judge_config()],
        ground_truth=LocomoQAGroundTruth(
            task_description="top-level-async",
            qa=[
                LocomoQA(
                    question="Who likes cycling?",
                    expected_answer="Alex likes cycling.",
                    category="preference",
                ),
                LocomoQA(
                    question="Where does Sam work?",
                    expected_answer="Sam works at Mozilla.",
                    category="work",
                ),
            ],
        ),
        execution_sequence=[
            UserStimulus(
                content="Who likes cycling?",
                step_evaluation_markers={
                    "expected_answer": "Alex likes cycling.",
                    "category": "preference",
                },
            ),
            UserStimulus(
                content="Where does Sam work?",
                step_evaluation_markers={
                    "expected_answer": "Sam works at Mozilla.",
                    "category": "work",
                },
            ),
        ],
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path / "ws",
            num_samples=2,
        )
    )
    orchestrator.run()

    assert len(captured["responses"]) == 2
    assert len(captured["ground_truth"].qa) == 2


def test_deterministic_memory_usage_emits_isolated_per_run_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_tiny_agent_create(agent_config):
        class _StubTinyAgent:
            def run(self, messages):
                return MagicMock(output="deterministic answer", final_output="")

        return _StubTinyAgent()

    monkeypatch.setattr(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async",
        _fake_tiny_agent_create,
    )

    scenario = Scenario(
        id="DET-LEDGER",
        name="det-ledger",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[
            AgentParameters(
                memory_strategy="deterministic",
                model_id="stub",
                instructions="",
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(
                content="who am i",
                step_evaluation_markers={"expected_answer": "a"},
            ),
            UserStimulus(
                content="what next",
                step_evaluation_markers={"expected_answer": "b"},
            ),
        ],
    )

    orchestrator = Orchestrator(
        BenchmarkConfig(
            scenarios=[scenario],
            workspace_dir=tmp_path / "ws",
            num_samples=2,
        )
    )
    runs = orchestrator.run()[scenario.id]

    assert len(runs) == 2
    assert runs[0].memory_ledger is not None
    assert runs[1].memory_ledger is not None
    assert runs[0].memory_ledger is not runs[1].memory_ledger

    for run in runs:
        assert run.memory_ledger is not None
        assert [event.operation_type.value for event in run.memory_ledger] == [
            "recall",
            "retain",
            "recall",
            "retain",
        ]
        assert all(
            event.payload["run_context"]["sample_index"] == run.sample_index
            for event in run.memory_ledger
        )
        assert run.component_metrics is not None
        assert run.component_metrics[0].operation_counts == {"recall": 2, "retain": 2}

        payload = json.loads(run.model_dump_json())
        assert "memory_ledger" in payload
        assert "component_metrics" in payload


def test_tool_use_memory_usage_emits_run_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_tiny_agent_create(agent_config):
        class _StubTinyAgent:
            def run(self, messages):
                recall_tool = next(
                    tool
                    for tool in agent_config.tools
                    if getattr(tool, "__name__", "") == "recall"
                )
                retain_tool = next(
                    tool
                    for tool in agent_config.tools
                    if getattr(tool, "__name__", "") == "retain"
                )
                _ = recall_tool("find fact", top_k=1)
                _ = retain_tool("tool retained fact", metadata={"source": "tool"})
                return MagicMock(output="tool-use answer", final_output="")

        return _StubTinyAgent()

    monkeypatch.setattr(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async",
        _fake_tiny_agent_create,
    )

    scenario = Scenario(
        id="TOOL-LEDGER",
        name="tool-ledger",
        scenario_priming=ScenarioPriming(
            environment=EnvironmentPriming(config=None),
            memory=MemoryPriming(entries=[]),
        ),
        agents=[
            AgentParameters(
                memory_strategy="tool_use",
                model_id="stub",
                instructions="",
            )
        ],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[{"type": "task_qa"}],
        execution_sequence=[
            UserStimulus(
                content="call memory tools",
                step_evaluation_markers={"expected_answer": "a"},
            )
        ],
    )

    run = Orchestrator(
        BenchmarkConfig(scenarios=[scenario], workspace_dir=tmp_path / "ws")
    ).run()[scenario.id][0]

    assert run.memory_ledger is not None
    assert [event.operation_type.value for event in run.memory_ledger] == [
        "recall",
        "retain",
    ]
    assert run.component_metrics is not None
    assert run.component_metrics[0].operation_counts == {"recall": 1, "retain": 1}


def test_set_retention_policy_dispatches_to_the_agents_memory_system(
    tmp_path,
) -> None:
    statement = "Do not retain the user's pet information."
    memory = KeywordGovernedMemory(storage_dir=tmp_path, drop_substrings=["dog"])
    agent = _RecordingAgent()
    agent.memory_system = memory
    scenario = Scenario(
        id="SET-POLICY",
        name="set policy",
        execution_sequence=[
            MemoryManipulation(
                command="set_retention_policy",
                parameters={"policies": [statement]},
            ),
            MemoryManipulation(
                command="retain", parameters={"content": "My dog is Rex."}
            ),
        ],
    )

    _, _, responses = _execute_with_recorders(scenario, agent, _RecordingEvaluator())

    assert [response.answer for response in responses] == [
        f"Enforcing {[statement]}",
        "Not stored: the policy forbids it.",
    ]


def test_memory_manipulation_responses_carry_the_operation_status(tmp_path) -> None:
    memory = KeywordGovernedMemory(storage_dir=tmp_path, drop_substrings=["dog"])
    ledger: list = []
    memory.attach_event_sink(ledger)
    agent = _RecordingAgent()
    agent.memory_system = memory
    scenario = Scenario(
        id="STATUS",
        name="status",
        execution_sequence=[
            MemoryManipulation(
                command="set_retention_policy",
                parameters={"policies": ["Do not retain the user's pet information."]},
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
        evaluator=_RecordingEvaluator(),
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
