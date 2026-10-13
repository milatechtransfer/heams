from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from heams.benchmarks.factory import CIBenchmarkConfig, LocomoBenchmarkConfig
from heams.evaluation.schemas import AuditRecord
from heams.memory_systems import InContextConfig
from heams.orchestration import (
    RunDirectoryError,
    execute_run,
    prepare_run,
    resolve_run_config,
)
from heams.orchestration.run_orchestration import build_experiment_execution_configs
from heams.orchestration.scenario_runtime import (
    setup_scenario_runtime,
    teardown_scenario_runtime,
)
from heams.result_ledger import ResultLedger
from heams.results import RunResult
from heams.scenarios import Scenario, UserStimulus
from heams.scenarios.config import ScenarioExecutionConfig
from heams.utils.settings import settings


def test_locomo_campaign_resolves_directory_owned_storage(
    tmp_path: Path,
) -> None:
    run_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "runs"
        / "in_context_all_benchmarks"
        / "locomo.yaml"
    )

    [experiment_config] = build_experiment_execution_configs(
        resolve_run_config(run_path)
    )

    assert experiment_config.experiment_name == ("gpt56_luna_in_context_locomo")
    assert isinstance(
        experiment_config.memory_system_config.backend_config,
        InContextConfig,
    )
    backend_config = experiment_config.memory_system_config.backend_config
    authored_config = backend_config.model_dump(mode="json")

    scenario = Scenario(
        id="locomo-smoke",
        name="locomo-smoke",
        execution_sequence=[UserStimulus(content="question")],
    )
    _, memory, storage_dir, _, _, _ = setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=experiment_config.agent_config,
        mem_cfg=experiment_config.memory_system_config,
        eval_cfg=MagicMock(),
        combo_key="a0-m0-e0",
        sample_index=0,
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert storage_dir == tmp_path / "locomo-smoke-a0-m0-e0-s0-memory"
    assert memory.storage_dir == storage_dir
    assert backend_config.model_dump(mode="json") == authored_config
    serialized_experiment = experiment_config.model_dump(mode="json")
    assert str(storage_dir) not in json.dumps(serialized_experiment)

    teardown_scenario_runtime(None, storage_dir)
    assert not storage_dir.exists()


def test_build_experiment_execution_configs_loads_typed_component_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    resolved_run_config = resolve_run_config(run_path, output_dir=results_dir)

    experiment_configs = build_experiment_execution_configs(resolved_run_config)

    assert len(experiment_configs) == 1
    assert experiment_configs[0].experiment_name == "gpt4o_mini_nomemory_locomo"
    assert experiment_configs[0].agent_config.type == "agent"
    assert experiment_configs[0].memory_system_config.backend_type == "no_memory"
    assert isinstance(experiment_configs[0].benchmark_config, LocomoBenchmarkConfig)


def test_execute_run_writes_run_and_experiment_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _NoopOrchestrator:
        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config

        def run(self, **kwargs):
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _NoopOrchestrator,
    )

    resolved_run_config = resolve_run_config(run_path, output_dir=results_dir)

    execute_run(resolved_run_config)

    run_artifact_path = (
        resolved_run_config.results_dir
        / f"run_config.{resolved_run_config.run_id}.yaml"
    )
    experiment_dir = resolved_run_config.results_dir / "gpt4o_mini_nomemory_locomo"
    experiment_artifact_path = (
        experiment_dir / f"experiment_config.{resolved_run_config.run_id}.yaml"
    )

    assert run_artifact_path.is_file()
    assert experiment_dir.is_dir()
    assert experiment_artifact_path.is_file()
    assert (experiment_dir / "execution_traces.jsonl").is_file()

    run_artifact = yaml.safe_load(run_artifact_path.read_text(encoding="utf-8"))
    experiment_artifact = yaml.safe_load(
        experiment_artifact_path.read_text(encoding="utf-8")
    )

    assert run_artifact["run_id"] == resolved_run_config.run_id
    assert experiment_artifact["experiment_name"] == "gpt4o_mini_nomemory_locomo"
    assert experiment_artifact["agent_config"]["type"] == "agent"
    assert (
        experiment_artifact["memory_system_config"]["backend_config"]["type"]
        == "no_memory"
    )
    assert experiment_artifact["benchmark_config"]["type"] == "locomo"
    assert (
        experiment_artifact["benchmark_config"]["agent_conversation"] == "independent"
    )
    assert _NoopOrchestrator.last_config is not None
    assert len(_NoopOrchestrator.last_config.scenarios) == 1
    assert not hasattr(_NoopOrchestrator.last_config, "concurrency")


def test_prepare_run_initializes_exact_directory_and_resumes_from_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "results" / "exact-run"

    first = prepare_run(run_path, output_dir)

    assert first.resumed is False
    assert first.run_config.results_dir == output_dir.resolve()
    experiment_dir = output_dir / "gpt4o_mini_nomemory_locomo"
    assert (output_dir / f"run_config.{first.run_config.run_id}.yaml").is_file()
    assert (
        experiment_dir / f"experiment_config.{first.run_config.run_id}.yaml"
    ).is_file()
    assert (experiment_dir / "runs.jsonl").read_text(encoding="utf-8") == ""

    run_path.write_text("run_name: edited-source\nexperiments: []\n", encoding="utf-8")
    resumed = prepare_run(run_path, output_dir)

    assert resumed.resumed is True
    assert resumed.run_config.run_id == first.run_config.run_id
    assert resumed.run_config.run_name == "example_campaign"
    assert resumed.experiment_configs[0].experiment_name == (
        "gpt4o_mini_nomemory_locomo"
    )


def test_prepare_run_refuses_non_mapb_directory_without_modifying_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "not-a-run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(RunDirectoryError):
        prepare_run(Path("configs/runs/smoke_test_benchmark.yaml"), output_dir)

    assert marker.read_text(encoding="utf-8") == "untouched"


def test_prepare_run_refuses_mismatched_snapshot_without_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "results" / "mismatched-run"
    prepared = prepare_run(run_path, output_dir)
    snapshot_path = output_dir / f"run_config.{prepared.run_config.run_id}.yaml"
    original_snapshot = snapshot_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(original_snapshot)
    payload["results_dir"] = str(tmp_path / "another-run")
    snapshot_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mismatched_snapshot = snapshot_path.read_text(encoding="utf-8")

    with pytest.raises(RunDirectoryError):
        prepare_run(run_path, output_dir)

    assert snapshot_path.read_text(encoding="utf-8") == mismatched_snapshot
    assert original_snapshot != mismatched_snapshot


def test_prepare_run_refuses_malformed_ledger_without_writing_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "results" / "malformed-ledger"
    prepared = prepare_run(run_path, output_dir)
    ledger_path = output_dir / "gpt4o_mini_nomemory_locomo" / "runs.jsonl"
    ledger_path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(RunDirectoryError):
        prepare_run(run_path, output_dir)

    assert ledger_path.read_text(encoding="utf-8") == "{not valid json}\n"
    assert not (output_dir / "gpt4o_mini_nomemory_locomo" / "run_status.json").exists()
    assert prepared.run_config.run_id


def test_overwrite_clears_only_a_recognized_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "results" / "overwrite-run"
    first = prepare_run(run_path, output_dir)
    marker = output_dir / "user-workspace.txt"
    marker.write_text("old", encoding="utf-8")

    restarted = prepare_run(run_path, output_dir, overwrite=True)

    assert restarted.resumed is False
    assert restarted.run_config.run_id != first.run_config.run_id
    assert not marker.exists()
    assert len(list(output_dir.glob("run_config.*.yaml"))) == 1


def test_overwrite_refuses_non_mapb_directory_without_clearing_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "not-a-run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(RunDirectoryError):
        prepare_run(
            Path("configs/runs/smoke_test_benchmark.yaml"), output_dir, overwrite=True
        )

    assert marker.read_text(encoding="utf-8") == "untouched"


def test_complete_run_does_not_construct_runtime_on_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "results" / "complete-run"
    prepared = prepare_run(run_path, output_dir)

    class _RecordingOrchestrator:
        calls = 0

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).calls += 1
            self.scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord
            from heams.results import RunResult

            on_run_complete = kwargs["on_run_complete"]
            for scenario in self.scenario_config.scenarios:
                on_run_complete(
                    RunResult(
                        scenario_id=scenario.id,
                        agent_index=0,
                        memory_index=0,
                        evaluator_index=0,
                        sample_index=0,
                        audit=AuditRecord(
                            is_successful=True,
                            accuracy_score=1.0,
                            passed_keys=[],
                            failed_keys=[],
                            audit_notes="stub",
                        ),
                    )
                )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingOrchestrator,
    )
    execute_run(prepared.run_config, prepared.experiment_configs)
    assert _RecordingOrchestrator.calls == 1

    run_path.write_text("run_name: edited-source\nexperiments: []\n", encoding="utf-8")

    class _UnexpectedOrchestrator:
        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            raise AssertionError("complete run constructed runtime orchestrator")

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _UnexpectedOrchestrator,
    )
    resumed = prepare_run(run_path, output_dir)
    execute_run(resumed.run_config, resumed.experiment_configs)


def test_interrupted_run_resumes_only_missing_durable_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    dataset_path = tmp_path / "locomo_fixture.json"
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    expanded_records = []
    for index in range(3):
        record = json.loads(json.dumps(records[0]))
        record["sample_id"] = f"LOC-FIXTURE-{index + 1}"
        record["conversation"]["session_1"][0]["dia_id"] = f"D{index + 1}:1"
        record["qa"][0]["evidence"] = [f"D{index + 1}:1"]
        expanded_records.append(record)
    dataset_path.write_text(json.dumps(expanded_records), encoding="utf-8")

    benchmark_path = run_path.parents[1] / "benchmarks" / "locomo.yaml"
    benchmark_payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    benchmark_payload["limit"] = 3
    benchmark_path.write_text(
        yaml.safe_dump(benchmark_payload, sort_keys=False),
        encoding="utf-8",
    )

    phase = "interrupted"
    initialized_by_phase: dict[str, list[str]] = {
        "interrupted": [],
        "resumed": [],
        "baseline": [],
    }
    executed_by_phase: dict[str, list[str]] = {
        "interrupted": [],
        "resumed": [],
        "baseline": [],
    }
    pending_keys_by_phase: dict[str, set[tuple[str, int]]] = {}

    def fake_setup(self, scenario, **kwargs) -> None:
        initialized_by_phase[phase].append(scenario.id)
        pending_keys_by_phase[phase] = set(self.config.pending_keys or set())
        self.active_environment = None
        self.active_memory = None
        self._active_memory_path = None
        self.active_agent = None
        self.active_evaluator = None
        self.active_memory_ledger = []

    def fake_teardown(self) -> None:
        self.active_environment = None
        self.active_memory = None
        self._active_memory_path = None
        self.active_agent = None
        self.active_evaluator = None
        self.active_memory_ledger = None

    def fake_execute_scenario(*, scenario, **kwargs):
        executed_by_phase[phase].append(scenario.id)
        if phase == "interrupted" and len(executed_by_phase[phase]) == 3:
            raise ValueError("simulated interruption")
        return (
            AuditRecord(
                is_successful=True,
                accuracy_score=1.0,
                passed_keys=["deterministic"],
                failed_keys=[],
                audit_notes="deterministic test audit",
            ),
            [],
            [],
        )

    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._setup_scenario_runtime",
        fake_setup,
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.Orchestrator._teardown_scenario_runtime",
        fake_teardown,
    )
    monkeypatch.setattr(
        "heams.orchestration.orchestrator.execute_scenario",
        fake_execute_scenario,
    )

    output_dir = tmp_path / "results" / "interrupted-run"
    prepared = prepare_run(run_path, output_dir)
    with pytest.raises(ValueError, match="simulated interruption"):
        execute_run(prepared.run_config, prepared.experiment_configs)

    experiment_dir = output_dir / "gpt4o_mini_nomemory_locomo"
    ledger_path = experiment_dir / "runs.jsonl"
    interrupted_ledger = ResultLedger(ledger_path)
    assert [run.scenario_id for run in interrupted_ledger.results] == [
        "LOC-FIXTURE-1",
        "LOC-FIXTURE-2",
    ]
    interrupted_ledger.close()
    assert initialized_by_phase["interrupted"] == [
        "LOC-FIXTURE-1",
        "LOC-FIXTURE-2",
        "LOC-FIXTURE-3",
    ]

    baseline_dir = tmp_path / "results" / "baseline-run"
    phase = "baseline"
    baseline = prepare_run(run_path, baseline_dir)
    execute_run(baseline.run_config, baseline.experiment_configs)

    run_path.write_text("run_name: edited-source\nexperiments: []\n", encoding="utf-8")
    phase = "resumed"
    resumed = prepare_run(run_path, output_dir)
    assert resumed.resumed is True
    assert resumed.run_config.run_id == prepared.run_config.run_id
    assert resumed.run_config.run_name == "example_campaign"
    execute_run(resumed.run_config, resumed.experiment_configs)

    assert pending_keys_by_phase["resumed"] == {("LOC-FIXTURE-3", 0)}
    assert initialized_by_phase["resumed"] == ["LOC-FIXTURE-3"]
    assert executed_by_phase["resumed"] == ["LOC-FIXTURE-3"]

    def ledger_signature(path: Path) -> dict[tuple[str, int], dict[str, object]]:
        ledger = ResultLedger(path)
        try:
            return {
                (run.scenario_id, run.sample_index): run.model_dump(
                    mode="json",
                    exclude={"system_metrics", "trace_id"},
                )
                for run in ledger.results
            }
        finally:
            ledger.close()

    resumed_signature = ledger_signature(ledger_path)
    baseline_signature = ledger_signature(
        baseline_dir / "gpt4o_mini_nomemory_locomo" / "runs.jsonl"
    )
    assert resumed_signature == baseline_signature
    assert set(resumed_signature) == {
        ("LOC-FIXTURE-1", 0),
        ("LOC-FIXTURE-2", 0),
        ("LOC-FIXTURE-3", 0),
    }

    resumed_runs = ResultLedger(ledger_path)
    try:
        run_trace_ids = [run.trace_id for run in resumed_runs.results]
    finally:
        resumed_runs.close()
    assert all(trace_id is not None for trace_id in run_trace_ids)
    trace_counts: dict[str, int] = {}
    for raw_line in (
        (experiment_dir / "execution_traces.jsonl").read_bytes().split(b"\n")
    ):
        if not raw_line.strip():
            continue
        trace_id = json.loads(raw_line)["trace_id"]
        trace_counts[trace_id] = trace_counts.get(trace_id, 0) + 1
    for trace_id in run_trace_ids:
        assert trace_counts.get(trace_id) == 1


def test_execute_run_passes_resolved_component_configs_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(
        tmp_path,
        memory_top_k=5,
        include_basic_memory=True,
    )
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _RecordingOrchestrator:
        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config
            self._scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord
            from heams.results import RunResult

            on_run_complete = kwargs.get("on_run_complete")
            for scenario in self._scenario_config.scenarios:
                if on_run_complete is not None:
                    on_run_complete(
                        RunResult(
                            scenario_id=scenario.id,
                            agent_index=0,
                            memory_index=0,
                            evaluator_index=0,
                            sample_index=0,
                            audit=AuditRecord(
                                is_successful=True,
                                accuracy_score=1.0,
                                passed_keys=["q0:4"],
                                failed_keys=[],
                                audit_notes="stub",
                            ),
                        )
                    )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingOrchestrator,
    )

    execute_run(resolve_run_config(run_path, output_dir=results_dir))

    assert _RecordingOrchestrator.last_config is not None
    scenario = _RecordingOrchestrator.last_config.scenarios[0]
    assert scenario.agents == []
    assert scenario.memory_systems == []
    assert _RecordingOrchestrator.last_config.agent_config is not None
    assert _RecordingOrchestrator.last_config.agent_config.type == "agent"
    assert (
        _RecordingOrchestrator.last_config.agent_config.memory_strategy.value
        == "tool_use"
    )
    assert _RecordingOrchestrator.last_config.memory_system_config is not None
    assert (
        _RecordingOrchestrator.last_config.memory_system_config.backend_config.top_k
        == 5
    )
    assert scenario.ground_truth is not None


def test_execute_run_records_benchmark_materialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    def fail_materialization(*args, **kwargs):
        raise RuntimeError("required source unavailable")

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.create_benchmark",
        fail_materialization,
    )

    resolved_run_config = resolve_run_config(run_path, output_dir=results_dir)
    with pytest.raises(RuntimeError, match="required source unavailable"):
        execute_run(resolved_run_config)

    status_path = (
        resolved_run_config.results_dir
        / "gpt4o_mini_nomemory_locomo"
        / "run_status.json"
    )
    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "success": False,
        "runnable_count": 0,
        "skipped_count": 0,
        "error": "required source unavailable",
    }


def test_execute_run_records_duplicate_scenario_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _Sample:
        def __init__(self, scenario_id: str) -> None:
            self.scenario_id = scenario_id

        def to_scenario(self) -> Scenario:
            return Scenario(
                id=self.scenario_id,
                name=self.scenario_id,
                execution_sequence=[UserStimulus(content="question")],
            )

    class _DuplicateBenchmark:
        skipped: list[object] = []

        def __iter__(self):
            return iter((_Sample("duplicate"), _Sample("duplicate")))

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.create_benchmark",
        lambda config: _DuplicateBenchmark(),
    )

    resolved_run_config = resolve_run_config(run_path, output_dir=results_dir)
    with pytest.raises(
        ValueError,
        match="Experiment 'gpt4o_mini_nomemory_locomo' produced duplicate "
        "scenario IDs: duplicate\\.",
    ):
        execute_run(resolved_run_config)

    status_path = (
        resolved_run_config.results_dir
        / "gpt4o_mini_nomemory_locomo"
        / "run_status.json"
    )
    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "success": False,
        "runnable_count": 2,
        "skipped_count": 0,
        "error": (
            "Experiment 'gpt4o_mini_nomemory_locomo' produced duplicate "
            "scenario IDs: duplicate."
        ),
    }


def test_execute_run_records_unexpected_ledger_key_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    results_dir = tmp_path / "results" / "example_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    prepared = prepare_run(run_path, results_dir)
    ledger_path = (
        prepared.run_config.results_dir / "gpt4o_mini_nomemory_locomo" / "runs.jsonl"
    )
    with ResultLedger(ledger_path) as ledger:
        ledger.append(
            RunResult(
                scenario_id="unexpected",
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=[],
                    failed_keys=[],
                    audit_notes="stub",
                ),
            )
        )

    message = (
        f"Result ledger {ledger_path} contains keys outside the materialized "
        "benchmark: [('unexpected', 0)]."
    )
    with pytest.raises(ValueError, match="contains keys outside the materialized"):
        execute_run(prepared.run_config, prepared.experiment_configs)

    status_path = ledger_path.parent / "run_status.json"
    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "success": False,
        "runnable_count": 1,
        "skipped_count": 0,
        "error": message,
    }


def test_execute_run_materializes_transparency_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heams.benchmarks.factory import TransparencyBenchmarkConfig

    run_path = _write_transparency_config_tree(tmp_path)
    results_dir = tmp_path / "results" / "transparency_campaign"
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    def load_test_beam(*, split: str) -> list[dict]:
        assert split == "100K"
        return [{}]

    def convert_test_beam(*_args: object, **_kwargs: object) -> list[dict]:
        source = (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "smoke_e2e"
            / "transparency"
            / "smoke.json"
        )
        return json.loads(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "heams.benchmarks.transparency.dataset.load_beam_records",
        load_test_beam,
    )
    monkeypatch.setattr(
        "heams.benchmarks.transparency.dataset.convert_beam_records_to_transparency",
        convert_test_beam,
    )

    class _RecordingOrchestrator:
        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config
            self._scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord, EvaluationRecord
            from heams.results import RunResult

            on_run_complete = kwargs.get("on_run_complete")
            for scenario in self._scenario_config.scenarios:
                if on_run_complete is not None:
                    on_run_complete(
                        RunResult(
                            scenario_id=scenario.id,
                            agent_index=0,
                            memory_index=0,
                            evaluator_index=0,
                            sample_index=0,
                            audit=AuditRecord(
                                is_successful=True,
                                accuracy_score=1.0,
                                passed_keys=["q0:ok"],
                                failed_keys=[],
                                audit_notes="stub",
                                report_records=[
                                    EvaluationRecord(
                                        key="q0:ok",
                                        passed=True,
                                        score=1.0,
                                        tags={
                                            "dimension": "completeness",
                                            "scenario_type": "direct_inspection",
                                            "privacy_domain": "health",
                                            "persona_id": "persona_a",
                                            "difficulty_level": "easy",
                                        },
                                    )
                                ],
                            ),
                        )
                    )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingOrchestrator,
    )

    resolved = resolve_run_config(run_path, output_dir=results_dir)
    experiment_configs = build_experiment_execution_configs(resolved)
    assert isinstance(
        experiment_configs[0].benchmark_config, TransparencyBenchmarkConfig
    )
    assert experiment_configs[0].benchmark_config.source.type == "beam"
    assert (
        experiment_configs[0].agent_config.memory_strategy.value
        == "recall_only_tool_use"
    )
    assert experiment_configs[0].memory_system_config.backend_config.top_k == 50

    execute_run(resolved)

    assert _RecordingOrchestrator.last_config is not None
    assert _RecordingOrchestrator.last_config.agent_config is not None
    assert "transparent access" in (
        _RecordingOrchestrator.last_config.agent_config.instructions or ""
    )
    assert (
        _RecordingOrchestrator.last_config.memory_system_config.backend_config.top_k
        == 50
    )
    scenario = _RecordingOrchestrator.last_config.scenarios[0]
    assert scenario.ground_truth is not None

    runs_path = resolved.results_dir / "gpt4o_mini_basic_transparency" / "runs.jsonl"
    assert runs_path.is_file()
    assert not (
        tmp_path
        / "results"
        / "transparency_campaign"
        / "_rollup"
        / "transparency"
        / "report.md"
    ).exists()


def test_execute_run_materializes_stale_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heams.benchmarks.factory import StaleBenchmarkConfig

    results_dir = tmp_path / "results" / "stale_campaign"
    run_path = _write_stale_config_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _RecordingOrchestrator:
        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config
            self._scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord
            from heams.results import RunResult

            on_run_complete = kwargs.get("on_run_complete")
            for scenario in self._scenario_config.scenarios:
                if on_run_complete is not None:
                    on_run_complete(
                        RunResult(
                            scenario_id=scenario.id,
                            agent_index=0,
                            memory_index=0,
                            evaluator_index=0,
                            sample_index=0,
                            audit=AuditRecord(
                                is_successful=True,
                                accuracy_score=1.0,
                                passed_keys=["q0:STALE-WIRE-1:dim1:T1"],
                                failed_keys=[],
                                audit_notes="stub",
                            ),
                        )
                    )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingOrchestrator,
    )

    resolved = resolve_run_config(run_path, output_dir=results_dir)
    experiment_configs = build_experiment_execution_configs(resolved)
    assert isinstance(experiment_configs[0].benchmark_config, StaleBenchmarkConfig)
    assert experiment_configs[0].benchmark_config.source.local_path is not None
    # The stale agent config resolves into a wrapper agent (strategy/instructions
    # are tunable component config, so we don't pin their exact values here).
    assert experiment_configs[0].agent_config.type == "agent"

    execute_run(resolved)

    assert _RecordingOrchestrator.last_config is not None
    assert _RecordingOrchestrator.last_config.agent_config is not None
    # One source sample -> one scenario with three probing stimuli (dim1/2/3).
    scenarios = _RecordingOrchestrator.last_config.scenarios
    assert [scenario.id for scenario in scenarios] == ["STALE-WIRE-1"]
    assert len(scenarios[0].execution_sequence) == 3
    assert scenarios[0].ground_truth is not None
    assert scenarios[0].agents == []
    assert scenarios[0].memory_systems == []

    experiment_dir = resolved.results_dir / "gpt4o_mini_nomemory_stale"
    assert (experiment_dir / f"experiment_config.{resolved.run_id}.yaml").is_file()
    assert (experiment_dir / "runs.jsonl").is_file()
    assert (resolved.results_dir / f"run_config.{resolved.run_id}.yaml").is_file()


def _write_stale_config_tree(tmp_path: Path) -> Path:
    package_root = tmp_path / "python_package"
    config_root = package_root / "configs"
    (config_root / "agents").mkdir(parents=True)
    (config_root / "memory_systems").mkdir(parents=True)
    (config_root / "benchmarks").mkdir(parents=True)
    (config_root / "runs").mkdir(parents=True)

    package_agent = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "agents"
        / "testing"
        / "stale_gpt4o_mini_recall_only_tool_use.yaml"
    )
    (config_root / "agents" / "stale_gpt4o_mini_recall_only_tool_use.yaml").write_text(
        package_agent.read_text(encoding="utf-8"), encoding="utf-8"
    )

    (config_root / "memory_systems" / "no_memory.yaml").write_text(
        "\n".join(
            [
                "id: no_memory",
                "backend_config:",
                "  type: no_memory",
                "  name: no_memory",
                "  top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fixture = tmp_path / "stale_fixture.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "uid": "STALE-WIRE-1",
                    "conflict_type": "T1",
                    "M_old": "The user is vegetarian.",
                    "M_new": "The user eats meat again.",
                    "explanation": "M_new supersedes M_old.",
                    "haystack_session": [
                        {"role": "user", "content": "I'm vegetarian."},
                        {"role": "user", "content": "Update: I eat meat again now."},
                    ],
                    "probing_queries": {
                        "dim1_query": "What are my current dietary preferences?",
                        "dim2_query": "Since I'm vegetarian, suggest a meatless dinner.",
                        "dim3_query": "Plan my dinner tonight.",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    package_bench = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "testing"
        / "stale_smoke.yaml"
    )
    payload = yaml.safe_load(package_bench.read_text(encoding="utf-8"))
    payload["source"] = {"local_path": str(fixture)}
    payload["limit"] = 1
    (config_root / "benchmarks" / "stale_smoke.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    run_path = config_root / "runs" / "stale_campaign.yaml"
    run_path.write_text(
        "\n".join(
            [
                "run_name: stale_campaign",
                "experiments:",
                "  - name: gpt4o_mini_nomemory_stale",
                "    agent_config_file: stale_gpt4o_mini_recall_only_tool_use.yaml",
                "    memory_system_config_file: no_memory.yaml",
                "    benchmark_config_file: stale_smoke.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_path


def _write_transparency_config_tree(tmp_path: Path) -> Path:
    package_root = tmp_path / "python_package"
    config_root = package_root / "configs"
    (config_root / "agents").mkdir(parents=True)
    (config_root / "memory_systems").mkdir(parents=True)
    (config_root / "benchmarks").mkdir(parents=True)
    (config_root / "runs").mkdir(parents=True)

    package_agent = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "agents"
        / "testing"
        / "transparency_gpt4o_mini_recall_only_tool_use.yaml"
    )
    (
        config_root / "agents" / "transparency_gpt4o_mini_recall_only_tool_use.yaml"
    ).write_text(
        package_agent.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (config_root / "memory_systems" / "example_memory_transparency.yaml").write_text(
        "\n".join(
            [
                "id: example.in_memory",
                "backend_config:",
                "  type: example.in_memory",
                "  name: example.in_memory",
                "  top_k: 50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    package_bench = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "testing"
        / "transparency_smoke.yaml"
    )
    payload = yaml.safe_load(package_bench.read_text(encoding="utf-8"))
    payload["source"]["materialization_limit"] = 1
    payload["limit"] = 1
    (config_root / "benchmarks" / "transparency_smoke.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    run_path = config_root / "runs" / "transparency_campaign.yaml"
    run_path.write_text(
        "\n".join(
            [
                "run_name: transparency_campaign",
                "experiments:",
                "  - name: gpt4o_mini_basic_transparency",
                "    agent_config_file: "
                "transparency_gpt4o_mini_recall_only_tool_use.yaml",
                "    memory_system_config_file: example_memory_transparency.yaml",
                "    benchmark_config_file: transparency_smoke.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_path


def test_execute_run_drives_ci_benchmark_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    run_path = _write_ci_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _RecordingCIOrchestrator:
        """Stub agent+judge loop emitting canned CI audit records."""

        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config
            self._scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord, EvaluationRecord
            from heams.results import RunResult

            on_run_complete = kwargs.get("on_run_complete")
            for scenario in self._scenario_config.scenarios:
                records = [
                    EvaluationRecord(
                        key=f"{scenario.id}:summary",
                        passed=True,
                        score=1.0,
                        tags={
                            "record_type": "ci_summary",
                            "leaked": "false",
                            "helpful": "true",
                            "allowed_total": "4",
                            "allowed_covered": "3",
                        },
                    ),
                    EvaluationRecord(
                        key=f"{scenario.id}:secret:ssn",
                        passed=True,
                        score=1.0,
                        tags={"record_type": "secret", "field": "ssn"},
                    ),
                ]
                if on_run_complete is not None:
                    on_run_complete(
                        RunResult(
                            scenario_id=scenario.id,
                            agent_index=0,
                            memory_index=0,
                            evaluator_index=0,
                            sample_index=0,
                            audit=AuditRecord(
                                is_successful=True,
                                accuracy_score=1.0,
                                audit_notes="stub",
                                report_records=records,
                            ),
                        )
                    )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingCIOrchestrator,
    )

    resolved = resolve_run_config(run_path, output_dir=results_dir)
    execute_run(resolved)

    experiment_dir = resolved.results_dir / "gpt4o_mini_no_memory_contextual_integrity"

    # The benchmark produced scenarios and carried no agent/memory policy of its
    # own into them.
    scenarios = _RecordingCIOrchestrator.last_config.scenarios
    assert len(scenarios) == 2
    assert scenarios[0].agents == []
    assert scenarios[0].memory_systems == []
    assert scenarios[0].evaluators[0].type == "ci_llm_judge"

    # Resolved-config artifact captures the full benchmark settings tree.
    artifact = yaml.safe_load(
        (experiment_dir / f"experiment_config.{resolved.run_id}.yaml").read_text()
    )
    assert artifact["benchmark_config"]["type"] == "ci"
    assert artifact["agent_config"]["model_args"]["timeout"] == 120
    assert artifact["memory_system_config"]["backend_config"]["top_k"] == 5
    # Prompts are inlined, so the run is reproducible from the artifact alone.
    reparsed = CIBenchmarkConfig.model_validate(artifact["benchmark_config"])
    assert reparsed.evaluator.leakage_prompt_template

    # Per-run and status artifacts.
    runs = [
        json.loads(line)
        for line in (experiment_dir / "runs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(runs) == 2
    status = json.loads((experiment_dir / "run_status.json").read_text())
    assert status["success"] is True

    # Rollup is generated on demand from saved artifacts (execute_run no longer
    # writes it inline; it uses the shared typed rollup path).
    from heams.benchmarks.contextual_integrity.report import write_ci_rollup_report

    report_path = write_ci_rollup_report(resolved.results_dir)
    assert report_path == resolved.results_dir / "_rollup" / "ci" / "report.md"
    report = report_path.read_text(encoding="utf-8")
    assert "# ci benchmark" in report
    assert "Contextual Integrity metrics" in report
    assert "no_memory" in report


def _write_fixture_tree(
    tmp_path: Path,
    *,
    memory_top_k: int = 5,
    include_basic_memory: bool = False,
) -> Path:
    package_root = tmp_path / "python_package"
    config_root = package_root / "configs"
    (config_root / "agents").mkdir(parents=True)
    (config_root / "memory_systems").mkdir(parents=True)
    (config_root / "benchmarks").mkdir(parents=True)
    (config_root / "runs").mkdir(parents=True)

    package_agent_yaml = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "agents"
        / "testing"
        / "gpt4o_mini_tool_use.yaml"
    )
    (config_root / "agents" / "gpt4o_mini_tool_use.yaml").write_text(
        package_agent_yaml.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (config_root / "memory_systems" / "no_memory.yaml").write_text(
        "\n".join(
            [
                "id: no_memory",
                "backend_config:",
                "  type: no_memory",
                "  name: no_memory",
                f"  top_k: {memory_top_k}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if include_basic_memory:
        (config_root / "memory_systems" / "example_memory.yaml").write_text(
            "\n".join(
                [
                    "id: example.in_memory",
                    "backend_config:",
                    "  type: example.in_memory",
                    "  name: example.in_memory",
                    f"  top_k: {memory_top_k}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    dataset = tmp_path / "locomo_fixture.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "sample_id": "LOC-FIXTURE-1",
                    "conversation": {
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "text": "I moved to Montreal last spring.",
                                "dia_id": "D1:1",
                            }
                        ],
                    },
                    "qa": [
                        {
                            "question": "Where does Alice live?",
                            "answer": "Montreal",
                            "evidence": ["D1:1"],
                            "category": 4,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    package_locomo_yaml = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "full_campaign"
        / "locomo.yaml"
    )
    locomo_payload = yaml.safe_load(package_locomo_yaml.read_text(encoding="utf-8"))
    locomo_payload.update(
        {
            "examples_dir": str(dataset),
            "limit": 1,
            "limit_stimuli": 1,
        }
    )
    (config_root / "benchmarks" / "locomo.yaml").write_text(
        yaml.safe_dump(locomo_payload, sort_keys=False),
        encoding="utf-8",
    )

    memory_file = "example_memory.yaml" if include_basic_memory else "no_memory.yaml"
    run_path = config_root / "runs" / "example_campaign.yaml"
    run_path.write_text(
        "\n".join(
            [
                "run_name: example_campaign",
                "experiments:",
                "  - name: gpt4o_mini_nomemory_locomo",
                "    agent_config_file: gpt4o_mini_tool_use.yaml",
                f"    memory_system_config_file: {memory_file}",
                "    benchmark_config_file: locomo.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_path


def _write_ci_fixture_tree(tmp_path: Path) -> Path:
    package_root = Path(__file__).resolve().parents[2]
    fixture = (
        package_root
        / "tests"
        / "fixtures"
        / "smoke_e2e"
        / "contextual_integrity"
        / "smoke.json"
    )
    config_root = tmp_path / "python_package" / "configs"
    for subdir in ("agents", "memory_systems", "benchmarks", "runs"):
        (config_root / subdir).mkdir(parents=True)

    def _copy(subdir: str, name: str) -> dict:
        return yaml.safe_load(
            (package_root / "configs" / subdir / name).read_text(encoding="utf-8")
        )

    # Shipped benchmark config, repointed at the local fixture so the HF
    # bootstrap never runs in tests.
    benchmark = _copy("benchmarks", "testing/contextual_integrity_smoke.yaml")
    benchmark["examples_dir"] = str(fixture)
    benchmark["limit"] = 2
    (config_root / "benchmarks" / "contextual_integrity_smoke.yaml").write_text(
        yaml.safe_dump(benchmark, sort_keys=False), encoding="utf-8"
    )

    for subdir, name in (
        (
            "agents",
            "testing/contextual_integrity_gpt4o_mini_recall_only_deterministic.yaml",
        ),
        ("memory_systems", "testing/no_memory.yaml"),
    ):
        (config_root / subdir / Path(name).name).write_text(
            yaml.safe_dump(_copy(subdir, name), sort_keys=False), encoding="utf-8"
        )

    run_path = config_root / "runs" / "contextual_integrity_smoke.yaml"
    run_path.write_text(
        yaml.safe_dump(
            {
                "run_name": "contextual_integrity_smoke",
                "experiments": [
                    {
                        "name": "gpt4o_mini_no_memory_contextual_integrity",
                        "agent_config_file": (
                            "contextual_integrity_gpt4o_mini_recall_only_deterministic.yaml"
                        ),
                        "memory_system_config_file": "no_memory.yaml",
                        "benchmark_config_file": "contextual_integrity_smoke.yaml",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_path


def test_execute_run_materializes_erasability_editability_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heams.benchmarks.factory import EEBenchmarkConfig

    results_dir = tmp_path / "results" / "ee_campaign"
    run_path = _write_erasability_editability_config_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    class _RecordingOrchestrator:
        last_config: ScenarioExecutionConfig | None = None

        def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
            type(self).last_config = scenario_config
            self._scenario_config = scenario_config

        def run(self, **kwargs):
            from heams.evaluation.schemas import AuditRecord
            from heams.results import RunResult

            on_run_complete = kwargs.get("on_run_complete")
            for scenario in self._scenario_config.scenarios:
                if on_run_complete is not None:
                    on_run_complete(
                        RunResult(
                            scenario_id=scenario.id,
                            agent_index=0,
                            memory_index=0,
                            evaluator_index=0,
                            sample_index=0,
                            audit=AuditRecord(
                                is_successful=True,
                                accuracy_score=1.0,
                                passed_keys=["after0:pl_001:Cas"],
                                failed_keys=[],
                                audit_notes="stub",
                            ),
                        )
                    )
            return {}

    monkeypatch.setattr(
        "heams.orchestration.experiment_orchestration.Orchestrator",
        _RecordingOrchestrator,
    )

    resolved = resolve_run_config(run_path, output_dir=results_dir)
    experiment_configs = build_experiment_execution_configs(resolved)
    assert isinstance(experiment_configs[0].benchmark_config, EEBenchmarkConfig)
    assert experiment_configs[0].benchmark_config.source.local_path is not None
    assert experiment_configs[0].agent_config.type == "agent"

    execute_run(resolved)

    assert _RecordingOrchestrator.last_config is not None
    assert _RecordingOrchestrator.last_config.agent_config is not None
    scenarios = _RecordingOrchestrator.last_config.scenarios
    assert [scenario.id for scenario in scenarios] == ["pl_001"]
    # 3 before-probes -> 4 replayed change+delete turns -> 3 after-probes.
    assert len(scenarios[0].execution_sequence) == 10
    assert scenarios[0].ground_truth is not None
    # The change+delete session must reach the agent live, not via priming.
    assert any(
        "remove that from your memory" in step.content
        for step in scenarios[0].execution_sequence
    )
    assert not any(
        "remove that from your memory" in entry.content
        for entry in scenarios[0].scenario_priming.memory.entries
    )
    assert scenarios[0].agents == []
    assert scenarios[0].memory_systems == []

    experiment_dir = resolved.results_dir / "gpt4o_mini_nomemory_ee"
    assert (experiment_dir / f"experiment_config.{resolved.run_id}.yaml").is_file()
    assert (experiment_dir / "runs.jsonl").is_file()
    assert (experiment_dir / "run_status.json").is_file()
    assert (resolved.results_dir / f"run_config.{resolved.run_id}.yaml").is_file()


def _write_erasability_editability_config_tree(tmp_path: Path) -> Path:
    package_root = tmp_path / "python_package"
    config_root = package_root / "configs"
    (config_root / "agents").mkdir(parents=True)
    (config_root / "memory_systems").mkdir(parents=True)
    (config_root / "benchmarks").mkdir(parents=True)
    (config_root / "runs").mkdir(parents=True)

    repo_configs = Path(__file__).resolve().parents[2] / "configs"
    agent_name = "erasability_editability_gpt4o_mini_tool_use.yaml"
    (config_root / "agents" / agent_name).write_text(
        (repo_configs / "agents" / "testing" / agent_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    (config_root / "memory_systems" / "no_memory.yaml").write_text(
        "\n".join(
            [
                "id: no_memory",
                "backend_config:",
                "  type: no_memory",
                "  name: no_memory",
                "  top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "smoke_e2e"
        / "erasability_editability"
        / "smoke.json"
    )

    package_bench = (
        repo_configs / "benchmarks" / "testing" / "erasability_editability_smoke.yaml"
    )
    payload = yaml.safe_load(package_bench.read_text(encoding="utf-8"))
    payload["source"] = {"local_path": str(fixture)}
    payload["limit"] = 1
    (config_root / "benchmarks" / "erasability_editability_smoke.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    run_path = config_root / "runs" / "ee_campaign.yaml"
    run_path.write_text(
        "\n".join(
            [
                "run_name: ee_campaign",
                "experiments:",
                "  - name: gpt4o_mini_nomemory_ee",
                f"    agent_config_file: {agent_name}",
                "    memory_system_config_file: no_memory.yaml",
                "    benchmark_config_file: erasability_editability_smoke.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_path
