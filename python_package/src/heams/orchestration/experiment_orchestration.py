from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from heams.benchmarks.factory import create_benchmark
from heams.execution_traces import ExecutionTraceLedger
from heams.orchestration.orchestrator import Orchestrator
from heams.orchestration.experiment_config import ResolvedExperimentExecutionConfig
from heams.result_ledger import ResultLedger, expected_result_keys
from heams.scenarios.base import Scenario
from heams.scenarios.config import ScenarioExecutionConfig


_EXPERIMENT_CONFIG_ARTIFACT_PREFIX = "experiment_config"


def _experiment_config_artifact_path(
    experiment_dir: Path,
    run_id: str,
) -> Path:
    return experiment_dir / (f"{_EXPERIMENT_CONFIG_ARTIFACT_PREFIX}.{run_id}.yaml")


def _ensure_experiment_config_artifact(
    experiment_config: ResolvedExperimentExecutionConfig,
) -> Path:
    artifact_path = _experiment_config_artifact_path(
        experiment_config.experiment_dir,
        experiment_config.run_id,
    )
    if artifact_path.exists():
        return artifact_path

    artifact_path.write_text(
        yaml.safe_dump(
            experiment_config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return artifact_path


def _prepare_experiment_runtime_settings(
    experiment_config: ResolvedExperimentExecutionConfig,
) -> ExperimentRuntimeSettings:
    """Prepare runtime information for an experiment .

    Long term, this helper can materialize a variety of runtime artifacts,
    such as memory storage paths.
    """

    return ExperimentRuntimeSettings(
        memory_storage_path=experiment_config.experiment_dir / "memory.json",
        runs_jsonl_path=experiment_config.experiment_dir / "runs.jsonl",
        execution_traces_jsonl_path=(
            experiment_config.experiment_dir / "execution_traces.jsonl"
        ),
        run_status_path=experiment_config.experiment_dir / "run_status.json",
    )


class ExperimentRuntimeSettings(BaseModel):
    """Runtime-derived paths prepared after the experiment is materialized.

    This should hold only data that is derived at runtime, such
    as concrete artifact paths, and not re-state resolved execution policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    memory_storage_path: Path
    runs_jsonl_path: Path
    execution_traces_jsonl_path: Path
    run_status_path: Path


def _write_run_status(
    path: Path,
    *,
    success: bool,
    error: str | None = None,
    runnable_count: int = 0,
    skipped_count: int = 0,
) -> None:
    payload: dict[str, object] = {
        "success": success,
        "runnable_count": runnable_count,
        "skipped_count": skipped_count,
    }
    if error is not None:
        payload["error"] = error
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_experiment(
    experiment_config: ResolvedExperimentExecutionConfig,
) -> None:
    print(f"Executing experiment '{experiment_config.experiment_name}'")

    # SETUP
    # This block should stay focused on experiment-scoped preparation only:
    # create the run directory, write the resolved config artifact, and derive
    # any runtime-only paths that execution will need.
    #
    # Concerns that likely belong here long term:
    # - materialize stable experiment-local paths for memory, checkpoints, and
    #   per-experiment output artifacts
    # - write the pre-execution resolved config artifact before any runtime work
    # - perform only non-mutating compatibility checks that depend on the
    #   already-resolved experiment config
    # - avoid scenario-driven policy rewrites or any reopening of config
    #   precedence decisions
    experiment_config.experiment_dir.mkdir(parents=True, exist_ok=True)
    _ensure_experiment_config_artifact(experiment_config)
    runtime_settings = _prepare_experiment_runtime_settings(experiment_config)

    # EXECUTION
    # This block is the future home of the actual benchmark runtime.
    #
    # Concerns inherited from the old orchestrator/runtime that need to be
    # accounted for here eventually:
    # - load the typed benchmark workload/config once that contract exists
    # - decide execution mode for the workload (sync, async, stateful async)
    #   without mutating resolved config
    # - iterate any benchmark-owned units of work such as scenarios, samples,
    #   and stimuli in the order the benchmark declares
    # - perform scenario-local setup when needed: environment creation,
    #   evaluator creation, memory priming, tool assembly, and agent creation
    # - keep runtime-only derivations explicit, such as per-scenario memory
    #   paths or per-scenario event ledgers
    # - collect runtime outputs needed later for metrics, reporting, and status
    # - contain runtime failures at the experiment boundary and record enough
    #   information to debug what failed
    # - tear down runtime state deterministically so temporary memory files,
    #   environments, and closable agents do not leak across work units
    #
    # Things that should not come back here from the old design:
    # - Cartesian product expansion of agents/memory/evaluators
    # - scenario-driven mutation of resolved agent or memory policy
    # - hidden config precedence layers or tag-driven reinterpretation of the
    #   experiment config

    scenarios: list[Scenario] = []
    skipped_count = 0
    try:
        benchmark = create_benchmark(experiment_config.benchmark_config)
        skipped_count = len(benchmark.skipped)
        scenarios = [sample.to_scenario() for sample in benchmark]
        if not scenarios:
            _write_run_status(
                runtime_settings.run_status_path,
                success=False,
                error="No runnable scenarios after benchmark materialization.",
                skipped_count=skipped_count,
            )
            raise ValueError(
                f"Experiment '{experiment_config.experiment_name}' produced no "
                "runnable scenarios."
            )
    except Exception as exc:
        # OUTPUT (failure) — includes create_benchmark / materialization errors
        _write_run_status(
            runtime_settings.run_status_path,
            success=False,
            error=str(exc),
            runnable_count=len(scenarios),
            skipped_count=skipped_count,
        )
        raise

    scenario_ids = [scenario.id for scenario in scenarios]
    scenario_id_counts: dict[str, int] = {}
    for scenario_id in scenario_ids:
        scenario_id_counts[scenario_id] = scenario_id_counts.get(scenario_id, 0) + 1
    duplicates = sorted(
        scenario_id for scenario_id, count in scenario_id_counts.items() if count > 1
    )
    if duplicates:
        message = (
            f"Experiment '{experiment_config.experiment_name}' produced duplicate "
            f"scenario IDs: {', '.join(duplicates)}."
        )
        _write_run_status(
            runtime_settings.run_status_path,
            success=False,
            error=message,
            runnable_count=len(scenarios),
            skipped_count=skipped_count,
        )
        raise ValueError(message)

    ledger = ResultLedger(runtime_settings.runs_jsonl_path)
    trace_ledger = ExecutionTraceLedger(runtime_settings.execution_traces_jsonl_path)
    expected_keys = expected_result_keys(scenario_ids)
    unexpected_keys = ledger.completed_keys - expected_keys
    if unexpected_keys:
        message = (
            f"Result ledger {runtime_settings.runs_jsonl_path} contains keys outside "
            f"the materialized benchmark: {sorted(unexpected_keys)!r}."
        )
        _write_run_status(
            runtime_settings.run_status_path,
            success=False,
            error=message,
            runnable_count=len(scenarios),
            skipped_count=skipped_count,
        )
        ledger.close()
        trace_ledger.close()
        raise ValueError(message)
    pending_keys = expected_keys - ledger.completed_keys
    if not pending_keys:
        ledger.close()
        trace_ledger.close()
        print(f"Experiment '{experiment_config.experiment_name}' is complete.")
        return

    try:
        with ledger, trace_ledger:
            orchestrator = Orchestrator(
                ScenarioExecutionConfig(
                    scenarios=scenarios,
                    workspace_dir=experiment_config.experiment_dir / "workspace",
                    agent_config=experiment_config.agent_config,
                    memory_system_config=experiment_config.memory_system_config,
                    pending_keys=pending_keys,
                )
            )
            orchestrator.run(
                on_run_complete=ledger.append,
                on_execution_trace_complete=trace_ledger.append,
                execution_trace_run_id=experiment_config.run_id,
            )
    except Exception as exc:
        # OUTPUT (failure) — includes runtime setup and execution errors.
        _write_run_status(
            runtime_settings.run_status_path,
            success=False,
            error=str(exc),
            runnable_count=len(scenarios),
            skipped_count=skipped_count,
        )
        raise

    # OUTPUT
    # This block should persist experiment-scoped artifacts only.
    #
    # Concerns that likely belong here long term:
    # - write run_status.json with success/failure and any runtime error details
    # - write stable per-experiment outputs produced by benchmark execution
    # - write any checkpoint or trace artifacts that are part of the experiment
    #   contract, not ad hoc console behavior
    # - keep aggregate reporting and cross-experiment summaries out of this
    #   function; those belong to rollup/reporting over saved artifacts
    _write_run_status(
        runtime_settings.run_status_path,
        success=True,
        runnable_count=len(scenarios),
        skipped_count=skipped_count,
    )
