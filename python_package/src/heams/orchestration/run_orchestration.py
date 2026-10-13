from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from heams.agents.factory import AgentConfig
from heams.benchmarks.factory import BenchmarkConfig
from heams.memory_systems.factory import MemorySystemVariant
from heams.orchestration.experiment_config import (
    ResolvedExperimentExecutionConfig,
)
from heams.orchestration.experiment_orchestration import (
    _ensure_experiment_config_artifact,
    _experiment_config_artifact_path,
    execute_experiment,
)
from heams.orchestration.run_config import (
    ExperimentConfig,
    ResolvedRunConfig,
    resolve_run_config,
)
from heams.result_ledger import ResultLedger

_AGENT_CONFIG_ADAPTER = TypeAdapter(AgentConfig)
_MEMORY_SYSTEM_CONFIG_ADAPTER = TypeAdapter(MemorySystemVariant)
_BENCHMARK_CONFIG_ADAPTER = TypeAdapter(BenchmarkConfig)

_RUN_CONFIG_ARTIFACT_PREFIX = "run_config"


class RunDirectoryError(ValueError):
    """Raised when an exact output directory cannot be trusted or resumed."""


@dataclass(frozen=True)
class PreparedRun:
    """Resolved run data and experiment snapshots ready for execution."""

    run_config: ResolvedRunConfig
    experiment_configs: tuple[ResolvedExperimentExecutionConfig, ...]
    resumed: bool
    overwritten: bool = False


def _load_agent_config(path: Path) -> AgentConfig:
    payload = _load_yaml_mapping(path)
    return _AGENT_CONFIG_ADAPTER.validate_python(payload)


def _load_memory_system_config(path: Path) -> MemorySystemVariant:
    payload = _load_yaml_mapping(path)
    return _MEMORY_SYSTEM_CONFIG_ADAPTER.validate_python(payload)


def _load_benchmark_config(path: Path) -> BenchmarkConfig:
    payload = _load_yaml_mapping(path)
    return _BENCHMARK_CONFIG_ADAPTER.validate_python(payload)


def build_experiment_execution_configs(
    run_config: ResolvedRunConfig,
) -> tuple[ResolvedExperimentExecutionConfig, ...]:
    """Resolve supported component configs into per-experiment execution data."""

    return tuple(
        resolve_experiment_execution_config(run_config, experiment)
        for experiment in run_config.experiments
    )


def load_run_snapshot(path: Path) -> ResolvedRunConfig:
    """Load a previously resolved run without consulting source YAML."""
    payload = _load_yaml_mapping(path)
    try:
        return ResolvedRunConfig.model_validate(payload)
    except ValueError as exc:
        raise RunDirectoryError(f"Malformed run snapshot {path}: {exc}") from exc


def load_experiment_snapshot(path: Path) -> ResolvedExperimentExecutionConfig:
    """Load a previously resolved experiment without resolving component YAML."""
    payload = _load_yaml_mapping(path)
    try:
        return ResolvedExperimentExecutionConfig.model_validate(payload)
    except ValueError as exc:
        raise RunDirectoryError(f"Malformed experiment snapshot {path}: {exc}") from exc


def resolve_experiment_execution_config(
    run_config: ResolvedRunConfig,
    experiment: ExperimentConfig,
) -> ResolvedExperimentExecutionConfig:
    return ResolvedExperimentExecutionConfig(
        run_id=run_config.run_id,
        run_name=run_config.run_name,
        experiment_name=experiment.name,
        experiment_dir=run_config.results_dir / experiment.name,
        agent_config_file=experiment.agent_config_file.file_name,
        memory_system_config_file=experiment.memory_system_config_file.file_name,
        benchmark_config_file=experiment.benchmark_config_file.file_name,
        agent_config=_load_agent_config(experiment.agent_config_file.path),
        memory_system_config=_load_memory_system_config(
            experiment.memory_system_config_file.path
        ),
        benchmark_config=_load_benchmark_config(experiment.benchmark_config_file.path),
    )


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunDirectoryError(f"Cannot read snapshot {path}: {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise RunDirectoryError(f"Snapshot at {path} must be a YAML mapping/object.")
    return payload


def _write_run_config_artifact(run_config: ResolvedRunConfig) -> Path:
    artifact_path = (
        run_config.results_dir
        / f"{_RUN_CONFIG_ARTIFACT_PREFIX}.{run_config.run_id}.yaml"
    )
    artifact_path.write_text(
        yaml.safe_dump(
            run_config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return artifact_path


def _run_snapshot_path(run_config: ResolvedRunConfig) -> Path:
    return (
        run_config.results_dir
        / f"{_RUN_CONFIG_ARTIFACT_PREFIX}.{run_config.run_id}.yaml"
    )


def _require_directory(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RunDirectoryError(f"Expected {description} directory at {path}.")


def _load_existing_run(
    output_dir: Path,
) -> PreparedRun | None:
    if output_dir.is_symlink():
        raise RunDirectoryError(
            f"Output target {output_dir} must not be a symbolic link."
        )
    if not output_dir.exists():
        return None
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RunDirectoryError(
            f"Output target {output_dir} is not an exact HEAMS run directory."
        )
    if not any(output_dir.iterdir()):
        return None

    run_snapshots = sorted(output_dir.glob(f"{_RUN_CONFIG_ARTIFACT_PREFIX}.*.yaml"))
    if len(run_snapshots) != 1:
        raise RunDirectoryError(
            f"Expected exactly one HEAMS run snapshot in {output_dir}; "
            f"found {len(run_snapshots)}."
        )
    run_snapshot_path = run_snapshots[0]
    if run_snapshot_path.is_symlink():
        raise RunDirectoryError(f"Run snapshot {run_snapshot_path} must not be a link.")
    run_config = load_run_snapshot(run_snapshot_path)
    experiment_names = [experiment.name for experiment in run_config.experiments]
    if len(experiment_names) != len(set(experiment_names)):
        raise RunDirectoryError(
            f"Run snapshot {run_snapshot_path} contains duplicate experiment names."
        )
    expected_run_snapshot = _run_snapshot_path(run_config)
    if run_snapshot_path != expected_run_snapshot:
        raise RunDirectoryError(
            f"Run snapshot {run_snapshot_path} does not match run ID "
            f"{run_config.run_id!r}."
        )
    if run_config.results_dir.resolve(strict=False) != output_dir.resolve(strict=False):
        raise RunDirectoryError(
            f"Run snapshot results_dir does not match exact output directory "
            f"{output_dir}."
        )

    experiment_configs: list[ResolvedExperimentExecutionConfig] = []
    for experiment in run_config.experiments:
        experiment_dir = output_dir / experiment.name
        _require_directory(experiment_dir, f"experiment {experiment.name!r}")
        snapshots = sorted(experiment_dir.glob("experiment_config.*.yaml"))
        if len(snapshots) != 1:
            raise RunDirectoryError(
                f"Expected exactly one experiment snapshot in {experiment_dir}; "
                f"found {len(snapshots)}."
            )
        snapshot_path = snapshots[0]
        if snapshot_path.is_symlink():
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} must not be a link."
            )
        expected_snapshot = _experiment_config_artifact_path(
            experiment_dir,
            run_config.run_id,
        )
        if snapshot_path != expected_snapshot:
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} does not match run ID "
                f"{run_config.run_id!r}."
            )
        experiment_config = load_experiment_snapshot(snapshot_path)
        if experiment_config.run_id != run_config.run_id:
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} has a mismatched run ID."
            )
        if experiment_config.run_name != run_config.run_name:
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} has a mismatched run name."
            )
        if experiment_config.experiment_name != experiment.name:
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} has a mismatched experiment name."
            )
        if (
            str(experiment_config.agent_config_file)
            != experiment.agent_config_file.file_name
            or str(experiment_config.memory_system_config_file)
            != experiment.memory_system_config_file.file_name
            or str(experiment_config.benchmark_config_file)
            != experiment.benchmark_config_file.file_name
        ):
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} does not match its run entry."
            )
        if experiment_config.experiment_dir.resolve(
            strict=False
        ) != experiment_dir.resolve(strict=False):
            raise RunDirectoryError(
                f"Experiment snapshot {snapshot_path} has a mismatched directory."
            )

        ledger_path = experiment_dir / "runs.jsonl"
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise RunDirectoryError(f"Missing result ledger at {ledger_path}.")
        try:
            ResultLedger(ledger_path).close()
        except ValueError as exc:
            raise RunDirectoryError(
                f"Malformed result ledger {ledger_path}: {exc}"
            ) from exc
        experiment_configs.append(experiment_config)

    return PreparedRun(run_config, tuple(experiment_configs), resumed=True)


def _initialize_run(
    config_path: Path,
    output_dir: Path,
    *,
    overwritten: bool = False,
) -> PreparedRun:
    run_config = resolve_run_config(config_path, output_dir=output_dir)
    experiment_configs = build_experiment_execution_configs(run_config)
    run_config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config_artifact(run_config)
    for experiment_config in experiment_configs:
        experiment_config.experiment_dir.mkdir(parents=True, exist_ok=False)
        _ensure_experiment_config_artifact(experiment_config)
        (experiment_config.experiment_dir / "runs.jsonl").touch()
    return PreparedRun(
        run_config,
        experiment_configs,
        resumed=False,
        overwritten=overwritten,
    )


def prepare_run(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> PreparedRun:
    """Classify an exact output directory and prepare it for execution."""
    exact_output_dir = output_dir.resolve(strict=False)
    existing_run = _load_existing_run(exact_output_dir)
    if existing_run is None:
        return _initialize_run(config_path, exact_output_dir)
    if not overwrite:
        return existing_run

    shutil.rmtree(exact_output_dir)
    return _initialize_run(config_path, exact_output_dir, overwritten=True)


def _ensure_run_artifacts(
    run_config: ResolvedRunConfig,
    experiment_configs: tuple[ResolvedExperimentExecutionConfig, ...],
) -> None:
    """Keep the direct orchestration API usable outside the CLI preparation path."""
    run_config.results_dir.mkdir(parents=True, exist_ok=True)
    run_snapshot_path = _run_snapshot_path(run_config)
    if not run_snapshot_path.exists():
        _write_run_config_artifact(run_config)
    for experiment_config in experiment_configs:
        experiment_config.experiment_dir.mkdir(parents=True, exist_ok=True)
        _ensure_experiment_config_artifact(experiment_config)
        (experiment_config.experiment_dir / "runs.jsonl").touch(exist_ok=True)


def execute_run(
    run_config: ResolvedRunConfig,
    experiment_configs: tuple[ResolvedExperimentExecutionConfig, ...] | None = None,
) -> None:
    """Execute a run"""
    if experiment_configs is None:
        existing_run = _load_existing_run(run_config.results_dir)
        if existing_run is not None:
            experiment_execution_configs = existing_run.experiment_configs
        else:
            experiment_execution_configs = build_experiment_execution_configs(
                run_config
            )
    else:
        experiment_execution_configs = experiment_configs
    _ensure_run_artifacts(run_config, experiment_execution_configs)

    print(f"Executing run '{run_config.run_id}'")
    for experiment_execution_config in experiment_execution_configs:
        execute_experiment(experiment_execution_config)
