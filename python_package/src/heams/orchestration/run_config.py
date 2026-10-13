from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heams.utils.settings import settings


def resolve_run_config(
    run_config_path: Path,
    output_dir: Path | None = None,
) -> ResolvedRunConfig:
    source_config_path = run_config_path.resolve(strict=False)
    run_config = RunConfig.load(source_config_path)
    executed_at = datetime.now(timezone.utc)
    run_id = _build_run_id(executed_at)
    results_dir = (
        output_dir or settings.project_dir / settings.default_results_dir
    ).resolve(strict=False)

    return ResolvedRunConfig(
        run_id=run_id,
        executed_at=executed_at,
        source_config_path=source_config_path,
        run_name=run_config.run_name,
        results_dir=results_dir,
        experiments=run_config.experiments,
    )


class ResolvedRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(..., min_length=1)
    executed_at: datetime
    source_config_path: Path

    run_name: str
    results_dir: Path
    experiments: tuple[ExperimentConfig, ...] = Field(..., min_length=1)


class RunConfig(BaseModel):
    """Run configuration without a results directory in its YAML schema.

    The exact results directory is supplied by :func:`resolve_run_config` through
    the CLI or library API.
    """

    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(..., min_length=1)
    experiments: list[ExperimentConfig] = Field(..., min_length=1)

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        runs_dir = settings.config_dir.resolve() / "runs"
        try:
            path.resolve(strict=False).relative_to(runs_dir.resolve(strict=False))
        except ValueError as exc:
            raise ValueError(
                f"Run configs must live under {runs_dir}; got {path}."
            ) from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping/object.")
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def _experiment_names_are_unique(self) -> "RunConfig":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for experiment in self.experiments:
            if experiment.name in seen:
                duplicates.add(experiment.name)
            seen.add(experiment.name)
        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(
                f"Duplicate experiment names are not allowed: {duplicate_list}."
            )
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    agent_config_file: AgentComponentRef
    memory_system_config_file: MemorySystemComponentRef
    benchmark_config_file: BenchmarkComponentRef

    @field_validator("name")
    @classmethod
    def _name_is_safe_directory_component(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or path.name != value or value in {".", ".."}:
            raise ValueError("Experiment names must be a single directory name.")
        return value


class ComponentConfigRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subdirectory: ClassVar[str] = ""
    # Subclasses enable this for component references that may include subdirectories.
    allow_nested_paths: ClassVar[bool] = False
    file_name: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_validate_input(cls, value: object) -> object:
        if isinstance(value, str):
            value = {"file_name": value}
        if not isinstance(value, dict):
            return value

        file_name = value.get("file_name")
        if isinstance(file_name, str):
            path = Path(file_name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or (not cls.allow_nested_paths and path.name != file_name)
            ):
                raise ValueError(
                    "Referenced config entries must be file names only, not paths."
                )
        return value

    @model_validator(mode="after")
    def _validate_component_file(self) -> "ComponentConfigRef":
        path = self.absolute_path()
        component_dir = (settings.config_dir / self.subdirectory).resolve(strict=False)
        try:
            path.relative_to(component_dir)
        except ValueError as exc:
            raise ValueError(
                f"Referenced config file must live under {component_dir}: {path}"
            ) from exc
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Referenced config file must be a YAML file: {path}")
        if not path.is_file():
            raise ValueError(f"Referenced config file does not exist: {path}")
        return self

    def absolute_path(self) -> Path:
        return (settings.config_dir / self.subdirectory / self.file_name).resolve(
            strict=False
        )

    @property
    def path(self) -> Path:
        return self.absolute_path()


class AgentComponentRef(ComponentConfigRef):
    subdirectory: ClassVar[str] = "agents"
    allow_nested_paths: ClassVar[bool] = True


class MemorySystemComponentRef(ComponentConfigRef):
    subdirectory: ClassVar[str] = "memory_systems"
    allow_nested_paths: ClassVar[bool] = True


class BenchmarkComponentRef(ComponentConfigRef):
    subdirectory: ClassVar[str] = "benchmarks"
    allow_nested_paths: ClassVar[bool] = True


def _build_run_id(executed_at: datetime) -> str:
    return f"{executed_at.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
