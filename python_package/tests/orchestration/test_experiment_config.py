"""Tests for `ResolvedExperimentExecutionConfig`'s cross-config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from heams.agents.config import AgentParameters
from heams.benchmarks.evomemory.config import EvoMemorySingleTurnBenchmarkConfig
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.memory_systems.factory import MemorySystemVariant
from heams.memory_systems.in_context import InContextConfig
from heams.memory_systems.plugins import (
    BaseMemoryWrapperConfig,
    MemoryPluginCatalog,
    MemoryWrapperPlugin,
    _builtin_plugins,
    set_memory_plugin_catalog,
)
from heams.orchestration.experiment_config import ResolvedExperimentExecutionConfig

AGENT_CONFIG = AgentParameters(model_id="openai:gpt-4o-mini", instructions="test")
SMOKE_POLICY = "Do not retain the user's health information."


class _CheckedConfig(InContextConfig):
    """A backend config recording, or rejecting, the policies it is asked about."""

    reject: bool = False
    seen: list[list[str]] = []

    def check_retention_policies(self, policies: list[str]) -> None:
        self.seen.append(policies)
        if self.reject:
            raise ValueError("this memory system can't enforce these policies")


class _CheckedWrapperConfig(BaseMemoryWrapperConfig):
    reject: bool = False

    def check_retention_policies(self, policies: list[str]) -> None:
        if self.reject:
            raise ValueError("this wrapper can't enforce these policies")


@pytest.fixture()
def wrapper_catalog():
    set_memory_plugin_catalog(
        MemoryPluginCatalog(
            _builtin_plugins(),
            [
                MemoryWrapperPlugin(
                    config_key="checked_wrapper",
                    config_model=_CheckedWrapperConfig,
                    wrap=lambda memory, config: memory,
                )
            ],
        )
    )
    yield
    set_memory_plugin_catalog(None)


def _resolved_config(
    *, benchmark_config, memory_system_config: MemorySystemVariant
) -> ResolvedExperimentExecutionConfig:
    return ResolvedExperimentExecutionConfig(
        run_id="run-1",
        run_name="run-1",
        experiment_name="exp-1",
        experiment_dir=Path("/tmp/exp-1"),
        agent_config_file=Path("agent.yaml"),
        memory_system_config_file=Path("memory.yaml"),
        benchmark_config_file=Path("benchmark.yaml"),
        agent_config=AGENT_CONFIG,
        memory_system_config=memory_system_config,
        benchmark_config=benchmark_config,
    )


def _smoke_benchmark() -> RetentionPolicyBenchmarkConfig:
    return RetentionPolicyBenchmarkConfig(policy_ids=["RP-01"])


def test_retention_policy_accepts_an_ungoverned_memory_system():
    """Without governance the system runs, and is scored, ungoverned."""
    _resolved_config(
        benchmark_config=_smoke_benchmark(),
        memory_system_config=MemorySystemVariant(backend_config=InContextConfig()),
    )


def test_retention_policy_asks_the_backend_about_the_benchmark_policies():
    config = _CheckedConfig(seen=[])

    _resolved_config(
        benchmark_config=_smoke_benchmark(),
        memory_system_config=MemorySystemVariant(backend_config=config),
    )

    assert config.seen == [[SMOKE_POLICY]]


def test_retention_policy_rejects_a_backend_that_refuses_the_policies():
    with pytest.raises(
        ValueError, match=r"Experiment 'exp-1' \(memory.yaml\): this memory system"
    ):
        _resolved_config(
            benchmark_config=_smoke_benchmark(),
            memory_system_config=MemorySystemVariant(
                backend_config=_CheckedConfig(reject=True, seen=[])
            ),
        )


@pytest.mark.usefixtures("wrapper_catalog")
def test_retention_policy_rejects_a_wrapper_that_refuses_the_policies():
    variant = MemorySystemVariant.model_validate(
        {
            "backend_config": {"type": "in_context"},
            "checked_wrapper": {"reject": True},
        }
    )

    with pytest.raises(ValueError, match="this wrapper can't enforce"):
        _resolved_config(
            benchmark_config=_smoke_benchmark(), memory_system_config=variant
        )


@pytest.mark.usefixtures("wrapper_catalog")
def test_other_benchmark_types_skip_the_retention_check():
    """The check is scoped to retention_policy; it must not affect other benchmarks."""
    variant = MemorySystemVariant.model_validate(
        {
            "backend_config": {"type": "in_context"},
            "checked_wrapper": {"reject": True},
        }
    )

    _resolved_config(
        benchmark_config=EvoMemorySingleTurnBenchmarkConfig(),
        memory_system_config=variant,
    )
