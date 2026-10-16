"""Tests for `ResolvedExperimentExecutionConfig`'s cross-config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from heams.agents.config import AgentParameters
from heams.benchmarks.evomemory.config import EvoMemorySingleTurnBenchmarkConfig
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.memory_systems.factory import MemorySystemVariant
from heams.memory_systems.in_context import InContextConfig
from heams.memory_systems.privacy import PrivacyConfig
from heams.orchestration.experiment_config import ResolvedExperimentExecutionConfig

AGENT_CONFIG = AgentParameters(model_id="openai:gpt-4o-mini", instructions="test")


def _resolved_config(
    *, benchmark_config, privacy_config: PrivacyConfig | None
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
        memory_system_config=MemorySystemVariant(
            backend_config=InContextConfig(), privacy_config=privacy_config
        ),
        benchmark_config=benchmark_config,
    )


def test_retention_policy_with_active_recall_policy_is_rejected():
    privacy = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": True, "mode": "deterministic"},
            "rules": [{"rule_id": "r1", "content_substrings": ["x"]}],
        }
    )

    with pytest.raises(ValueError, match="recall.enabled: false"):
        _resolved_config(
            benchmark_config=RetentionPolicyBenchmarkConfig(),
            privacy_config=privacy,
        )


def test_retention_policy_with_recall_disabled_is_allowed():
    privacy = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": False},
            "rules": [{"rule_id": "r1", "content_substrings": ["x"]}],
        }
    )

    resolved = _resolved_config(
        benchmark_config=RetentionPolicyBenchmarkConfig(),
        privacy_config=privacy,
    )
    assert resolved.benchmark_config.type == "retention_policy"


def test_retention_policy_with_ruleless_recall_enabled_is_allowed():
    """`recall.enabled: true` that resolves to a no-op must not be rejected."""
    privacy = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": True, "mode": "deterministic"},
            "rules": [],
        }
    )

    _resolved_config(
        benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
    )


def test_retention_policy_with_no_privacy_config_is_rejected():
    """An unwrapped backend can't take the policy each sample sets."""
    with pytest.raises(ValueError, match="needs an active privacy_config"):
        _resolved_config(
            benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=None
        )


def test_retention_policy_with_inactive_privacy_config_is_rejected():
    privacy = PrivacyConfig.model_validate(
        {"retention": {"enabled": False}, "recall": {"enabled": False}}
    )

    with pytest.raises(ValueError, match="needs an active privacy_config"):
        _resolved_config(
            benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
        )


def test_retention_policy_with_recall_activated_by_policy_rules_is_rejected():
    """A recall layer that is rule-less in YAML turns on once policies are set."""
    privacy = PrivacyConfig.model_validate(
        {
            "retention": {
                "mode": "deterministic",
                "policy_rules": {"p": [{"rule_id": "r1", "content_substrings": ["x"]}]},
            },
            "recall": {"enabled": True, "mode": "deterministic"},
        }
    )

    with pytest.raises(ValueError, match="recall.enabled: false"):
        _resolved_config(
            benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
        )


def test_retention_policy_llm_mode_policy_rules_do_not_activate_recall():
    """llm mode never appends `policy_rules`, so they can't switch recall on."""
    privacy = PrivacyConfig.model_validate(
        {
            "retention": {
                "mode": "llm",
                "policy_rules": {"p": [{"rule_id": "r1", "content_substrings": ["x"]}]},
            },
            "recall": {"enabled": True, "mode": "deterministic"},
        }
    )

    _resolved_config(
        benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
    )


def test_retention_policy_with_policy_rules_missing_a_policy_is_rejected():
    privacy = PrivacyConfig.model_validate(
        {
            "retention": {
                "mode": "deterministic",
                "policy_rules": {
                    "Do not retain the user's health information.": [
                        {"rule_id": "r1", "content_substrings": ["health"]}
                    ]
                },
            },
            "recall": {"enabled": False},
        }
    )

    with pytest.raises(ValueError, match="employment information"):
        _resolved_config(
            benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
        )

    # Narrowed to the one policy the table covers, the same config is fine.
    _resolved_config(
        benchmark_config=RetentionPolicyBenchmarkConfig(policy_ids=["RP-01"]),
        privacy_config=privacy,
    )


def test_retention_policy_skips_policy_rules_coverage_in_llm_mode():
    """llm mode reads the statement itself, so a partial table is irrelevant."""
    privacy = PrivacyConfig.model_validate(
        {
            "retention": {
                "mode": "llm",
                "policy_rules": {"p": [{"rule_id": "r1", "content_substrings": ["x"]}]},
            },
            "recall": {"enabled": False},
        }
    )

    _resolved_config(
        benchmark_config=RetentionPolicyBenchmarkConfig(), privacy_config=privacy
    )


def test_other_benchmark_types_ignore_the_recall_precondition():
    """The check is scoped to retention_policy; it must not affect other benchmarks."""
    privacy = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": True, "mode": "deterministic"},
            "rules": [{"rule_id": "r1", "content_substrings": ["x"]}],
        }
    )

    _resolved_config(
        benchmark_config=EvoMemorySingleTurnBenchmarkConfig(), privacy_config=privacy
    )
