"""The Retention policy benchmark run through the privacy wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from heams.benchmarks.factory import create_benchmark
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.retention_policy.metrics import compute_metrics
from heams.evaluation.factory import create_evaluator
from heams.evaluation.schemas import AuditRecord
from heams.memory_systems.factory import create_memory
from heams.memory_systems.in_context import InContextConfig
from heams.orchestration.scenario_execution import execute_scenario
from heams_memory_privacy_wrapper import PrivacyConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "python_package"
CONFIGS = PACKAGE_ROOT / "configs"


class _AgentStub:
    """Holds a memory system; Retention policy scenarios never call the agent."""

    def __init__(self, memory_system):
        self.memory_system = memory_system


def _benchmark(name: str):
    path = CONFIGS / "benchmarks" / f"{name}.yaml"
    # The relative dataset path resolves against the heams package root.
    return create_benchmark(
        RetentionPolicyBenchmarkConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    )


def _run(sample, privacy_config: PrivacyConfig, tmp_path: Path) -> AuditRecord:
    memory = create_memory(
        InContextConfig(),
        storage_dir=tmp_path,
        wrapper_configs={"privacy_config": privacy_config},
    )
    ledger: list = []
    memory.attach_event_sink(ledger)
    scenario = sample.to_scenario()
    eval_cfg = scenario.evaluators[0]
    audit, _, _ = execute_scenario(
        scenario=scenario,
        agent=_AgentStub(memory),
        environment=None,
        evaluator=create_evaluator(eval_cfg, task_name=scenario.name),
        eval_cfg=eval_cfg,
        build_ground_truth=lambda s, c: c.to_ground_truth(s),
        memory_ledger=ledger,
    )
    return audit


def _shipped_privacy_config() -> PrivacyConfig:
    doc = yaml.safe_load(
        (CONFIGS / "memory_systems" / "milamem_retention_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    return PrivacyConfig.model_validate(doc["privacy_config"])


def test_a_privacy_config_with_no_rules_stores_everything(tmp_path):
    """Governed but forbidding nothing: leak 100%, allowed 100%."""
    sample = _benchmark("retention_policy_smoke").samples[0]
    permissive = PrivacyConfig.model_validate(
        {
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": False},
            "rules": [],
        }
    )

    audit = _run(sample, permissive, tmp_path)

    metrics = compute_metrics(
        [
            {
                "observed": record.observed,
                "expected": record.expected,
                "policy_id": record.tags["policy_id"],
                "category": record.tags.get("category", "?"),
            }
            for record in audit.report_records
        ]
    )
    assert metrics.leak_rate == 1.0
    assert metrics.allowed_rate == 1.0
    assert "Policy not enforced" not in audit.audit_notes


def test_a_policy_missing_from_policy_rules_fails_the_scenario(tmp_path):
    sample = _benchmark("retention_policy_smoke").samples[0]
    privacy = PrivacyConfig.model_validate(
        {
            "retention": {
                "mode": "deterministic",
                "policy_rules": {
                    "some other policy": [
                        {"rule_id": "r1", "content_substrings": ["x"]}
                    ]
                },
            },
            "recall": {"enabled": False},
        }
    )

    with pytest.raises(ValueError, match="no rules for"):
        _run(sample, privacy, tmp_path)


def test_the_shipped_config_passes_the_pre_run_checks_for_every_policy():
    benchmark = _benchmark("retention_policy")
    policies = sorted({sample.ground_truth.policy for sample in benchmark.samples})

    _shipped_privacy_config().check_retention_policies(policies)

    assert len(policies) == 12


def test_the_shipped_config_drops_each_samples_own_category(tmp_path):
    """One config serves every policy: each sample's policy reaches the layer."""
    privacy = _shipped_privacy_config()

    for index, sample in enumerate(_benchmark("retention_policy").samples):
        audit = _run(sample, privacy, tmp_path / str(index))
        dropped = {
            record.tags["category"]
            for record in audit.report_records
            if record.observed == "drop"
        }
        prohibited = {
            fact.category
            for fact in sample.ground_truth.facts
            if fact.expected == "drop"
        }
        # Keyword rules leak some prohibited facts, so assert the sample's policy
        # reached the layer (its category is blocked), not a perfect score.
        assert prohibited <= dropped, sample.id
