from __future__ import annotations

from pathlib import Path

import pytest

from heams.benchmarks.factory import create_benchmark
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.retention_policy.metrics import compute_metrics
from heams.evaluation.factory import create_evaluator
from heams.memory_systems.privacy import PrivacyConfig
from heams.orchestration.scenario_execution import execute_scenario
from tests.benchmarks.retention_policy.fixtures import (
    _AgentStub,
    _load_config,
    _memory,
    _run,
)


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    import json

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _row(row_id: str, policy_id: str, **overrides) -> dict:
    row = {
        "id": row_id,
        "policy_id": policy_id,
        "policy": f"policy for {policy_id}",
        "text": f"fact {row_id}",
        "category": "misc",
        "expected": "store",
    }
    row.update(overrides)
    return row


def test_smoke_config_scopes_to_one_policy():
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    assert len(benchmark.samples) == 1
    sample = benchmark.samples[0]
    assert sample.tags[1] == "RP-01"
    assert {f.expected for f in sample.ground_truth.facts} == {"store", "drop"}


def test_full_config_covers_every_policy():
    benchmark = create_benchmark(_load_config("retention_policy"))
    smoke = create_benchmark(_load_config("retention_policy_smoke"))
    assert len(benchmark.samples) == 12
    assert len(benchmark.samples) > len(smoke.samples)
    assert benchmark.skipped == []


def test_one_bad_row_only_skips_that_row_not_its_whole_policy(tmp_path):
    """A malformed row must cost itself, not every valid fact under its policy."""
    rows = [
        _row("r1", "P1"),
        _row("r2", "P1", text=""),  # missing required field -> invalid
        _row("r3", "P1"),
    ]
    config = RetentionPolicyBenchmarkConfig(dataset_path=_write_dataset(tmp_path, rows))
    benchmark = create_benchmark(config)

    assert len(benchmark.skipped) == 1
    assert benchmark.skipped[0].path.name == "r2"
    assert len(benchmark.samples) == 1
    facts = benchmark.samples[0].ground_truth.facts
    assert {f.row_id for f in facts} == {"r1", "r3"}


def test_row_missing_policy_id_key_entirely_is_skipped_not_a_crash(tmp_path):
    """A row missing `policy_id` outright must not raise a bare KeyError."""
    good = _row("r1", "P1")
    broken = _row("r2", "P1")
    del broken["policy_id"]
    config = RetentionPolicyBenchmarkConfig(
        dataset_path=_write_dataset(tmp_path, [good, broken])
    )
    benchmark = create_benchmark(config)

    assert len(benchmark.skipped) == 1
    assert benchmark.skipped[0].path.name == "r2"
    assert len(benchmark.samples) == 1
    assert {f.row_id for f in benchmark.samples[0].ground_truth.facts} == {"r1"}


def test_limit_caps_policies_not_rows(tmp_path):
    """`limit` must never truncate a policy's fact set mid-way."""
    rows = [
        _row("p1-a", "P1"),
        _row("p1-b", "P1", expected="drop"),
        _row("p2-a", "P2"),
        _row("p2-b", "P2", expected="drop"),
    ]
    config = RetentionPolicyBenchmarkConfig(
        dataset_path=_write_dataset(tmp_path, rows), limit=1
    )
    benchmark = create_benchmark(config)

    assert len(benchmark.samples) == 1
    facts = benchmark.samples[0].ground_truth.facts
    assert len(facts) == 2
    assert {f.expected for f in facts} == {"store", "drop"}


def test_policy_with_no_rules_stores_everything(tmp_path):
    """A policy that forbids nothing stores and recalls everything: leak 100%, allowed 100%.

    Expressed as an empty rule list rather than `privacy_config=None`. A null
    config leaves the backend unwrapped, so there is no governance layer to ask;
    an empty one still builds the wrapper and resolves to a no-op policy, which
    is the ungoverned baseline this benchmark wants to measure.
    """
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    permissive = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": False},
            "rules": [],
        }
    )
    audit = _run(sample, _memory(tmp_path, permissive))

    decisions = [
        {
            "observed": record.observed,
            "expected": record.expected,
            "policy_id": record.tags["policy_id"],
            "category": record.tags.get("category", "?"),
        }
        for record in audit.report_records
    ]
    metrics = compute_metrics(decisions)
    assert metrics.total == len(sample.ground_truth.facts)
    assert metrics.leak_rate == 1.0
    assert metrics.allowed_rate == 1.0


def test_deterministic_rule_drops_matching_chunk(tmp_path):
    """A rule that matches drops the payload; the evaluator scores it correct."""
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    target = next(f for f in sample.ground_truth.facts if f.expected == "drop")
    substring = target.text.split()[0]

    privacy = PrivacyConfig.model_validate(
        {
            "enabled": True,
            "retention": {"enabled": True, "mode": "deterministic"},
            "recall": {"enabled": False},
            "rules": [{"rule_id": "r1", "content_substrings": [substring]}],
        }
    )
    audit = _run(sample, _memory(tmp_path, privacy))

    record = next(r for r in audit.report_records if r.key == target.row_id)
    assert record.passed is True
    assert record.observed == "drop"


def test_scenarios_set_policy_then_retain_then_recall_every_fact():
    """Set the policy, write every fact, then read each back."""
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    total = len(sample.ground_truth.facts)
    assert len(sample.execution_sequence) == 1 + 2 * total
    commands = [step.command for step in sample.execution_sequence]
    assert commands[0] == "set_retention_policy"
    assert commands[1 : 1 + total] == ["retain"] * total
    assert commands[1 + total :] == ["recall"] * total


def test_each_sample_sets_its_own_policy_statement_not_its_id():
    """The system gets the user's words; the id and categories stay ground truth."""
    benchmark = create_benchmark(_load_config("retention_policy"))
    for sample in benchmark.samples:
        step = sample.execution_sequence[0]
        assert step.parameters == {"policies": [sample.ground_truth.policy]}
        assert sample.ground_truth.policy_id not in step.parameters["policies"][0]


def test_category_is_not_leaked_into_retain_metadata():
    """Passing the category would reduce the task to a metadata lookup."""
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    total = len(sample.ground_truth.facts)
    for step in sample.execution_sequence[1 : 1 + total]:
        assert "category" not in step.parameters["metadata"]


def test_unknown_policy_id_is_rejected():
    config = _load_config("retention_policy_smoke").model_copy(
        update={"policy_ids": ["RP-99"]}
    )
    with pytest.raises(ValueError, match="RP-99"):
        create_benchmark(config)


def test_memory_manipulation_rejects_commands_outside_the_allowlist(tmp_path):
    """A scenario must not be able to drive arbitrary memory methods by name.

    Mirrors `BaseEnvironment.dispatch_command`, whose contract is a narrow
    allowlist. Without this, naming `delete` or `consolidate` in a scenario file
    would execute it.
    """
    from heams.scenarios.actions import MemoryManipulation

    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    scenario = sample.to_scenario()
    scenario.execution_sequence = [
        MemoryManipulation(command="delete", parameters={"query": "everything"})
    ]
    eval_cfg = scenario.evaluators[0]

    with pytest.raises(ValueError, match="not dispatchable"):
        execute_scenario(
            scenario=scenario,
            agent=_AgentStub(_memory(tmp_path, None)),
            environment=None,
            evaluator=create_evaluator(eval_cfg, task_name=scenario.name),
            eval_cfg=eval_cfg,
            build_ground_truth=lambda s, c: c.to_ground_truth(s),
        )


def test_policy_missing_from_policy_rules_fails_the_scenario(tmp_path):
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
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
        _run(benchmark.samples[0], _memory(tmp_path, privacy))
