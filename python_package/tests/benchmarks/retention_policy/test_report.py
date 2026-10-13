from __future__ import annotations

from heams.benchmarks.factory import create_benchmark
from heams.benchmarks.retention_policy.metrics import compute_metrics, render_markdown
from heams.evaluation.schemas import AuditRecord
from tests.benchmarks.retention_policy.fixtures import _load_config


def test_metrics_separate_the_two_degenerate_layers():
    """Recall-everything and recall-nothing must not look alike."""
    rows = [
        {"observed": "store", "expected": "drop", "policy_id": "P", "category": "c"},
        {"observed": "store", "expected": "store", "policy_id": "P", "category": "c"},
    ]
    store_all = compute_metrics(rows)
    drop_all = compute_metrics([{**r, "observed": "drop"} for r in rows])

    assert store_all.accuracy == drop_all.accuracy == 0.5
    assert (store_all.leak_rate, store_all.allowed_rate) == (1.0, 1.0)
    assert (drop_all.leak_rate, drop_all.allowed_rate) == (0.0, 0.0)


def test_unscored_rows_excluded_from_rates():
    """A run with no outcome is not evidence of a wrong result."""
    metrics = compute_metrics(
        [
            {"observed": None, "expected": "drop", "policy_id": "P", "category": "c"},
            {"observed": "drop", "expected": "drop", "policy_id": "P", "category": "c"},
        ]
    )
    assert metrics.total == 1
    assert metrics.leak_rate == 0.0


def test_render_markdown_reports_both_rates():
    metrics = compute_metrics(
        [
            {
                "observed": "store",
                "expected": "drop",
                "policy_id": "P",
                "category": "c",
            },
            {
                "observed": "store",
                "expected": "store",
                "policy_id": "P",
                "category": "c",
            },
        ]
    )
    text = "\n".join(render_markdown(metrics))
    assert "Leak rate" in text
    assert "Allowed rate" in text


def test_rollup_renderer_is_registered():
    """Metrics must reach the run report, not just exist in metrics.py."""
    from heams.reporting.benchmark_rollup import _ROLLUP_RENDERERS

    assert "retention_policy" in _ROLLUP_RENDERERS


def test_rollup_recovers_decisions_and_renders(tmp_path):
    from heams.benchmarks.retention_policy.report import (
        collect_decisions,
        render_retention_policy_rollup_sections,
    )
    from heams.evaluation.schemas import EvaluationRecord
    from heams.results import RunResult

    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    scenarios = [s.to_scenario() for s in benchmark.samples]

    # Every fact scored correctly -> observed must equal expected for each fact.
    results = {
        "mem": [
            RunResult(
                scenario_id=s.id,
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    report_records=[
                        EvaluationRecord(
                            key=fact.row_id,
                            passed=True,
                            expected=fact.expected,
                            observed=fact.expected,
                            tags={
                                "policy_id": s.ground_truth.policy_id,
                                "category": fact.category,
                            },
                        )
                        for fact in s.ground_truth.facts
                    ],
                ),
            )
            for s in scenarios
        ]
    }
    decisions = collect_decisions(scenarios, results)
    assert len(decisions) == sum(len(s.ground_truth.facts) for s in scenarios)
    assert all(d["observed"] == d["expected"] for d in decisions)

    text = "\n".join(
        render_retention_policy_rollup_sections(scenarios, results, ["mem"])
    )
    assert "Leak rate" in text and "0.0%" in text

    # A record with no usable outcome must not be counted as a wrong result.
    first_fact = scenarios[0].ground_truth.facts[0]
    unscored = {
        "mem": [
            RunResult(
                scenario_id=scenarios[0].id,
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=False,
                    accuracy_score=0.0,
                    score_available=False,
                    score_unavailable_reason="audit_error",
                    report_records=[
                        EvaluationRecord(
                            key=first_fact.row_id,
                            passed=False,
                            expected=first_fact.expected,
                            observed=None,
                            tags={
                                "policy_id": scenarios[0].ground_truth.policy_id,
                                "category": first_fact.category,
                            },
                        )
                    ],
                ),
            )
        ]
    }
    assert collect_decisions(scenarios, unscored)[0]["observed"] is None
