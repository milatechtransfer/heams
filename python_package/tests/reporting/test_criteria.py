"""Criterion assignment from benchmark declarations."""

from __future__ import annotations

import pytest

from heams.benchmarks.info import benchmark_info
from heams.reporting.criteria import (
    OVERALL,
    UNDECLARED_DIMENSION,
    criterion_from_tags,
    dimension,
    sample_scores,
)


@pytest.mark.parametrize(
    ("benchmark", "tags", "criterion"),
    [
        ("beam", {"question_type": "abstention", "token_scale": "1M"}, "abstention"),
        ("ee", {"task_type": "erase", "phase": "pre"}, "erase / pre"),
        ("ee", {"phase": "post"}, "post"),
        ("mpbench", {"attack_class": "explicit", "domain": "email"}, "explicit"),
        ("stale", {"dimension": "dim1", "dimension_label": "explicit"}, "explicit"),
        ("locomo", {"category": "temporal"}, "temporal"),
        ("locomo", {}, OVERALL),
    ],
)
def test_records_use_the_benchmark_declared_tags(
    benchmark: str, tags: dict[str, str], criterion: str
) -> None:
    assert criterion_from_tags(benchmark_info(benchmark), tags) == criterion


def test_a_criterion_tag_wins_for_any_benchmark() -> None:
    tags = {"criterion": "custom", "category": "temporal"}

    assert criterion_from_tags(benchmark_info("locomo"), tags) == "custom"
    assert criterion_from_tags(None, tags) == "custom"
    assert criterion_from_tags(None, {"category": "temporal"}) == OVERALL


def test_sections_come_from_the_benchmark_declaration() -> None:
    assert dimension("ci") == "Privacy"
    assert dimension("retention_policy") == "Privacy"
    assert dimension("beam") == "Performance"
    assert dimension("not_a_benchmark") == UNDECLARED_DIMENSION


def test_sample_scores_average_records_per_criterion() -> None:
    audit = {
        "report_records": [
            {"passed": True, "score": 0.5, "tags": {"category": "temporal"}},
            {"passed": True, "tags": {"category": "temporal"}},
            {"passed": False, "tags": {"category": "multi-hop"}},
        ]
    }

    assert sample_scores("locomo", audit) == {"temporal": 0.75, "multi-hop": 0.0}


def test_samples_without_records_fall_back_to_keys_then_accuracy() -> None:
    keys = {"passed_keys": ["ADV_1:explicit"], "failed_keys": ["ADV_2:implicit"]}

    assert sample_scores("mpbench", keys) == {"explicit": 1.0, "implicit": 0.0}
    assert sample_scores("stale", {"accuracy_score": 0.25}) == {OVERALL: 0.25}
    assert sample_scores("stale", {}) == {}
