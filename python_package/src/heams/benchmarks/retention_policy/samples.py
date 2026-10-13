"""Turn dataset rows, grouped by policy, into BenchmarkSample instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.evaluation.scenarios.retention_policy import (
    RetentionPolicyFact,
    RetentionPolicyGroundTruth,
)
from heams.scenarios.actions import MemoryManipulation
from heams.utils import settings

REQUIRED_FIELDS = ("id", "policy_id", "policy", "text", "category", "expected")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_absolute():
        path = settings.project_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Retention Policy dataset not found at {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of rows.")
    return rows


def group_rows_by_policy(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("policy_id", "?"), []).append(row)
    return groups


def validate_row(row: dict[str, Any]) -> None:
    """Raise ValueError if `row` can't become a `RetentionPolicyFact`."""
    missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"row is missing required fields: {missing}")
    if row["expected"] not in {"store", "drop"}:
        raise ValueError(f"row {row['id']!r} has expected={row['expected']!r}")


def policy_rows_to_sample(
    rows: list[dict[str, Any]], *, config: RetentionPolicyBenchmarkConfig
) -> BenchmarkSample:
    """Build one sample covering every fact under one policy.

    `rows` must already be validated (`validate_row`) -- this only shapes
    them into a sample, so a single malformed row doesn't cost the whole
    policy's sample the way raising here would. The policy is set on the
    memory system first; then all facts are retained, then all are
    recall-probed, so recall reflects the fully-ingested state rather than
    an artificially small or order-dependent context.
    """
    policy_id = rows[0]["policy_id"]
    policy = rows[0]["policy"]

    facts: list[RetentionPolicyFact] = [
        RetentionPolicyFact(
            row_id=row["id"],
            category=row["category"],
            text=row["text"],
            expected=row["expected"],
        )
        for row in rows
    ]

    execution_sequence: list[MemoryManipulation] = [
        # Policy first: the same memory-system config then serves every
        # policy, rather than one config per policy pinning its own rules.
        MemoryManipulation(
            command="set_retention_policy",
            parameters={"policies": [policy]},
            description=f"enforce {policy_id}: {policy}",
        )
    ]
    execution_sequence += [
        MemoryManipulation(
            command="retain",
            parameters={
                "content": row["text"],
                # No category: it is ground truth.
                "metadata": {"source": "retention_policy_benchmark"},
            },
            description=f"retain chunk {row['id']} under {policy_id}",
        )
        for row in rows
    ]
    execution_sequence += [
        MemoryManipulation(
            command="recall",
            # Self-probe: query with the fact's own text, so recall success
            # means "this fact is findable", independent of any particular
            # backend's ranking behaviour for a different query.
            parameters={"query": row["text"]},
            description=f"recall probe for chunk {row['id']} under {policy_id}",
        )
        for row in rows
    ]

    ground_truth = RetentionPolicyGroundTruth(
        task_description=f"Retention+recall under {policy_id}",
        policy_id=policy_id,
        policy=policy,
        facts=facts,
    )

    return BenchmarkSample(
        id=policy_id,
        name=f"{policy_id}: retain+recall ({len(rows)} facts)",
        tags=["retention_policy", policy_id],
        execution_sequence=execution_sequence,
        evaluators=[config.evaluator],
        ground_truth=ground_truth,
    )


__all__ = [
    "group_rows_by_policy",
    "load_rows",
    "policy_rows_to_sample",
    "validate_row",
]
