"""Criterion-level scoring of judge records, driven by benchmark declarations.

Each judge record is assigned to a criterion:

1. its ``criterion`` tag, when the judge sets one (any benchmark can name its
   criteria this way without further declarations);
2. else the first of the benchmark's ``BenchmarkInfo.criterion_tags`` groups
   whose tags are all set ("erase / pre" for ``("task_type", "phase")``);
3. else ``Overall``.

A sample's score for a criterion is the mean of its records' scores
(``passed`` when a record has no score). Samples without records fall back
to their audit keys (``BenchmarkInfo.criterion_from_key``), then to their
overall accuracy.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from heams.benchmarks.info import BenchmarkInfo, benchmark_info

OVERALL = "Overall"
#: Judge-record tag that names a record's criterion directly.
CRITERION_TAG = "criterion"
#: Section for results of a benchmark this version does not declare.
UNDECLARED_DIMENSION = "Other"


def dimension(benchmark_type: str) -> str:
    """Report section of a benchmark type, as the benchmark declares it."""
    info = benchmark_info(benchmark_type)
    return info.dimension.value if info else UNDECLARED_DIMENSION


def criterion_from_tags(info: BenchmarkInfo | None, tags: dict[str, Any]) -> str:
    if tags.get(CRITERION_TAG) not in (None, ""):
        return str(tags[CRITERION_TAG])
    for group in info.criterion_tags if info else ():
        if all(tags.get(tag) not in (None, "") for tag in group):
            return " / ".join(str(tags[tag]) for tag in group)
    return OVERALL


def criterion_from_key(info: BenchmarkInfo | None, key: str) -> str:
    if info and info.criterion_from_key:
        return info.criterion_from_key(key) or OVERALL
    return OVERALL


def sample_scores(benchmark_type: str, audit: dict[str, Any]) -> dict[str, float]:
    """``{criterion: score in [0, 1]}`` for one saved ``RunResult.audit``."""
    info = benchmark_info(benchmark_type)
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in audit.get("report_records") or []:
        score = record.get("score")
        grouped[criterion_from_tags(info, record.get("tags") or {})].append(
            float(score) if score is not None else float(bool(record.get("passed")))
        )
    if not grouped:
        for key in audit.get("passed_keys") or []:
            grouped[criterion_from_key(info, str(key))].append(1.0)
        for key in audit.get("failed_keys") or []:
            grouped[criterion_from_key(info, str(key))].append(0.0)
    if grouped:
        return {name: statistics.mean(values) for name, values in grouped.items()}
    accuracy = audit.get("accuracy_score")
    return {OVERALL: float(accuracy)} if accuracy is not None else {}


__all__ = [
    "CRITERION_TAG",
    "OVERALL",
    "UNDECLARED_DIMENSION",
    "criterion_from_key",
    "criterion_from_tags",
    "dimension",
    "sample_scores",
]
