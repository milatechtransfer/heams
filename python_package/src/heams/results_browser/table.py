"""Criterion table across memory systems, one source run per benchmark.

Results for one memory system and benchmark can be spread over many runs
(smoke tests, subsets, retries, campaigns). For each pair the table picks one
experiment according to a :class:`Pick` mode, computes its criterion scores
from that experiment's ``runs.jsonl`` index, and reports which run each
column came from, and whether it finished, so mixed sources stay visible.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from heams.benchmarks.info import ReportDimension, RunScope
from heams.reporting.criteria import OVERALL, UNDECLARED_DIMENSION, dimension
from heams.results_browser.catalog import (
    Experiment,
    ResultsCatalog,
    RunIndex,
    distinct_labels,
)

#: Sections in display order: as declared by the framework, then results of
#: benchmarks this version does not declare.
DIMENSIONS = (*(d.value for d in ReportDimension), UNDECLARED_DIMENSION)
_EFFICIENCY = ReportDimension.EFFICIENCY.value
_EFFICIENCY_ROWS = (
    ("Duration", "s"),
    ("LLM tokens", "tokens"),
    ("Memory operations", "operations"),
    ("Memory operation error rate", "%"),
)


class Pick(StrEnum):
    """How to choose one experiment per memory system × benchmark."""

    LARGEST = "largest"
    """Most samples recorded, finished or not; the newest run breaks ties."""
    LATEST = "latest"
    """Newest finished run, whatever its size."""
    LATEST_FULL = "latest_full"
    """Newest finished run over the whole dataset (``RunScope.FULL``)."""


def in_table(experiment: Experiment) -> bool:
    """Whether an experiment can be a table source.

    A run over a different slice of the benchmark (``RunScope.PARTIAL``, as
    the benchmark declares it) cannot stand for the benchmark; it stays
    browsable under Experiments.
    """
    return experiment.recorded_samples > 0 and experiment.scope is not RunScope.PARTIAL


_PICK_KEYS: dict[Pick, Callable[[Experiment], tuple[Any, ...]]] = {
    Pick.LARGEST: lambda e: (e.recorded_samples, e.run_id, e.id),
    Pick.LATEST: lambda e: (e.run_id, e.recorded_samples, e.id),
    Pick.LATEST_FULL: lambda e: (e.run_id, e.recorded_samples, e.id),
}


def select_sources(
    experiments: list[Experiment], pick: Pick
) -> dict[tuple[str, str], Experiment]:
    """The chosen experiment for each (memory system, benchmark)."""
    candidates: dict[tuple[str, str], list[Experiment]] = defaultdict(list)
    for experiment in experiments:
        if not in_table(experiment):
            continue
        # "Latest" would otherwise favour runs that started minutes ago.
        if pick is not Pick.LARGEST and experiment.status != "complete":
            continue
        if pick is Pick.LATEST_FULL and experiment.scope is not RunScope.FULL:
            continue
        candidates[(experiment.memory_system, experiment.benchmark_key)].append(
            experiment
        )
    return {key: max(group, key=_PICK_KEYS[pick]) for key, group in candidates.items()}


def latest_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary per sample identity: the last record written for it.

    A resumed run can append a sample twice. Each summary keeps its own
    ``i``, the position of that record in ``runs.jsonl``.
    """
    return list({tuple(s["identity"]): s for s in summaries}.values())


def experiment_cells(
    benchmark: str, index: RunIndex
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """``{(dimension, criterion, unit): cell}`` for one experiment's samples.

    Samples without a valid score (``score_available`` false) are excluded
    from every score and counted in the ``Overall`` cell's ``excluded``; a
    cell with no scored sample has ``value`` ``None``.
    """
    summaries = index.summaries
    if not summaries:
        return {}
    latest = latest_summaries(summaries)
    scored = [s for s in latest if s["score_available"]]
    bench_dimension = dimension(benchmark)

    scores: dict[str, list[float]] = defaultdict(list)
    for summary in scored:
        for criterion, score in summary["scores"].items():
            scores[criterion].append(score)
    cells: dict[tuple[str, str, str], dict[str, Any]] = {
        (bench_dimension, criterion, "%"): {
            "value": 100.0 * statistics.mean(values),
            "n": len(values),
            "excluded": 0,
        }
        for criterion, values in scores.items()
    }
    # Benchmarks whose records carry criteria get their Overall row from the
    # task accuracy of the scored samples.
    overall_key = (bench_dimension, OVERALL, "%")
    if overall_key not in cells:
        accuracies = [s["accuracy"] for s in scored if s["accuracy"] is not None]
        cells[overall_key] = {
            "value": 100.0 * statistics.mean(accuracies) if accuracies else None,
            "n": len(accuracies),
            "excluded": 0,
        }
    cells[overall_key]["excluded"] = len(latest) - len(scored)

    efficiency: dict[tuple[str, str], list[float]] = defaultdict(list)
    for summary in latest:
        if not summary["has_metrics"]:
            continue
        if summary["duration_s"] is not None:
            efficiency[("Duration", "s")].append(summary["duration_s"])
        efficiency[("LLM tokens", "tokens")].append(float(summary["tokens"] or 0))
        operations = float(summary["memory_ops"] or 0)
        efficiency[("Memory operations", "operations")].append(operations)
        errors = float(summary["memory_errors"] or 0)
        efficiency[("Memory operation error rate", "%")].append(
            100.0 * errors / operations if operations else 0.0
        )
    for (criterion, unit), values in efficiency.items():
        cells[(_EFFICIENCY, criterion, unit)] = {
            "value": statistics.mean(values),
            "n": len(values),
            "excluded": 0,
        }
    return cells


def _source_json(experiment: Experiment) -> dict[str, Any]:
    return {
        "experiment": experiment.id,
        "run_dir": experiment.run_dir,
        "started": experiment.started,
        "samples": experiment.recorded_samples,
        "limit": experiment.limit,
        "subset": experiment.subset,
        "status": experiment.status,
    }


def build_table(
    catalog: ResultsCatalog, pick: Pick = Pick.LARGEST, *, wait: bool = False
) -> dict[str, Any]:
    """Criterion table for `pick`.

    With ``wait=False`` missing ``runs.jsonl`` indexes are requested in the
    background and reported under ``pending``; callers poll until it is empty.
    """
    experiments = list(catalog.experiments.values())
    selected = select_sources(experiments, pick)
    if wait:
        for experiment in selected.values():
            catalog.run_index(experiment.id)
    else:
        # The chosen runs first, then the other modes' so switching is quick.
        others = [
            experiment.id
            for other in Pick
            if other is not pick
            for experiment in select_sources(experiments, other).values()
        ]
        catalog.request_indexes([e.id for e in selected.values()] + others)

    by_pair: dict[tuple[str, str], list[Experiment]] = defaultdict(list)
    for experiment in experiments:
        if in_table(experiment):
            by_pair[(experiment.memory_system, experiment.benchmark_key)].append(
                experiment
            )
    labels = {e.memory_system: e.memory_label for e in experiments}
    columns = distinct_labels({memory: labels[memory] for memory, _ in by_pair})
    # Columns read alphabetically by the names shown, not by config ids.
    memories = sorted(columns, key=lambda memory: (columns[memory].lower(), memory))
    benchmarks = {e.benchmark_key: e for group in by_pair.values() for e in group}

    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    pending: list[str] = []
    sources: list[dict[str, Any]] = []
    for (memory, benchmark_key), group in sorted(by_pair.items()):
        chosen = selected.get((memory, benchmark_key))
        newest_first = sorted(group, key=lambda e: (e.run_id, e.id), reverse=True)
        entry: dict[str, Any] = {
            "memory_system": memory,
            "benchmark": benchmark_key,
            "selected": _source_json(chosen) if chosen else None,
            "state": chosen.status if chosen else newest_first[0].status,
            "alternatives": [
                _source_json(e)
                for e in newest_first
                if e is not chosen and e.recorded_samples
            ],
        }
        sources.append(entry)
        if chosen is None:
            continue
        index = catalog.indexed(chosen.id)
        if index is None:
            error = catalog.index_error(chosen.id)
            entry["state"] = "unreadable" if error else "indexing"
            if not error:
                pending.append(chosen.id)
            continue
        cells = experiment_cells(chosen.benchmark, index)
        for (dim, criterion, unit), cell in cells.items():
            row = rows.setdefault(
                (dim, benchmark_key, criterion, unit),
                {
                    "dimension": dim,
                    "benchmark": benchmark_key,
                    "benchmark_label": chosen.benchmark_label,
                    "criterion": criterion,
                    "unit": unit,
                    "cells": {},
                },
            )
            row["cells"][memory] = {**cell, "experiment": chosen.id}

    def benchmark_order(key: str) -> tuple[Any, ...]:
        # Declared context size first (smaller histories first), then name.
        experiment = benchmarks[key]
        info = experiment.info
        tokens = info.tokens(experiment.variant) if info else None
        return (tokens is None, tokens or 0, experiment.benchmark_label)

    ordered = sorted(
        rows.items(),
        key=lambda item: (
            DIMENSIONS.index(item[0][0])
            if item[0][0] in DIMENSIONS
            else len(DIMENSIONS),
            benchmark_order(item[0][1]),
            item[0][2] != OVERALL,
            [name for name, _ in _EFFICIENCY_ROWS].index(item[0][2])
            if item[0][0] == _EFFICIENCY
            else 0,
            item[0][2],
        ),
    )
    return {
        "pick": pick.value,
        "dimensions": list(DIMENSIONS),
        "memory_systems": memories,
        "memory_labels": {memory: columns[memory] for memory in memories},
        "benchmark_labels": {key: e.benchmark_label for key, e in benchmarks.items()},
        "rows": [row for _, row in ordered],
        "sources": sources,
        "pending": pending,
    }


__all__ = ["Pick", "build_table", "in_table", "select_sources"]
