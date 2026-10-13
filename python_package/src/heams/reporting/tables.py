"""Markdown table helpers shared by benchmark report renderers."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any


def format_metric_cell(value: object, *, precision: int | None = None) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if precision is not None:
            return f"{value:.{precision}f}"
        return f"{value:.2f}"
    return str(value)


def mean_metric_value(values: Sequence[object]) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return None
    return statistics.mean(numeric_values)


def render_metric_rows(
    metric_specs: Sequence[tuple[str, Any, int | None]],
    summaries: Sequence[object | None],
) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for metric_name, getter, precision in metric_specs:
        rows.append(
            (
                metric_name,
                [
                    format_metric_cell(getter(summary), precision=precision)
                    if summary is not None
                    else "na"
                    for summary in summaries
                ],
            )
        )
    return rows


def render_mean_metric_rows(
    metric_specs: Sequence[tuple[str, Any, int | None]],
    memory_labels: Sequence[str],
    summaries_by_memory: dict[str, list[object]],
) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for metric_name, getter, precision in metric_specs:
        values: list[str] = []
        for memory_label_str in memory_labels:
            mean_value = mean_metric_value(
                [
                    getter(summary)
                    for summary in summaries_by_memory.get(memory_label_str, [])
                ]
            )
            values.append(format_metric_cell(mean_value, precision=precision))
        rows.append((metric_name, values))
    return rows


def render_comparison_table(
    title: str,
    memory_labels: Sequence[str],
    metric_rows: Sequence[tuple[str, list[str]]],
) -> list[str]:
    if not metric_rows:
        return []

    lines = [f"**{title}:**", ""]
    lines.append(f"| Metric | {' | '.join(memory_labels)} |")
    lines.append(f"|---|{'|'.join('---:' for _ in memory_labels)}|")
    for metric_name, values in metric_rows:
        lines.append(f"| {metric_name} | {' | '.join(values)} |")
    return lines


def render_stratified_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric_from: int = 0,
) -> list[str]:
    """Render a free-form stratified breakdown table."""
    if not rows:
        return []

    alignment = [
        "---:" if index >= numeric_from else "---" for index in range(len(headers))
    ]
    lines = [f"**{title}:**", ""]
    lines.append(f"| {' | '.join(headers)} |")
    lines.append(f"|{'|'.join(alignment)}|")
    for row in rows:
        lines.append(f"| {' | '.join(row)} |")
    return lines
