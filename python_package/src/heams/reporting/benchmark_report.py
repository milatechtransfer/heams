"""Shared markdown helpers for benchmark rollups and experiment reports."""

from __future__ import annotations

import json
from typing import Any, Sequence

from heams.reporting.aggregation import memory_label
from heams.results import RunResult
from heams.scenarios.base import Scenario


def render_run_outcome_comparison(
    scenario: Scenario,
    runs: Sequence[RunResult],
) -> list[str]:
    lines = ["**Run outcomes:**", ""]
    lines.append("| Memory | Success | Accuracy | Failed keys |")
    lines.append("|---|---|---:|---|")
    for run in sorted(runs, key=lambda current: current.memory_index):
        mlabel = memory_label(scenario, run.memory_index)
        failed_keys = ", ".join(run.audit.failed_keys) if run.audit.failed_keys else "-"
        lines.append(
            f"| {mlabel} | {run.audit.is_successful} | "
            f"{run.audit.accuracy_score:.2f} | {failed_keys} |"
        )
    return lines


def render_run_trace_details(
    scenario: Scenario, runs: Sequence[RunResult]
) -> list[str]:
    lines: list[str] = []
    for run in sorted(runs, key=lambda current: current.memory_index):
        mlabel = memory_label(scenario, run.memory_index)
        lines.append(f"<details><summary>{mlabel} trace</summary>")
        lines.append("")
        lines.append(f"- is_successful: {run.audit.is_successful}")
        lines.append(f"- accuracy: {run.audit.accuracy_score:.2f}")
        if run.audit.passed_keys:
            lines.append(f"- passed_keys: {run.audit.passed_keys}")
        if run.audit.failed_keys:
            lines.append(f"- failed_keys: {run.audit.failed_keys}")
        lines.append(f"- notes: {run.audit.audit_notes}")
        lines.append("")
        if run.transcript:
            for turn_idx, turn in enumerate(run.transcript):
                lines.append(f"**Q{turn_idx}:** {turn.question}")
                lines.append("")
                lines.append(f"**A{turn_idx}:** {turn.answer}")
                if turn.target_answer is not None:
                    lines.append("")
                    lines.append(f"**Target{turn_idx}:** {turn.target_answer}")
                lines.append("")
                if turn.tool_results:
                    lines.append("Tool results:")
                    lines.append("")
                    for result in turn.tool_results:
                        lines.append(f"- `{result.tool_name}`")
                        lines.append("")
                        lines.append("```json")
                        lines.append(render_tool_result_output(result.output))
                        lines.append("```")
                        lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def render_tool_result_output(output: Any) -> str:
    return json.dumps(output, indent=2, sort_keys=True, default=json_default)
