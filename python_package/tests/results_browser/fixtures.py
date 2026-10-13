"""Builders for results-browser tests: saved runs, experiments and traces."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from heams.evaluation.schemas import AuditRecord, EvaluationRecord, SystemMetricsSummary
from heams.results import AgentTurn, RunResult, ToolResult

_CSV_FIELDS = [
    "campaign",
    "experiment",
    "memory_system",
    "benchmark",
    "scope",
    "dimension",
    "criterion",
    "metric_name",
    "aggregation",
    "metric_value",
    "unit",
    "sample_count",
    "excluded_count",
]


def make_run(
    scenario_id: str, *, passed: bool, category: str, scored: bool = True
) -> RunResult:
    """A one-question sample with a judge record, a tool call, and metrics."""
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=passed,
            accuracy_score=1.0 if passed else 0.0,
            passed_keys=["q0"] if passed else [],
            failed_keys=[] if passed else ["q0"],
            score_available=scored,
            report_records=[
                EvaluationRecord(key="q0", passed=passed, tags={"category": category})
            ],
        ),
        transcript=[
            AgentTurn(
                question=f"What happened in {scenario_id}?",
                answer="An answer.",
                tool_results=[
                    ToolResult(
                        tool_name="recall",
                        arguments={"query": "x"},
                        output={"memories": ["m" * 100] * 50},
                    )
                ],
            )
        ],
        system_metrics=SystemMetricsSummary(
            run_duration_ms=1500.0, llm_total_tokens=42
        ),
    )


def write_experiment(
    run_dir: Path,
    name: str,
    *,
    benchmark: str,
    memory: str,
    runs: list[RunResult],
    run_id: str = "20260101-000000-abcd1234",
    limit: int | None = None,
    policy_ids: list[str] | None = None,
    complete: bool = True,
    benchmark_fields: dict[str, Any] | None = None,
    memory_backend: str = "no_memory",
) -> Path:
    """An experiment directory as `heams run` leaves it."""
    experiment_dir = run_dir / name
    experiment_dir.mkdir(parents=True)
    benchmark_config: dict[str, Any] = {"type": benchmark, "limit": limit}
    if benchmark == "retention_policy":
        benchmark_config["policy_ids"] = policy_ids
    benchmark_config.update(benchmark_fields or {})
    (experiment_dir / f"experiment_config.{run_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_name": name,
                "benchmark_config": benchmark_config,
                "agent_config": {"model_id": "openai:gpt-test"},
                "memory_system_config": {
                    "id": memory,
                    "backend_config": {"type": memory_backend},
                },
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "runs.jsonl").write_text(
        "".join(f"{run.model_dump_json()}\n" for run in runs), encoding="utf-8"
    )
    if complete:
        (experiment_dir / "run_status.json").write_text(
            json.dumps({"success": True, "runnable_count": len(runs)}),
            encoding="utf-8",
        )
    return experiment_dir


def write_submetrics(run_dir: Path, rows: list[dict[str, str]]) -> None:
    with (run_dir / "submetrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _CSV_FIELDS})


def trace_event(sequence: int, start_ms: float, **fields: object) -> dict[str, object]:
    return {
        "sequence_index": sequence,
        "phase": "execution",
        "arguments": {},
        "start_offset_ms": start_ms,
        "status": "ok",
        **fields,
    }


def trace_row(trace_id: str, scenario_id: str) -> dict[str, object]:
    """One `execution_traces.jsonl` row (`ExecutionTraceRecord`, schema 1)."""
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "run_id": "20260101-000000-abcd1234",
        "scenario_id": scenario_id,
        "agent_index": 0,
        "memory_index": 0,
        "evaluator_index": 0,
        "sample_index": 0,
        "outcome": "completed",
        "events": [
            trace_event(
                0,
                0.0,
                phase="priming",
                type="memory_priming",
                name="prime_memory_entries",
                content="Melanie painted a sunrise in 2022.",
                duration_ms=3.0,
                step_index=0,
            ),
            trace_event(
                1,
                5.0,
                type="user_stimulus",
                name="user_stimulus",
                content="When did Melanie paint a sunrise?",
                step_index=1,
                stimulus_index=0,
            ),
            trace_event(
                2,
                6.0,
                type="llm_call",
                name="gpt-test",
                content="call recall",
                input_messages=[
                    {"role": "system", "content": "You have a memory."},
                    {"role": "user", "content": "x" * 5_000},
                ],
                usage={"input_tokens": 120, "output_tokens": 8},
                duration_ms=40.0,
                source="tinyagent",
                step_index=1,
                stimulus_index=0,
            ),
            trace_event(
                3,
                50.0,
                type="tool_call",
                name="recall",
                content="Melanie painted a sunrise in 2022.",
                arguments={"query": "sunrise"},
                result={"memories": ["Melanie painted a sunrise in 2022."]},
                duration_ms=12.5,
                status="error",
                error="timeout after retry",
                source="tinyagent",
                step_index=1,
                stimulus_index=0,
            ),
        ],
    }


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())
