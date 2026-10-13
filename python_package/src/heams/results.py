from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from heams.evaluation.schemas import (
    AuditRecord,
    ComponentMetricsSummary,
    ComponentPerformanceMetricsSummary,
    SystemMetricsSummary,
)
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationStatus,
)

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario


class ToolResult(BaseModel):
    """Tool output captured for report/debug artefacts."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class AgentTurn(BaseModel):
    """Single user stimulus + agent answer captured during a run."""

    question: str
    answer: str
    target_answer: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    tool_results: list[ToolResult] = Field(default_factory=list)
    context_truncated: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="Whether the model request required context history truncation.",
    )


class StimulusEvent(BaseModel):
    """Progress signal fired once per `UserStimulus` invocation."""

    scenario_id: str
    agent_index: int
    memory_index: int
    evaluator_index: int
    sample_index: int
    stimulus_index: int
    ok: bool


class RunResult(BaseModel):
    """Outcome of a single (scenario, agent, memory, evaluator, sample) run."""

    scenario_id: str
    agent_index: int
    memory_index: int
    evaluator_index: int
    sample_index: int
    audit: AuditRecord
    trace_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Join key for this sample's optional execution trace.",
    )
    transcript: list[AgentTurn] = Field(default_factory=list)
    system_metrics: SystemMetricsSummary | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional system-level efficiency metrics derived from run data and ledger.",
    )
    component_metrics: list[ComponentMetricsSummary] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional component-level efficiency summaries derived from runtime ledgers.",
    )
    component_performance_metrics: list[ComponentPerformanceMetricsSummary] | None = (
        Field(
            default=None,
            exclude_if=lambda value: value is None,
            description="Optional component-level performance summaries derived from recall operations.",
        )
    )
    memory_ledger: list[MemoryOperationEvent] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional compact ledger of memory-system operations for the run.",
    )


def summarize_memory_ledger(
    ledger: list[MemoryOperationEvent],
) -> list[ComponentMetricsSummary]:
    """Build compact component-level summaries from raw memory ledger events."""

    grouped: dict[str, list[MemoryOperationEvent]] = defaultdict(list)
    for event in ledger:
        component = event.component or "unknown"
        grouped[component].append(event)

    summaries: list[ComponentMetricsSummary] = []
    for component, events in grouped.items():
        operation_counts: dict[str, int] = defaultdict(int)
        status_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)
        duration_values: list[float] = []

        for event in events:
            operation_counts[event.operation_type.value] += 1
            status_counts[event.status.value] += 1
            source_counts[event.source.value] += 1
            if (
                event.duration_ms is not None
                and event.status != MemoryOperationStatus.SKIPPED
            ):
                duration_values.append(event.duration_ms)

        summaries.append(
            ComponentMetricsSummary(
                component=component,
                total_operations=len(events),
                successful_operations=sum(
                    1
                    for event in events
                    if event.status == MemoryOperationStatus.SUCCESS
                ),
                failed_operations=sum(
                    1 for event in events if event.status == MemoryOperationStatus.ERROR
                ),
                skipped_operations=sum(
                    1
                    for event in events
                    if event.status == MemoryOperationStatus.SKIPPED
                ),
                operation_counts=dict(operation_counts),
                status_counts=dict(status_counts),
                source_counts=dict(source_counts),
                total_duration_ms=sum(duration_values) if duration_values else None,
            )
        )

    return summaries


def stimulus_indices(scenario: Scenario) -> list[int]:
    from heams.scenarios.actions import UserStimulus

    return [
        index
        for index, step in enumerate(scenario.execution_sequence)
        if isinstance(step, UserStimulus)
    ]


def progress_indices(scenario: Scenario) -> list[int]:
    return stimulus_indices(scenario) or [0]


def make_stimulus_event(
    *,
    scenario_id: str,
    agent_index: int,
    memory_index: int,
    evaluator_index: int,
    sample_index: int,
    stimulus_index: int,
    ok: bool,
) -> StimulusEvent:
    return StimulusEvent(
        scenario_id=scenario_id,
        agent_index=agent_index,
        memory_index=memory_index,
        evaluator_index=evaluator_index,
        sample_index=sample_index,
        stimulus_index=stimulus_index,
        ok=ok,
    )


def backfill_missing_stimulus_events(
    *,
    expected_indices: list[int],
    emitted_indices: set[int],
    emit_stimulus: Callable[[int, bool], None] | None,
    ok: bool,
) -> None:
    if emit_stimulus is None:
        return
    for stimulus_index in expected_indices:
        if stimulus_index not in emitted_indices:
            emit_stimulus(stimulus_index, ok)
