"""Task QA adapter for interleaved EvoMemory corrective-feedback turns."""

from __future__ import annotations

from pydantic import Field

from heams.agents.base import AgentResponse
from heams.evaluation.schemas import AuditRecord
from heams.evaluation.scenarios.task_qa import (
    TaskQAEvaluator,
    TaskQAGroundTruth,
)
from heams.utils import register


class EvoMemorySingleTurnGroundTruth(TaskQAGroundTruth):
    """Task QA targets and their positions in the full response stream."""

    response_indices: list[int] = Field(default_factory=list)


@register("evomemory_single_turn_task_qa")
class EvoMemorySingleTurnTaskQAEvaluator(TaskQAEvaluator):
    """Score only task responses from an interleaved task/feedback stream."""

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: EvoMemorySingleTurnGroundTruth,
    ) -> AuditRecord:
        if len(ground_truth.response_indices) != len(ground_truth.tasks):
            raise ValueError(
                "EvoMemory single-turn ground truth must provide one response "
                "index per task."
            )
        if not ground_truth.tasks:
            raise ValueError("EvoMemory single-turn audit received no tasks.")

        highest_index = max(ground_truth.response_indices)
        if min(ground_truth.response_indices) < 0 or highest_index >= len(
            agent_responses
        ):
            raise ValueError(
                "EvoMemory single-turn audit response indices are outside the "
                f"available {len(agent_responses)} responses."
            )

        task_responses = [
            agent_responses[index] for index in ground_truth.response_indices
        ]
        return super().audit(
            task_responses,
            TaskQAGroundTruth(
                task_description=ground_truth.task_description,
                tasks=ground_truth.tasks,
            ),
        )


__all__ = [
    "EvoMemorySingleTurnGroundTruth",
    "EvoMemorySingleTurnTaskQAEvaluator",
]
