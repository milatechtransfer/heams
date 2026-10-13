from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.evaluation.trace_utils import iter_tool_calls
from heams.utils import register

_TERMINAL_TOOL_NAMES = {"finish_task", "step_task"}


class TaskEnvironmentGroundTruth(BaseGroundTruth):
    expected_task_ids: list[str]
    success_threshold: float = 1.0
    tags: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Benchmark-supplied dimensions (e.g. dataset name) stamped onto "
            "every report_records entry for this task environment run."
        ),
    )


@register("task_environment_eval")
class TaskEnvironmentEvaluator(BaseEvaluator):
    """Evaluate a stateful task environment from terminal task outputs."""

    def _extract_task_results(
        self, agent_responses: list[AgentResponse]
    ) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for call in iter_tool_calls(agent_responses):
            if call.tool_name not in _TERMINAL_TOOL_NAMES:
                continue
            output = call.output
            if not isinstance(output, Mapping):
                continue
            for result in self.results_from_output(output):
                task_id = result.get("task_id")
                if task_id is None:
                    continue
                results[str(task_id)] = bool(result.get("success", False))

        if not results:
            raise ValueError(
                "Agent trace did not contain terminal task environment results."
            )
        return results

    @staticmethod
    def results_from_output(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        result = output.get("result")
        if isinstance(result, Mapping):
            return [result]
        completed = output.get("completed_tasks")
        if isinstance(completed, list):
            return [item for item in completed if isinstance(item, Mapping)]
        return []

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: TaskEnvironmentGroundTruth,
    ) -> AuditRecord:
        observed = self._extract_task_results(agent_responses)
        passed_keys = [
            task_id
            for task_id in ground_truth.expected_task_ids
            if observed.get(task_id) is True
        ]
        failed_keys = [
            task_id
            for task_id in ground_truth.expected_task_ids
            if observed.get(task_id) is not True
        ]
        total = len(passed_keys) + len(failed_keys)
        accuracy = len(passed_keys) / total if total else 0.0
        records = [
            EvaluationRecord(
                key=task_id,
                passed=task_id in passed_keys,
                score=1.0 if task_id in passed_keys else 0.0,
                tags=ground_truth.tags,
            )
            for task_id in ground_truth.expected_task_ids
        ]
        self._last_audit = AuditRecord(
            is_successful=accuracy >= ground_truth.success_threshold,
            accuracy_score=accuracy,
            passed_keys=passed_keys,
            failed_keys=failed_keys,
            audit_notes=f"Evaluated {total} environment tasks.",
            report_records=records,
        )
        return self._last_audit
