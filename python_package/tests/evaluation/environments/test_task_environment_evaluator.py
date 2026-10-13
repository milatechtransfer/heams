from __future__ import annotations

from types import SimpleNamespace

import pytest

from heams.agents.base import AgentResponse
from heams.evaluation.environments.task_environment_evaluator import (
    TaskEnvironmentEvaluator,
    TaskEnvironmentGroundTruth,
)


def _execute_tool_span(tool_name: str, output: object) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"execute_tool {tool_name}",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.args": "{}",
            "gen_ai.output": repr(output),
        },
    )


def _response_with_result(task_id: str, *, success: bool) -> AgentResponse:
    trace = SimpleNamespace(
        spans=[
            _execute_tool_span(
                "finish_task",
                {
                    "status": "success",
                    "result": {"task_id": task_id, "success": success},
                },
            )
        ]
    )
    return AgentResponse(answer="", raw_trace=trace)


def test_task_environment_evaluator_scores_submitted_answer() -> None:
    evaluator = TaskEnvironmentEvaluator(task_name="TASK-ENV")
    audit = evaluator.audit(
        [_response_with_result("alfworld_0", success=True)],
        TaskEnvironmentGroundTruth(
            task_description="TASK-ENV",
            expected_task_ids=["alfworld_0"],
        ),
    )

    assert audit.is_successful is True
    assert audit.accuracy_score == 1.0
    assert audit.passed_keys == ["alfworld_0"]
    assert audit.report_records is not None
    assert len(audit.report_records) == 1
    assert audit.report_records[0].key == "alfworld_0"
    assert audit.report_records[0].passed is True
    assert audit.report_records[0].score == 1.0


def test_task_environment_evaluator_tags_report_records_from_ground_truth() -> None:
    evaluator = TaskEnvironmentEvaluator(task_name="TASK-ENV")
    audit = evaluator.audit(
        [
            _response_with_result("alfworld_0", success=True),
            _response_with_result("alfworld_1", success=False),
        ],
        TaskEnvironmentGroundTruth(
            task_description="TASK-ENV",
            expected_task_ids=["alfworld_0", "alfworld_1"],
            tags={"dataset": "alfworld"},
        ),
    )

    assert audit.report_records is not None
    assert len(audit.report_records) == 2
    by_key = {record.key: record for record in audit.report_records}
    assert by_key["alfworld_0"].passed is True
    assert by_key["alfworld_0"].score == 1.0
    assert by_key["alfworld_0"].tags == {"dataset": "alfworld"}
    assert by_key["alfworld_1"].passed is False
    assert by_key["alfworld_1"].score == 0.0
    assert by_key["alfworld_1"].tags == {"dataset": "alfworld"}


def test_task_environment_evaluator_requires_submit_tool_call() -> None:
    evaluator = TaskEnvironmentEvaluator(task_name="TASK-ENV")
    response = AgentResponse(answer="Firefox", raw_trace=SimpleNamespace(spans=[]))

    with pytest.raises(ValueError, match="terminal task environment results"):
        evaluator.audit(
            [response],
            TaskEnvironmentGroundTruth(
                task_description="TASK-ENV",
                expected_task_ids=["alfworld_0"],
            ),
        )


def test_task_environment_evaluator_results_from_output_parses_supported_shapes() -> (
    None
):
    assert TaskEnvironmentEvaluator.results_from_output(
        {"result": {"task_id": "alfworld_0", "success": True}}
    ) == [{"task_id": "alfworld_0", "success": True}]

    assert TaskEnvironmentEvaluator.results_from_output(
        {
            "completed_tasks": [
                {"task_id": "alfworld_0", "success": True},
                "ignore-me",
                {"task_id": "alfworld_1", "success": False},
            ]
        }
    ) == [
        {"task_id": "alfworld_0", "success": True},
        {"task_id": "alfworld_1", "success": False},
    ]
