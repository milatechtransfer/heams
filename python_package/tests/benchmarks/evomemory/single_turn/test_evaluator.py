from __future__ import annotations

from heams.agents.base import AgentResponse
from heams.evaluation.factory import (
    EvoMemorySingleTurnTaskQAEvaluatorConfig,
    create_evaluator,
)
from heams.evaluation.scenarios.evomemory_single_turn_task_qa import (
    EvoMemorySingleTurnTaskQAEvaluator,
)
from heams.scenarios import AgentManipulation, Scenario, UserStimulus


def test_evomemory_single_turn_evaluator_scores_only_task_responses() -> None:
    scenario = Scenario(
        id="EVOMEM-FEEDBACK",
        name="EvoMemory feedback",
        execution_sequence=[
            UserStimulus(
                content="Q1",
                step_evaluation_markers={
                    "evomemory_turn": "task",
                    "expected_answer": "A1",
                    "task_id": "task-1",
                },
            ),
            UserStimulus(
                content="Correct answer: A1",
                step_evaluation_markers={"evomemory_turn": "feedback"},
            ),
            AgentManipulation(command="reset_state"),
            UserStimulus(
                content="Q2",
                step_evaluation_markers={
                    "evomemory_turn": "task",
                    "expected_answer": "A2",
                    "task_id": "task-2",
                },
            ),
            UserStimulus(
                content="Correct answer: A2",
                step_evaluation_markers={"evomemory_turn": "feedback"},
            ),
            AgentManipulation(command="reset_state"),
        ],
    )
    config = EvoMemorySingleTurnTaskQAEvaluatorConfig()
    ground_truth = config.to_ground_truth(scenario)
    evaluator = create_evaluator(config, task_name=scenario.id)

    assert ground_truth.response_indices == [0, 2]
    assert isinstance(evaluator, EvoMemorySingleTurnTaskQAEvaluator)

    audit = evaluator.audit(
        [
            AgentResponse(answer="A1", raw_trace=None),
            AgentResponse(answer="feedback acknowledged", raw_trace=None),
            AgentResponse(answer="wrong", raw_trace=None),
            AgentResponse(answer="feedback acknowledged", raw_trace=None),
        ],
        ground_truth,
    )

    assert audit.accuracy_score == 0.5
    assert audit.passed_keys == ["task-1"]
    assert audit.failed_keys == ["task-2"]
