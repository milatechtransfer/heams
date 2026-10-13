from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from heams.agents.base import AgentResponse
from heams.evaluation.factory import LongMemEvalLlmJudgeConfig, create_evaluator
from heams.evaluation.scenarios.long_mem_eval_llm_judge import (
    LongMemEvalLlmJudgeEvaluator,
    LongMemEvalQA,
    LongMemEvalQAGroundTruth,
    build_long_mem_eval_judge_prompt,
)
from heams.scenarios import Scenario, UserStimulus


def test_long_mem_eval_config_builds_ground_truth_contract() -> None:
    scenario = Scenario(
        id="LME-TEST",
        name="long mem eval contract",
        execution_sequence=[
            UserStimulus(
                content="Q1",
                step_evaluation_markers={
                    "expected_answer": "A1",
                    "question_id": "LME-Q1",
                    "question_type": "knowledge-update",
                    "question_date": "2024-03-02",
                    "answer_session_ids": ["S1"],
                    "is_abstention": False,
                },
            )
        ],
    )

    ground_truth = LongMemEvalLlmJudgeConfig().to_ground_truth(scenario)

    assert isinstance(ground_truth, LongMemEvalQAGroundTruth)
    assert len(ground_truth.qa) == 1
    assert isinstance(ground_truth.qa[0], LongMemEvalQA)
    assert ground_truth.qa[0].question_id == "LME-Q1"
    assert ground_truth.qa[0].question_type == "knowledge-update"
    assert ground_truth.qa[0].answer_session_ids == ["S1"]


def test_long_mem_eval_evaluator_tags_report_records_from_qa() -> None:
    evaluator = create_evaluator(LongMemEvalLlmJudgeConfig(), task_name="LME-TEST")
    assert isinstance(evaluator, LongMemEvalLlmJudgeEvaluator)
    evaluator._judge = MagicMock()
    evaluator._judge.run.return_value = SimpleNamespace(
        passed=True, reasoning="Matches expected answer."
    )

    ground_truth = LongMemEvalQAGroundTruth(
        task_description="LME-TEST",
        qa=[
            LongMemEvalQA(
                question_id="LME-Q1",
                question="Q1",
                expected_answer="A1",
                question_type="temporal-reasoning",
                is_abstention=True,
            )
        ],
    )

    audit = evaluator.audit(
        [AgentResponse(answer="A1, confirmed", raw_trace=None)],
        ground_truth,
    )

    assert audit.report_records is not None
    assert len(audit.report_records) == 1
    record = audit.report_records[0]
    assert record.passed is True
    assert record.score == 1.0
    assert record.tags == {
        "question_type": "temporal-reasoning",
        "abstention": "abstention",
    }
    assert record.expected == "A1"
    assert record.observed == "A1, confirmed"
    assert record.reason == "Matches expected answer."


def test_long_mem_eval_evaluator_rejects_response_count_mismatch() -> None:
    evaluator = create_evaluator(LongMemEvalLlmJudgeConfig(), task_name="LME-TEST")
    assert isinstance(evaluator, LongMemEvalLlmJudgeEvaluator)
    evaluator._judge = MagicMock()

    ground_truth = LongMemEvalQAGroundTruth(
        task_description="LME-TEST",
        qa=[
            LongMemEvalQA(
                question_id="LME-Q1",
                question="Q1",
                expected_answer="A1",
            )
        ],
    )
    responses = [
        AgentResponse(answer="A1", raw_trace=None),
        AgentResponse(answer="extra", raw_trace=None),
    ]

    with pytest.raises(ValueError, match="LongMemEval audit expected"):
        evaluator.audit(responses, ground_truth)


@pytest.mark.parametrize(
    ("question_type", "expected_snippet"),
    [
        (
            "single-session-user",
            "If the response only contains a subset of the information required",
        ),
        (
            "single-session-assistant",
            "If the response only contains a subset of the information required",
        ),
        (
            "multi-session",
            "If the response only contains a subset of the information required",
        ),
        ("temporal-reasoning", "do not penalize off-by-one errors"),
        (
            "knowledge-update",
            "some previous information along with an updated answer",
        ),
        ("single-session-preference", "rubric for desired personalized response"),
    ],
)
def test_long_mem_eval_judge_prompt_matches_upstream_task_rubrics(
    question_type: str,
    expected_snippet: str,
) -> None:
    prompt = build_long_mem_eval_judge_prompt(
        LongMemEvalQA(
            question_id="LME-Q1",
            question="Q?",
            expected_answer="A",
            question_type=question_type,
        ),
        "hypothesis",
    )

    assert expected_snippet in prompt
    assert "Question: Q?" in prompt
    assert "Model Response: hypothesis" in prompt
    assert "Answer yes or no only." in prompt


def test_long_mem_eval_judge_prompt_handles_abstention() -> None:
    prompt = build_long_mem_eval_judge_prompt(
        LongMemEvalQA(
            question_id="LME-Q1_abs",
            question="Q?",
            expected_answer="Explanation",
            question_type="single-session-user",
            is_abstention=True,
        ),
        "I cannot determine that from memory.",
    )

    assert "unanswerable question" in prompt
    assert "Explanation: Explanation" in prompt
    assert "Does the model correctly identify the question as unanswerable?" in prompt
