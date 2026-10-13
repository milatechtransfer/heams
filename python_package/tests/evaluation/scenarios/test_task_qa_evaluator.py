from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest

from heams.agents.base import AgentResponse
from heams.evaluation.factory import TaskQAEvaluatorConfig, create_evaluator
from heams.evaluation.scenarios.task_qa import (
    TaskQA,
    TaskQAEvaluator,
    TaskQAGroundTruth,
    compare_tool_call_json,
    extract_choice_letter,
    extract_evomemory_aime_answer,
    extract_gpqa_choice_letter,
    extract_mmlu_pro_choice_letter,
    extract_numeric_answer,
    normalize_answer,
    score_bfcl_ast,
)
from heams.scenarios import Scenario, UserStimulus


_BFCL_FUNCTIONS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "dict",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "parameters": {
            "type": "dict",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]
_BFCL_GROUND_TRUTH = json.dumps(
    [
        {
            "get_weather": {
                "location": ["San Francisco", "SF"],
                "unit": ["", "celsius"],
                "days": ["", 3],
            }
        }
    ]
)


def test_task_qa_config_builds_ground_truth_contract() -> None:
    scenario = Scenario(
        id="TASK-QA-TEST",
        name="task qa contract",
        execution_sequence=[
            UserStimulus(
                content="Q1",
                step_evaluation_markers={
                    "expected_answer": "A1",
                    "task_id": "task-1",
                    "metadata": {"source": "unit"},
                    "difficulty": "easy",
                    "domain": "general",
                },
            )
        ],
    )

    ground_truth = TaskQAEvaluatorConfig().to_ground_truth(scenario)

    assert isinstance(ground_truth, TaskQAGroundTruth)
    assert ground_truth.tasks == [
        TaskQA(
            question="Q1",
            expected_answer="A1",
            task_id="task-1",
            metadata={"source": "unit"},
            difficulty="easy",
            domain="general",
        )
    ]


def test_task_qa_config_accepts_target_fallback_key() -> None:
    scenario = Scenario(
        id="TASK-QA-TARGET",
        name="task qa target fallback",
        execution_sequence=[
            UserStimulus(
                content="Q1",
                step_evaluation_markers={"target": "A1"},
            )
        ],
    )

    ground_truth = TaskQAEvaluatorConfig().to_ground_truth(scenario)

    assert ground_truth.tasks[0].expected_answer == "A1"


def test_task_qa_config_missing_answer_error_lists_accepted_keys() -> None:
    scenario = Scenario(
        id="TASK-QA-MISSING",
        name="task qa missing target",
        execution_sequence=[
            UserStimulus(
                content="Q1",
                step_evaluation_markers={},
            )
        ],
    )

    with pytest.raises(ValueError) as error:
        TaskQAEvaluatorConfig().to_ground_truth(scenario)

    message = str(error.value)
    assert "`step_evaluation_markers.expected_answer`" in message
    assert "`step_evaluation_markers.target`" in message


def test_normalize_answer_uses_common_exact_match_convention() -> None:
    assert normalize_answer("The Firefox, browser!") == "firefox browser"


def test_task_qa_evaluator_scores_normalized_exact_matches() -> None:
    evaluator = create_evaluator(TaskQAEvaluatorConfig(), task_name="TASK-QA-TEST")
    assert isinstance(evaluator, TaskQAEvaluator)

    audit = evaluator.audit(
        [AgentResponse(answer="The Firefox browser.", raw_trace=None)],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[
                TaskQA(
                    question="Which browser?",
                    expected_answer="firefox browser",
                    task_id="task-1",
                )
            ],
        ),
    )

    assert audit.is_successful is True
    assert audit.accuracy_score == 1.0
    assert audit.passed_keys == ["task-1"]
    assert audit.failed_keys == []


def test_task_qa_evaluator_scores_multiple_choice_letters() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="multiple_choice_letter"),
        task_name="TASK-QA-TEST",
    )

    audit = evaluator.audit(
        [AgentResponse(answer="The answer is C.", raw_trace=None)],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="C", task_id="mc")],
        ),
    )

    assert audit.is_successful is True
    assert extract_choice_letter("Option D") == "D"


def test_task_qa_evaluator_scores_final_multiple_choice_letter() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="multiple_choice_letter"),
        task_name="TASK-QA-TEST",
    )

    audit = evaluator.audit(
        [
            AgentResponse(
                answer=(
                    "To find the probability, use the overlap integral. "
                    "So the correct answer is D. (8/9)^(1/4)."
                ),
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="D", task_id="mc")],
        ),
    )

    assert audit.is_successful is True


def test_extract_choice_letter_does_not_treat_articles_as_answers() -> None:
    assert extract_choice_letter(
        "A quantum oscillator calculation gives option D."
    ) == ("D")
    assert extract_choice_letter("Thus, the correct option is:\n\n**B. value**") == "B"
    assert extract_choice_letter("Thus, the correct answer is:\n\n**A. $13,724**") == (
        "A"
    )
    assert extract_choice_letter("a long explanation without final choice") == ""


def test_task_qa_evaluator_scores_aime_numeric_answers() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="numeric_answer"),
        task_name="TASK-QA-TEST",
    )

    audit = evaluator.audit(
        [
            AgentResponse(
                answer=(
                    "The positive integers are 1, 9, 10, and 18. "
                    "Thus the sum is \\boxed{38}."
                ),
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="38", task_id="aime")],
        ),
    )

    assert audit.is_successful is True
    assert extract_numeric_answer("Final answer: 038") == "38"


def test_task_qa_evaluator_scores_evomemory_gpqa_labeled_final_choice() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="evomemory_gpqa"),
        task_name="TASK-QA-TEST",
    )

    audit = evaluator.audit(
        [
            AgentResponse(
                answer=(
                    "A calculation gives several intermediate values. The correct "
                    "answer is:\n\n\\boxed{\\text{B. }10^{-4}\\ \\text{eV}}"
                ),
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="B", task_id="gpqa")],
        ),
    )

    assert audit.is_successful is True
    assert extract_gpqa_choice_letter("Therefore D is correct.") == "D"
    assert extract_gpqa_choice_letter("A preliminary option. Final: B.") == "B"
    assert extract_gpqa_choice_letter("The correct answer is E.") == ""


def test_task_qa_evaluator_scores_evomemory_mmlu_pro_choices_through_j() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="evomemory_mmlu_pro"),
        task_name="TASK-QA-TEST",
    )

    passing = evaluator.audit(
        [
            AgentResponse(
                answer="I. Unsafe practices, Distress, Fear, Serious",
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="I", task_id="mmlu")],
        ),
    )
    incorrect = evaluator.audit(
        [
            AgentResponse(
                answer="The correct answer is J. Quantitative.", raw_trace=None
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="I", task_id="mmlu")],
        ),
    )

    assert passing.is_successful is True
    assert incorrect.is_successful is False
    assert extract_mmlu_pro_choice_letter("The correct answer is J.") == "J"
    assert extract_mmlu_pro_choice_letter("Therefore D is correct.") == "D"


def test_task_qa_evaluator_scores_evomemory_aime_answers() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="evomemory_aime"),
        task_name="TASK-QA-TEST",
    )

    audit = evaluator.audit(
        [
            AgentResponse(
                answer="After working it out, the answer is 038.",
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="38", task_id="aime")],
        ),
    )

    assert audit.is_successful is True
    assert extract_evomemory_aime_answer("Thus \\boxed{038}.") == "38"
    assert extract_evomemory_aime_answer("We compute result: 1001 then 038") == ""
    assert extract_evomemory_aime_answer("Thus \\boxed{1001}.") == ""
    assert extract_evomemory_aime_answer("We compute 1001 and then 038") == "38"


def test_task_qa_records_keep_answers_out_of_the_reason() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="evomemory_aime"),
        task_name="TASK-QA-TEST",
    )
    answer = "Let the time be \\(t\\).\nSo\n\\[\n\\boxed{214}\n\\]"

    audit = evaluator.audit(
        [AgentResponse(answer=answer, raw_trace=None)],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer="204", task_id="aime")],
        ),
    )

    (record,) = audit.report_records
    assert record.expected == "204"
    assert record.observed == answer
    assert record.reason == "extracted AIME answer '214', expected '204'"


def test_score_bfcl_ast_matches_with_optional_param_omitted() -> None:
    prediction = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is True


def test_score_bfcl_ast_accepts_alternative_acceptable_value() -> None:
    prediction = '{"name": "get_weather", "arguments": {"location": "SF"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is True


def test_score_bfcl_ast_is_case_insensitive_for_strings() -> None:
    prediction = '{"name": "get_weather", "arguments": {"location": "san francisco"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is True


def test_score_bfcl_ast_coerces_numeric_types() -> None:
    prediction = '{"name": "get_weather", "arguments": {"location": "SF", "days": 3.0}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is True


def test_score_bfcl_ast_rejects_wrong_function_name() -> None:
    prediction = '{"name": "search_web", "arguments": {"query": "weather"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is False


def test_score_bfcl_ast_rejects_wrong_required_value() -> None:
    prediction = '{"name": "get_weather", "arguments": {"location": "Boston"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is False


def test_score_bfcl_ast_rejects_missing_required_param() -> None:
    prediction = '{"name": "get_weather", "arguments": {"unit": "celsius"}}'

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is False


def test_score_bfcl_ast_rejects_hallucinated_param() -> None:
    prediction = (
        '{"name": "get_weather", "arguments": {"location": "SF", "humidity": "high"}}'
    )

    assert score_bfcl_ast(prediction, _BFCL_GROUND_TRUTH, _BFCL_FUNCTIONS) is False


def test_task_qa_evaluator_scores_bfcl_ast_through_audit() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="bfcl_ast"),
        task_name="TASK-QA-TEST",
    )
    prediction = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'

    audit = evaluator.audit(
        [AgentResponse(answer=prediction, raw_trace=None)],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[
                TaskQA(
                    question="Q",
                    expected_answer=_BFCL_GROUND_TRUTH,
                    task_id="tool",
                    metadata={"functions": _BFCL_FUNCTIONS},
                )
            ],
        ),
    )

    assert audit.is_successful is True


def test_task_qa_evaluator_scores_toolbench_json_calls() -> None:
    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="json_tool_call"),
        task_name="TASK-QA-TEST",
    )
    target = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'

    audit = evaluator.audit(
        [
            AgentResponse(
                answer=(
                    'I would call {"name": "get_weather", '
                    '"arguments": {"location": "San Francisco"}}'
                ),
                raw_trace=None,
            )
        ],
        TaskQAGroundTruth(
            task_description="TASK-QA-TEST",
            tasks=[TaskQA(question="Q", expected_answer=target, task_id="tool")],
        ),
    )

    assert audit.is_successful is True
    assert compare_tool_call_json(target, target) is True


def test_task_qa_llm_judge_does_not_filter_judge_warnings() -> None:
    class WarningJudge:
        def run(self, *, context: str, question: str) -> SimpleNamespace:
            warnings.warn(
                "PydanticSerializationUnexpectedValue: Expected `None`",
                UserWarning,
                stacklevel=2,
            )
            return SimpleNamespace(passed=True, reasoning="matches")

    evaluator = create_evaluator(
        TaskQAEvaluatorConfig(match_mode="llm_judge"),
        task_name="TASK-QA-TEST",
    )
    assert isinstance(evaluator, TaskQAEvaluator)
    evaluator._judge = WarningJudge()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        audit = evaluator.audit(
            [AgentResponse(answer="blue", raw_trace=None)],
            TaskQAGroundTruth(
                task_description="TASK-QA-TEST",
                tasks=[TaskQA(question="Q", expected_answer="blue", task_id="judge")],
            ),
        )

    assert audit.is_successful is True
    assert any(
        "PydanticSerializationUnexpectedValue" in str(warning.message)
        for warning in captured
    )


def test_task_qa_evaluator_scores_fenced_toolbench_json_calls() -> None:
    target = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
    prediction = (
        "I would use:\n```json\n"
        '{"arguments": {"location": "San Francisco"}, "name": "get_weather"}'
        "\n```"
    )

    assert compare_tool_call_json(prediction, target) is True


def test_task_qa_evaluator_scores_later_tool_call_json_object() -> None:
    target = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
    prediction = (
        'First consider {"name": "search_docs", "arguments": {"query": "weather"}}. '
        'The final call is {"name": "get_weather", '
        '"arguments": {"location": "San Francisco"}}.'
    )

    assert compare_tool_call_json(prediction, target) is True


def test_task_qa_evaluator_rejects_response_count_mismatch() -> None:
    evaluator = create_evaluator(TaskQAEvaluatorConfig(), task_name="TASK-QA-TEST")
    ground_truth = TaskQAGroundTruth(
        task_description="TASK-QA-TEST",
        tasks=[TaskQA(question="Q1", expected_answer="A1")],
    )

    with pytest.raises(ValueError, match="task_qa audit expected"):
        evaluator.audit(
            [
                AgentResponse(answer="A1", raw_trace=None),
                AgentResponse(answer="extra", raw_trace=None),
            ],
            ground_truth,
        )
