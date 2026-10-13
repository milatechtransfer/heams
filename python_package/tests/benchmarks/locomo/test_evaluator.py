from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from heams.evaluation.llm_judge import LlmJudge

from heams.agents.base import AgentResponse
from heams.benchmarks.locomo.judge import (
    LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE,
    LocomoLlmJudgeConfig,
    LocomoLlmJudgeEvaluator,
    LocomoQA,
    LocomoQAGroundTruth,
    _LocomoJudgeOutput,
    build_locomo_judge_prompt,
    is_scored_locomo_category,
    normalize_locomo_category,
)
from heams.evaluation.factory import create_evaluator
from heams.scenarios import Scenario, UserStimulus
from tests.benchmarks.locomo.fixtures import load_locomo_judge_config


def test_locomo_config_reads_scenario_ground_truth() -> None:
    ground_truth = LocomoQAGroundTruth(
        task_description="locomo contract",
        qa=[
            LocomoQA(
                question="Q1",
                expected_answer="A1",
                evidence_dia_ids=["D1"],
                evidence_texts=["Alice: I moved to Montreal."],
                category=4,
            )
        ],
    )
    scenario = Scenario(
        id="LOC-TEST",
        name="locomo contract",
        execution_sequence=[
            UserStimulus(
                content="Question: Q1 Short answer:",
                step_evaluation_markers={"question": "Q1"},
            )
        ],
        ground_truth=ground_truth,
    )

    resolved = load_locomo_judge_config().to_ground_truth(scenario)

    assert resolved is ground_truth
    assert len(resolved.qa) == 1
    assert isinstance(resolved.qa[0], LocomoQA)
    assert resolved.qa[0].question == "Q1"
    assert resolved.qa[0].evidence_texts == ["Alice: I moved to Montreal."]
    assert resolved.qa[0].category == 4


def test_locomo_config_requires_ground_truth() -> None:
    scenario = Scenario(
        id="LOC-TEST",
        name="locomo contract",
        execution_sequence=[UserStimulus(content="Q")],
    )
    with pytest.raises(ValueError, match="missing ground_truth"):
        load_locomo_judge_config().to_ground_truth(scenario)


def test_locomo_evaluator_rejects_response_count_mismatch() -> None:
    evaluator = create_evaluator(load_locomo_judge_config(), task_name="LOC-TEST")
    assert isinstance(evaluator, LocomoLlmJudgeEvaluator)
    evaluator._judge = MagicMock(spec=LlmJudge)

    ground_truth = LocomoQAGroundTruth(
        task_description="LOC-TEST",
        qa=[LocomoQA(question="Q1", expected_answer="A1")],
    )
    responses = [
        AgentResponse(answer="A1", raw_trace=None),
        AgentResponse(answer="extra", raw_trace=None),
    ]

    with pytest.raises(ValueError, match="LOCOMO audit expected"):
        evaluator.audit(responses, ground_truth)


def test_locomo_category_normalization_and_scored_predicate_share_contract() -> None:
    assert normalize_locomo_category(5) == 5
    assert normalize_locomo_category("5") == 5
    assert normalize_locomo_category("adversarial") == 5
    assert normalize_locomo_category("single-hop") == 4
    assert is_scored_locomo_category(None) is True
    assert is_scored_locomo_category("adversarial") is False
    assert is_scored_locomo_category(5) is False


def test_locomo_judge_prompt_includes_category_and_evidence_context() -> None:
    judge_config = load_locomo_judge_config()
    prompt = build_locomo_judge_prompt(
        LocomoQA(
            question="What desserts did Alice order?",
            expected_answer="chocolate raspberry tart; lemon sorbet",
            evidence_dia_ids=["D1:1", "D2:3"],
            category=4,
        ),
        "Alice ordered the lemon sorbet.",
        config=judge_config,
    )

    assert (
        "1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT."
        in prompt
    )
    assert (
        "4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT."
        in prompt
    )
    assert "Gold answer: chocolate raspberry tart; lemon sorbet" in prompt
    assert "Generated answer: Alice ordered the lemon sorbet." in prompt


def test_locomo_judge_prompt_includes_evidence_and_preprocesses_category_3() -> None:
    judge_config = load_locomo_judge_config()
    prompt = build_locomo_judge_prompt(
        LocomoQA(
            question="Would Caroline adopt?",
            expected_answer="yes; she has researched agencies",
            evidence_dia_ids=["D1:1"],
            evidence_texts=["Caroline: I researched adoption agencies this week."],
            category=3,
        ),
        "Yes, she seems ready to adopt.",
        config=judge_config,
    )

    assert "## Evidence (actual conversation messages containing the answer)" in prompt
    assert "Caroline: I researched adoption agencies this week." in prompt
    assert (
        "5. **EVIDENCE SUPPORTS ANSWER**: If the evidence corroborates the generated answer, mark CORRECT"
        in prompt
    )
    assert "6. **SEMANTIC OVERLAP**" in prompt
    assert "Gold answer: yes" in prompt
    assert "Gold answer: yes; she has researched agencies" not in prompt


def test_locomo_evaluator_configures_llm_judge_wrapper() -> None:
    judge_config = load_locomo_judge_config()
    evaluator = create_evaluator(judge_config, task_name="LOC-TEST")
    assert isinstance(evaluator, LocomoLlmJudgeEvaluator)

    judge = evaluator._get_judge()

    assert isinstance(judge, LlmJudge)
    assert judge.output_type is _LocomoJudgeOutput
    assert judge.model_args["temperature"] == 0
    assert judge.model_args["response_format"] is _LocomoJudgeOutput
    assert judge.system_prompt == judge_config.judge_system_prompt


def test_locomo_evaluator_uses_evidence_fed_llm_judge_wrapper() -> None:
    judge_config = load_locomo_judge_config()
    evaluator = create_evaluator(judge_config, task_name="LOC-TEST")
    assert isinstance(evaluator, LocomoLlmJudgeEvaluator)
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = _LocomoJudgeOutput(
        label="CORRECT",
        reasoning="Evidence supports the answer.",
    )

    ground_truth = LocomoQAGroundTruth(
        task_description="LOC-TEST",
        qa=[
            LocomoQA(
                question="When did Caroline go to the LGBTQ support group?",
                expected_answer="May 7, 2023",
                evidence_dia_ids=["D1:3"],
                evidence_texts=[
                    "Caroline: I went to a LGBTQ support group yesterday and it was powerful."
                ],
                category=2,
            )
        ],
    )

    audit = evaluator.audit(
        [AgentResponse(answer="May 10, 2023", raw_trace=None)],
        ground_truth,
    )

    assert audit.accuracy_score == 1.0
    assert audit.passed_keys == ["q0:2"]
    evaluator._judge.run.assert_called_once_with(
        context=build_locomo_judge_prompt(
            ground_truth.qa[0], "May 10, 2023", config=judge_config
        ),
        question="",
        prompt_template=LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE,
    )
    assert audit.report_records is not None
    assert len(audit.report_records) == 1
    record = audit.report_records[0]
    assert record.key == "q0:2"
    assert record.passed is True
    assert record.score == 1.0
    assert record.tags == {"category": "temporal"}
    assert record.expected == "May 7, 2023"
    assert record.observed == "May 10, 2023"


def test_locomo_evaluator_report_record_expected_matches_preprocessed_answer() -> None:
    """`expected` must reflect what the judge actually saw, not the raw gold answer."""
    judge_config = load_locomo_judge_config()
    evaluator = create_evaluator(judge_config, task_name="LOC-TEST")
    assert isinstance(evaluator, LocomoLlmJudgeEvaluator)
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = _LocomoJudgeOutput(label="CORRECT")

    ground_truth = LocomoQAGroundTruth(
        task_description="LOC-TEST",
        qa=[
            LocomoQA(
                question="What desserts did Alice order?",
                expected_answer="chocolate raspberry tart; lemon sorbet",
                category=3,
            )
        ],
    )

    audit = evaluator.audit(
        [AgentResponse(answer="chocolate raspberry tart", raw_trace=None)],
        ground_truth,
    )

    record = audit.report_records[0]
    assert record.expected == "chocolate raspberry tart"
    assert record.observed == "chocolate raspberry tart"


def test_locomo_evaluator_tags_report_records_with_category_label() -> None:
    judge_config = load_locomo_judge_config()
    evaluator = create_evaluator(judge_config, task_name="LOC-TEST")
    assert isinstance(evaluator, LocomoLlmJudgeEvaluator)
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = _LocomoJudgeOutput(
        label="WRONG",
        reasoning="No match found.",
    )

    ground_truth = LocomoQAGroundTruth(
        task_description="LOC-TEST",
        qa=[
            LocomoQA(question="Q1", expected_answer="A1", category=4),
        ],
    )

    audit = evaluator.audit(
        [AgentResponse(answer="wrong answer", raw_trace=None)],
        ground_truth,
    )

    assert audit.report_records is not None
    record = audit.report_records[0]
    assert record.passed is False
    assert record.score == 0.0
    assert record.tags == {"category": "single-hop"}
    assert record.reason == "No match found."


def test_locomo_judge_config_requires_prompt_fields() -> None:
    with pytest.raises(Exception):
        LocomoLlmJudgeConfig()
