from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml

from heams.agents.base import AgentResponse
from heams.benchmarks.beam.judge import (
    BeamLlmJudgeConfig,
    BeamLlmJudgeEvaluator,
    BeamQA,
    BeamQAGroundTruth,
)
from heams.benchmarks.beam.metrics import clamp_nugget_score, compute_kendall_tau_b
from heams.evaluation.schemas import EvaluationRecord
from tests.benchmarks.beam.fixtures import _SMOKE_YAML


def _judge_from_smoke() -> BeamLlmJudgeEvaluator:
    payload = yaml.safe_load(_SMOKE_YAML.read_text(encoding="utf-8"))
    return BeamLlmJudgeEvaluator(
        task_name="beam",
        config=BeamLlmJudgeConfig.model_validate(payload["evaluator"]),
    )


def _event_qa(*nuggets: str) -> BeamQA:
    return BeamQA(
        question="Order?",
        question_type="event_ordering",
        nuggets=list(nuggets),
        token_scale="100K",
    )


def test_clamp_and_kendall_tau_b() -> None:
    assert clamp_nugget_score(0.9) == 1.0
    assert clamp_nugget_score(0.5) == 0.5
    assert clamp_nugget_score(0.1) == 0.0
    assert compute_kendall_tau_b([0, 1, 2], [0, 1, 2]) == 1.0
    assert compute_kendall_tau_b([2, 1, 0], [0, 1, 2]) == -1.0
    assert compute_kendall_tau_b([0], [0, 1]) == 0.0


def test_event_ordering_reversed_complete_list_fails() -> None:
    evaluator = _judge_from_smoke()
    evaluator._event_ordering_predicted = MagicMock(  # type: ignore[method-assign]
        return_value=[2, 1, 0]
    )
    evaluator._judge_nugget = MagicMock()  # type: ignore[method-assign]
    score, _ = evaluator._score_question(
        AgentResponse(answer="c then b then a"),
        _event_qa("a", "b", "c"),
    )
    assert score == pytest.approx(0.0)
    assert score < evaluator.config.pass_threshold
    evaluator._judge_nugget.assert_not_called()


def test_event_ordering_partial_correct_subset_is_k_over_n() -> None:
    evaluator = _judge_from_smoke()
    evaluator._event_ordering_predicted = MagicMock(  # type: ignore[method-assign]
        return_value=[0, 1]
    )
    score, _ = evaluator._score_question(
        AgentResponse(answer="a then b"),
        _event_qa("a", "b", "c", "d"),
    )
    assert score == pytest.approx(0.5)


def test_event_ordering_duplicates_do_not_inflate_recall() -> None:
    evaluator = _judge_from_smoke()
    evaluator._event_ordering_predicted = MagicMock(  # type: ignore[method-assign]
        return_value=[0, 0, 1, 1]
    )
    score, _ = evaluator._score_question(
        AgentResponse(answer="a a b b"),
        _event_qa("a", "b", "c", "d"),
    )
    assert score == pytest.approx(0.5)
    assert score <= 1.0


def test_event_ordering_empty_extract_is_zero() -> None:
    evaluator = _judge_from_smoke()
    evaluator._event_ordering_predicted = MagicMock(  # type: ignore[method-assign]
        return_value=[]
    )
    score, _ = evaluator._score_question(
        AgentResponse(answer=""),
        _event_qa("a", "b", "c"),
    )
    assert score == pytest.approx(0.0)


def test_event_ordering_full_correct_is_one() -> None:
    evaluator = _judge_from_smoke()
    evaluator._event_ordering_predicted = MagicMock(  # type: ignore[method-assign]
        return_value=[0, 1, 2]
    )
    score, _ = evaluator._score_question(
        AgentResponse(answer="a then b then c"),
        _event_qa("a", "b", "c"),
    )
    assert score == pytest.approx(1.0)


def test_judge_aggregates_nugget_pass_threshold() -> None:
    evaluator = _judge_from_smoke()
    evaluator._score_question = MagicMock(  # type: ignore[method-assign]
        side_effect=[(1.0, "stub=1"), (0.0, "stub=0")]
    )
    ground_truth = BeamQAGroundTruth(
        task_description="fixture",
        token_scale="100K",
        qa=[
            BeamQA(
                question="Where?",
                question_type="information_extraction",
                nuggets=["Montreal"],
                token_scale="100K",
            ),
            BeamQA(
                question="Order?",
                question_type="event_ordering",
                nuggets=["a", "b"],
                token_scale="100K",
            ),
        ],
    )
    audit = evaluator.audit(
        [
            AgentResponse(answer="Alex lives in Montreal."),
            AgentResponse(answer="I don't know."),
        ],
        ground_truth,
    )
    assert audit.accuracy_score == pytest.approx(0.5)
    assert audit.passed_keys == ["q0:information_extraction"]
    assert audit.failed_keys == ["q1:event_ordering"]
    assert audit.report_records is not None
    assert isinstance(audit.report_records[0], EvaluationRecord)
    assert audit.report_records[0].tags["question_type"] == "information_extraction"
    assert audit.report_records[0].expected == "Montreal"
    assert audit.report_records[0].observed == "Alex lives in Montreal."
    assert audit.report_records[1].expected == "a\nb"
    assert audit.report_records[1].observed == "I don't know."
    assert audit.is_successful is True
